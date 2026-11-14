"""
procesamiento_tdem.py
==================================================================
De los barridos de campo a la curva de decaimiento de cada momento:
apilado robusto, error del apilado, gates útiles, resistividad aparente
de tiempo tardío y curva combinada LM + HM.

Apilado robusto (por momento y por gate), en tres pasos
--------------------------------------------------------------
Cada barrido del archivo ya es un promedio de cientos de transitorios.
Para cada momento se tienen N barridos (624 en un sondeo típico) de G
gates cada uno.

**Paso 1 · Barridos completos (Isolation Forest + LOF).** Cada barrido
es un vector de G desviaciones normalizadas respecto a la mediana de su
gate (en log₁₀ ρa donde el gate tiene señal; en dB/dt lineal donde es
ruido, ver abajo). Isolation Forest y Local Outlier Factor dan un puntaje
de anomalía por barrido. El umbral **no** es una fracción fija de
contaminación: un barrido es atípico para un detector si su puntaje pasa
de mediana + k·1.4826·MAD de los puntajes de ese sondeo. Por omisión se
descarta el barrido cuando **ambos** detectores coinciden. Con datos de
campo, la contaminación automática de Isolation Forest marcaba 5–14 % de
los barridos, que eran solo la cola normal del ruido gaussiano; quitarlos
subestima el error y sesga la media. El umbral adaptativo deja en ~1 %
los barridos que de verdad se apartan (ráfagas, caídas de corriente).

**Paso 2 · Valores sueltos (MAD por gate, k = 3) en log₁₀ ρa.** Con los
barridos que quedan, en cada gate:

- *Régimen log* (la señal domina: mediana / σ_robusta del barrido
  individual ≥ 3). Se calcula la ρa tardía de **cada barrido** y la
  mediana y la MAD de log₁₀ ρa. Se descartan los barridos con
  |log₁₀ ρa − mediana| > 3 · 1.4826 · MAD. El corte es *relativo* (un
  factor de ρa), así que no depende de si el subsuelo es de 5 o de
  500 Ω·m. Un valor ≤ 0 en este régimen es un pico y se descarta.
- *Régimen lineal* (el ruido domina; muchos barridos individuales son
  negativos y su log no existe). Mediana y MAD de dB/dt lineal y el
  mismo corte |z| > 3. Promediar en log aquí sesgaría la curva: con datos
  de campo, la media de log de los barridos positivos daba dB/dt 2 a 40
  veces mayor y ρa 2 a 10 veces menor en los gates tardíos.

**Paso 3 · Valor apilado.** La **media lineal** de dB/dt de los barridos
conservados y su error estándar (σ/√n). Es el estimador insesgado para
ruido aditivo y el único que sigue funcionando al acercarse al piso de
ruido. La ρa se calcula después, sobre la curva apilada. Para leer la
incertidumbre en resistividad se reportan, en log₁₀ ρa:

- ``sigma_log10_rho_barrido``: dispersión de log₁₀ ρa entre barridos
  (solo régimen log);
- ``sigma_log10_rho``: incertidumbre de la ρa apilada,
  σ(log₁₀ ρa) = ⅔ · (error estándar / dB/dt) / ln 10.

El error estándar solo baja el ruido **aleatorio**. El ruido coherente
(acoplamiento con metal, armónicos de la red) está en todos los barridos
por igual: ningún detector de atípicos lo ve ni el apilado lo cancela.
Por eso la inversión agrega un piso de error relativo (ver
:mod:`inversion_tdem`) y el usuario puede excluir gates.

Gate útil
------------
Positivo, con SNR = |valor| / error estándar ≥ 3, y con todos los
gates anteriores del mismo momento también útiles: al cruzar el piso de
ruido lo que sigue ya no se usa, aunque algún gate suelto vuelva a salir
bien.

Repuntes
-----------
Una curva dB/dt de lazo central sobre un medio 1D decae siempre. Si el
valor apilado **sube** más de 3 σ entre dos gates útiles, el gate que
sube se marca como ``repunte`` y, por omisión, no se invierte.

Curva combinada LM + HM
--------------------------
Para ver e interpretar (ρa, pseudosecciones) se arma una sola curva: el
LM desde su primer gate útil hasta el tiempo de cambio, y el HM de ahí
en adelante. El cambio ocurre en el primer gate del HM que (a) ya está
fuera de la influencia de su rampa y (b) tiene un error relativo menor o
igual que el del LM en ese tiempo. La **inversión no usa esta curva**:
invierte LM y HM juntos, cada uno con su forma de onda.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("tdem.procesamiento")

MU0 = 4e-7 * np.pi
FACTOR_MAD = 1.4826


@dataclass
class ConfigProcesamiento:
    """Parámetros del apilado y de la selección de gates."""

    # Paso 1: barridos completos
    deteccion_barridos: str = "if_lof"  # "if_lof" | "ninguna"
    consenso: str = "ambos"            # "ambos" (los dos detectores) | "cualquiera"
    k_puntaje: float = 3.0             # umbral = mediana + k·1.4826·MAD de los puntajes
    max_frac_descartados: float = 0.20  # tope de seguridad de barridos descartados
    vecinos_lof: int = 35
    semilla: int = 0
    # Paso 2: valores sueltos por gate
    k_mad: float = 3.0                 # |z robusto| > k → valor atípico en ese gate
    snr_regimen_log: float = 3.0       # mediana/σ por barrido ≥ esto → MAD en log₁₀ ρa
    # Selección de gates
    snr_min: float = 3.0               # SNR mínimo para que un gate sea útil
    factor_rampa: float = 2.0          # t < factor · rampa → gate con efecto de rampa
    excluir_repuntes: bool = True      # no invertir gates donde la señal sube
    excluir_efecto_rampa: bool = False  # no invertir gates con efecto de rampa

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ======================================================================
# Apilado
# ======================================================================


def _robusto(x: np.ndarray) -> tuple[float, float]:
    """Mediana y σ robusta (1.4826·MAD) ignorando NaN."""
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    else:
        pass
    mediana = float(np.median(x))
    return mediana, FACTOR_MAD * float(np.median(np.abs(x - mediana)))


def _log_rho_barridos(tiempo: float, valores: np.ndarray, area_lazo: Optional[float],
                      vueltas: float) -> np.ndarray:
    """log₁₀ ρa de cada barrido (NaN donde dB/dt ≤ 0).

    Sin área del lazo se usa −⅔·log₁₀ dB/dt: difiere de log₁₀ ρa solo por
    una constante del gate, así que la mediana, la MAD y los z robustos
    son idénticos.
    """
    positivos = np.where(valores > 0, valores, np.nan)
    if area_lazo is None:
        return -2.0 / 3.0 * np.log10(positivos)
    else:
        return np.log10(rho_a_tardia(np.full(len(valores), tiempo), positivos, area_lazo, vueltas))


def _regimen(valores: np.ndarray, snr_log: float) -> str:
    """'log' si la señal domina en el barrido individual; si no, 'lineal'."""
    mediana, sigma = _robusto(valores)
    if np.isfinite(mediana) and mediana > 0 and (sigma == 0 or mediana / sigma >= snr_log):
        return "log"
    else:
        return "lineal"


def _desviaciones_normalizadas(tabla: pd.DataFrame, tiempos: pd.Series, config: ConfigProcesamiento,
                               area_lazo: Optional[float], vueltas: float) -> np.ndarray:
    """Matriz barridos × gates de z robustos (log₁₀ ρa o lineal según el gate).

    Es la entrada de Isolation Forest y LOF: normalizar por gate hace que
    un gate temprano (dB/dt ~1e-4) y uno tardío (~1e-9) pesen igual.
    """
    columnas = []
    for gate in tabla.columns:
        v = tabla[gate].to_numpy(float)
        if _regimen(v, config.snr_regimen_log) == "log":
            x = _log_rho_barridos(float(tiempos[gate]), v, area_lazo, vueltas)
            mediana, sigma = _robusto(x)
            z = (x - mediana) / sigma if sigma > 0 else np.zeros_like(x)
            z = np.where(np.isfinite(v) & (v <= 0), 20.0, z)  # un negativo con señal clara es un pico
        else:
            mediana, sigma = _robusto(v)
            z = (v - mediana) / sigma if sigma > 0 else np.zeros_like(v)
        columnas.append(z)
    return np.clip(np.nan_to_num(np.column_stack(columnas), nan=0.0), -20.0, 20.0)


def _umbral_adaptativo(puntaje: np.ndarray, k: float) -> float:
    mediana, sigma = _robusto(puntaje)
    return mediana + k * sigma


def puntuar_barridos(gates: pd.DataFrame, barridos: pd.DataFrame, config: ConfigProcesamiento,
                     area_lazo: Optional[float] = None, vueltas: float = 1.0) -> pd.DataFrame:
    """Paso 1: puntaje de anomalía de cada barrido con Isolation Forest y LOF.

    Returns
    -------
    DataFrame
        ``momento, barrido, indice, puntaje_if, umbral_if, atipico_if,
        puntaje_lof, umbral_lof, atipico_lof, rms_z, descartado``. Puntajes
        más altos = más anómalo.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    datos = gates.merge(barridos[["barrido", "es_ruido"]], on="barrido")
    datos = datos[~datos["es_ruido"].astype(bool)]
    salida = []
    for momento, grupo in datos.groupby("momento"):
        tabla = grupo.pivot_table(index="barrido", columns="gate", values="dbdt", aggfunc="first").sort_index()
        tiempos = grupo.groupby("gate")["tiempo"].median()
        n = len(tabla)
        base = pd.DataFrame({"momento": momento, "barrido": tabla.index.to_numpy(), "indice": np.arange(1, n + 1)})
        if config.deteccion_barridos != "if_lof" or n < 20:
            salida.append(base.assign(puntaje_if=np.nan, umbral_if=np.nan, atipico_if=False, puntaje_lof=np.nan,
                                      umbral_lof=np.nan, atipico_lof=False, rms_z=np.nan, descartado=False))
            continue
        else:
            pass
        z = _desviaciones_normalizadas(tabla, tiempos, config, area_lazo, vueltas)
        bosque = IsolationForest(n_estimators=200, random_state=config.semilla).fit(z)
        puntaje_if = -bosque.score_samples(z)
        # LOF degenera si el vecindario es casi toda la muestra (todos los
        # puntos quedan con LOF ≈ 1, incluso el atípico): como mucho n/4.
        vecinos = max(5, min(int(config.vecinos_lof), n // 4))
        lof = LocalOutlierFactor(n_neighbors=vecinos).fit(z)
        puntaje_lof = -lof.negative_outlier_factor_
        umbral_if = _umbral_adaptativo(puntaje_if, config.k_puntaje)
        umbral_lof = _umbral_adaptativo(puntaje_lof, config.k_puntaje)
        atipico_if = puntaje_if > umbral_if
        atipico_lof = puntaje_lof > umbral_lof
        if config.consenso == "cualquiera":
            descartado = atipico_if | atipico_lof
        else:
            descartado = atipico_if & atipico_lof
        # Tope de seguridad: si se marca demasiado no es contaminación sino
        # un sondeo heterogéneo (deriva, cambio de corriente); se conservan
        # solo los más anómalos por el rango combinado de ambos puntajes.
        tope = int(np.floor(config.max_frac_descartados * n))
        if descartado.sum() > tope:
            rango = pd.Series(puntaje_if).rank().to_numpy() + pd.Series(puntaje_lof).rank().to_numpy()
            elegidos = np.argsort(-np.where(descartado, rango, -np.inf))[:tope]
            descartado = np.zeros(n, dtype=bool)
            descartado[elegidos] = True
            logger.warning("%s: más del %.0f %% de barridos marcados; se descartan solo los %d más anómalos.",
                           momento, 100 * config.max_frac_descartados, tope)
        else:
            pass
        salida.append(base.assign(puntaje_if=puntaje_if, umbral_if=umbral_if, atipico_if=atipico_if,
                                  puntaje_lof=puntaje_lof, umbral_lof=umbral_lof, atipico_lof=atipico_lof,
                                  rms_z=np.sqrt((z ** 2).mean(axis=1)), descartado=descartado))
    if not salida:
        return pd.DataFrame(columns=["momento", "barrido", "indice", "puntaje_if", "umbral_if", "atipico_if",
                                     "puntaje_lof", "umbral_lof", "atipico_lof", "rms_z", "descartado"])
    else:
        return pd.concat(salida, ignore_index=True)


def _apilar_gate(valores: np.ndarray, tiempo: float, config: ConfigProcesamiento,
                 area_lazo: Optional[float], vueltas: float) -> dict[str, Any]:
    """Pasos 2 y 3 en un gate: MAD (log₁₀ ρa o lineal) y media lineal."""
    valores = valores[np.isfinite(valores)]
    n = len(valores)
    if n == 0:
        return {"n_stacks": 0}
    else:
        pass
    regimen = _regimen(valores, config.snr_regimen_log)
    mediana, sigma = _robusto(valores)
    sigma_log_barrido = np.nan
    rho_mediana_barridos = np.nan
    if regimen == "log":
        x = _log_rho_barridos(tiempo, valores, area_lazo, vueltas)
        mediana_x, sigma_x = _robusto(x)
        if sigma_x > 0:
            dentro = np.isfinite(x) & (np.abs(x - mediana_x) <= config.k_mad * sigma_x)
        else:
            dentro = np.isfinite(x)
        if area_lazo is not None:
            rho_mediana_barridos = 10 ** mediana_x
        else:
            pass
        sigma_log_barrido = float(np.std(x[dentro], ddof=1)) if dentro.sum() > 1 else np.nan
    else:
        if sigma > 0:
            dentro = np.abs(valores - mediana) <= config.k_mad * sigma
        else:
            dentro = np.ones(n, dtype=bool)
    conservados = valores[dentro]
    desviacion = float(conservados.std(ddof=1)) if len(conservados) > 1 else np.nan
    media = float(conservados.mean()) if len(conservados) else np.nan
    error = desviacion / np.sqrt(len(conservados)) if len(conservados) > 1 else np.nan
    return {
        "n_stacks": n,
        "n_atipicos": int(n - dentro.sum()),
        "regimen": regimen,
        "mediana": mediana,
        "sigma_robusta": sigma,
        "dbdt": media,
        "desv_std": desviacion,
        "error_std": error,
        "frac_negativos": float((valores < 0).mean()),
        "rho_a_mediana_barridos": rho_mediana_barridos,
        "sigma_log10_rho_barrido": sigma_log_barrido,
        "sigma_log10_rho": (2.0 / 3.0) * (error / abs(media)) / np.log(10)
        if np.isfinite(media) and media != 0 and np.isfinite(error) else np.nan,
    }


def apilar(gates: pd.DataFrame, barridos: pd.DataFrame, config: ConfigProcesamiento,
           area_lazo: Optional[float] = None, vueltas: float = 1.0,
           puntajes: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Curva apilada por momento y gate (sin barridos de ruido).

    Parameters
    ----------
    gates : DataFrame
        Tabla ``gates`` de la etapa cruda (``barrido, momento, gate,
        tiempo, t_ini, t_fin, dbdt, error_rel``).
    barridos : DataFrame
        Tabla ``barridos`` (``barrido, es_ruido, rampa_s``).
    area_lazo, vueltas : float, optional
        Para calcular ρa por barrido. Sin área, el criterio en log usa
        −⅔·log₁₀ dB/dt, que da exactamente los mismos descartes.
    puntajes : DataFrame, optional
        Resultado de :func:`puntuar_barridos` (si no se da, se calcula).

    Returns
    -------
    DataFrame
        Una fila por (momento, gate) con valor apilado, error, SNR,
        incertidumbre en log₁₀ ρa y banderas ``util``, ``repunte``,
        ``efecto_rampa`` y ``usar``.
    """
    if puntajes is None:
        puntajes = puntuar_barridos(gates, barridos, config, area_lazo, vueltas)
    else:
        pass
    descartados = set(puntajes.loc[puntajes["descartado"].astype(bool), "barrido"]) if len(puntajes) else set()
    datos = gates.merge(barridos[["barrido", "es_ruido", "rampa_s"]], on="barrido")
    datos = datos[~datos["es_ruido"].astype(bool)]
    n_descartados = datos[datos["barrido"].isin(descartados)].groupby("momento")["barrido"].nunique()
    datos = datos[~datos["barrido"].isin(descartados)]
    filas = []
    for (momento, gate), grupo in datos.groupby(["momento", "gate"]):
        tiempo = float(grupo["tiempo"].median())
        filas.append({
            "momento": momento, "gate": int(gate), "tiempo": tiempo,
            "t_ini": float(grupo["t_ini"].median()), "t_fin": float(grupo["t_fin"].median()),
            "rampa_s": float(grupo["rampa_s"].median()),
            "error_equipo": float(grupo["error_rel"].median()),
            "n_barridos_descartados": int(n_descartados.get(momento, 0)),
            **_apilar_gate(grupo["dbdt"].to_numpy(float), tiempo, config, area_lazo, vueltas),
        })
    apilado = pd.DataFrame(filas).sort_values(["momento", "gate"]).reset_index(drop=True)
    if apilado.empty:
        return apilado
    else:
        pass
    apilado["error_rel_emp"] = apilado["desv_std"] / apilado["dbdt"].abs()
    apilado["error_rel_se"] = apilado["error_std"] / apilado["dbdt"].abs()
    apilado["snr"] = apilado["dbdt"].abs() / apilado["error_std"]
    apilado["efecto_rampa"] = apilado["tiempo"] < config.factor_rampa * apilado["rampa_s"]
    apilado["util"] = _gates_utiles(apilado, config.snr_min)
    apilado["repunte"] = _repuntes(apilado)
    apilado["excluido_usuario"] = False
    apilado["usar"] = seleccion_para_invertir(apilado, config)
    return apilado


def _gates_utiles(apilado: pd.DataFrame, snr_min: float) -> pd.Series:
    """Positivo, SNR ≥ mínimo y todos los anteriores útiles (por momento)."""
    bueno = (apilado["dbdt"] > 0) & (apilado["snr"] >= snr_min)
    ordenado = apilado.assign(bueno=bueno).sort_values(["momento", "gate"])
    contiguo = ordenado.groupby("momento")["bueno"].cummin()
    return contiguo.reindex(apilado.index).astype(bool)


def _repuntes(apilado: pd.DataFrame) -> pd.Series:
    """Gates útiles donde la señal sube más de 3σ respecto al anterior útil."""
    marca = pd.Series(False, index=apilado.index)
    for _, grupo in apilado[apilado["util"]].groupby("momento"):
        g = grupo.sort_values("gate")
        v = g["dbdt"].to_numpy(float)
        se = g["error_std"].to_numpy(float)
        sube = np.diff(v) > 3 * np.hypot(se[1:], se[:-1])
        marca.loc[g.index[1:][sube]] = True
    return marca


def seleccion_para_invertir(apilado: pd.DataFrame, config: ConfigProcesamiento) -> pd.Series:
    """Gates que entran a la inversión según las banderas y la configuración."""
    usar = apilado["util"].astype(bool) & ~apilado["excluido_usuario"].astype(bool)
    if config.excluir_repuntes:
        usar &= ~apilado["repunte"].astype(bool)
    else:
        pass
    if config.excluir_efecto_rampa:
        usar &= ~apilado["efecto_rampa"].astype(bool)
    else:
        pass
    return usar


def aplicar_exclusiones(apilado: pd.DataFrame, excluidos: pd.DataFrame,
                        config: ConfigProcesamiento) -> pd.DataFrame:
    """Aplica la selección manual de gates (``momento, gate, excluir``)."""
    salida = apilado.drop(columns=["excluido_usuario"], errors="ignore").merge(
        excluidos[["momento", "gate", "excluir"]].rename(columns={"excluir": "excluido_usuario"}),
        on=["momento", "gate"], how="left")
    salida["excluido_usuario"] = salida["excluido_usuario"].fillna(False).astype(bool)
    salida["usar"] = seleccion_para_invertir(salida, config)
    return salida


# ======================================================================
# Resistividad aparente y profundidad de difusión
# ======================================================================


def rho_a_tardia(tiempo, dbdt, area_lazo: float, vueltas: float = 1.0) -> np.ndarray:
    """Resistividad aparente de tiempo tardío, lazo central.

        ρa = μ0 / (4π t) · (2 μ0 A N / (5 t V))^(2/3)

    ``V`` es dB/dt normalizado por corriente y área de Rx [V/(A·m²)].
    Solo vale en tiempos tardíos; con V ≤ 0 devuelve NaN.
    """
    t = np.asarray(tiempo, float)
    v = np.asarray(dbdt, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = MU0 / (4 * np.pi * t) * (2 * MU0 * area_lazo * vueltas / (5 * t * v)) ** (2 / 3)
    return np.where(v > 0, rho, np.nan)


def profundidad_difusion(tiempo, rho) -> np.ndarray:
    """Profundidad de difusión d = √(2 t ρ / μ0) en metros."""
    return np.sqrt(2 * np.asarray(tiempo, float) * np.asarray(rho, float) / MU0)


def agregar_rho_a(apilado: pd.DataFrame, area_lazo: float, vueltas: float = 1.0) -> pd.DataFrame:
    salida = apilado.copy()
    salida["rho_a"] = rho_a_tardia(salida["tiempo"], salida["dbdt"], area_lazo, vueltas)
    salida.loc[~salida["util"].astype(bool), "rho_a"] = np.nan
    salida["prof_difusion"] = profundidad_difusion(salida["tiempo"], salida["rho_a"])
    # Banda ±1σ de la ρa apilada, simétrica en log₁₀ ρa.
    if "sigma_log10_rho" in salida:
        factor = 10 ** salida["sigma_log10_rho"]
        salida["rho_a_min"] = salida["rho_a"] / factor
        salida["rho_a_max"] = salida["rho_a"] * factor
    else:
        pass
    return salida


# ======================================================================
# Curva combinada
# ======================================================================


def combinar_momentos(apilado: pd.DataFrame, factor_rampa: float = 2.0) -> tuple[pd.DataFrame, float]:
    """Una sola curva LM + HM para visualización e interpretación.

    Returns
    -------
    (curva, t_cambio)
        ``curva`` con los gates útiles de ambos momentos y la columna
        ``en_curva``; ``t_cambio`` es el tiempo desde el que manda el HM
        (NaN si falta uno de los momentos).
    """
    util = apilado[apilado["util"].astype(bool)].copy()
    lm = util[util["momento"] == "LM"].sort_values("tiempo")
    hm = util[util["momento"] == "HM"].sort_values("tiempo")
    if len(lm) == 0 or len(hm) == 0:
        util["en_curva"] = True
        return util.sort_values("tiempo").reset_index(drop=True), np.nan
    else:
        pass
    t_cambio = float(hm["tiempo"].iloc[0])
    candidatos = hm[hm["tiempo"] >= factor_rampa * hm["rampa_s"]]
    for _, fila in candidatos.iterrows():
        if fila["tiempo"] > lm["tiempo"].max():
            t_cambio = float(fila["tiempo"])
            break
        else:
            pass
        error_lm = float(np.interp(np.log(fila["tiempo"]), np.log(lm["tiempo"]), lm["error_rel_se"]))
        if fila["error_rel_se"] <= error_lm:
            t_cambio = float(fila["tiempo"])
            break
        else:
            pass
    util["en_curva"] = np.where(util["momento"] == "LM", util["tiempo"] < t_cambio, util["tiempo"] >= t_cambio)
    return util.sort_values(["tiempo", "momento"]).reset_index(drop=True), t_cambio


def solape_lm_hm(apilado: pd.DataFrame) -> pd.DataFrame:
    """Diferencia del HM respecto al LM (interpolado en log-log) en el solape.

    Se dejan fuera los gates con efecto de rampa: ahí LM y HM difieren por
    diseño (rampas distintas) y la comparación no dice nada de la
    calibración.
    """
    util = apilado[apilado["util"].astype(bool)]
    if "efecto_rampa" in util:
        util = util[~util["efecto_rampa"].astype(bool)]
    else:
        pass
    lm = util[util["momento"] == "LM"].sort_values("tiempo")
    hm = util[util["momento"] == "HM"].sort_values("tiempo")
    if len(lm) < 2 or len(hm) < 1:
        return pd.DataFrame()
    else:
        pass
    t_bajo = max(lm["tiempo"].min(), hm["tiempo"].min())
    t_alto = min(lm["tiempo"].max(), hm["tiempo"].max())
    dentro = hm[(hm["tiempo"] >= t_bajo) & (hm["tiempo"] <= t_alto)]
    if dentro.empty:
        return pd.DataFrame()
    else:
        pass
    v_lm = np.exp(np.interp(np.log(dentro["tiempo"]), np.log(lm["tiempo"]), np.log(lm["dbdt"])))
    return pd.DataFrame({"gate_hm": dentro["gate"].to_numpy(), "tiempo": dentro["tiempo"].to_numpy(),
                         "dbdt_hm": dentro["dbdt"].to_numpy(), "dbdt_lm": v_lm,
                         "diferencia_pct": 100 * (dentro["dbdt"].to_numpy() / v_lm - 1)})


def pendientes(apilado: pd.DataFrame) -> pd.DataFrame:
    """Pendiente local d log V / d log t entre gates útiles consecutivos."""
    partes = []
    for momento, grupo in apilado[apilado["util"].astype(bool)].groupby("momento"):
        g = grupo.sort_values("tiempo")
        if len(g) < 2:
            continue
        else:
            pass
        pendiente = np.diff(np.log(g["dbdt"])) / np.diff(np.log(g["tiempo"]))
        partes.append(pd.DataFrame({"momento": momento, "gate": g["gate"].to_numpy()[1:],
                                    "tiempo": g["tiempo"].to_numpy()[1:], "pendiente": pendiente}))
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()


def atipicos_por_barrido(gates: pd.DataFrame, k_mad: float) -> pd.DataFrame:
    """Fracción de gates atípicos de cada barrido (detecta picos)."""
    g = gates[["momento", "gate", "barrido", "dbdt"]].copy()
    mediana = g.groupby(["momento", "gate"])["dbdt"].transform("median")
    desvio = (g["dbdt"] - mediana).abs()
    sigma = FACTOR_MAD * desvio.groupby([g["momento"], g["gate"]]).transform("median")
    g["atipico"] = desvio > k_mad * sigma.replace(0, np.nan)
    return (g.groupby(["momento", "barrido"])["atipico"].mean()
            .rename("frac_atipicos").reset_index())
