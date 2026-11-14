"""
procesamiento_tdem.py
==================================================================
De los barridos de campo a la curva de decaimiento de cada momento:
apilado robusto, error del apilado, gates útiles, resistividad aparente
de tiempo tardío y curva combinada LM + HM.

Apilado robusto (por momento y por gate)
-------------------------------------------
Cada barrido del archivo ya es un promedio de cientos de transitorios.
Para cada gate se tienen N barridos (624 en un sondeo típico):

1. mediana y MAD de los N valores; σ_robusta = 1.4826 · MAD;
2. se descartan los valores con |v − mediana| > k · σ_robusta (k = 5);
3. el valor apilado es la **media** de los que quedan, y su error
   estándar = desviación estándar / √n.

La MAD detecta picos (rayos lejanos, cercas eléctricas) sin dejarse
inflar por ellos; la media de lo que queda es más eficiente que la
mediana con ruido gaussiano. k = 5 casi nunca toca datos buenos.

El error estándar solo baja el ruido **aleatorio**. El ruido coherente
(acoplamiento con metal, armónicos de la red) no se cancela al apilar:
por eso la inversión agrega un piso de error relativo (ver
:mod:`inversion_tdem`).

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
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("tdem.procesamiento")

MU0 = 4e-7 * np.pi
FACTOR_MAD = 1.4826


@dataclass
class ConfigProcesamiento:
    """Parámetros del apilado y de la selección de gates."""

    k_mad: float = 5.0                 # |z robusto| > k → valor atípico en ese gate
    snr_min: float = 3.0               # SNR mínimo para que un gate sea útil
    factor_rampa: float = 2.0          # t < factor · rampa → gate con efecto de rampa
    excluir_repuntes: bool = True      # no invertir gates donde la señal sube
    excluir_efecto_rampa: bool = False  # no invertir gates con efecto de rampa

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ======================================================================
# Apilado
# ======================================================================


def _apilar_gate(valores: np.ndarray, k_mad: float) -> dict[str, Any]:
    """Apila los valores de un gate (todos los barridos de un momento)."""
    valores = valores[np.isfinite(valores)]
    n = len(valores)
    if n == 0:
        return {"n_stacks": 0}
    else:
        pass
    mediana = float(np.median(valores))
    sigma = FACTOR_MAD * float(np.median(np.abs(valores - mediana)))
    if sigma > 0:
        dentro = np.abs(valores - mediana) <= k_mad * sigma
    else:
        dentro = np.ones(n, dtype=bool)
    conservados = valores[dentro]
    desviacion = float(conservados.std(ddof=1)) if len(conservados) > 1 else np.nan
    return {
        "n_stacks": n,
        "n_atipicos": int(n - dentro.sum()),
        "mediana": mediana,
        "sigma_robusta": sigma,
        "dbdt": float(conservados.mean()),
        "desv_std": desviacion,
        "error_std": desviacion / np.sqrt(len(conservados)) if len(conservados) > 1 else np.nan,
        "frac_negativos": float((valores < 0).mean()),
    }


def apilar(gates: pd.DataFrame, barridos: pd.DataFrame, config: ConfigProcesamiento) -> pd.DataFrame:
    """Curva apilada por momento y gate (sin barridos de ruido).

    Parameters
    ----------
    gates : DataFrame
        Tabla ``gates`` de la etapa cruda (``barrido, momento, gate,
        tiempo, t_ini, t_fin, dbdt, error_rel``).
    barridos : DataFrame
        Tabla ``barridos`` (``barrido, es_ruido, rampa_s``).

    Returns
    -------
    DataFrame
        Una fila por (momento, gate) con valor apilado, error, SNR y
        banderas ``util``, ``repunte``, ``efecto_rampa`` y ``usar``.
    """
    datos = gates.merge(barridos[["barrido", "es_ruido", "rampa_s"]], on="barrido")
    datos = datos[~datos["es_ruido"].astype(bool)]
    filas = []
    for (momento, gate), grupo in datos.groupby(["momento", "gate"]):
        filas.append({
            "momento": momento, "gate": int(gate),
            "tiempo": float(grupo["tiempo"].median()),
            "t_ini": float(grupo["t_ini"].median()), "t_fin": float(grupo["t_fin"].median()),
            "rampa_s": float(grupo["rampa_s"].median()),
            "error_equipo": float(grupo["error_rel"].median()),
            **_apilar_gate(grupo["dbdt"].to_numpy(float), config.k_mad),
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
