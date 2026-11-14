"""
inversion_tdem.py
==================================================================
Inversión 1D de un sondeo TDEM: modelo suave multicapa, profundidad de
investigación (DOI) y modelo de bloques.

La física (respuesta y jacobiano) la calcula **SimPEG** a través de
:mod:`forward_tdem`. La optimización se escribe aquí, explícita, para
que cada decisión se pueda auditar.

Datos
--------
- Se invierten **juntos** los gates marcados ``usar`` de ambos momentos
  (o solo LM / solo HM), cada momento con su forma de onda.
- Se trabaja en logaritmo del dato: r = (ln d_obs − ln d_calc) / σ_ln,
  con σ_ln = σ / d_obs.
- Error de cada dato: σ² = (error estándar del apilado)² + (p · d)²,
  con p = piso relativo (3 % por omisión). El piso cubre lo que el
  apilado no ve: ruido coherente, calibración, efectos 2D/3D menores.

Paso 1 — Modelo suave (multicapa)
------------------------------------
- Malla: ``n_capas`` capas con interfaces logarítmicas entre z₁ y
  z_max. z₁ = 1/10 de la profundidad de difusión del primer gate; z_max =
  ``factor_prof`` × profundidad de difusión del último gate (con la ρa de
  tiempo tardío mediana). Se puede fijar a mano.
- Parámetro: m = ln ρ por capa. Modelo inicial y de referencia: semiespacio
  con la ρa tardía mediana.
- Objetivo: Φ = Σ r² + λ ‖R (m − m_ref)‖², con R = [primera diferencia
  (suavidad vertical); √α · I (cercanía a la referencia)].
- Gauss–Newton con búsqueda lineal. λ empieza grande (10 × la razón de
  trazas JᵀJ / RᵀR) y se **enfría** (× 0.5) mientras χ²/N > objetivo:
  es el principio de discrepancia — el modelo más suave que ajusta los
  datos a su nivel de error. Si el objetivo no se alcanza (datos con
  efectos no 1D o error subestimado), se elige el modelo más suave cuyo χ²
  está a menos de 10 % del mínimo alcanzado, en vez del último, que solo
  ganaría décimas de χ² a cambio de mucha rugosidad.

Paso 2 — DOI (Oldenburg y Li, 1999)
--------------------------------------
Se repite la inversión con el λ final y dos referencias distintas
(ρ₀ × 10 y ρ₀ / 10) y más peso de cercanía. Donde los datos mandan, los
dos modelos coinciden; donde no, cada uno vuelve a su referencia. El
índice R(z) = |m₁ − m₂| / |m_ref1 − m_ref2| vale ≈ 0 donde hay control y
crece donde no lo hay. La DOI es el techo del tramo inferior continuo con
R > umbral (0.1), buscado desde el fondo: los primeros metros también
pueden tener R alto (el primer gate no los ve) y no son la DOI.

Paso 3 — Modelo de bloques
------------------------------
Para k = 2 … k_max capas:

1. **Arranque:** segmentación óptima del modelo suave (por encima de la
   DOI) en k tramos de ln ρ casi constante (programación dinámica);
   también se prueban las interfaces escaladas × 0.7 y × 1.4.
2. **Ajuste:** mínimos cuadrados acotados (SciPy, ``trf``) sobre
   ln ρ y la posición de las interfaces, con jacobiano de SimPEG.
3. **Espesor mínimo** obligatorio: la primera capa mide al menos z₁ y
   cada capa siguiente al menos una fracción f de la profundidad de su
   techo. Va dentro del ajuste con la parametrización
   ln z₁ = ln h_min + w₁ ; ln zᵢ = ln zᵢ₋₁ + ln(1 + f) + wᵢ ; w ≥ 0.

El número de capas se elige por **BIC** (o AIC): el de menor criterio y,
entre los que difieren menos de 2, el más simple. Las incertidumbres ±1σ
salen de la covarianza linealizada en el mínimo.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

import forward_tdem as fw
import procesamiento_tdem as pr

logger = logging.getLogger("tdem.inversion")

LN_RHO_MIN = np.log(fw.RHO_MIN)
LN_RHO_MAX = np.log(fw.RHO_MAX)


@dataclass
class ConfigInversion:
    """Parámetros de la inversión (los valores por omisión son los recomendados)."""

    momentos: str = "ambos"                # ambos | LM | HM
    integrar_gates: bool = True
    n_sub_gate: int = 3
    aplicar_retardo: bool = False          # sumar TIME_DELAY a los tiempos de gate
    aplicar_factor_campo: bool = False     # multiplicar el dato por FIELD_SHIFT_FACTOR
    error_relativo_minimo: float = 0.03
    # Paso 1 · multicapa
    n_capas: int = 30
    prof_max: float = 0.0                  # 0 = automática
    factor_prof: float = 1.5
    chi2_objetivo: float = 1.0
    lambda_inicial: float = 0.0            # 0 = automático
    factor_enfriamiento: float = 0.5
    max_iter: int = 25
    alfa_cercania: float = 1e-3
    # Paso 2 · DOI
    calcular_doi: bool = True
    doi_umbral: float = 0.1
    doi_factor_referencia: float = 10.0
    doi_alfa: float = 0.01
    # Paso 3 · bloques
    k_max: int = 6
    k_fijo: int = 0
    criterio: str = "BIC"
    espesor_min_fraccion: float = 0.2

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResultadoInversion:
    multicapa: pd.DataFrame
    bloques: pd.DataFrame
    ajuste: pd.DataFrame
    seleccion: pd.DataFrame
    iteraciones: pd.DataFrame
    proceso: pd.DataFrame
    metricas: dict[str, Any] = field(default_factory=dict)
    advertencias: list[str] = field(default_factory=list)


@dataclass
class DatosInversion:
    """Datos preparados para invertir (vector concatenado por momento)."""

    modelo: fw.ModeloDirecto
    observado: np.ndarray
    sigma: np.ndarray
    tabla: pd.DataFrame

    @property
    def sigma_ln(self) -> np.ndarray:
        return self.sigma / self.observado

    def residuales(self, calculado: np.ndarray) -> np.ndarray:
        return (np.log(self.observado) - np.log(np.maximum(calculado, 1e-30))) / self.sigma_ln

    def chi2(self, calculado: np.ndarray) -> float:
        r = self.residuales(calculado)
        return float(np.sum(r ** 2) / len(r))


# ======================================================================
# Preparación de datos
# ======================================================================


def preparar_datos(apilado: pd.DataFrame, formas_onda: pd.DataFrame, barridos: pd.DataFrame,
                   geometria: fw.GeometriaLazo, config: ConfigInversion) -> tuple["DatosInversion", "DatosInversion"]:
    """Arma el modelo directo y los vectores de datos y errores.

    Returns
    -------
    (datos, datos_utiles)
        ``datos``: los gates marcados ``usar`` (los que se invierten).
        ``datos_utiles``: todos los gates útiles, para mostrar el ajuste
        también en los excluidos. Comparten el mismo modelo directo.
    """
    momentos = ["LM", "HM"] if config.momentos == "ambos" else [config.momentos]
    # El modelo directo se arma con TODOS los gates de cada momento y luego
    # se eligen filas: así la transformada a tiempo (lo caro) es la misma para
    # todos los sondeos con la misma configuración y se calcula una sola vez.
    tabla = apilado[apilado["momento"].isin(momentos)].copy()
    tabla["es_util"] = tabla["util"].astype(bool) & (tabla["dbdt"] > 0)
    tabla["es_usado"] = tabla["es_util"] & tabla["usar"].astype(bool)
    if int(tabla["es_usado"].sum()) < 4:
        raise ValueError(f"Solo hay {int(tabla['es_usado'].sum())} gates para invertir: hacen falta al menos 4.")
    else:
        pass
    parametros = barridos.groupby("momento")[["retardo_s", "factor_campo", "rampa_s"]].median()
    datos_momentos, partes = [], []
    for momento in [m for m in momentos if m in set(tabla.loc[tabla["es_util"], "momento"])]:
        t = tabla[tabla["momento"] == momento].sort_values("tiempo").copy()
        retardo = float(parametros.loc[momento, "retardo_s"]) if config.aplicar_retardo else 0.0
        retardo = retardo if np.isfinite(retardo) else 0.0
        factor = float(parametros.loc[momento, "factor_campo"]) if config.aplicar_factor_campo else 1.0
        factor = factor if np.isfinite(factor) and factor > 0 else 1.0
        onda = formas_onda[formas_onda["momento"] == momento].sort_values("punto")
        if len(onda) >= 3:
            onda_t, onda_a = onda["tiempo"].to_numpy(float), onda["amplitud"].to_numpy(float)
        else:
            onda_t, onda_a = fw.onda_por_omision(float(parametros.loc[momento, "rampa_s"]))
        datos_momentos.append(fw.DatosMomento(
            momento, t["tiempo"].to_numpy(float) + retardo, t["t_ini"].to_numpy(float) + retardo,
            t["t_fin"].to_numpy(float) + retardo, onda_t, onda_a))
        t["dbdt_inv"] = t["dbdt"] * factor
        t["error_std_inv"] = t["error_std"] * factor
        partes.append(t)
    tabla = pd.concat(partes, ignore_index=True)
    observado = tabla["dbdt_inv"].to_numpy(float)
    sigma = np.sqrt(tabla["error_std_inv"].fillna(0).to_numpy(float) ** 2
                    + (config.error_relativo_minimo * np.abs(observado)) ** 2)
    modelo = fw.ModeloDirecto(geometria, datos_momentos, config.integrar_gates, config.n_sub_gate)

    def subconjunto(filas: np.ndarray) -> DatosInversion:
        return DatosInversion(modelo.con_filas(filas), observado[filas], sigma[filas],
                              tabla.iloc[filas].reset_index(drop=True))

    return (subconjunto(np.where(tabla["es_usado"].to_numpy())[0]),
            subconjunto(np.where(tabla["es_util"].to_numpy())[0]))


def malla_profundidad(datos: DatosInversion, geometria: fw.GeometriaLazo,
                      config: ConfigInversion) -> tuple[np.ndarray, float]:
    """Espesores de la malla logarítmica y ρ de arranque (ρa tardía mediana)."""
    t = datos.tabla["tiempo"].to_numpy(float)
    rho_a = pr.rho_a_tardia(t, datos.observado, geometria.area, geometria.vueltas)
    rho0 = float(np.nanmedian(rho_a)) if np.isfinite(rho_a).any() else 100.0
    rho0 = float(np.clip(rho0, 1.0, 5000.0))
    z_min = float(pr.profundidad_difusion(t.min(), rho0)) / 10.0
    z_max = config.prof_max if config.prof_max > 0 else config.factor_prof * float(pr.profundidad_difusion(t.max(), rho0))
    z_min = float(np.clip(z_min, 0.5, 20.0))
    z_max = float(max(z_max, 10.0 * z_min))
    interfaces = np.geomspace(z_min, z_max, config.n_capas - 1)
    return np.diff(np.concatenate([[0.0], interfaces])), rho0


# ======================================================================
# Paso 1 · modelo suave
# ======================================================================


def _regularizacion(n: int, alfa: float) -> np.ndarray:
    suavidad = np.diff(np.eye(n), axis=0)
    return np.vstack([suavidad, np.sqrt(alfa) * np.eye(n)])


def invertir_suave(datos: DatosInversion, espesores: np.ndarray, m_ref: np.ndarray, config: ConfigInversion,
                   m_inicial: Optional[np.ndarray] = None, lam_fijo: Optional[float] = None,
                   alfa: Optional[float] = None, max_iter: Optional[int] = None,
                   verificar: Optional[Callable] = None) -> dict[str, Any]:
    """Gauss–Newton regularizado con enfriamiento de λ.

    - Una iteración de Gauss–Newton por cada λ; λ se multiplica por
      ``factor_enfriamiento`` hasta que χ²/N ≤ objetivo.
    - Si el objetivo **no se alcanza** (errores subestimados, efectos no 1D),
      no se toma el último modelo —el más rugoso— sino el más suave cuyo χ²
      está a menos de 10 % del mínimo alcanzado.
    - Con ``lam_fijo`` no se enfría λ (inversiones de la DOI).
    """
    n = len(m_ref)
    R = _regularizacion(n, config.alfa_cercania if alfa is None else alfa)
    RtR = R.T @ R
    m = np.array(m_inicial if m_inicial is not None else m_ref, float)
    calculado, J = datos.modelo.jacobiano_rho(np.exp(m), espesores)
    w = 1.0 / datos.sigma_ln
    if lam_fijo is not None:
        lam = float(lam_fijo)
    else:
        if config.lambda_inicial > 0:
            lam = float(config.lambda_inicial)
        else:
            Jw = J / calculado[:, None] * w[:, None]
            lam = 10.0 * float(np.trace(Jw.T @ Jw) / np.trace(RtR))
    lam_minimo = lam * 1e-4
    historial, modelos = [], []
    chi2 = datos.chi2(calculado)
    sin_mejora = 0
    for iteracion in range(1, (max_iter or config.max_iter) + 1):
        if verificar is not None:
            verificar()
        else:
            pass
        r = datos.residuales(calculado)
        Jw = J / calculado[:, None] * w[:, None]          # ∂(ln d)/∂m pesado por 1/σ_ln
        gradiente = Jw.T @ r - lam * RtR @ (m - m_ref)
        paso = np.linalg.solve(Jw.T @ Jw + lam * RtR, gradiente)
        phi_actual = np.sum(r ** 2) + lam * np.sum((R @ (m - m_ref)) ** 2)
        aceptado = False
        for escala in (1.0, 0.5, 0.25, 0.125):
            m_nuevo = np.clip(m + escala * paso, LN_RHO_MIN, LN_RHO_MAX)
            calc_nuevo = datos.modelo.respuesta(np.exp(m_nuevo), espesores)
            r_nuevo = datos.residuales(calc_nuevo)
            phi_nuevo = np.sum(r_nuevo ** 2) + lam * np.sum((R @ (m_nuevo - m_ref)) ** 2)
            if phi_nuevo < phi_actual:
                aceptado = True
                break
            else:
                pass
        if aceptado:
            mejora = (chi2 - datos.chi2(calc_nuevo)) / max(chi2, 1e-12)
            m = m_nuevo
            calculado, J = datos.modelo.jacobiano_rho(np.exp(m), espesores)
            chi2 = datos.chi2(calculado)
            sin_mejora = sin_mejora + 1 if mejora < 0.01 else 0
        else:
            sin_mejora += 1
        rugosidad = float(np.sum(np.diff(m) ** 2))
        historial.append({"iteracion": iteracion, "chi2": chi2, "lambda": lam, "rugosidad": rugosidad,
                          "paso_aceptado": aceptado})
        modelos.append((m.copy(), calculado.copy(), J.copy(), chi2, lam))
        if lam_fijo is None and chi2 <= config.chi2_objetivo:
            break
        else:
            pass
        if lam_fijo is None:
            if lam <= lam_minimo or sin_mejora >= 5:
                break
            else:
                pass
            lam = max(lam * config.factor_enfriamiento, lam_minimo)
        else:
            pass
    historial = pd.DataFrame(historial)
    elegido = len(modelos) - 1
    criterio = "objetivo de χ² alcanzado" if chi2 <= config.chi2_objetivo else "última iteración"
    if lam_fijo is None and chi2 > config.chi2_objetivo and len(modelos) >= 3:
        # El más suave (λ más grande) de los que ajustan casi tan bien como el
        # mejor alcanzado (≤ 10 % más χ²). Evita quedarse con el último modelo,
        # que solo gana décimas de χ² a cambio de mucha rugosidad.
        chi2s = historial["chi2"].to_numpy()
        elegido = int(np.argmax(chi2s <= 1.10 * chi2s.min()))
        criterio = "el más suave con χ² a menos de 10 % del mínimo (el objetivo no se alcanzó)"
    else:
        pass
    historial["elegido"] = historial.index == elegido
    m, calculado, J, chi2, lam = modelos[elegido]
    return {"m": m, "calculado": calculado, "J": J, "chi2": chi2, "lambda": lam,
            "historial": historial, "criterio": criterio}


def calcular_doi(datos: DatosInversion, espesores: np.ndarray, m_final: np.ndarray, rho0: float,
                 lam: float, config: ConfigInversion, verificar: Optional[Callable] = None) -> tuple[float, np.ndarray]:
    """DOI de Oldenburg y Li con dos referencias."""
    salto = np.log(config.doi_factor_referencia)
    referencia_alta = np.full(len(m_final), np.log(rho0) + salto)
    referencia_baja = np.full(len(m_final), np.log(rho0) - salto)
    alta = invertir_suave(datos, espesores, referencia_alta, config, m_inicial=m_final, lam_fijo=lam,
                          alfa=config.doi_alfa, max_iter=8, verificar=verificar)
    baja = invertir_suave(datos, espesores, referencia_baja, config, m_inicial=m_final, lam_fijo=lam,
                          alfa=config.doi_alfa, max_iter=8, verificar=verificar)
    indice = np.abs(alta["m"] - baja["m"]) / (2 * salto)
    bases = np.cumsum(espesores)
    centros = np.concatenate([[0.5 * bases[0]], 0.5 * (bases[:-1] + bases[1:]), [bases[-1] * 1.2]])
    # Se busca desde el fondo: la DOI es el techo del tramo inferior continuo
    # sin control (R > umbral). Las celdas más someras también pueden tener R
    # alto (el primer gate no ve los primeros metros) y no deben confundirse
    # con la profundidad de investigación.
    sin_control = indice > config.doi_umbral
    if not sin_control[-1]:
        return float(bases[-1]), indice
    else:
        pass
    i = len(indice) - 1
    while i > 0 and sin_control[i - 1]:
        i -= 1
    if i == 0:
        return float(centros[0]), indice
    else:
        pass
    z0, z1 = np.log(centros[i - 1]), np.log(centros[i])
    f = (config.doi_umbral - indice[i - 1]) / max(indice[i] - indice[i - 1], 1e-9)
    return float(np.exp(z0 + np.clip(f, 0, 1) * (z1 - z0))), indice


# ======================================================================
# Paso 3 · bloques
# ======================================================================


def segmentar(espesores: np.ndarray, m: np.ndarray, k: int, z_limite: float) -> tuple[np.ndarray, np.ndarray]:
    """Segmentación óptima de un modelo suave en k capas (programación dinámica).

    Minimiza la varianza de ln ρ dentro de cada tramo, ponderada por
    espesor, usando solo las celdas por encima de ``z_limite``.

    Returns
    -------
    (ln_rho, interfaces) de k capas (k−1 interfaces).
    """
    bases = np.cumsum(espesores)
    n_validas = int(np.clip(np.searchsorted(bases, z_limite) + 1, k, len(m)))
    valores = m[:n_validas]
    pesos = np.concatenate([espesores, [espesores[-1]]])[:n_validas]

    def costo(a: int, b: int) -> float:
        w, v = pesos[a:b], valores[a:b]
        media = np.sum(w * v) / np.sum(w)
        return float(np.sum(w * (v - media) ** 2))

    n = n_validas
    tabla = np.full((k + 1, n + 1), np.inf)
    corte = np.zeros((k + 1, n + 1), dtype=int)
    tabla[0, 0] = 0.0
    for j in range(1, k + 1):
        for b in range(j, n + 1):
            for a in range(j - 1, b):
                valor = tabla[j - 1, a] + costo(a, b)
                if valor < tabla[j, b]:
                    tabla[j, b] = valor
                    corte[j, b] = a
                else:
                    pass
    limites = [n]
    for j in range(k, 0, -1):
        limites.append(corte[j, limites[-1]])
    limites = limites[::-1]
    ln_rho = np.array([np.sum(pesos[a:b] * valores[a:b]) / np.sum(pesos[a:b])
                       for a, b in zip(limites[:-1], limites[1:])])
    interfaces = np.array([bases[b - 1] for b in limites[1:-1]])
    return ln_rho, interfaces


class _Parametrizacion:
    """Interfaces con espesor mínimo: w ≥ 0 ↔ profundidades z."""

    def __init__(self, h_min: float, fraccion: float) -> None:
        self.ln_h_min = np.log(h_min)
        self.ln_paso = np.log1p(fraccion)

    def a_profundidades(self, w: np.ndarray) -> np.ndarray:
        incrementos = np.concatenate([[self.ln_h_min], np.full(len(w) - 1, self.ln_paso)]) + w
        return np.exp(np.cumsum(incrementos))

    def desde_profundidades(self, z: np.ndarray) -> np.ndarray:
        ln_z = np.log(np.asarray(z, float))
        w = np.empty_like(ln_z)
        w[0] = ln_z[0] - self.ln_h_min
        w[1:] = np.diff(ln_z) - self.ln_paso
        return np.maximum(w, 0.0)

    def jacobiano_espesores(self, w: np.ndarray) -> np.ndarray:
        """∂h/∂w (k−1 × k−1)."""
        z = self.a_profundidades(w)
        n = len(z)
        dz = np.tril(np.ones((n, n))) * z[:, None]
        dh = dz.copy()
        dh[1:] -= dz[:-1]
        return dh


#: Peso de la penalización que mantiene la última interfaz dentro de la
#: malla del multicapa (por unidad de ln z de exceso).
PESO_TOPE_PROFUNDIDAD = 30.0


def _ajustar_bloques(datos: DatosInversion, ln_rho0: np.ndarray, z0: np.ndarray,
                     param: _Parametrizacion, limites_ln_rho: tuple[float, float],
                     z_tope: float = np.inf) -> dict[str, Any]:
    """Ajuste de un modelo de k bloques por mínimos cuadrados acotados.

    ``z_tope`` es la profundidad máxima permitida a la última interfaz
    (el fondo de la malla del multicapa). Sin ese tope, cuando los datos
    no ven el fondo, la última interfaz puede irse a cientos de km: cada
    ``w`` admite hasta e¹² veces la profundidad anterior y el ajuste no
    cambia. El tope se aplica como penalización tipo bisagra sobre
    ln z_última (su derivada respecto a cada ``w`` vale 1).
    """
    from scipy.optimize import least_squares

    k = len(ln_rho0)
    if np.isfinite(z_tope):
        z0 = np.minimum(np.asarray(z0, float), 0.95 * z_tope)
    else:
        pass
    w0 = param.desde_profundidades(z0)
    ln_tope = np.log(z_tope) if np.isfinite(z_tope) else np.inf

    def exceso(w):
        return max(0.0, float(np.log(param.a_profundidades(w)[-1])) - ln_tope) if k > 1 else 0.0

    def separar(p):
        return p[:k], p[k:]

    def espesores(w):
        return np.diff(np.concatenate([[0.0], param.a_profundidades(w)]))

    def residual(p):
        ln_rho, w = separar(p)
        r = -datos.residuales(datos.modelo.respuesta_bloques(np.exp(ln_rho), espesores(w)))
        return np.append(r, PESO_TOPE_PROFUNDIDAD * exceso(w))

    def jacobiano(p):
        ln_rho, w = separar(p)
        h = espesores(w)
        d, J = datos.modelo.jacobiano_bloques(np.exp(ln_rho), h)
        escala = (1.0 / datos.sigma_ln / d)[:, None]
        J_rho = J[:, :k] * escala
        J_w = (J[:, k:] * escala / h[None, :]) @ param.jacobiano_espesores(w)
        fila_tope = np.zeros(2 * k - 1)
        if exceso(w) > 0:
            fila_tope[k:] = PESO_TOPE_PROFUNDIDAD
        else:
            pass
        return np.vstack([np.hstack([J_rho, J_w]), fila_tope])

    bajo, alto = limites_ln_rho
    p0 = np.concatenate([np.clip(ln_rho0, bajo + 0.01, alto - 0.01), w0])
    inferior = np.concatenate([np.full(k, bajo), np.zeros(k - 1)])
    superior = np.concatenate([np.full(k, alto), np.full(k - 1, 12.0)])
    p0 = np.clip(p0, inferior + 1e-9, superior - 1e-9)
    solucion = least_squares(residual, p0, jac=jacobiano, bounds=(inferior, superior), method="trf",
                             x_scale="jac", max_nfev=30, xtol=1e-5, ftol=1e-5)
    ln_rho, w = separar(solucion.x)
    h = espesores(w)
    d, J = datos.modelo.jacobiano_bloques(np.exp(ln_rho), h)
    return {"ln_rho": ln_rho, "espesores": h, "calculado": d, "J": J,
            "phi": float(np.sum(datos.residuales(d) ** 2)), "chi2": datos.chi2(d)}


def _incertidumbre(datos: DatosInversion, ajuste: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """σ de ln ρ y de ln h a partir de la covarianza linealizada."""
    escala = (1.0 / datos.sigma_ln / ajuste["calculado"])[:, None]
    Jw = ajuste["J"] * escala
    try:
        covarianza = np.linalg.pinv(Jw.T @ Jw) * max(1.0, ajuste["chi2"])
        sigma = np.sqrt(np.clip(np.diag(covarianza), 0, None))
    except np.linalg.LinAlgError:
        sigma = np.full(Jw.shape[1], np.nan)
    k = len(ajuste["ln_rho"])
    return sigma[:k], sigma[k:]


def invertir_bloques(datos: DatosInversion, espesores_suave: np.ndarray, m_suave: np.ndarray,
                     z_limite: float, config: ConfigInversion, chi2_suave: float = 1.0,
                     verificar: Optional[Callable] = None) -> tuple[dict[str, Any], pd.DataFrame, list[str]]:
    """Modelos de 2…k_max capas y elección por criterio de información.

    - La resistividad de cada bloque se acota a una década por debajo y por
      encima del rango del multicapa: evita capas "fantasma" de 10⁴ Ω·m que
      ningún dato exige.
    - Si el multicapa no alcanzó el nivel de ruido (χ² > 1), los criterios
      de información se calculan con los errores escalados por √χ²: así la
      penalización por capa se compara con un ajuste realista y no se
      agregan capas para "explicar" error sistemático.
    """
    n = len(datos.observado)
    escala = max(1.0, float(chi2_suave))
    limites = (max(LN_RHO_MIN, float(m_suave.min()) - np.log(10.0)),
               min(LN_RHO_MAX, float(m_suave.max()) + np.log(10.0)))
    param = _Parametrizacion(h_min=float(espesores_suave[0]), fraccion=config.espesor_min_fraccion)
    z_tope = float(np.sum(espesores_suave))  # fondo de la malla del multicapa
    valores_k = [int(config.k_fijo)] if config.k_fijo >= 2 else list(range(2, int(config.k_max) + 1))
    resultados, filas, notas = {}, [], []
    for k in valores_k:
        if verificar is not None:
            verificar()
        else:
            pass
        ln_rho0, z0 = segmentar(espesores_suave, m_suave, k, z_limite)
        mejor = None
        for etiqueta, factor in (("segmentación del multicapa", 1.0), ("interfaces × 0.7", 0.7),
                                 ("interfaces × 1.4", 1.4)):
            try:
                ajuste = _ajustar_bloques(datos, ln_rho0, z0 * factor, param, limites, z_tope)
            except Exception as exc:  # un arranque malo no detiene la búsqueda
                logger.warning("k=%d, arranque %s falló: %s", k, etiqueta, exc)
                continue
            ajuste["arranque"] = etiqueta
            if mejor is None or ajuste["phi"] < mejor["phi"]:
                mejor = ajuste
            else:
                pass
        if mejor is None:
            continue
        else:
            pass
        p = 2 * k - 1
        resultados[k] = mejor
        filas.append({"k": k, "chi2": mejor["chi2"], "phi": mejor["phi"],
                      "BIC": mejor["phi"] / escala + p * np.log(n), "AIC": mejor["phi"] / escala + 2 * p,
                      "arranque": mejor["arranque"]})
        # Parada temprana: dos k seguidos claramente peores que el mejor (ΔIC > 6)
        # ya no van a ganar, porque cada capa extra solo suma penalización.
        criterio_actual = pd.DataFrame(filas)[config.criterio if config.criterio in ("BIC", "AIC") else "BIC"]
        if len(filas) >= 3 and (criterio_actual.iloc[-2:] > criterio_actual.min() + 6).all():
            notas.append(f"Búsqueda de capas detenida en k = {k}: los dos últimos modelos tienen un criterio "
                         "claramente peor que el mejor.")
            break
        else:
            pass
    seleccion = pd.DataFrame(filas)
    if seleccion.empty:
        raise ValueError("Ningún modelo de bloques convergió.")
    else:
        pass
    criterio = config.criterio if config.criterio in ("BIC", "AIC") else "BIC"
    minimo = seleccion[criterio].min()
    candidatos = seleccion[seleccion[criterio] - minimo < 2.0]
    k_elegido = int(candidatos["k"].min())
    seleccion["elegido"] = seleccion["k"] == k_elegido
    if len(candidatos) > 1:
        notas.append(f"{len(candidatos)} modelos con Δ{criterio} < 2: se eligió el más simple (k = {k_elegido}).")
    else:
        pass
    if escala > 1.0:
        notas.append(f"Criterio de información con errores escalados por √χ² del multicapa ({np.sqrt(escala):.2f}).")
    else:
        pass
    if k_elegido == max(valores_k) and len(valores_k) > 1:
        notas.append(f"El número de capas elegido ({k_elegido}) es el máximo probado: considere subir k_max.")
    else:
        pass
    return resultados[k_elegido], seleccion, notas


# ======================================================================
# Orquestación
# ======================================================================


def _tabla_bloques(ajuste: dict[str, Any], sigma_rho: np.ndarray, sigma_h: np.ndarray,
                   doi: Optional[float]) -> pd.DataFrame:
    rho = np.exp(ajuste["ln_rho"])
    tabla = fw.modelo_a_tabla(ajuste["espesores"], rho)
    sigma_rho = np.minimum(np.nan_to_num(sigma_rho, nan=10.0), 10.0)
    sigma_h = np.minimum(np.nan_to_num(sigma_h, nan=10.0), 10.0)
    sigma_h_completo = np.concatenate([sigma_h, [np.nan]])
    tabla["rho_min"] = rho * np.exp(-sigma_rho)
    tabla["rho_max"] = rho * np.exp(sigma_rho)
    tabla["espesor_min"] = tabla["espesor"] * np.exp(-sigma_h_completo)
    tabla["espesor_max"] = tabla["espesor"] * np.exp(sigma_h_completo)
    tabla["sigma_ln_rho"] = sigma_rho
    tabla["sigma_ln_h"] = sigma_h_completo
    resolucion = []
    for s_r, s_h in zip(sigma_rho, sigma_h_completo):
        peor = np.nanmax([s_r, s_h if np.isfinite(s_h) else 0.0])
        if peor < 0.2:
            resolucion.append("bien resuelta")
        else:
            resolucion.append("parcial" if peor < 0.7 else "pobre")
    tabla["resolucion"] = resolucion
    finito = np.isfinite(tabla["espesor"])
    tabla["S"] = np.where(finito, tabla["espesor"] / tabla["rho"], np.nan)
    tabla["T"] = np.where(finito, tabla["espesor"] * tabla["rho"], np.nan)
    tabla["bajo_doi"] = tabla["z_techo"] > (doi if doi else np.inf)
    return tabla


def _tabla_ajuste(datos_todos: DatosInversion, usados: pd.DataFrame, geometria: fw.GeometriaLazo,
                  multicapa: pd.DataFrame, bloques: pd.DataFrame) -> pd.DataFrame:
    """Observado contra calculado en todos los gates útiles."""
    tabla = datos_todos.tabla[["momento", "gate", "tiempo", "dbdt_inv"]].rename(columns={"dbdt_inv": "dbdt_obs"})
    tabla["error"] = datos_todos.sigma
    for nombre, modelo in (("multicapa", multicapa), ("bloques", bloques)):
        espesores = modelo["espesor"].to_numpy(float)[:-1]
        calculado = datos_todos.modelo.respuesta(modelo["rho"].to_numpy(float), espesores)
        tabla[f"dbdt_{nombre}"] = calculado
        tabla[f"error_{nombre}_pct"] = 100 * (calculado / tabla["dbdt_obs"] - 1)
    claves = set(zip(usados["momento"], usados["gate"]))
    tabla["usado"] = [(m, g) in claves for m, g in zip(tabla["momento"], tabla["gate"])]
    for columna in ("obs", "multicapa", "bloques"):
        tabla[f"rho_a_{columna}"] = pr.rho_a_tardia(tabla["tiempo"], tabla[f"dbdt_{columna}"], geometria.area,
                                                   geometria.vueltas)
    return tabla


def invertir_sondeo(apilado: pd.DataFrame, formas_onda: pd.DataFrame, barridos: pd.DataFrame,
                    geometria: fw.GeometriaLazo, config: Optional[ConfigInversion] = None,
                    verificar_cancelacion: Optional[Callable] = None) -> ResultadoInversion:
    """Inversión completa de un sondeo (pasos 1, 2 y 3)."""
    cfg = config or ConfigInversion()
    inicio = time.time()
    advertencias: list[str] = []
    proceso: list[str] = []

    datos, datos_utiles = preparar_datos(apilado, formas_onda, barridos, geometria, cfg)
    espesores, rho0 = malla_profundidad(datos, geometria, cfg)
    z_max = float(np.sum(espesores))
    conteo = datos.tabla.groupby("momento").size().to_dict()
    proceso.append(f"Datos: {len(datos.observado)} gates ({', '.join(f'{k} {v}' for k, v in conteo.items())}); "
                   f"error = error estándar del apilado ⊕ {100 * cfg.error_relativo_minimo:.0f} % del dato; "
                   f"gates {'integrados' if cfg.integrar_gates else 'en su centro'}; forma de onda del archivo.")

    m_ref = np.full(cfg.n_capas, np.log(rho0))
    suave = invertir_suave(datos, espesores, m_ref, cfg, verificar=verificar_cancelacion)
    proceso.append(f"Paso 1 · multicapa: {cfg.n_capas} capas logarítmicas de {espesores[0]:.1f} m a "
                   f"{z_max:.0f} m, arranque en semiespacio de {rho0:.1f} Ω·m (ρa tardía mediana); "
                   f"{len(suave['historial'])} iteraciones de Gauss–Newton enfriando λ; modelo elegido por "
                   f"{suave['criterio']}: χ² = {suave['chi2']:.2f}, λ = {suave['lambda']:.3g}.")
    if suave["chi2"] > 2 * cfg.chi2_objetivo:
        advertencias.append(f"El multicapa no alcanza el nivel de ruido (χ² = {suave['chi2']:.2f}): error "
                            "subestimado, gates con efectos no 1D o acoplamiento.")
    else:
        pass

    doi, indice_doi = None, np.full(cfg.n_capas, np.nan)
    if cfg.calcular_doi:
        doi, indice_doi = calcular_doi(datos, espesores, suave["m"], rho0, suave["lambda"], cfg,
                                       verificar=verificar_cancelacion)
        proceso.append(f"Paso 2 · DOI de Oldenburg–Li (referencias ×{cfg.doi_factor_referencia:g} y "
                       f"÷{cfg.doi_factor_referencia:g}, umbral {cfg.doi_umbral:g}): ≈ {doi:.0f} m.")
    else:
        pass

    multicapa = fw.modelo_a_tabla(espesores, np.exp(suave["m"]))
    multicapa["indice_doi"] = indice_doi
    finito = np.isfinite(multicapa["espesor"])
    multicapa["S"] = np.where(finito, multicapa["espesor"] / multicapa["rho"], np.nan)
    multicapa["T"] = np.where(finito, multicapa["espesor"] * multicapa["rho"], np.nan)

    z_limite = doi if doi else z_max
    mejor, seleccion, notas = invertir_bloques(datos, espesores, suave["m"], z_limite, cfg,
                                               chi2_suave=suave["chi2"], verificar=verificar_cancelacion)
    advertencias += notas
    sigma_rho, sigma_h = _incertidumbre(datos, mejor)
    bloques = _tabla_bloques(mejor, sigma_rho, sigma_h, doi)
    proceso.append(f"Paso 3 · bloques: k = {', '.join(map(str, seleccion['k']))} ajustados por mínimos cuadrados "
                   f"acotados (espesor mínimo: {espesores[0]:.1f} m la primera capa y "
                   f"{100 * cfg.espesor_min_fraccion:.0f} % de la profundidad del techo las demás); "
                   f"{cfg.criterio} elige {len(bloques)} capas (arranque: {mejor['arranque']}), χ² = {mejor['chi2']:.2f}.")
    if bloques["bajo_doi"].any():
        proceso.append(f"{int(bloques['bajo_doi'].sum())} capa(s) del modelo de bloques empiezan por debajo de la DOI: "
                       "su resistividad no está controlada por los datos.")
    else:
        pass

    ajuste = _tabla_ajuste(datos_utiles, datos.tabla, geometria, multicapa, bloques)
    usados = ajuste[ajuste["usado"]]
    metricas = {
        "n_datos": int(len(datos.observado)), "rho_inicial": rho0, "prof_max": z_max,
        "multicapa_chi2": suave["chi2"], "multicapa_lambda": suave["lambda"],
        "multicapa_iteraciones": int(len(suave["historial"])),
        "multicapa_rms": float(100 * np.sqrt(np.mean((usados["error_multicapa_pct"] / 100) ** 2))),
        "bloques_n_capas": int(len(bloques)), "bloques_chi2": mejor["chi2"],
        "bloques_rms": float(100 * np.sqrt(np.mean((usados["error_bloques_pct"] / 100) ** 2))),
        "criterio": cfg.criterio, "doi": doi if doi else np.nan, "duracion_s": time.time() - inicio,
    }
    return ResultadoInversion(
        multicapa=multicapa, bloques=bloques, ajuste=ajuste, seleccion=seleccion,
        iteraciones=suave["historial"], proceso=pd.DataFrame({"descripcion": proceso}),
        metricas=metricas, advertencias=advertencias,
    )
