"""
forward_tdem.py
==================================================================
Modelado directo 1D de TDEM con **SimPEG** (``Simulation1DLayered``).

Qué se modela
----------------
- **Transmisor:** lazo cuadrado o rectangular de ``lado_x × lado_y`` en
  superficie (``LineCurrent`` cerrado), con ``vueltas`` vueltas. La
  orientación del lazo no importa con el receptor en el centro de un
  medio 1D.
- **Receptor:** dB/dt vertical en la posición de la bobina (por omisión,
  el centro del lazo). La respuesta se normaliza por corriente y por área
  de Rx: queda en V/(A·m²), las mismas unidades que el USF.
- **Forma de onda:** la del archivo (``TX_RAMP``), por momento, como
  forma de onda lineal por tramos. Incluye el encendido, el pulso de
  polaridad opuesta anterior y la rampa de apagado: así los gates
  tempranos (afectados por la rampa) y los tardíos (afectados por el
  pulso anterior) se modelan sin corregir los datos.
- **Gates:** el instrumento integra cada gate entre su apertura y su
  cierre. Si ``integrar_gates`` está activo, la respuesta se calcula en
  ``n_sub`` tiempos dentro de cada ventana y se promedia (promedio en
  tiempo lineal); si no, se evalúa en el centro del gate.

LM y HM son dos fuentes del mismo ``Survey`` de SimPEG, cada una con su
forma de onda y sus tiempos: así se invierten juntos.

Signo y unidades
-------------------
SimPEG devuelve dBz/dt con z hacia arriba; tras el apagado es negativo.
Aquí se devuelve ``−dBz/dt`` (positivo, como el archivo con
``Z_DIRECTION: DOWN``).

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("tdem.forward")

RHO_MIN = 0.1
RHO_MAX = 1.0e5

#: Coeficientes de la transformada frecuencia→tiempo ya calculados, por
#: firma de (tiempos de los receptores, forma de onda). Los sondeos de una
#: misma campaña suelen compartir gates y forma de onda: así el cálculo
#: caro (segundos) se hace una sola vez por configuración.
_CACHE_COEFICIENTES: dict = {}
_MAX_CACHE = 16


@dataclass
class GeometriaLazo:
    """Lazo transmisor y posición de la bobina receptora (m)."""

    lado_x: float = 40.0
    lado_y: float = 40.0
    vueltas: float = 1.0
    rx_x: float = 0.0
    rx_y: float = 0.0
    rx_z: float = 0.0

    @property
    def area(self) -> float:
        return float(self.lado_x) * float(self.lado_y)


@dataclass
class DatosMomento:
    """Gates de un momento y su forma de onda."""

    momento: str
    tiempo: np.ndarray
    t_ini: np.ndarray
    t_fin: np.ndarray
    onda_tiempo: np.ndarray
    onda_amplitud: np.ndarray


def _tiempos_de_gate(datos: DatosMomento, integrar: bool, n_sub: int) -> tuple[np.ndarray, np.ndarray]:
    """Tiempos de cálculo y matriz de promediado (n_gates × n_tiempos)."""
    n = len(datos.tiempo)
    if not integrar or n_sub <= 1 or not np.all(np.isfinite(datos.t_ini)) or not np.all(np.isfinite(datos.t_fin)):
        return np.asarray(datos.tiempo, float), np.eye(n)
    else:
        pass
    tiempos, filas = [], []
    for i, (a, b) in enumerate(zip(datos.t_ini, datos.t_fin)):
        bordes = np.geomspace(a, b, n_sub + 1)
        centros = np.sqrt(bordes[:-1] * bordes[1:])
        pesos = np.diff(bordes) / (b - a)
        fila = np.zeros(n * n_sub)
        fila[i * n_sub:(i + 1) * n_sub] = pesos
        tiempos.append(centros)
        filas.append(fila)
    return np.concatenate(tiempos), np.vstack(filas)


def _onda_valida(tiempo: np.ndarray, amplitud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forma de onda normalizada a pico 1, con tiempos crecientes y que
    termina en cero (el apagado)."""
    t = np.asarray(tiempo, float)
    a = np.asarray(amplitud, float)
    orden = np.argsort(t, kind="stable")
    t, a = t[orden], a[orden]
    pico = np.max(np.abs(a)) if len(a) else 1.0
    a = a / (pico if pico > 0 else 1.0)
    unicos = np.concatenate([[True], np.diff(t) > 0])
    return t[unicos], a[unicos]


def onda_por_omision(rampa_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Escalón con rampa lineal de apagado, si el archivo no trae TX_RAMP."""
    rampa = float(rampa_s) if np.isfinite(rampa_s) and rampa_s > 0 else 1e-6
    return np.array([-1e-3, -1e-3 + rampa, 0.0, rampa]) - rampa, np.array([0.0, 1.0, 1.0, 0.0])


class ModeloDirecto:
    """Respuesta y jacobiano de un sondeo (todos sus momentos).

    Parameters
    ----------
    geometria : GeometriaLazo
    momentos : list[DatosMomento]
        Uno por momento; el vector de datos es la concatenación en ese orden.
    integrar_gates : bool
        Promediar la respuesta dentro de cada ventana de gate.
    n_sub : int
        Tiempos por gate cuando se integra.
    """

    def __init__(self, geometria: GeometriaLazo, momentos: list[DatosMomento],
                 integrar_gates: bool = True, n_sub: int = 3) -> None:
        from simpeg.electromagnetics import time_domain as tdem

        self.geometria = geometria
        self.momentos = momentos
        self._promedios = []
        fuentes = []
        hx, hy = 0.5 * geometria.lado_x, 0.5 * geometria.lado_y
        lazo = np.array([[-hx, -hy, 0.0], [hx, -hy, 0.0], [hx, hy, 0.0], [-hx, hy, 0.0], [-hx, -hy, 0.0]])
        for datos in momentos:
            tiempos, promedio = _tiempos_de_gate(datos, integrar_gates, n_sub)
            t_onda, a_onda = _onda_valida(datos.onda_tiempo, datos.onda_amplitud)
            tiempos = np.maximum(tiempos, t_onda[-1] * 1.0001 + 1e-9)
            receptor = tdem.receivers.PointMagneticFluxTimeDerivative(
                np.array([[geometria.rx_x, geometria.rx_y, geometria.rx_z]]), tiempos, orientation="z")
            onda = tdem.sources.PiecewiseLinearWaveform(t_onda, a_onda)
            fuentes.append(tdem.sources.LineCurrent([receptor], location=lazo, waveform=onda,
                                                    current=float(geometria.vueltas)))
            self._promedios.append(promedio)
        self.survey = tdem.Survey(fuentes)
        self.promedio = _bloques_diagonales(self._promedios)
        self._simulaciones: dict = {}
        self._firma = _firma_survey(self.survey)
        self._activo = self.promedio

    def con_filas(self, filas: np.ndarray) -> "ModeloDirecto":
        """Copia que solo devuelve los gates ``filas`` y comparte las
        simulaciones (no recalcula coeficientes)."""
        import copy

        copia = copy.copy(self)
        copia._activo = self.promedio[np.asarray(filas)]
        return copia

    def _preparar(self, simulacion):
        """Comparte entre simulaciones los coeficientes de Hankel y de la
        transformada a tiempo (dependen solo del survey y la forma de onda).

        Calcularlos cuesta segundos por simulación; con esto se calculan
        una sola vez por sondeo, aunque cambie el número de capas.
        """
        guardados = _CACHE_COEFICIENTES.get(self._firma)
        if guardados is None:
            guardados = simulacion.get_coefficients()
            if len(_CACHE_COEFICIENTES) >= _MAX_CACHE:
                _CACHE_COEFICIENTES.pop(next(iter(_CACHE_COEFICIENTES)))
            else:
                pass
            _CACHE_COEFICIENTES[self._firma] = guardados
        else:
            # Los coeficientes de Hankel (rápidos, dependen de la geometría) se
            # recalculan; la transformada a tiempo con la forma de onda se copia.
            simulacion._compute_hankel_coefficients()
            simulacion._set_coefficients(guardados)
        return simulacion

    @property
    def n_datos(self) -> int:
        return self._activo.shape[0]

    # ------------------------------------------------------------------
    # Simulaciones de SimPEG (se crean una vez por tipo de modelo)
    # ------------------------------------------------------------------

    def _simulacion_espesores_fijos(self, espesores: np.ndarray):
        from simpeg import maps
        from simpeg.electromagnetics import time_domain as tdem

        espesores = np.array(espesores, dtype=float)  # copia escribible (geoana la exige)
        clave = ("fijos", tuple(np.round(espesores, 9)))
        if clave not in self._simulaciones:
            self._simulaciones = {k: v for k, v in self._simulaciones.items() if k[0] != "fijos"}
            self._simulaciones[clave] = self._preparar(tdem.Simulation1DLayered(
                survey=self.survey, thicknesses=espesores,
                sigmaMap=maps.ExpMap(nP=len(espesores) + 1)))
        else:
            pass
        return self._simulaciones[clave]

    def _simulacion_bloques(self, n_capas: int):
        from simpeg import maps
        from simpeg.electromagnetics import time_domain as tdem

        clave = ("bloques", int(n_capas))
        if clave not in self._simulaciones:
            cables = maps.Wires(("sigma", n_capas), ("espesor", n_capas - 1))
            self._simulaciones[clave] = self._preparar(tdem.Simulation1DLayered(
                survey=self.survey, sigmaMap=maps.ExpMap() * cables.sigma,
                thicknessesMap=maps.ExpMap() * cables.espesor))
        else:
            pass
        return self._simulaciones[clave]

    # ------------------------------------------------------------------
    # Respuesta y jacobianos
    # ------------------------------------------------------------------

    def respuesta(self, rho: np.ndarray, espesores: np.ndarray) -> np.ndarray:
        """−dBz/dt por gate [V/(A·m²)]."""
        rho = np.clip(np.asarray(rho, float), RHO_MIN, RHO_MAX)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(rho) == 1:
                simulacion = self._simulacion_espesores_fijos(np.array([]))
            else:
                simulacion = self._simulacion_espesores_fijos(np.asarray(espesores, float))
            crudo = simulacion.dpred(np.log(1.0 / rho))
        return -self._activo @ crudo

    def jacobiano_rho(self, rho: np.ndarray, espesores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Respuesta y ∂d/∂(ln ρ) con espesores fijos."""
        rho = np.clip(np.asarray(rho, float), RHO_MIN, RHO_MAX)
        m = np.log(1.0 / rho)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            simulacion = self._simulacion_espesores_fijos(np.asarray(espesores, float))
            crudo = simulacion.dpred(m)
            j_sigma = simulacion.getJ(m)
        # d = −P·dpred(ln σ) y ln σ = −ln ρ  →  ∂d/∂ln ρ = P·J
        return -self._activo @ crudo, self._activo @ np.asarray(j_sigma)

    def respuesta_bloques(self, rho: np.ndarray, espesores: np.ndarray) -> np.ndarray:
        """Respuesta de un modelo de bloques (sin jacobiano, más barata)."""
        rho = np.clip(np.asarray(rho, float), RHO_MIN, RHO_MAX)
        m = np.concatenate([np.log(1.0 / rho), np.log(np.array(espesores, dtype=float))])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            crudo = self._simulacion_bloques(len(rho)).dpred(m)
        return -self._activo @ crudo

    def jacobiano_bloques(self, rho: np.ndarray, espesores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Respuesta y ∂d/∂[ln ρ (k), ln h (k−1)] para un modelo de bloques."""
        rho = np.clip(np.asarray(rho, float), RHO_MIN, RHO_MAX)
        espesores = np.array(espesores, dtype=float)
        m = np.concatenate([np.log(1.0 / rho), np.log(espesores)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            simulacion = self._simulacion_bloques(len(rho))
            crudo = simulacion.dpred(m)
            j = np.asarray(simulacion.getJ(m))
        d = -self._activo @ crudo
        jac = -self._activo @ j
        jac[:, :len(rho)] *= -1.0
        return d, jac


def _firma_survey(survey) -> tuple:
    """Firma de lo que determina la transformada a tiempo: tiempos y onda."""
    partes = []
    for fuente in survey.source_list:
        onda = fuente.waveform
        partes.append((tuple(np.round(fuente.receiver_list[0].times, 12)),
                       tuple(np.round(onda.times, 12)), tuple(np.round(onda.currents, 9))))
    return tuple(partes)


def _bloques_diagonales(matrices: list[np.ndarray]) -> np.ndarray:
    filas = sum(m.shape[0] for m in matrices)
    columnas = sum(m.shape[1] for m in matrices)
    salida = np.zeros((filas, columnas))
    f = c = 0
    for m in matrices:
        salida[f:f + m.shape[0], c:c + m.shape[1]] = m
        f += m.shape[0]
        c += m.shape[1]
    return salida


def muestrear_modelo(espesores: np.ndarray, valores: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Valor del modelo de capas en las profundidades ``z``."""
    bases = np.cumsum(np.asarray(espesores, float))
    indices = np.searchsorted(bases, np.asarray(z, float), side="right")
    return np.asarray(valores, float)[np.minimum(indices, len(valores) - 1)]


def modelo_a_tabla(espesores: np.ndarray, rho: np.ndarray) -> "pd.DataFrame":
    """Esquema de modelo de la aplicación: ``capa, z_techo, z_base, espesor, rho``."""
    import pandas as pd

    espesores = np.asarray(espesores, float)
    bases = np.concatenate([np.cumsum(espesores), [np.inf]])
    techos = np.concatenate([[0.0], bases[:-1]])
    return pd.DataFrame({"capa": np.arange(1, len(rho) + 1), "z_techo": techos, "z_base": bases,
                         "espesor": bases - techos, "rho": np.asarray(rho, float)})


def verificar_semiespacio(rho: float = 100.0) -> float:
    """Error relativo del modelado frente a la solución de semiespacio
    de tiempo tardío (lazo circular equivalente, apagado instantáneo).

    Sirve como prueba de humo de la física: en tiempos tardíos, y con un
    encendido largo (campo primario estacionario), debe ser menor que 1 %.
    Con un encendido corto la diferencia crece: por eso la inversión usa
    la forma de onda real del archivo.
    """
    lado = 40.0
    t = np.geomspace(5e-4, 5e-3, 6)
    datos = DatosMomento("HM", t, t, t, np.array([-0.1, -0.1 + 1e-9, -1e-9, 0.0]),
                         np.array([0.0, 1.0, 1.0, 0.0]))
    modelo = ModeloDirecto(GeometriaLazo(lado, lado), [datos], integrar_gates=False)
    calculado = modelo.respuesta(np.array([rho]), np.array([]))
    mu0 = 4e-7 * np.pi
    analitico = (2 * mu0 * lado ** 2 / (5 * t)) * (mu0 / (4 * np.pi * t * rho)) ** 1.5
    return float(np.max(np.abs(calculado / analitico - 1)))
