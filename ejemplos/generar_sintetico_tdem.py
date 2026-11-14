"""
generar_sintetico_tdem.py
==================================================================
Genera archivos USF sintéticos (LM + HM, barridos crudos con ruido)
para probar la aplicación sin datos de cliente.

Los datos se calculan con el mismo modelo directo de la aplicación
(``forward_tdem``, SimPEG): lazo de 40 × 40 m, receptor al centro,
rampa trapezoidal declarada en ``TX_RAMP``. Cada barrido lleva ruido
gaussiano (5 % del dato ⊕ un piso por momento), así que el apilado,
el QC y la inversión se ejercitan igual que con datos de campo.

Uso::

    python ejemplos/generar_sintetico_tdem.py            # escribe en ejemplos/sinteticos/

Escribe tres sondeos a lo largo de una línea N–S (el modelo cambia
lateralmente) y ``coordenadas_ejemplo.csv``. El tercer sondeo se
escribe **sin ubicación** (``LOCATION`` con valores dummy) para probar
la captura manual de coordenadas.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

import forward_tdem as fw  # noqa: E402

DUMMY = 99999

#: Ventanas de cada momento: (primer borde, último borde, número de gates,
#: rampa de apagado, corriente, piso de ruido por barrido en V/(A·m²)).
MOMENTOS = {
    "LM": dict(canal=1, t0=1.0e-5, t1=1.0e-3, n=20, rampa=7.0e-6, corriente=1.2, piso=2.0e-7, frecuencia=480.08),
    "HM": dict(canal=2, t0=2.5e-5, t1=1.2e-2, n=28, rampa=1.3e-5, corriente=7.5, piso=1.0e-9, frecuencia=30.0),
}


def ventanas(t0: float, t1: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gates logarítmicos contiguos: (inicio, centro geométrico, fin)."""
    bordes = np.logspace(np.log10(t0), np.log10(t1), n + 1)
    return bordes[:-1], np.sqrt(bordes[:-1] * bordes[1:]), bordes[1:]


def onda(rampa: float, encendido: float = 2.0e-3) -> tuple[np.ndarray, np.ndarray]:
    """Trapecio: sube en 0.1 ms, se sostiene y se apaga de 0 a ``rampa``
    (misma convención de tiempos que los USF del WalkTEM)."""
    return (np.array([-encendido, -encendido + 1e-4, 0.0, rampa]), np.array([0.0, 1.0, 1.0, 0.0]))


def respuesta_modelo(espesores, rho, lado: float = 40.0) -> dict[str, np.ndarray]:
    """dB/dt normalizado de cada momento para un modelo de capas."""
    datos = []
    for nombre, p in MOMENTOS.items():
        ini, centro, fin = ventanas(p["t0"], p["t1"], p["n"])
        t_onda, a_onda = onda(p["rampa"])
        datos.append(fw.DatosMomento(nombre, centro, ini, fin, t_onda, a_onda))
    modelo = fw.ModeloDirecto(fw.GeometriaLazo(lado, lado), datos, integrar_gates=True, n_sub=3)
    d = modelo.respuesta(np.asarray(rho, float), np.asarray(espesores, float))
    n_lm = MOMENTOS["LM"]["n"]
    return {"LM": d[:n_lm], "HM": d[n_lm:]}


def _formato(valores, patron: str = "{:.6e}") -> str:
    return ", ".join(patron.format(v) for v in valores)


def escribir_usf(ruta: Path, nombre: str, espesores, rho, lon: float | None, lat: float | None,
                 elev: float = 1500.0, n_barridos: int = 40, inicio: datetime | None = None,
                 semilla: int = 0) -> Path:
    """Escribe un USF con ``n_barridos`` barridos crudos por momento."""
    rng = np.random.default_rng(semilla)
    inicio = inicio or datetime(2026, 11, 12, 10, 0, 0)
    verdad = respuesta_modelo(espesores, rho)
    ubicacion = f"{lon:.7f}, {lat:.7f}, {elev:.2f}" if lon is not None and lat is not None \
        else f"{DUMMY}, {DUMMY}, {DUMMY}"
    marca = inicio.strftime("%Y_%m%d_%H%M%S")
    lineas = [
        "//Data from TEMcompany instrument (SINTÉTICO, TlalliRes1D_TDEM)",
        f"//stb-file name: sintetico\\{marca}.stb",
        "//StackType: RawStack", "//InstrumentType: sTEM", "//InstrumentID: SINTETICO_0001",
        "", "//USF: Universal Sounding Format", "//SOUNDING_GROUP_NAME: Sintetico", "//SOUNDINGS: 1",
        "//EPSG: 4326", f"//DUMMY: {DUMMY}", "//END", "",
        f"/DATE: {inicio:%Y%m%d}", f"/DAYTIME: {inicio.hour + inicio.minute / 60 + inicio.second / 3600:.5f}",
        "/ARRAY: FIXED LOOP TEM", "/LOOP_SIZE: 40.000, 40.000", f"/SOUNDING_NAME: {nombre}",
        f"/SWEEPS: {2 * n_barridos}", f"/LOCATION: {ubicacion}", "/COIL_LOCATION: 0.00, 0.00",
        "/Z_DIRECTION: DOWN", "/LENGTH_UNITS: M", "/VOLTAGE_UNITS: V/AM2", "",
    ]
    numero = 0
    for momento, p in MOMENTOS.items():
        ini, centro, fin = ventanas(p["t0"], p["t1"], p["n"])
        t_onda, a_onda = onda(p["rampa"])
        rampa_txt = _formato([v for par in zip(t_onda, a_onda) for v in par], "{:.6e}")
        for _ in range(n_barridos):
            numero += 1
            corriente = p["corriente"] * (1 + rng.normal(0, 0.005))
            sigma = np.sqrt((0.05 * verdad[momento]) ** 2 + p["piso"] ** 2)
            valor = verdad[momento] + rng.normal(0, 1, len(centro)) * sigma
            lineas += [
                f"/SWEEP_NUMBER: {numero}", f"/CURRENT: {corriente:.2f}", f"/FREQUENCY: {p['frecuencia']}",
                "/SWEEP_IS_NOISE: 0", "/FIELD_SHIFT_FACTOR: 1.0000", "/TIME_DELAY: 0.000e+00",
                f"/RAMP_TIME: {p['rampa']:.6e}", f"/TX_TURNONTIME: {DUMMY}", f"/RAMP_TIME_ON: {DUMMY}",
                f"/TX_RAMP: {rampa_txt}", f"/POINTS: {len(centro)}", "/LOW_PASS: 650000, 1.00, 800000, 1.00",
                f"/CHANNEL: {p['canal']}", "/END", "",
                "        TIME,      VOLTAGE, ERROR_BAR, QUALITY,     TIMEOPEN,    TIMECLOSE",
            ]
            for t, v, s, a, b in zip(centro, valor, sigma, ini, fin):
                lineas.append(f"{t:.5e}, {v:.5e}, {100 * s / abs(v):.2f}, 1, {a:.5e}, {b:.5e}")
            lineas += ["/END", ""]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\r\n".join(lineas) + "\r\n", encoding="utf-8")
    return ruta


#: Tres sondeos sobre una línea N–S; el conductor se adelgaza hacia el sur.
SONDEOS = (
    ("SINT_001", (20.0, 40.0, 80.0), (60.0, 8.0, 40.0, 150.0), 20.6800),
    ("SINT_002", (20.0, 30.0, 90.0), (60.0, 10.0, 40.0, 150.0), 20.6791),
    ("SINT_003", (25.0, 20.0, 90.0), (60.0, 15.0, 40.0, 150.0), 20.6782),
)
LONGITUD = -103.3400


def generar(carpeta: Path, n_barridos: int = 40) -> list[Path]:
    """Escribe los tres USF de ejemplo y el archivo de coordenadas."""
    rutas = []
    inicio = datetime(2026, 11, 12, 10, 0, 0)
    for i, (nombre, espesores, rho, lat) in enumerate(SONDEOS):
        momento = inicio + timedelta(minutes=40 * i)
        con_gps = i < len(SONDEOS) - 1
        ruta = carpeta / f"{nombre}_{momento:%Y_%m%d_%H%M%S}.usf"
        rutas.append(escribir_usf(ruta, nombre, espesores, rho, LONGITUD if con_gps else None,
                                  lat if con_gps else None, n_barridos=n_barridos, inicio=momento, semilla=i))
    # Coordenadas del tercero (sin GPS en el archivo), en UTM 13N.
    import geodesia_tdem as geo

    x, y = geo.geograficas_a_proyectadas([LONGITUD], [SONDEOS[-1][3]], 32613)
    pd.DataFrame({"Sondeo": [SONDEOS[-1][0]], "X": [round(float(x[0]), 2)], "Y": [round(float(y[0]), 2)],
                  "Z": [1500.0]}).to_csv(carpeta / "coordenadas_ejemplo.csv", index=False)
    return rutas


if __name__ == "__main__":
    destino = Path(__file__).resolve().parent / "sinteticos"
    for r in generar(destino):
        print(f"Escrito: {r}")
    print(f"Escrito: {destino / 'coordenadas_ejemplo.csv'}")
