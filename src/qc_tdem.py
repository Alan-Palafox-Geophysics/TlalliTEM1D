"""
qc_tdem.py
==================================================================
Control de calidad (QC) de sondeos TDEM.

Cada revisión es una función pequeña que recibe el contexto de un
sondeo y devuelve una lista de :class:`Hallazgo`. Para agregar una
revisión basta con escribir la función y añadirla a
``REVISIONES_SONDEO``. Los umbrales viven en :class:`ConfigQC`.

Severidades
--------------
====  ===============================================================
OK    la revisión pasó
INFO  dato informativo, no cuenta para la valoración
WARN  observación: los datos sirven, pero hay que tenerla en cuenta
FAIL  problema serio: no invertir sin revisar / considerar re-medir
====  ===============================================================

Valoración por sondeo y momento: **Buena** (sin observaciones),
**Aceptable** (1–2 WARN), **Revisar** (≥ 3 WARN), **Deficiente** (algún
FAIL). Puntaje = 100 − 25·FAIL − 8·WARN.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

import procesamiento_tdem as pr

logger = logging.getLogger("tdem.qc")

ORDEN_SEVERIDAD = {"OK": 0, "INFO": 1, "WARN": 2, "FAIL": 3}
RADIO_TIERRA_M = 6_371_000.0


@dataclass
class ConfigQC:
    """Umbrales del control de calidad."""

    corriente_cv_aviso_pct: float = 3.0
    corriente_cv_falla_pct: float = 10.0
    corriente_deriva_aviso_pct: float = 5.0
    corriente_deriva_falla_pct: float = 15.0
    gps_dispersion_aviso_m: float = 5.0
    gps_dispersion_falla_m: float = 15.0
    velocidad_aviso_ms: float = 0.5
    tolerancia_centro_m: float = 10.0
    frac_barridos_atipicos_aviso: float = 0.05
    gates_utiles_minimos: dict = field(default_factory=lambda: {"LM": 8, "HM": 12})
    solape_aviso_pct: float = 10.0
    solape_falla_pct: float = 25.0
    rho_a_min: float = 0.1
    rho_a_max: float = 10000.0
    factor_pausa: float = 5.0
    tolerancia_usf_xyz: float = 1e-3
    k_mad: float = 5.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Hallazgo:
    sondeo: str
    momento: Optional[str]      # LM, HM o None si aplica a todo el sondeo
    categoria: str              # metadatos | adquisicion | posicion | datos | consistencia | campana
    revision: str
    severidad: str
    valor: Optional[float]
    umbral: Optional[str]
    mensaje: str


@dataclass
class Contexto:
    """Lo que una revisión necesita de un sondeo."""

    sondeo: str
    meta: dict[str, Any]
    barridos: pd.DataFrame
    gates: pd.DataFrame
    ventanas: pd.DataFrame
    apilado: pd.DataFrame
    config: ConfigQC

    def hallazgo(self, momento, categoria, revision, severidad, mensaje, valor=None, umbral=None) -> Hallazgo:
        try:
            valor = None if valor is None or not np.isfinite(float(valor)) else float(valor)
        except (TypeError, ValueError):
            valor = None
        return Hallazgo(self.sondeo, momento, categoria, revision, severidad, valor, umbral, mensaje)


def _severidad(valor: float, aviso: float, falla: float) -> str:
    if valor is None or not np.isfinite(valor):
        return "INFO"
    else:
        pass
    if valor >= falla:
        return "FAIL"
    else:
        return "WARN" if valor >= aviso else "OK"


def xy_local_m(lat, lon, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """Proyección local equirectangular (suficiente a menos de 10 km)."""
    x = np.radians(np.asarray(lon, float) - lon0) * RADIO_TIERRA_M * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat, float) - lat0) * RADIO_TIERRA_M
    return x, y


def _numero(valor: Any) -> float:
    try:
        return float(valor)
    except (TypeError, ValueError):
        return np.nan


# ======================================================================
# 1. Metadatos
# ======================================================================


def revisar_encabezado(ctx: Contexto) -> list[Hallazgo]:
    requeridos = {"lon": "WARN", "lat": "WARN", "lado_x": "FAIL", "lado_y": "FAIL",
                  "inicio": "WARN", "id_instrumento": "INFO"}
    faltan = [k for k in requeridos if not np.isfinite(_numero(ctx.meta.get(k))) and k in ("lon", "lat", "lado_x", "lado_y")]
    faltan += [k for k in ("inicio", "id_instrumento") if not ctx.meta.get(k)]
    if not faltan:
        return [ctx.hallazgo(None, "metadatos", "encabezado_completo", "OK",
                             "Ubicación, tamaño del lazo, fecha e instrumento presentes.")]
    else:
        pass
    peor = max((requeridos[k] for k in faltan), key=ORDEN_SEVERIDAD.get)
    texto = f"Faltan en el encabezado: {', '.join(faltan)}."
    if "lon" in faltan or "lat" in faltan:
        texto += " Capture las coordenadas en la pantalla «Proyecto y datos»."
    else:
        pass
    return [ctx.hallazgo(None, "metadatos", "encabezado_completo", peor, texto)]


def revisar_barridos(ctx: Contexto) -> list[Hallazgo]:
    salida = []
    declarados, leidos = _numero(ctx.meta.get("n_barridos_declarados")), len(ctx.barridos)
    if np.isfinite(declarados) and declarados != leidos:
        salida.append(ctx.hallazgo(None, "metadatos", "barridos_declarados", "WARN",
                                   f"El encabezado declara {declarados:.0f} barridos y se leyeron {leidos}.",
                                   leidos, f"= {declarados:.0f}"))
    else:
        salida.append(ctx.hallazgo(None, "metadatos", "barridos_declarados", "OK",
                                   f"{leidos} barridos leídos, igual a lo declarado."))

    filas = ctx.gates.groupby("barrido").size().rename("filas")
    tabla = ctx.barridos.set_index("barrido")[["n_gates_declarados"]].join(filas)
    distintos = tabla[tabla["filas"] != tabla["n_gates_declarados"]]
    if len(distintos):
        salida.append(ctx.hallazgo(None, "metadatos", "gates_por_barrido", "WARN",
                                   f"{len(distintos)} barridos con un número de gates distinto a POINTS.",
                                   len(distintos), "0"))
    else:
        salida.append(ctx.hallazgo(None, "metadatos", "gates_por_barrido", "OK",
                                   "Todos los barridos tienen los gates declarados."))

    n_lm = int((ctx.barridos["momento"] == "LM").sum())
    n_hm = int((ctx.barridos["momento"] == "HM").sum())
    salida.append(ctx.hallazgo(None, "metadatos", "balance_momentos",
                               "OK" if n_lm == n_hm and n_lm > 0 else "WARN",
                               f"{n_lm} barridos LM y {n_hm} barridos HM.", n_lm - n_hm, "0"))
    if int(ctx.barridos["es_ruido"].astype(bool).sum()) == 0:
        salida.append(ctx.hallazgo(None, "metadatos", "medicion_de_ruido", "INFO",
                                   "Sin barridos de ruido (Tx apagado): el ruido de fondo se estima solo "
                                   "con la dispersión entre stacks."))
    else:
        pass
    return salida


def revisar_parametros_constantes(ctx: Contexto) -> list[Hallazgo]:
    salida = []
    parametros = ["frecuencia_hz", "rampa_s", "factor_campo", "retardo_s", "filtro1_hz", "filtro2_hz",
                  "n_gates_declarados"]
    for momento, grupo in ctx.barridos.groupby("momento"):
        cambian = [p for p in parametros if p in grupo and grupo[p].nunique(dropna=True) > 1]
        g = ctx.gates[ctx.gates["momento"] == momento]
        if g.groupby("gate")["tiempo"].nunique().max() > 1:
            cambian.append("tiempos de gate")
        else:
            pass
        if cambian:
            salida.append(ctx.hallazgo(momento, "metadatos", "parametros_constantes", "WARN",
                                       f"Parámetros que cambian entre barridos: {', '.join(cambian)}."))
        else:
            f = grupo.iloc[0]
            salida.append(ctx.hallazgo(
                momento, "metadatos", "parametros_constantes", "OK",
                f"Constantes en {len(grupo)} barridos: f = {f['frecuencia_hz']:g} Hz, rampa = "
                f"{f['rampa_s'] * 1e6:.1f} µs, factor = {f['factor_campo']:g}, retardo = "
                f"{f['retardo_s'] * 1e9:.0f} ns, {f['n_gates_declarados']:.0f} gates."))
    return salida


def revisar_ventanas(ctx: Contexto) -> list[Hallazgo]:
    """open < centro < close y sin traslapes entre gates vecinos."""
    salida = []
    for (fuente, momento), g in ctx.ventanas.groupby(["fuente", "momento"]):
        g = g.sort_values("gate")
        invertidas = int(((g["t_ini"] >= g["tiempo"]) | (g["tiempo"] >= g["t_fin"])).sum())
        traslapes = int((g["t_fin"].to_numpy()[:-1] > g["t_ini"].to_numpy()[1:] * (1 + 1e-6)).sum())
        principal = fuente == ctx.meta.get("fuente_principal", "USF")
        if invertidas:
            salida.append(ctx.hallazgo(momento, "consistencia", f"ventanas_{fuente}",
                                       "FAIL" if principal else "WARN",
                                       f"Tabla de gates {fuente}: {invertidas} gates con apertura/centro/cierre "
                                       "fuera de orden. La tabla de tiempos no es válida.", invertidas, "0"))
            continue
        else:
            pass
        if traslapes:
            ancho = float((g["t_fin"] / g["t_ini"]).median())
            separacion = float(np.exp(np.median(np.diff(np.log(g["tiempo"])))))
            salida.append(ctx.hallazgo(
                momento, "consistencia", f"ventanas_{fuente}", "WARN",
                f"Tabla de gates {fuente}: {traslapes} de {len(g) - 1} pares de ventanas vecinas se traslapan "
                f"(ancho cierre/apertura = {ancho:.3f}, separación entre centros = {separacion:.3f}). Si fue "
                "intencional, los gates vecinos quedan correlacionados; si no, la configuración de gates es "
                "incorrecta.", traslapes, "0"))
        else:
            salida.append(ctx.hallazgo(momento, "consistencia", f"ventanas_{fuente}", "OK",
                                       f"Tabla de gates {fuente} ordenada y sin traslapes ({len(g)} gates, "
                                       f"{g['tiempo'].min():.2e}–{g['tiempo'].max():.2e} s)."))
    return salida


def revisar_usf_contra_xyz(ctx: Contexto) -> list[Hallazgo]:
    if "dbdt_xyz" not in ctx.gates or ctx.gates["dbdt_xyz"].isna().all():
        return []
    else:
        pass
    salida = []
    tol = ctx.config.tolerancia_usf_xyz
    g = ctx.gates
    diferencia = float(((g["dbdt"] - g["dbdt_xyz"]).abs() / g["dbdt_xyz"].abs()).median())
    salida.append(ctx.hallazgo(None, "consistencia", "usf_xyz_dato", "OK" if diferencia <= tol else "WARN",
                               f"Diferencia relativa mediana USF vs XYZ/I: {diferencia:.1e}.", diferencia,
                               f"≤ {tol:g}"))
    if "corriente_xyz_a" in ctx.barridos:
        dif_i = float((ctx.barridos["corriente_a"] - ctx.barridos["corriente_xyz_a"]).abs().max())
        salida.append(ctx.hallazgo(None, "consistencia", "usf_xyz_corriente", "OK" if dif_i <= 0.006 else "WARN",
                                   f"Diferencia máxima de corriente USF vs XYZ: {dif_i:.3f} A (el USF redondea "
                                   "a 0.01 A).", dif_i, "≤ 0.006 A"))
    else:
        pass
    usf = ctx.ventanas[ctx.ventanas["fuente"] == "USF"]
    xyz = ctx.ventanas[ctx.ventanas["fuente"] == "XYZ"]
    if len(usf) and len(xyz):
        unidas = usf.merge(xyz, on=["momento", "gate"], suffixes=("", "_xyz"))
        for columna, etiqueta in (("t_ini", "apertura"), ("tiempo", "centro"), ("t_fin", "cierre")):
            dif = float(((unidas[columna] - unidas[f"{columna}_xyz"]).abs() / unidas[columna]).max())
            salida.append(ctx.hallazgo(None, "consistencia", f"usf_xyz_gate_{etiqueta}",
                                       "OK" if dif <= tol else "WARN",
                                       f"Diferencia máxima en el {etiqueta} de los gates USF vs XYZ: "
                                       f"{100 * dif:.2f} %.", dif, f"≤ {tol:g}"))
    else:
        pass
    return salida


def revisar_bandera_calidad(ctx: Contexto) -> list[Hallazgo]:
    if "calidad" not in ctx.gates or ctx.gates["calidad"].isna().all():
        return []
    else:
        pass
    salida = []
    for momento, g in ctx.gates.groupby("momento"):
        malos = float((g["calidad"] == 0).mean())
        severidad = "OK" if malos == 0 else ("WARN" if malos < 0.2 else "FAIL")
        salida.append(ctx.hallazgo(momento, "datos", "bandera_calidad", severidad,
                                   f"{100 * malos:.1f} % de datos con QUALITY = 0 (marcados por el equipo).",
                                   malos, "< 0.2"))
    return salida


# ======================================================================
# 2. Adquisición
# ======================================================================


def revisar_corriente(ctx: Contexto) -> list[Hallazgo]:
    salida = []
    c = ctx.config
    for momento, grupo in ctx.barridos.groupby("momento"):
        corriente = grupo.sort_values("indice_momento")["corriente_a"].to_numpy(float)
        media = float(np.mean(corriente))
        cv = 100 * float(np.std(corriente, ddof=1)) / media
        n10 = max(1, len(corriente) // 10)
        deriva = 100 * (corriente[-n10:].mean() - corriente[:n10].mean()) / media
        salida.append(ctx.hallazgo(
            momento, "adquisicion", "estabilidad_corriente",
            _severidad(cv, c.corriente_cv_aviso_pct, c.corriente_cv_falla_pct),
            f"Corriente media {media:.2f} A (mín {corriente.min():.2f}, máx {corriente.max():.2f}); "
            f"coeficiente de variación {cv:.1f} %.", cv, f"< {c.corriente_cv_aviso_pct} %"))
        salida.append(ctx.hallazgo(
            momento, "adquisicion", "deriva_corriente",
            _severidad(abs(deriva), c.corriente_deriva_aviso_pct, c.corriente_deriva_falla_pct),
            f"Deriva de corriente entre el inicio y el final: {deriva:+.1f} %. El dato está normalizado por "
            "corriente; una deriva grande indica batería baja o calentamiento.", deriva,
            f"|x| < {c.corriente_deriva_aviso_pct} %"))
        fuera = np.abs(corriente / np.median(corriente) - 1) > 0.05
        n_arranque = int(np.argmin(fuera)) if not fuera.all() else len(fuera)
        if n_arranque:
            salida.append(ctx.hallazgo(
                momento, "adquisicion", "arranque_corriente", "INFO",
                f"Los primeros {n_arranque} barridos tienen la corriente desviada más de 5 % de la mediana "
                f"(arranque del transmisor; el primero {100 * (corriente[0] / np.median(corriente) - 1):+.0f} %). "
                "Al estar normalizados por corriente no afectan el apilado.", n_arranque))
        else:
            pass
    return salida


def revisar_tiempos(ctx: Contexto) -> list[Hallazgo]:
    if "hora" not in ctx.barridos or ctx.barridos["hora"].isna().all():
        return [ctx.hallazgo(None, "adquisicion", "tiempos", "INFO",
                             "Sin hora por barrido (no hay XYZ): no se revisan pausas.")]
    else:
        pass
    salida = []
    horas = pd.to_datetime(ctx.barridos["hora"])
    duracion = float((horas.max() - horas.min()).total_seconds())
    salida.append(ctx.hallazgo(None, "adquisicion", "duracion", "INFO",
                               f"Duración del sondeo: {duracion / 60:.1f} min.", duracion))
    for momento, grupo in ctx.barridos.assign(hora=horas).groupby("momento"):
        dt = grupo.sort_values("hora")["hora"].diff().dt.total_seconds().dropna()
        cadencia = float(dt.median())
        pausas = dt[dt > ctx.config.factor_pausa * cadencia]
        texto = f"Un barrido cada {cadencia:.2f} s."
        if len(pausas):
            texto += f" {len(pausas)} pausa(s); la mayor de {pausas.max():.0f} s."
        else:
            pass
        salida.append(ctx.hallazgo(momento, "adquisicion", "pausas", "OK" if pausas.empty else "WARN",
                                   texto, len(pausas), "0"))
    return salida


# ======================================================================
# 3. Posición
# ======================================================================


def revisar_gps(ctx: Contexto) -> list[Hallazgo]:
    if "gps_lat" not in ctx.barridos or ctx.barridos["gps_lat"].isna().all():
        return []
    else:
        pass
    c, salida = ctx.config, []
    b = ctx.barridos.dropna(subset=["gps_lat", "gps_lon"])
    lat0, lon0 = float(b["gps_lat"].median()), float(b["gps_lon"].median())
    x, y = xy_local_m(b["gps_lat"], b["gps_lon"], lat0, lon0)
    p95 = float(np.percentile(np.hypot(x, y), 95))
    salida.append(ctx.hallazgo(None, "posicion", "dispersion_gps",
                               _severidad(p95, c.gps_dispersion_aviso_m, c.gps_dispersion_falla_m),
                               f"95 % de las posiciones GPS dentro de {p95:.1f} m de la mediana (el equipo debe "
                               "estar quieto).", p95, f"< {c.gps_dispersion_aviso_m} m"))
    vmax = float(b["gps_vel_ms"].max())
    salida.append(ctx.hallazgo(None, "posicion", "velocidad_gps", "OK" if vmax < c.velocidad_aviso_ms else "WARN",
                               f"Velocidad GPS máxima {vmax:.2f} m/s.", vmax, f"< {c.velocidad_aviso_ms} m/s"))

    lon, lat = _numero(ctx.meta.get("lon")), _numero(ctx.meta.get("lat"))
    if not (np.isfinite(lon) and np.isfinite(lat)):
        return salida
    else:
        pass
    xc, yc = xy_local_m([lat], [lon], lat0, lon0)
    distancia = float(np.hypot(xc[0], yc[0]))
    offset = float(np.hypot(_numero(ctx.meta.get("offset_gps_x")), _numero(ctx.meta.get("offset_gps_y"))))
    media_diagonal = float(np.hypot(_numero(ctx.meta.get("lado_x")), _numero(ctx.meta.get("lado_y"))) / 2)
    if np.isfinite(offset):
        diferencia = abs(distancia - offset)
        salida.append(ctx.hallazgo(None, "posicion", "centro_vs_gps",
                                   "OK" if diferencia <= c.tolerancia_centro_m else "WARN",
                                   f"Centro del lazo a {distancia:.1f} m de la mediana GPS; el desplazamiento "
                                   f"declarado del GPS es {offset:.1f} m (diferencia {diferencia:.1f} m).",
                                   diferencia, f"< {c.tolerancia_centro_m} m"))
        if offset > 1.05 * media_diagonal:
            esquina = abs(distancia - media_diagonal) < abs(distancia - offset)
            texto = (f"El desplazamiento GPS declarado ({ctx.meta.get('offset_gps_x')}, "
                     f"{ctx.meta.get('offset_gps_y')}) m pone al GPS a {offset:.1f} m del centro, fuera del lazo "
                     f"(esquina a {media_diagonal:.1f} m).")
            if esquina:
                texto += (f" La distancia observada ({distancia:.1f} m) es más compatible con un GPS en la esquina: "
                          "probablemente se capturó la diagonal en cada eje en lugar de (x, y) = "
                          f"({_numero(ctx.meta.get('lado_x')) / 2:.0f}, {_numero(ctx.meta.get('lado_y')) / 2:.0f}) m.")
            else:
                texto += " Verifique el desplazamiento en campo."
            salida.append(ctx.hallazgo(None, "posicion", "offset_gps", "WARN", texto, offset,
                                       f"≤ {media_diagonal:.1f} m"))
        else:
            pass
    else:
        pass
    return salida


# ======================================================================
# 4. Datos medidos
# ======================================================================


def revisar_gates_utiles(ctx: Contexto) -> list[Hallazgo]:
    salida = []
    for momento, ap in ctx.apilado.groupby("momento"):
        utiles = ap[ap["util"].astype(bool)]
        minimo = ctx.config.gates_utiles_minimos.get(momento, 8)
        n = len(utiles)
        severidad = "OK" if n >= minimo else ("WARN" if n >= minimo / 2 else "FAIL")
        if n:
            ultimo = utiles.iloc[-1]
            rho_mediana = float(np.nanmedian(utiles["rho_a"])) if "rho_a" in utiles else np.nan
            profundidad = float(pr.profundidad_difusion(ultimo["tiempo"], rho_mediana))
            texto = (f"{n} de {len(ap)} gates útiles (SNR ≥ umbral), {utiles['tiempo'].min():.2e}–"
                     f"{ultimo['tiempo']:.2e} s. Profundidad de difusión aproximada al último gate útil: "
                     f"{profundidad:.0f} m.")
        else:
            texto = "Ningún gate útil."
        salida.append(ctx.hallazgo(momento, "datos", "gates_utiles", severidad, texto, n, f"≥ {minimo}"))
        ruido = float(ap.tail(3)["error_std"].median())
        salida.append(ctx.hallazgo(momento, "datos", "piso_de_ruido", "INFO",
                                   f"Error estándar del apilado en los últimos 3 gates: {ruido:.1e} V/(A·m²).",
                                   ruido))
    return salida


def revisar_forma_curva(ctx: Contexto) -> list[Hallazgo]:
    """La curva dB/dt de lazo central debe decaer siempre."""
    salida = []
    tiempos_repunte: dict[str, np.ndarray] = {}
    for momento, ap in ctx.apilado.groupby("momento"):
        repuntes = ap[ap["repunte"].astype(bool)]
        tiempos_repunte[momento] = repuntes["tiempo"].to_numpy(float)
        if repuntes.empty:
            salida.append(ctx.hallazgo(momento, "datos", "decaimiento_monotono", "OK",
                                       "La curva apilada decae de forma monótona."))
        else:
            salida.append(ctx.hallazgo(
                momento, "datos", "decaimiento_monotono", "FAIL" if len(repuntes) >= 2 else "WARN",
                f"La señal SUBE de forma significativa (> 3σ) en los gates "
                f"{', '.join(map(str, repuntes['gate']))}. Un medio 1D no produce repuntes: suele ser "
                "acoplamiento con metal o cables (cercas, tuberías, líneas eléctricas) o un problema en la "
                "configuración de gates. Esos gates no se invierten salvo que se indique lo contrario.",
                len(repuntes), "0"))
        negativos = ap[(ap["dbdt"] < 0) & (ap["snr"] >= 3)]
        if len(negativos):
            salida.append(ctx.hallazgo(momento, "datos", "cambio_de_signo", "WARN",
                                       f"Valores negativos con SNR alto en los gates "
                                       f"{', '.join(map(str, negativos['gate']))}: posible efecto IP o "
                                       "acoplamiento.", len(negativos), "0"))
        else:
            pass
    lm, hm = tiempos_repunte.get("LM", np.array([])), tiempos_repunte.get("HM", np.array([]))
    if len(lm) and len(hm):
        coinciden = [t for t in hm if np.any(np.abs(np.log(lm / t)) < np.log(1.3))]
        if coinciden:
            salida.append(ctx.hallazgo(
                None, "datos", "repuntes_coinciden_lm_hm", "WARN",
                f"Los repuntes aparecen en LM y en HM a los mismos tiempos ({', '.join(f'{t:.1e}' for t in coinciden)} "
                "s). LM y HM son mediciones independientes, así que el rasgo está en la señal: lo más probable es "
                "acoplamiento con infraestructura metálica cerca del lazo.", len(coinciden), "0"))
        else:
            pass
    else:
        pass
    return salida


def revisar_pendientes(ctx: Contexto) -> list[Hallazgo]:
    tabla = pr.pendientes(ctx.apilado)
    salida = []
    if tabla.empty:
        return salida
    else:
        pass
    for momento, p in tabla.groupby("momento"):
        tardia = float(p.tail(3)["pendiente"].median())
        salida.append(ctx.hallazgo(
            momento, "datos", "pendiente_tardia", "WARN" if tardia > -1.0 or tardia < -4.5 else "OK",
            f"Pendiente log-log en los últimos gates útiles: {tardia:.2f} (semiespacio ≈ −2.5; > −1 sugiere "
            "ruido o acoplamiento; < −4.5 efecto IP o medio muy resistivo).", tardia, "[−4.5, −1.0]"))
        plano = (p["pendiente"] > -0.5).astype(int)
        racha = plano.groupby((plano != plano.shift()).cumsum()).transform("sum") * plano
        if (racha >= 2).any():
            t = p.loc[racha >= 2, "tiempo"]
            salida.append(ctx.hallazgo(
                momento, "datos", "tramo_plano", "WARN",
                f"La curva se aplana (pendiente > −0.5) entre {t.min():.1e} y {t.max():.1e} s. Un conductor "
                "profundo puede aplanarla, pero un tramo casi horizontal también es típico de ruido coherente o "
                "acoplamiento; valídelo con la inversión y con sondeos vecinos.", float((racha >= 2).sum()), "0"))
        else:
            pass
    return salida


def revisar_barras_error(ctx: Contexto) -> list[Hallazgo]:
    salida = []
    for momento, ap in ctx.apilado[ctx.apilado["util"].astype(bool)].groupby("momento"):
        razon = float((ap["error_equipo"] / ap["error_rel_emp"]).median())
        coincide = 0.5 <= razon <= 2
        salida.append(ctx.hallazgo(
            momento, "datos", "barras_de_error", "OK" if coincide else "WARN",
            f"Error reportado por el equipo / dispersión observada entre stacks = {razon:.2f}. " +
            ("Coinciden." if coincide else "No representan la variabilidad real: la inversión usa el error "
             "empírico del apilado más un piso relativo."), razon, "[0.5, 2]"))
    return salida


def revisar_barridos_atipicos(ctx: Contexto) -> list[Hallazgo]:
    salida = []
    por_barrido = pr.atipicos_por_barrido(ctx.gates, ctx.config.k_mad)
    for momento, g in por_barrido.groupby("momento"):
        frac = float((g["frac_atipicos"] > 0.2).mean())
        salida.append(ctx.hallazgo(
            momento, "datos", "barridos_atipicos",
            "OK" if frac <= ctx.config.frac_barridos_atipicos_aviso else "WARN",
            f"{100 * frac:.1f} % de los barridos tienen más del 20 % de sus gates atípicos (picos). El apilado "
            "robusto los descarta gate por gate.", frac, f"≤ {ctx.config.frac_barridos_atipicos_aviso}"))
    return salida


def revisar_estabilidad(ctx: Contexto) -> list[Hallazgo]:
    """La primera y la segunda mitad del sondeo deben dar la misma curva."""
    salida = []
    datos = ctx.gates.merge(ctx.barridos[["barrido", "indice_momento"]], on="barrido")
    for momento, g in datos.groupby("momento"):
        utiles = ctx.apilado[(ctx.apilado["momento"] == momento) & ctx.apilado["util"].astype(bool)]["gate"]
        g = g[g["gate"].isin(utiles)]
        if g.empty:
            continue
        else:
            pass
        mitad = g["indice_momento"].max() / 2
        a = g[g["indice_momento"] <= mitad].groupby("gate")["dbdt"].agg(["mean", "sem"])
        b = g[g["indice_momento"] > mitad].groupby("gate")["dbdt"].agg(["mean", "sem"])
        z = (a["mean"] - b["mean"]).abs() / np.hypot(a["sem"], b["sem"])
        relativo = (100 * (b["mean"] / a["mean"] - 1)).abs()
        n_malos = int(((z > 3) & (relativo > 2)).sum())
        salida.append(ctx.hallazgo(
            momento, "datos", "estabilidad_temporal", "OK" if n_malos <= 1 else "WARN",
            f"Primera vs segunda mitad de los stacks: diferencia mediana {relativo.median():.1f} %; {n_malos} "
            "gates difieren más de 2 % y 3σ.", n_malos, "≤ 1"))
    return salida


def revisar_rho_a(ctx: Contexto) -> list[Hallazgo]:
    salida = []
    c = ctx.config
    if "rho_a" not in ctx.apilado:
        return salida
    else:
        pass
    for momento, ap in ctx.apilado[ctx.apilado["util"].astype(bool)].groupby("momento"):
        rho = ap["rho_a"].dropna()
        if rho.empty:
            continue
        else:
            pass
        fuera = int(((rho < c.rho_a_min) | (rho > c.rho_a_max)).sum())
        salida.append(ctx.hallazgo(momento, "datos", "resistividad_aparente", "OK" if fuera == 0 else "WARN",
                                   f"ρa de tiempo tardío en gates útiles: {rho.min():.1f}–{rho.max():.1f} Ω·m "
                                   f"(mediana {rho.median():.1f}).", float(rho.median()),
                                   f"[{c.rho_a_min}, {c.rho_a_max}] Ω·m"))
    return salida


def revisar_rampa(ctx: Contexto) -> list[Hallazgo]:
    salida = []
    for momento, ap in ctx.apilado.groupby("momento"):
        n = int(ap["efecto_rampa"].astype(bool).sum())
        if n:
            salida.append(ctx.hallazgo(momento, "datos", "efecto_rampa", "INFO",
                                       f"{n} gate(s) muy cerca del apagado (t < 2 × rampa de "
                                       f"{ap['rampa_s'].iloc[0] * 1e6:.1f} µs): la inversión los modela con la "
                                       "forma de onda del archivo.", n))
        else:
            pass
    return salida


def revisar_solape(ctx: Contexto) -> list[Hallazgo]:
    tabla = pr.solape_lm_hm(ctx.apilado)
    c = ctx.config
    if tabla.empty:
        return [ctx.hallazgo(None, "consistencia", "solape_lm_hm", "INFO",
                             "No hay solape útil entre LM y HM para compararlos.")]
    else:
        pass
    diferencia = float(tabla["diferencia_pct"].median())
    return [ctx.hallazgo(
        None, "consistencia", "solape_lm_hm", _severidad(abs(diferencia), c.solape_aviso_pct, c.solape_falla_pct),
        f"En el solape ({tabla['tiempo'].min():.1e}–{tabla['tiempo'].max():.1e} s, {len(tabla)} gates) el HM "
        f"difiere del LM en {diferencia:+.1f} % (mediana). Parte se debe a las rampas distintas, que la inversión "
        "modela; si el ajuste no reconcilia ambos momentos, revise los factores de calibración.",
        diferencia, f"|x| < {c.solape_aviso_pct} %")]


REVISIONES_SONDEO = [
    revisar_encabezado, revisar_barridos, revisar_parametros_constantes, revisar_ventanas,
    revisar_usf_contra_xyz, revisar_bandera_calidad, revisar_corriente, revisar_tiempos, revisar_gps,
    revisar_gates_utiles, revisar_forma_curva, revisar_pendientes, revisar_barras_error,
    revisar_barridos_atipicos, revisar_estabilidad, revisar_rho_a, revisar_rampa, revisar_solape,
]


def evaluar_sondeo(sondeo: str, meta: dict[str, Any], barridos: pd.DataFrame, gates: pd.DataFrame,
                   ventanas: pd.DataFrame, apilado: pd.DataFrame,
                   config: Optional[ConfigQC] = None) -> pd.DataFrame:
    """Corre todas las revisiones de un sondeo y devuelve los hallazgos."""
    ctx = Contexto(sondeo, meta, barridos, gates, ventanas, apilado, config or ConfigQC())
    hallazgos: list[Hallazgo] = []
    for revision in REVISIONES_SONDEO:
        try:
            hallazgos.extend(revision(ctx))
        except Exception as exc:  # una revisión rota no debe tumbar el QC completo
            logger.exception("Revisión %s falló", revision.__name__)
            hallazgos.append(ctx.hallazgo(None, "metadatos", revision.__name__, "INFO",
                                          f"La revisión no se pudo ejecutar: {exc}"))
    return pd.DataFrame([asdict(h) for h in hallazgos])


# ======================================================================
# 5. Campaña (comparan sondeos entre sí)
# ======================================================================


def revisiones_campana(coordenadas: pd.DataFrame, ventanas: pd.DataFrame, barridos: pd.DataFrame,
                       apilados: pd.DataFrame) -> pd.DataFrame:
    """Revisiones entre sondeos: gates, parámetros, distancias y ruido.

    Las tablas traen la columna ``TDEM`` (sondeo) de la consulta
    multi-sondeo. ``coordenadas`` viene ordenada por hora de inicio.
    """
    hallazgos: list[Hallazgo] = []

    def agregar(sondeo, momento, revision, severidad, mensaje, valor=None, umbral=None):
        hallazgos.append(Hallazgo(sondeo, momento, "campana", revision, severidad,
                                  None if valor is None else float(valor), umbral, mensaje))

    nombres = list(dict.fromkeys(coordenadas["Sondeo"].astype(str))) if len(coordenadas) else []
    nombres += [n for n in sorted(ventanas["TDEM"].unique()) if n not in nombres] if len(ventanas) else []
    if len(nombres) < 2:
        return pd.DataFrame(columns=[f.name for f in Hallazgo.__dataclass_fields__.values()])
    else:
        pass
    referencia = nombres[0]

    if len(ventanas):
        principal = ventanas.groupby("TDEM")["fuente"].transform(lambda f: "USF" if (f == "USF").any() else "XYZ")
        v = ventanas[ventanas["fuente"] == principal]
        for momento, g in v.groupby("momento"):
            ref = g[g["TDEM"] == referencia].set_index("gate")["tiempo"]
            if ref.empty:
                continue
            else:
                pass
            for nombre, gs in g.groupby("TDEM"):
                if nombre == referencia:
                    continue
                else:
                    pass
                t = gs.set_index("gate")["tiempo"]
                comunes = ref.index.intersection(t.index)
                diferencia = float(((t[comunes] - ref[comunes]).abs() / ref[comunes]).max())
                igual = diferencia < 1e-3 and len(t) == len(ref)
                agregar(nombre, momento, "gates_entre_sondeos", "OK" if igual else "WARN",
                        f"Tiempos de gate vs {referencia}: diferencia máxima {100 * diferencia:.1f} %, {len(t)} vs "
                        f"{len(ref)} gates." + ("" if igual else " Configuraciones distintas en el mismo proyecto: "
                                                "confirme que fue intencional."), diferencia, "< 0.1 %")
    else:
        pass

    if len(barridos):
        parametros = ["frecuencia_hz", "rampa_s", "factor_campo", "retardo_s"]
        p = barridos.groupby(["TDEM", "momento"])[parametros].median().reset_index()
        for momento, g in p.groupby("momento"):
            cambian = [c for c in parametros if g[c].nunique() > 1]
            agregar(None, momento, "parametros_entre_sondeos", "OK" if not cambian else "WARN",
                    "Parámetros de adquisición iguales en todos los sondeos." if not cambian else
                    f"Parámetros que cambian entre sondeos: {', '.join(cambian)}.")
    else:
        pass

    if len(coordenadas) >= 2:
        c = coordenadas.reset_index(drop=True)
        for i in range(1, len(c)):
            d = float(np.hypot(c.loc[i, "X"] - c.loc[i - 1, "X"], c.loc[i, "Y"] - c.loc[i - 1, "Y"]))
            cerca = d < 0.5 * float(c.loc[i, "lado_x"] or 40.0)
            agregar(c.loc[i, "Sondeo"], None, "distancia_sondeo_anterior", "WARN" if cerca else "INFO",
                    f"{d:.0f} m desde {c.loc[i - 1, 'Sondeo']}." + (" Lazos casi superpuestos: ¿repetición del "
                                                                     "sondeo?" if cerca else ""), d)
    else:
        pass

    if len(apilados):
        tardio = (apilados.sort_values("gate").groupby(["TDEM", "momento"]).tail(3)
                  .groupby(["TDEM", "momento"])["error_std"].median().reset_index())
        for momento, g in tardio.groupby("momento"):
            if len(g) < 2:
                continue
            else:
                pass
            razon = float(g["error_std"].max() / g["error_std"].min())
            ruidoso = g.loc[g["error_std"].idxmax(), "TDEM"]
            agregar(ruidoso, momento, "ruido_entre_sondeos", "WARN" if razon > 10 else "INFO",
                    f"El piso de ruido varía {razon:.0f}× entre sondeos; el más ruidoso es este. Revise fuentes de "
                    "ruido cerca del sitio (líneas eléctricas, tráfico).", razon)
    else:
        pass
    return pd.DataFrame([asdict(h) for h in hallazgos])


# ======================================================================
# 6. Valoración
# ======================================================================

RANGO_VALORACION = {"Buena": 0, "Aceptable": 1, "Revisar": 2, "Deficiente": 3}


def valoracion(n_aviso: int, n_falla: int) -> str:
    if n_falla:
        return "Deficiente"
    else:
        pass
    if n_aviso >= 3:
        return "Revisar"
    else:
        return "Aceptable" if n_aviso else "Buena"


def resumen(sondeo: str, hallazgos: pd.DataFrame, apilado: pd.DataFrame, barridos: pd.DataFrame) -> pd.DataFrame:
    """Una fila por momento más una fila ``TODOS`` con la peor valoración."""
    filas = []
    for momento, ap in apilado.groupby("momento"):
        h = hallazgos[hallazgos["momento"].isna() | (hallazgos["momento"] == momento)]
        n_aviso = int((h["severidad"] == "WARN").sum())
        n_falla = int((h["severidad"] == "FAIL").sum())
        utiles = ap[ap["util"].astype(bool)]
        b = barridos[barridos["momento"] == momento]
        filas.append({
            "sondeo": sondeo, "momento": momento, "n_stacks": int(ap["n_stacks"].max()),
            "corriente_media_a": float(b["corriente_a"].mean()),
            "n_gates": len(ap), "n_utiles": len(utiles), "n_para_invertir": int(ap["usar"].astype(bool).sum()),
            "t_primero_util": float(utiles["tiempo"].min()) if len(utiles) else np.nan,
            "t_ultimo_util": float(utiles["tiempo"].max()) if len(utiles) else np.nan,
            "snr_mediano": float(utiles["snr"].median()) if len(utiles) else np.nan,
            "piso_ruido": float(ap.tail(3)["error_std"].median()),
            "rho_a_mediana": float(utiles["rho_a"].median()) if "rho_a" in utiles and len(utiles) else np.nan,
            "n_aviso": n_aviso, "n_falla": n_falla, "puntaje": max(0, 100 - 25 * n_falla - 8 * n_aviso),
            "valoracion": valoracion(n_aviso, n_falla),
            "observaciones": " | ".join(h.loc[h["severidad"].isin(["WARN", "FAIL"]), "revision"]),
        })
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return tabla
    else:
        pass
    peor = max(tabla["valoracion"], key=RANGO_VALORACION.get)
    filas.append({
        "sondeo": sondeo, "momento": "TODOS", "n_stacks": int(tabla["n_stacks"].sum()),
        "n_gates": int(tabla["n_gates"].sum()), "n_utiles": int(tabla["n_utiles"].sum()),
        "n_para_invertir": int(tabla["n_para_invertir"].sum()),
        "t_primero_util": float(tabla["t_primero_util"].min()), "t_ultimo_util": float(tabla["t_ultimo_util"].max()),
        "n_aviso": int((hallazgos["severidad"] == "WARN").sum()), "n_falla": int((hallazgos["severidad"] == "FAIL").sum()),
        "puntaje": int(tabla["puntaje"].min()), "valoracion": peor,
        "observaciones": " | ".join(sorted(set(hallazgos.loc[hallazgos["severidad"].isin(["WARN", "FAIL"]), "revision"]))),
    })
    return pd.DataFrame(filas)
