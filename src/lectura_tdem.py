"""
lectura_tdem.py
==================================================================
Lectura de sondeos TDEM en formato **USF** (prioritario) y **XYZ**
(TEMcompany / ABEM: WalkTEM, sTEM, GroundTEM), y su conversión a las
tablas normalizadas del proyecto.

Formato USF (Universal Sounding Format, especificación de Interpex)
----------------------------------------------------------------------
Texto ASCII (normalmente con fin de línea CRLF)::

    //CLAVE: valor        encabezado del archivo (instrumento, EPSG…) → //END
    /CLAVE: valor         encabezado del sondeo (ubicación, lazo…)
    /SWEEP_NUMBER: n      inicio de un barrido
    /CLAVE: valor         parámetros del barrido (corriente, rampa…)
    /END                  fin del encabezado del barrido
    TIME, VOLTAGE, …      columnas
    1.1e-05, 3.5e-04 …    una fila por gate
    /END                  fin de la tabla (opcional según el exportador)

Cada barrido del USF es ya un **stack crudo** (promedio de cientos de
transitorios hecho por el instrumento). ``VOLTAGE`` es dB/dt normalizado
por corriente y área de la bobina receptora, en V/(A·m²).

Formato XYZ (stb2xyz)
------------------------
Secciones ``[..]`` con ``Clave= valores`` y una tabla con una fila por
stack, en orden cronológico (LM y HM intercalados), con hora, GPS y
corriente. ``dbdtDat`` está normalizado solo por el área de Rx (V/m²) y
``dbdtStd`` es fracción del dato.

Reglas de combinación
------------------------
- El USF manda: datos y parámetros salen de él.
- El XYZ del mismo archivo ``.stb`` (se emparejan por la marca
  ``AAAA_MMDD_HHMMSS``) agrega hora y GPS por barrido y sirve para
  verificar dato a dato. El USF ordena por momento y el XYZ por tiempo:
  se emparejan por (momento, orden dentro del momento).
- Sin USF, el sondeo se arma con el XYZ (dato / corriente).
- El valor *dummy* del archivo (99999) se convierte en NaN.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("tdem.lectura")


class LecturaTDEMError(Exception):
    """El archivo no se pudo interpretar como datos TDEM."""


DUMMY_DEFECTO = 99999.0

#: Canal del USF → momento. Canal 1 = bajo momento, canal 2 = alto.
CANAL_A_MOMENTO = {1: "LM", 2: "HM"}

#: Código de momento del XYZ → momento.
CODIGO_XYZ_A_MOMENTO = {0: "LM", 1: "HM"}

#: Claves del USF que siempre se conservan como texto.
CLAVES_TEXTO = {
    "SOUNDING_NAME", "SOUNDING_GROUP_NAME", "DATE", "ARRAY", "PROFILE", "INSTRUMENT",
    "INSTRUMENTTYPE", "INSTRUMENTID", "USF", "Z_DIRECTION", "LENGTH_UNITS",
    "VOLTAGE_UNITS", "STACKTYPE", "STB-FILE NAME", "STB2UFS.EXE VERSION", "STB-FILEVERSION",
}

#: Columnas del USF → columnas del proyecto.
COLUMNAS_USF = {
    "TIME": "tiempo", "VOLTAGE": "dbdt", "ERROR_BAR": "error_pct", "STD_DEV": "desv_std",
    "QUALITY": "calidad", "TIMEOPEN": "t_ini", "TIMECLOSE": "t_fin",
}

COLUMNAS_REGISTRO_XYZ = {
    "Date": "fecha", "Time": "hora_txt", "Latitude": "gps_lat", "Longitude": "gps_lon",
    "Elevation": "gps_elev_m", "Speed": "gps_vel_ms", "TxCurrent": "corriente_a",
    "DatRec": "registro", "Moment": "codigo_momento", "TxON": "tx_encendido", "Ngates": "n_gates",
}

_SEPARADOR = re.compile(r"[,\s]+")
_NUMERO = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$|^nan$", re.IGNORECASE)
_MARCA_STB = re.compile(r"(\d{4}_\d{4}_\d{6})")


# ======================================================================
# Contenedores
# ======================================================================


@dataclass
class SondeoUSF:
    """Un sondeo tal como viene en un archivo USF."""

    archivo: Path
    encabezado_archivo: dict[str, Any]
    encabezado: dict[str, Any]
    barridos: list[dict[str, Any]]
    gates: pd.DataFrame

    @property
    def nombre(self) -> str:
        return str(self.encabezado.get("SOUNDING_NAME", self.archivo.stem))


@dataclass
class SondeoXYZ:
    """Un sondeo tal como viene en un archivo XYZ de TEMcompany."""

    archivo: Path
    secciones: dict[str, dict[str, Any]]
    registros: pd.DataFrame
    gates: pd.DataFrame
    ventanas: pd.DataFrame
    forma_onda: pd.DataFrame


@dataclass
class TablasSondeo:
    """Tablas normalizadas de un sondeo, listas para guardarse."""

    nombre: str
    meta: dict[str, Any]
    barridos: pd.DataFrame
    gates: pd.DataFrame
    forma_onda: pd.DataFrame
    ventanas: pd.DataFrame
    advertencias: list[str] = field(default_factory=list)


# ======================================================================
# Utilidades de texto
# ======================================================================


def _texto_desde(fuente: Union[str, Path, bytes]) -> str:
    """Contenido de texto de una ruta o de bytes."""
    if isinstance(fuente, (bytes, bytearray)):
        crudo = bytes(fuente)
    else:
        crudo = Path(fuente).read_bytes()
    for codificacion in ("utf-8-sig", "latin-1"):
        try:
            return crudo.decode(codificacion)
        except UnicodeDecodeError:
            continue
    return crudo.decode("utf-8", errors="replace")


def _valor(clave: str, crudo: str) -> Any:
    """Texto de una clave → str, float o lista de floats."""
    crudo = crudo.strip()
    if clave in CLAVES_TEXTO:
        return crudo
    else:
        pass
    partes = [p for p in _SEPARADOR.split(crudo) if p]
    if partes and all(_NUMERO.match(p) for p in partes):
        numeros = [float(p) for p in partes]
        return numeros[0] if len(numeros) == 1 else numeros
    else:
        return crudo


def _clave_valor(linea: str) -> tuple[str, Any]:
    """``'/LOOP_SIZE: 40, 40'`` → ``('LOOP_SIZE', [40.0, 40.0])``."""
    cuerpo = linea.lstrip("/")
    if ":" not in cuerpo:
        return cuerpo.strip().upper(), None
    else:
        pass
    clave, crudo = cuerpo.split(":", 1)
    clave = clave.strip().upper()
    return clave, _valor(clave, crudo)


def _es_fila_datos(linea: str) -> bool:
    partes = [p for p in _SEPARADOR.split(linea) if p]
    return bool(partes) and all(_NUMERO.match(p) for p in partes)


def _como_lista(valor: Any, n: int) -> list:
    if isinstance(valor, list):
        return (valor + [np.nan] * n)[:n]
    else:
        return [valor] + [np.nan] * (n - 1)


def _limpiar_dummy(valor: Any, dummy: float) -> Any:
    if isinstance(valor, (int, float)) and np.isclose(valor, dummy):
        return np.nan
    else:
        return valor


def marca_stb(texto: Any) -> Optional[str]:
    """Marca ``AAAA_MMDD_HHMMSS`` del archivo ``.stb`` de origen."""
    coincidencia = _MARCA_STB.search(str(texto or ""))
    return coincidencia.group(1) if coincidencia else None


def sha1_archivo(ruta: Union[str, Path]) -> str:
    return hashlib.sha1(Path(ruta).read_bytes()).hexdigest()


# ======================================================================
# USF
# ======================================================================


class _LectorUSF:
    """Máquina de estados que recorre el USF línea por línea.

    Estados: ``sondeo`` (claves del sondeo), ``encabezado_barrido`` y
    ``tabla``. Un sondeo nuevo empieza cuando aparece una clave de nivel
    sondeo después de que el actual ya tiene barridos.
    """

    def __init__(self) -> None:
        self.encabezado_archivo: dict[str, Any] = {}
        self.sondeos: list[dict[str, Any]] = []
        self.actual: Optional[dict[str, Any]] = None
        self.barrido: Optional[dict[str, Any]] = None
        self.columnas: Optional[list[str]] = None
        self.filas: list[list[float]] = []
        self.estado = "sondeo"

    def _nuevo_sondeo(self) -> None:
        self._cerrar_barrido()
        self.actual = {"encabezado": {}, "barridos": [], "tablas": []}
        self.sondeos.append(self.actual)

    def _cerrar_barrido(self) -> None:
        if self.barrido is None:
            return None
        else:
            pass
        tabla = None
        if self.filas:
            n = max(len(f) for f in self.filas)
            columnas = list(self.columnas or [])
            columnas = (columnas + [f"COL{i + 1}" for i in range(len(columnas), n)])[:n]
            filas = [f + [np.nan] * (n - len(f)) for f in self.filas]
            tabla = pd.DataFrame(filas, columns=columnas)
        else:
            pass
        self.actual["barridos"].append(self.barrido)
        self.actual["tablas"].append((self.barrido["SWEEP_NUMBER"], tabla))
        self.barrido, self.columnas, self.filas = None, None, []
        self.estado = "sondeo"
        return None

    def _clave(self, linea: str) -> None:
        clave, valor = _clave_valor(linea)
        if clave == "END":
            if self.estado == "encabezado_barrido":
                self.estado = "tabla"
            else:
                if self.estado == "tabla":
                    self._cerrar_barrido()
                else:
                    pass
            return None
        else:
            pass

        if self.estado == "tabla":
            self._cerrar_barrido()
        else:
            pass

        if clave == "SWEEP_NUMBER":
            if self.actual is None:
                self._nuevo_sondeo()
            else:
                pass
            self.barrido = {"SWEEP_NUMBER": int(valor)}
            self.estado = "encabezado_barrido"
            return None
        else:
            pass

        if self.estado == "encabezado_barrido":
            self.barrido[clave] = valor
            return None
        else:
            pass

        if self.actual is None or self.actual["barridos"]:
            self._nuevo_sondeo()
        else:
            pass
        self.actual["encabezado"][clave] = valor
        return None

    def _linea_tabla(self, linea: str) -> None:
        if _es_fila_datos(linea):
            self.filas.append([float(p) for p in _SEPARADOR.split(linea) if p])
        else:
            if self.columnas is None and not self.filas:
                self.columnas = [c.strip().upper() for c in linea.split(",") if c.strip()]
            else:
                pass

    def leer(self, texto: str) -> list[dict[str, Any]]:
        for cruda in texto.splitlines():
            linea = cruda.strip()
            if not linea:
                continue
            else:
                pass
            if linea.startswith("//"):
                clave, valor = _clave_valor(linea)
                if clave != "END":
                    self.encabezado_archivo[clave] = valor
                else:
                    pass
                continue
            else:
                pass
            if linea.startswith("/"):
                self._clave(linea)
            else:
                if self.estado == "tabla":
                    self._linea_tabla(linea)
                else:
                    pass
        self._cerrar_barrido()
        return self.sondeos


def _tablas_a_largo(tablas: list) -> pd.DataFrame:
    """Une las tablas de todos los barridos en una tabla larga."""
    partes = []
    for numero, tabla in tablas:
        if tabla is None or tabla.empty:
            continue
        else:
            pass
        df = tabla.rename(columns=COLUMNAS_USF).copy()
        df.insert(0, "barrido", int(numero))
        df.insert(1, "gate", np.arange(1, len(df) + 1))
        partes.append(df)
    if not partes:
        return pd.DataFrame(columns=["barrido", "gate", "tiempo", "dbdt"])
    else:
        return pd.concat(partes, ignore_index=True)


def leer_usf(fuente: Union[str, Path, bytes], nombre_archivo: Optional[str] = None) -> list[SondeoUSF]:
    """Lee un USF y devuelve sus sondeos (normalmente uno)."""
    texto = _texto_desde(fuente)
    if "SWEEP_NUMBER" not in texto.upper():
        raise LecturaTDEMError("El archivo no tiene barridos (/SWEEP_NUMBER): no parece un USF de TDEM.")
    else:
        pass
    lector = _LectorUSF()
    crudos = lector.leer(texto)
    ruta = Path(nombre_archivo or (fuente if not isinstance(fuente, (bytes, bytearray)) else "sondeo.usf"))
    return [SondeoUSF(archivo=ruta, encabezado_archivo=lector.encabezado_archivo,
                      encabezado=c["encabezado"], barridos=c["barridos"],
                      gates=_tablas_a_largo(c["tablas"])) for c in crudos]


# ======================================================================
# XYZ
# ======================================================================


def _leer_secciones(lineas: list[str]) -> tuple[dict[str, dict[str, Any]], int]:
    """Secciones ``[..]`` y el índice de la línea de columnas."""
    secciones: dict[str, dict[str, Any]] = {"Preambulo": {}}
    actual = "Preambulo"
    for i, linea in enumerate(lineas):
        limpia = linea.strip()
        if limpia.startswith("Date ") and "Time" in limpia:
            return secciones, i
        else:
            pass
        coincidencia = re.match(r"^\[(\w+)\]", limpia)
        if coincidencia:
            actual = coincidencia.group(1)
            secciones.setdefault(actual, {})
            continue
        else:
            pass
        if "=" in limpia:
            clave, crudo = limpia.split("=", 1)
            partes = crudo.split()
            if actual != "Units" and partes and all(_NUMERO.match(p) for p in partes):
                numeros = [float(p) for p in partes]
                secciones[actual][clave.strip()] = numeros[0] if len(numeros) == 1 else numeros
            else:
                secciones[actual][clave.strip()] = crudo.strip()
        else:
            if limpia:
                secciones["Preambulo"][f"linea_{i}"] = limpia
            else:
                pass
    raise LecturaTDEMError("No se encontró la línea de columnas 'Date Time …' en el XYZ.")


def leer_xyz(fuente: Union[str, Path, bytes], nombre_archivo: Optional[str] = None) -> SondeoXYZ:
    """Lee un XYZ de TEMcompany y lo separa en registros y gates."""
    lineas = _texto_desde(fuente).splitlines()
    secciones, indice = _leer_secciones(lineas)
    columnas = lineas[indice].split()
    filas = [l.split() for l in lineas[indice + 1:] if l.strip()]
    filas = [f for f in filas if len(f) == len(columnas)]
    tabla = pd.DataFrame(filas, columns=columnas)

    registros = tabla[[c for c in COLUMNAS_REGISTRO_XYZ if c in tabla]].rename(columns=COLUMNAS_REGISTRO_XYZ)
    for columna in registros.columns:
        if columna not in ("fecha", "hora_txt"):
            registros[columna] = pd.to_numeric(registros[columna], errors="coerce")
        else:
            pass
    registros["hora"] = pd.to_datetime(registros["fecha"] + " " + registros["hora_txt"],
                                       format="%Y%m%d %H:%M:%S.%f", errors="coerce")
    registros["momento"] = registros["codigo_momento"].map(CODIGO_XYZ_A_MOMENTO)
    registros["registro"] = registros["registro"].astype(int)
    registros = registros.drop(columns=["fecha", "hora_txt"])

    col_dat = sorted(c for c in columnas if c.startswith("dbdtDat"))
    col_std = sorted(c for c in columnas if c.startswith("dbdtStd"))
    valores = tabla[col_dat].apply(pd.to_numeric, errors="coerce").to_numpy()
    errores = tabla[col_std].apply(pd.to_numeric, errors="coerce").to_numpy()
    n = len(col_dat)
    gates = pd.DataFrame({
        "registro": np.repeat(registros["registro"].to_numpy(), n),
        "momento": np.repeat(registros["momento"].to_numpy(), n),
        "gate": np.tile(np.arange(1, n + 1), len(registros)),
        "dbdt_v_m2": valores.ravel(),
        "error_rel": errores.ravel(),
    })
    gates = gates[gates["gate"] <= np.repeat(registros["n_gates"].to_numpy(), n)].reset_index(drop=True)

    tiempos = secciones.get("ProcessedGateTimes", {})
    partes_v = []
    for momento in ("LM", "HM"):
        centro = tiempos.get(f"{momento}_CenterTime")
        if centro is None:
            continue
        else:
            pass
        centro = np.atleast_1d(centro)
        partes_v.append(pd.DataFrame({
            "momento": momento, "gate": np.arange(1, len(centro) + 1),
            "t_ini": np.atleast_1d(tiempos.get(f"{momento}_OpenTime")), "tiempo": centro,
            "t_fin": np.atleast_1d(tiempos.get(f"{momento}_CloseTime")),
        }))
    especificaciones = secciones.get("RxTxSpecs", {})
    partes_f = []
    for momento in ("LM", "HM"):
        t = especificaciones.get(f"{momento}_Waveform_Time")
        a = especificaciones.get(f"{momento}_Waveform_Amplitude")
        if t is None or a is None:
            continue
        else:
            pass
        partes_f.append(pd.DataFrame({"momento": momento, "punto": np.arange(1, len(t) + 1),
                                      "tiempo": t, "amplitud": a}))
    ruta = Path(nombre_archivo or (fuente if not isinstance(fuente, (bytes, bytearray)) else "sondeo.xyz"))
    return SondeoXYZ(
        archivo=ruta, secciones=secciones, registros=registros, gates=gates,
        ventanas=pd.concat(partes_v, ignore_index=True) if partes_v else pd.DataFrame(),
        forma_onda=pd.concat(partes_f, ignore_index=True) if partes_f else pd.DataFrame(),
    )


def marca_stb_usf(sondeo: SondeoUSF) -> Optional[str]:
    return marca_stb(sondeo.encabezado_archivo.get("STB-FILE NAME")) or marca_stb(sondeo.archivo.name)


def marca_stb_xyz(sondeo: SondeoXYZ) -> Optional[str]:
    preambulo = " ".join(str(v) for v in sondeo.secciones.get("Preambulo", {}).values())
    return marca_stb(preambulo) or marca_stb(sondeo.archivo.name)


# ======================================================================
# Tablas normalizadas
# ======================================================================


def _barridos_usf(usf: SondeoUSF, dummy: float) -> pd.DataFrame:
    filas = []
    for b in usf.barridos:
        filtros = _como_lista(b.get("LOW_PASS"), 4)
        canal = int(b.get("CHANNEL", 1))
        filas.append({
            "barrido": int(b["SWEEP_NUMBER"]), "canal": canal,
            "momento": CANAL_A_MOMENTO.get(canal, f"CH{canal}"),
            "corriente_a": _limpiar_dummy(b.get("CURRENT"), dummy),
            "frecuencia_hz": _limpiar_dummy(b.get("FREQUENCY"), dummy),
            "es_ruido": bool(b.get("SWEEP_IS_NOISE", 0)),
            "factor_campo": _limpiar_dummy(b.get("FIELD_SHIFT_FACTOR"), dummy),
            "retardo_s": _limpiar_dummy(b.get("TIME_DELAY"), dummy),
            "rampa_s": _limpiar_dummy(b.get("RAMP_TIME"), dummy),
            "n_gates_declarados": _limpiar_dummy(b.get("POINTS"), dummy),
            "filtro1_hz": filtros[0], "filtro2_hz": filtros[2],
        })
    barridos = pd.DataFrame(filas)
    barridos["indice_momento"] = barridos.groupby("momento").cumcount() + 1
    return barridos


def _forma_onda_usf(usf: SondeoUSF, barridos: pd.DataFrame) -> pd.DataFrame:
    """Rampa TX del primer barrido de cada momento (es igual en todos)."""
    por_numero = {b["SWEEP_NUMBER"]: b for b in usf.barridos}
    partes = []
    for momento, grupo in barridos.groupby("momento"):
        rampa = por_numero[int(grupo["barrido"].iloc[0])].get("TX_RAMP")
        if not isinstance(rampa, list) or len(rampa) < 4:
            continue
        else:
            pass
        pares = np.asarray(rampa, float).reshape(-1, 2)
        partes.append(pd.DataFrame({"momento": momento, "punto": np.arange(1, len(pares) + 1),
                                    "tiempo": pares[:, 0], "amplitud": pares[:, 1]}))
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()


def _gates_usf(usf: SondeoUSF, barridos: pd.DataFrame) -> pd.DataFrame:
    gates = usf.gates.merge(barridos[["barrido", "momento"]], on="barrido")
    if "error_pct" in gates:
        gates["error_rel"] = gates["error_pct"] / 100.0
    else:
        if "desv_std" in gates:
            gates["error_rel"] = gates["desv_std"] / gates["dbdt"].abs()
        else:
            gates["error_rel"] = np.nan
    gates["fuente"] = "USF"
    columnas = ["barrido", "momento", "gate", "tiempo", "t_ini", "t_fin", "dbdt", "error_rel", "calidad", "fuente"]
    return gates.reindex(columns=columnas)


def _unir_xyz(barridos: pd.DataFrame, gates: pd.DataFrame, xyz: SondeoXYZ,
              advertencias: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hora y GPS por barrido, y el dato del XYZ en unidades del USF."""
    registros = xyz.registros.copy()
    registros["indice_momento"] = registros.groupby("momento").cumcount() + 1
    registros = registros.rename(columns={"corriente_a": "corriente_xyz_a", "registro": "registro_xyz"})
    columnas = ["momento", "indice_momento", "registro_xyz", "hora", "gps_lat", "gps_lon",
                "gps_elev_m", "gps_vel_ms", "corriente_xyz_a"]
    barridos = barridos.merge(registros[[c for c in columnas if c in registros]],
                              on=["momento", "indice_momento"], how="left")
    for momento in barridos["momento"].unique():
        n_usf = int((barridos["momento"] == momento).sum())
        n_xyz = int((registros["momento"] == momento).sum())
        if n_usf != n_xyz:
            advertencias.append(f"{momento}: {n_usf} barridos en el USF y {n_xyz} en el XYZ.")
        else:
            pass

    datos = xyz.gates.merge(
        registros[["registro_xyz", "momento", "corriente_xyz_a", "indice_momento"]]
        .rename(columns={"registro_xyz": "registro"}), on=["registro", "momento"])
    datos["dbdt_xyz"] = datos["dbdt_v_m2"] / datos["corriente_xyz_a"]
    datos = datos.merge(barridos[["momento", "indice_momento", "barrido"]], on=["momento", "indice_momento"])
    gates = gates.merge(datos[["barrido", "gate", "dbdt_xyz", "error_rel"]]
                        .rename(columns={"error_rel": "error_rel_xyz"}),
                        on=["barrido", "gate"], how="left")
    return barridos, gates


def _desde_xyz(xyz: SondeoXYZ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Barridos y gates cuando solo hay XYZ."""
    info = xyz.secciones.get("InstrumentInfo", {})
    reg = xyz.registros.sort_values(["codigo_momento", "registro"]).reset_index(drop=True)
    reg["barrido"] = np.arange(1, len(reg) + 1)
    reg["indice_momento"] = reg.groupby("momento").cumcount() + 1
    barridos = pd.DataFrame({
        "barrido": reg["barrido"], "canal": reg["codigo_momento"].astype(int) + 1, "momento": reg["momento"],
        "corriente_a": reg["corriente_a"], "frecuencia_hz": np.nan, "es_ruido": reg["tx_encendido"] == 0,
        "factor_campo": reg["momento"].map({m: info.get(f"{m}_DataFactor") for m in ("LM", "HM")}),
        "retardo_s": reg["momento"].map({m: info.get(f"{m}_GateTimeShift") for m in ("LM", "HM")}),
        "rampa_s": np.nan, "n_gates_declarados": reg["n_gates"], "filtro1_hz": np.nan, "filtro2_hz": np.nan,
        "indice_momento": reg["indice_momento"], "registro_xyz": reg["registro"], "hora": reg["hora"],
        "gps_lat": reg["gps_lat"], "gps_lon": reg["gps_lon"], "gps_elev_m": reg["gps_elev_m"],
        "gps_vel_ms": reg["gps_vel_ms"], "corriente_xyz_a": reg["corriente_a"],
    })
    g = xyz.gates.merge(reg[["registro", "barrido", "corriente_a"]], on="registro")
    g = g.merge(xyz.ventanas, on=["momento", "gate"], how="left")
    gates = pd.DataFrame({
        "barrido": g["barrido"], "momento": g["momento"], "gate": g["gate"], "tiempo": g["tiempo"],
        "t_ini": g["t_ini"], "t_fin": g["t_fin"], "dbdt": g["dbdt_v_m2"] / g["corriente_a"],
        "error_rel": g["error_rel"], "calidad": np.nan, "fuente": "XYZ",
    })
    return barridos, gates


def _inicio_usf(encabezado: dict[str, Any]) -> Optional[datetime]:
    """DATE (AAAAMMDD) + DAYTIME (horas decimales)."""
    try:
        dia = datetime.strptime(str(encabezado["DATE"]), "%Y%m%d")
        return dia + timedelta(hours=float(encabezado["DAYTIME"]))
    except (KeyError, ValueError, TypeError):
        return None


def _ventanas(gates: pd.DataFrame, xyz: Optional[SondeoXYZ]) -> pd.DataFrame:
    """Ventanas de gates por momento de la fuente principal y, si hay, del XYZ."""
    columnas = ["momento", "gate", "t_ini", "tiempo", "t_fin"]
    primero = gates.groupby("momento")["barrido"].transform("min")
    principal = gates.loc[gates["barrido"] == primero, columnas].copy()
    principal["fuente"] = gates["fuente"].iloc[0] if len(gates) else "USF"
    partes = [principal]
    if xyz is not None and len(xyz.ventanas) and principal["fuente"].iloc[0] == "USF":
        extra = xyz.ventanas[columnas].copy()
        extra["fuente"] = "XYZ"
        partes.append(extra)
    else:
        pass
    return pd.concat(partes, ignore_index=True)


def construir_tablas(usf: Optional[SondeoUSF], xyz: Optional[SondeoXYZ]) -> TablasSondeo:
    """Tablas normalizadas de un sondeo a partir de USF y/o XYZ."""
    if usf is None and xyz is None:
        raise LecturaTDEMError("Se necesita al menos un USF o un XYZ.")
    else:
        pass
    advertencias: list[str] = []
    especificaciones = xyz.secciones.get("RxTxSpecs", {}) if xyz is not None else {}
    offset_gps = _como_lista(especificaciones.get("GPS_XYZPos"), 3)

    if usf is not None:
        dummy = float(usf.encabezado_archivo.get("DUMMY", DUMMY_DEFECTO) or DUMMY_DEFECTO)
        barridos = _barridos_usf(usf, dummy)
        gates = _gates_usf(usf, barridos)
        forma_onda = _forma_onda_usf(usf, barridos)
        if xyz is not None:
            barridos, gates = _unir_xyz(barridos, gates, xyz, advertencias)
        else:
            pass
        e, ea = usf.encabezado, usf.encabezado_archivo
        ubicacion = [_limpiar_dummy(v, dummy) for v in _como_lista(e.get("LOCATION"), 3)]
        lazo = _como_lista(e.get("LOOP_SIZE"), 2)
        bobina = _como_lista(e.get("COIL_LOCATION"), 2)
        nombre = usf.nombre
        inicio = _inicio_usf(e)
        meta = {
            "proyecto_campo": ea.get("SOUNDING_GROUP_NAME"), "instrumento": ea.get("INSTRUMENTTYPE"),
            "id_instrumento": ea.get("INSTRUMENTID"), "tipo_stack": ea.get("STACKTYPE"),
            "epsg_archivo": ea.get("EPSG"), "lon": ubicacion[0], "lat": ubicacion[1], "elev": ubicacion[2],
            "lado_x": lazo[0], "lado_y": lazo[1], "rx_x": bobina[0], "rx_y": bobina[1],
            "unidades": e.get("VOLTAGE_UNITS"), "n_barridos_declarados": e.get("SWEEPS"),
            "archivo_stb": ea.get("STB-FILE NAME"), "archivo_usf": usf.archivo.name,
            "fuente_principal": "USF",
        }
    else:
        barridos, gates = _desde_xyz(xyz)
        forma_onda = xyz.forma_onda.copy()
        info = xyz.secciones.get("InstrumentInfo", {})
        gps = xyz.registros[["gps_lon", "gps_lat", "gps_elev_m"]].median()
        lazo = _como_lista(especificaciones.get("TxLoop_XYLength"), 2)
        bobina = _como_lista(especificaciones.get("RxCoil_XYZPos"), 3)
        nombre = Path(xyz.archivo).stem
        inicio = xyz.registros["hora"].min()
        inicio = inicio.to_pydatetime() if pd.notna(inicio) else None
        advertencias.append("Sondeo armado solo con el XYZ: la ubicación es la mediana del GPS, sin "
                            "corregir el desplazamiento del GPS respecto al centro del lazo.")
        meta = {
            "proyecto_campo": None, "instrumento": info.get("InstrumentType"),
            "id_instrumento": info.get("InstrumentID"), "tipo_stack": None, "epsg_archivo": 4326,
            "lon": gps["gps_lon"], "lat": gps["gps_lat"], "elev": gps["gps_elev_m"],
            "lado_x": lazo[0], "lado_y": lazo[1], "rx_x": bobina[0], "rx_y": bobina[1],
            "unidades": "V/AM2 (convertido desde V/m2)", "n_barridos_declarados": len(xyz.registros),
            "archivo_stb": marca_stb_xyz(xyz), "archivo_usf": None, "fuente_principal": "XYZ",
        }

    con_hora = "hora" in barridos and barridos["hora"].notna().any()
    meta.update({
        "inicio": inicio.isoformat(timespec="seconds") if inicio else None,
        "fin": barridos["hora"].max().isoformat(timespec="seconds") if con_hora else None,
        "vueltas": float(especificaciones.get("TxLoop_NTurns", 1.0) or 1.0),
        "offset_gps_x": offset_gps[0], "offset_gps_y": offset_gps[1],
        "archivo_xyz": xyz.archivo.name if xyz is not None else None,
        "n_barridos": int(len(barridos)), "n_lm": int((barridos["momento"] == "LM").sum()),
        "n_hm": int((barridos["momento"] == "HM").sum()),
    })
    return TablasSondeo(nombre=nombre, meta=meta, barridos=barridos, gates=gates,
                        forma_onda=forma_onda, ventanas=_ventanas(gates, xyz), advertencias=advertencias)


def emparejar(rutas: list[Union[str, Path]]) -> list[tuple[Optional[Path], Optional[Path]]]:
    """Empareja USF con XYZ por la marca del ``.stb``.

    Si hay varios XYZ del mismo ``.stb`` (canales A y B), se prefiere el A.
    """
    usfs = [Path(r) for r in rutas if Path(r).suffix.lower() == ".usf"]
    xyzs = [Path(r) for r in rutas if Path(r).suffix.lower() == ".xyz"]
    por_marca: dict[str, Path] = {}
    for ruta in xyzs:
        marca = marca_stb(ruta.name)
        if marca is None:
            try:
                marca = marca_stb_xyz(leer_xyz(ruta))
            except LecturaTDEMError:
                logger.warning("XYZ ilegible, se omite: %s", ruta)
                continue
        else:
            pass
        if marca not in por_marca or "ChA" in ruta.name:
            por_marca[marca] = ruta
        else:
            pass
    pares, usadas = [], set()
    for ruta in usfs:
        marca = marca_stb(ruta.name)
        if marca is None:
            marca = marca_stb_usf(leer_usf(ruta)[0])
        else:
            pass
        pares.append((ruta, por_marca.get(marca)))
        if marca:
            usadas.add(marca)
        else:
            pass
    pares += [(None, ruta) for marca, ruta in por_marca.items() if marca not in usadas]
    return pares


def leer_par(ruta_usf: Optional[Path], ruta_xyz: Optional[Path]) -> list[TablasSondeo]:
    """Lee un par USF/XYZ y devuelve las tablas de cada sondeo."""
    xyz = leer_xyz(ruta_xyz) if ruta_xyz is not None else None
    if ruta_usf is None:
        return [construir_tablas(None, xyz)]
    else:
        pass
    sondeos = leer_usf(ruta_usf)
    if len(sondeos) > 1 and xyz is not None:
        logger.warning("%s trae %d sondeos; el XYZ solo se usa con el primero.", ruta_usf.name, len(sondeos))
    else:
        pass
    return [construir_tablas(s, xyz if i == 0 else None) for i, s in enumerate(sondeos)]


# ======================================================================
# Coordenadas y modelo de referencia
# ======================================================================

SINONIMOS_COORDENADAS: dict[str, tuple[str, ...]] = {
    "Sondeo": ("sondeo", "tdem", "tem", "sounding", "estacion", "punto", "nombre", "station"),
    "X": ("x", "este", "easting", "e", "utme", "xutm"),
    "Y": ("y", "norte", "northing", "n", "utmn", "yutm"),
    "Z": ("z", "cota", "elevacion", "elev", "altitud", "h", "msnm"),
    "azimut": ("azimut", "azimuth", "orientacion", "rumbo"),
}


def _normalizar(texto: Any) -> str:
    crudo = unicodedata.normalize("NFKD", str(texto))
    sin_acentos = "".join(c for c in crudo if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", sin_acentos.lower())


def _leer_tabla_texto(texto: str) -> pd.DataFrame:
    """Tabla de texto con separador detectado (tabulador, ;, coma o espacios)."""
    lineas = [l for l in texto.splitlines() if l.strip() and not l.strip().startswith(("#", "//"))]
    if not lineas:
        raise LecturaTDEMError("El archivo está vacío.")
    else:
        pass
    muestra = "\n".join(lineas[:20])
    separador = r"\s+"
    if "\t" in muestra:
        separador = "\t"
    else:
        if ";" in muestra:
            separador = ";"
        else:
            if "," in muestra:
                separador = ","
            else:
                pass
    # Con «;» o tabulador como separador, la coma solo puede ser decimal
    # (Excel en español): se lee como tal.
    decimal = "," if separador in (";", "\t") and "," in muestra else "."
    return pd.read_csv(io.StringIO("\n".join(lineas)), sep=separador, engine="python", skipinitialspace=True,
                       decimal=decimal)


def leer_coordenadas(fuente: Union[str, Path, bytes]) -> pd.DataFrame:
    """Archivo ``Sondeo, X, Y[, Z, azimut]`` (X, Y en el EPSG del proyecto)."""
    tabla = _leer_tabla_texto(_texto_desde(fuente))
    renombre = {}
    for original in tabla.columns:
        norm = _normalizar(original)
        for estandar, lista in SINONIMOS_COORDENADAS.items():
            if norm in lista and estandar not in renombre.values():
                renombre[original] = estandar
                break
            else:
                pass
    tabla = tabla.rename(columns=renombre)
    faltan = [c for c in ("Sondeo", "X", "Y") if c not in tabla.columns]
    if faltan:
        raise LecturaTDEMError(f"Al archivo de coordenadas le faltan las columnas {faltan}.")
    else:
        pass
    df = pd.DataFrame({
        "Sondeo": tabla["Sondeo"].astype(str).str.strip(),
        "X": pd.to_numeric(tabla["X"], errors="coerce"),
        "Y": pd.to_numeric(tabla["Y"], errors="coerce"),
        "Z": pd.to_numeric(tabla["Z"], errors="coerce") if "Z" in tabla else np.nan,
        "azimut": pd.to_numeric(tabla["azimut"], errors="coerce") if "azimut" in tabla else np.nan,
    })
    return df.dropna(subset=["X", "Y"]).drop_duplicates("Sondeo", keep="last").reset_index(drop=True)


def leer_modelo_referencia(fuente: Union[str, Path, bytes]) -> pd.DataFrame:
    """Modelo conocido (pozo o sintético): ``espesor, rho`` o ``z_techo, rho``."""
    tabla = _leer_tabla_texto(_texto_desde(fuente).replace("inf", "nan").replace("Inf", "nan"))
    norm = {_normalizar(c): c for c in tabla.columns}
    col_rho = next((norm[k] for k in ("rho", "resistividad", "res", "ohmm") if k in norm), None)
    if col_rho is None:
        raise LecturaTDEMError("El modelo de referencia necesita una columna 'rho'.")
    else:
        pass
    rho = pd.to_numeric(tabla[col_rho], errors="coerce").to_numpy(float)
    col_z = next((norm[k] for k in ("ztecho", "techo", "z") if k in norm), None)
    if col_z is not None:
        techos = pd.to_numeric(tabla[col_z], errors="coerce").to_numpy(float)
        bases = np.concatenate([techos[1:], [np.inf]])
    else:
        col_h = next((norm[k] for k in ("espesor", "h", "thk") if k in norm), None)
        if col_h is None:
            raise LecturaTDEMError("El modelo de referencia necesita 'espesor' o 'z_techo'.")
        else:
            pass
        h = pd.to_numeric(tabla[col_h], errors="coerce").to_numpy(float)
        h = np.where(np.isfinite(h), h, np.inf)
        h[-1] = np.inf
        bases = np.cumsum(h)
        techos = np.concatenate([[0.0], bases[:-1]])
    return pd.DataFrame({"capa": np.arange(1, len(rho) + 1), "z_techo": techos, "z_base": bases,
                         "espesor": bases - techos, "rho": rho})
