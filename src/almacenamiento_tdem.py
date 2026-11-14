"""
almacenamiento_tdem.py
==================================================================
Capa de persistencia de proyectos TDEM.

Misma arquitectura que TlalliRes1D (SEV) y TlalliRes2D: **Parquet
particionado por sondeo + catálogo SQLite + consultas DuckDB**.

- **Parquet** (``<etapa>/TDEM=<id>/<tabla>.parquet``): fuente de verdad
  de los datos de cada sondeo.
- **Catálogo SQLite**: qué sondeos existen, su lazo, coordenadas,
  estado por etapa y métricas. La interfaz navega sin abrir Parquet.
- **DuckDB**: consultas multi-sondeo (mapa, unidades, perfiles) leyendo
  todas las particiones como una sola tabla.

Etapas y tablas
------------------
=================  ===================================================
Etapa              Tablas
=================  ===================================================
``01_crudo``       ``meta``, ``barridos``, ``gates``, ``forma_onda``,
                   ``ventanas``, ``modelo_referencia``
``02_apilado``     ``apilado``, ``curva``, ``qc``, ``exclusiones``
``03_inversion``   ``multicapa``, ``bloques``, ``ajuste``,
                   ``seleccion``, ``iteraciones``, ``proceso``
``04_parametros``  ``parametros``, ``unidades``, ``resumen_unidades``,
                   ``qc_campana``
``05_perfiles``    ``perfil_sondeos``, ``perfil_ruta``
=================  ===================================================

Las tablas de campaña (unidades, revisiones entre sondeos) se guardan
bajo el sondeo reservado ``CAMPANA``; las de perfiles, bajo el nombre
del perfil.

Coordenadas
--------------
El catálogo guarda la ubicación geográfica que trae el archivo
(``lon, lat, elev``) y la proyectada (``x, y, z``) en el EPSG del
proyecto (UTM, ver :mod:`geodesia_tdem`). ``origen_coord`` indica si
las X, Y vienen del archivo o las capturó el usuario: al volver a cargar
un sondeo, las coordenadas capturadas a mano no se sobrescriben.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import duckdb
import numpy as np
import pandas as pd

import geodesia_tdem as geo

logger = logging.getLogger("tdem.almacenamiento")


class AlmacenamientoError(Exception):
    """Error base de la capa de persistencia."""


class ProyectoNoEncontradoError(AlmacenamientoError, FileNotFoundError):
    """La ruta indicada no contiene un proyecto TDEM válido."""


class EtapaDesconocidaError(AlmacenamientoError, ValueError):
    """Se solicitó una etapa/tabla que no existe en el esquema."""


ETAPAS: dict[str, str] = {
    "crudo": "01_crudo",
    "apilado": "02_apilado",
    "inversion": "03_inversion",
    "parametros": "04_parametros",
    "perfiles": "05_perfiles",
}

TABLAS_POR_ETAPA: dict[str, tuple[str, ...]] = {
    "crudo": ("meta", "barridos", "gates", "forma_onda", "ventanas", "modelo_referencia"),
    "apilado": ("apilado", "curva", "qc", "resumen_qc", "barridos_qc", "exclusiones"),
    "inversion": ("multicapa", "bloques", "ajuste", "seleccion", "iteraciones", "proceso"),
    "parametros": ("parametros", "unidades", "resumen_unidades", "qc_campana"),
    "perfiles": ("perfil_sondeos", "perfil_ruta"),
}

#: Identificador reservado para las tablas de campaña.
CAMPANA = "CAMPANA"

#: Prefijo de las carpetas de partición (estilo Hive: ``TDEM=<id>``).
PREFIJO_PARTICION = "TDEM"

NOMBRE_CATALOGO = "_catalogo.sqlite"
NOMBRE_CONFIG = "_proyecto.json"
CARPETA_ENTRADAS = "_entradas"
TIPO_PROYECTO = "TDEM"


def _validar_etapa(etapa: str) -> None:
    if etapa not in ETAPAS:
        raise EtapaDesconocidaError(f"Etapa '{etapa}' desconocida. Opciones: {list(ETAPAS)}")
    else:
        pass


def _validar_tabla(etapa: str, tabla: str) -> None:
    _validar_etapa(etapa)
    if tabla not in TABLAS_POR_ETAPA[etapa]:
        raise EtapaDesconocidaError(
            f"Tabla '{tabla}' no pertenece a la etapa '{etapa}'. Válidas: {TABLAS_POR_ETAPA[etapa]}")
    else:
        pass


def sanear_id(identificador: Any) -> str:
    """Nombre seguro para carpeta de partición."""
    texto = str(identificador).strip()
    for caracter in '/\\:*?"<>| \t\n=':
        texto = texto.replace(caracter, "_")
    return texto if texto else "SIN_NOMBRE"


@dataclass
class Proyecto:
    """Un proyecto TDEM en disco."""

    ruta: Path
    nombre: str
    cliente: Optional[str] = None
    anio: Optional[str] = None
    epsg: Optional[int] = None

    @property
    def ruta_catalogo(self) -> Path:
        return self.ruta / NOMBRE_CATALOGO

    def ruta_etapa(self, etapa: str) -> Path:
        _validar_etapa(etapa)
        return self.ruta / ETAPAS[etapa]

    def ruta_tabla(self, etapa: str, tabla: str, sondeo: Any) -> Path:
        _validar_tabla(etapa, tabla)
        return self.ruta_etapa(etapa) / f"{PREFIJO_PARTICION}={sanear_id(sondeo)}" / f"{tabla}.parquet"

    def as_dict(self) -> dict[str, Any]:
        return {"nombre": self.nombre, "cliente": self.cliente, "anio": self.anio,
                "epsg": self.epsg, "ruta": str(self.ruta), "tipo": TIPO_PROYECTO}


def _escribir_config(proyecto: Proyecto) -> None:
    (proyecto.ruta / NOMBRE_CONFIG).write_text(
        json.dumps(proyecto.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def crear_proyecto(ruta_base: Union[str, Path], nombre: str, cliente: Optional[str] = None,
                   anio: Optional[str] = None, sobrescribir: bool = False) -> Proyecto:
    """Crea la estructura de carpetas y el catálogo de un proyecto."""
    ruta = Path(ruta_base) / sanear_id(nombre)
    if ruta.exists() and sobrescribir:
        shutil.rmtree(ruta)
    else:
        pass
    ruta.mkdir(parents=True, exist_ok=True)
    for carpeta in ETAPAS.values():
        (ruta / carpeta).mkdir(exist_ok=True)
    proyecto = Proyecto(ruta=ruta, nombre=nombre, cliente=cliente, anio=anio)
    _escribir_config(proyecto)
    _inicializar_catalogo(proyecto)
    logger.info("Proyecto creado: %s", ruta)
    return proyecto


def abrir_proyecto(ruta: Union[str, Path]) -> Proyecto:
    """Abre un proyecto existente."""
    ruta = Path(ruta)
    config = ruta / NOMBRE_CONFIG
    if not config.exists():
        raise ProyectoNoEncontradoError(f"'{ruta}' no es un proyecto válido (falta {NOMBRE_CONFIG}).")
    else:
        pass
    datos = json.loads(config.read_text(encoding="utf-8"))
    if datos.get("tipo", TIPO_PROYECTO) != TIPO_PROYECTO:
        raise ProyectoNoEncontradoError("El archivo es un proyecto de otra aplicación (no TDEM).")
    else:
        pass
    proyecto = Proyecto(ruta=ruta, nombre=datos.get("nombre", ruta.name), cliente=datos.get("cliente"),
                        anio=datos.get("anio"), epsg=datos.get("epsg"))
    _inicializar_catalogo(proyecto)
    return proyecto


def fijar_epsg(proyecto: Proyecto, epsg: int) -> Proyecto:
    """Fija el sistema de coordenadas proyectadas del proyecto."""
    proyecto.epsg = int(epsg)
    _escribir_config(proyecto)
    return proyecto


# ======================================================================
# Catálogo
# ======================================================================

_SQL_CREAR = """
CREATE TABLE IF NOT EXISTS sondeos (
    sondeo         TEXT PRIMARY KEY,
    archivo_usf    TEXT,
    archivo_xyz    TEXT,
    instrumento    TEXT,
    id_instrumento TEXT,
    inicio         TEXT,
    n_barridos     INTEGER,
    n_lm           INTEGER,
    n_hm           INTEGER,
    lado_x         REAL,
    lado_y         REAL,
    vueltas        REAL,
    lon            REAL,
    lat            REAL,
    elev           REAL,
    x              REAL,
    y              REAL,
    z              REAL,
    azimut         REAL,
    origen_coord   TEXT,
    creado_en      TEXT
);
CREATE TABLE IF NOT EXISTS etapas (
    sondeo TEXT NOT NULL, etapa TEXT NOT NULL, tabla TEXT NOT NULL,
    n_filas INTEGER, ruta_relativa TEXT, actualizado_en TEXT,
    PRIMARY KEY (sondeo, etapa, tabla)
);
CREATE TABLE IF NOT EXISTS metricas (
    sondeo TEXT NOT NULL, etapa TEXT NOT NULL, clave TEXT NOT NULL,
    valor_num REAL, valor_texto TEXT, actualizado_en TEXT,
    PRIMARY KEY (sondeo, etapa, clave)
);
"""


@contextmanager
def _conectar(proyecto: Proyecto):
    """Conexión al catálogo que confirma y cierra al salir."""
    conexion = sqlite3.connect(proyecto.ruta_catalogo)
    try:
        yield conexion
        conexion.commit()
    finally:
        conexion.close()


def _inicializar_catalogo(proyecto: Proyecto) -> None:
    with _conectar(proyecto) as conexion:
        conexion.executescript(_SQL_CREAR)


def _ahora() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _numero(valor: Any) -> Optional[float]:
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def registrar_sondeo(proyecto: Proyecto, sondeo: Any, meta: dict[str, Any]) -> None:
    """Registra o actualiza un sondeo a partir de los metadatos del archivo.

    Si el archivo trae ubicación válida, se proyecta al EPSG del proyecto
    (que se fija con el primer sondeo). Las coordenadas que el usuario
    capturó a mano (``origen_coord = 'usuario'``) no se sobrescriben.
    """
    lon, lat = _numero(meta.get("lon")), _numero(meta.get("lat"))
    x = y = None
    if geo.coordenadas_validas(lon, lat):
        if proyecto.epsg is None:
            fijar_epsg(proyecto, geo.epsg_utm(lon, lat))
        else:
            pass
        xs, ys = geo.geograficas_a_proyectadas([lon], [lat], proyecto.epsg)
        x, y = float(xs[0]), float(ys[0])
    else:
        lon = lat = None
    with _conectar(proyecto) as conexion:
        previa = conexion.execute("SELECT origen_coord FROM sondeos WHERE sondeo=?", (str(sondeo),)).fetchone()
        conservar = previa is not None and previa[0] == "usuario"
        conexion.execute(
            """
            INSERT INTO sondeos (sondeo, archivo_usf, archivo_xyz, instrumento, id_instrumento, inicio,
                                 n_barridos, n_lm, n_hm, lado_x, lado_y, vueltas, lon, lat, elev,
                                 x, y, z, azimut, origen_coord, creado_en)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            ON CONFLICT(sondeo) DO UPDATE SET
                archivo_usf=excluded.archivo_usf, archivo_xyz=excluded.archivo_xyz,
                instrumento=excluded.instrumento, id_instrumento=excluded.id_instrumento,
                inicio=excluded.inicio, n_barridos=excluded.n_barridos, n_lm=excluded.n_lm,
                n_hm=excluded.n_hm, lado_x=excluded.lado_x, lado_y=excluded.lado_y,
                vueltas=excluded.vueltas, lon=excluded.lon, lat=excluded.lat, elev=excluded.elev
            """,
            (str(sondeo), meta.get("archivo_usf"), meta.get("archivo_xyz"), meta.get("instrumento"),
             meta.get("id_instrumento"), meta.get("inicio"), meta.get("n_barridos"), meta.get("n_lm"),
             meta.get("n_hm"), _numero(meta.get("lado_x")), _numero(meta.get("lado_y")),
             _numero(meta.get("vueltas")) or 1.0, lon, lat, _numero(meta.get("elev")),
             x, y, _numero(meta.get("elev")), "archivo" if x is not None else None, _ahora()),
        )
        if not conservar and x is not None:
            conexion.execute("UPDATE sondeos SET x=?, y=?, z=?, origen_coord='archivo' WHERE sondeo=?",
                             (x, y, _numero(meta.get("elev")), str(sondeo)))
        else:
            pass


def fijar_coordenadas(proyecto: Proyecto, coordenadas: pd.DataFrame) -> int:
    """Asigna X, Y, Z (y azimut y lados del lazo) capturados por el usuario.

    Las X, Y se interpretan en el EPSG del proyecto. Devuelve cuántos
    sondeos cambiaron.
    """
    cambiados = 0
    with _conectar(proyecto) as conexion:
        for _, fila in coordenadas.iterrows():
            # Azimut y lados son opcionales: si la columna no viene o está
            # vacía se conserva lo que ya tenía el catálogo (COALESCE).
            azimut = _numero(fila.get("azimut")) if "azimut" in fila else None
            lado_x = _numero(fila.get("lado_x")) if "lado_x" in fila else None
            lado_y = _numero(fila.get("lado_y")) if "lado_y" in fila else None
            cursor = conexion.execute(
                "UPDATE sondeos SET x=?, y=?, z=?, azimut=COALESCE(?, azimut), lado_x=COALESCE(?, lado_x), "
                "lado_y=COALESCE(?, lado_y), origen_coord='usuario' WHERE sondeo=?",
                (float(fila["X"]), float(fila["Y"]), _numero(fila.get("Z")) or 0.0, azimut, lado_x, lado_y,
                 str(fila["Sondeo"])),
            )
            cambiados += cursor.rowcount
    return cambiados


def coordenadas(proyecto: Proyecto) -> pd.DataFrame:
    """Sondeos con coordenadas: ``Sondeo, X, Y, Z, azimut, lado_x, lado_y``."""
    with _conectar(proyecto) as conexion:
        df = pd.read_sql_query(
            "SELECT sondeo AS Sondeo, x AS X, y AS Y, z AS Z, azimut, lado_x, lado_y, origen_coord "
            "FROM sondeos WHERE x IS NOT NULL AND y IS NOT NULL ORDER BY inicio, sondeo", conexion)
    df["Z"] = df["Z"].fillna(0.0)
    df["azimut"] = df["azimut"].fillna(0.0)
    return df


def registrar_metricas(proyecto: Proyecto, sondeo: Any, etapa: str, metricas: dict[str, Any]) -> None:
    """Guarda métricas escalares (numéricas o de texto) en el catálogo."""
    _validar_etapa(etapa)
    ahora = _ahora()
    filas = []
    for clave, valor in metricas.items():
        if isinstance(valor, bool):
            filas.append((str(sondeo), etapa, clave, None, str(valor), ahora))
            continue
        else:
            pass
        if isinstance(valor, (int, float, np.integer, np.floating)):
            filas.append((str(sondeo), etapa, clave, float(valor), None, ahora))
            continue
        else:
            pass
        try:
            texto = valor if isinstance(valor, str) else json.dumps(valor, default=str)
            filas.append((str(sondeo), etapa, clave, None, texto, ahora))
        except (TypeError, ValueError):
            logger.debug("Métrica no serializable: %s", clave)
    if not filas:
        return None
    else:
        pass
    with _conectar(proyecto) as conexion:
        conexion.executemany(
            """
            INSERT INTO metricas (sondeo, etapa, clave, valor_num, valor_texto, actualizado_en)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(sondeo, etapa, clave) DO UPDATE SET
                valor_num=excluded.valor_num, valor_texto=excluded.valor_texto,
                actualizado_en=excluded.actualizado_en
            """, filas)
    return None


def guardar_tabla(proyecto: Proyecto, df: pd.DataFrame, etapa: str, tabla: str, sondeo: Any,
                  metricas: Optional[dict[str, Any]] = None) -> Path:
    """Guarda un DataFrame en la partición del sondeo y lo anota."""
    ruta = proyecto.ruta_tabla(etapa, tabla, sondeo)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    a_guardar = df.copy()
    for columna in a_guardar.columns:
        if a_guardar[columna].dtype == object:
            a_guardar[columna] = a_guardar[columna].astype(str)
        else:
            pass
    a_guardar.to_parquet(ruta, index=False, compression="snappy")
    with _conectar(proyecto) as conexion:
        conexion.execute(
            """
            INSERT INTO etapas (sondeo, etapa, tabla, n_filas, ruta_relativa, actualizado_en)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(sondeo, etapa, tabla) DO UPDATE SET
                n_filas=excluded.n_filas, ruta_relativa=excluded.ruta_relativa,
                actualizado_en=excluded.actualizado_en
            """,
            (str(sondeo), etapa, tabla, len(a_guardar), str(ruta.relative_to(proyecto.ruta)), _ahora()),
        )
    if metricas:
        registrar_metricas(proyecto, sondeo, etapa, metricas)
    else:
        pass
    return ruta


def cargar_tabla(proyecto: Proyecto, etapa: str, tabla: str, sondeo: Any,
                 columnas: Optional[list[str]] = None) -> Optional[pd.DataFrame]:
    """Carga una tabla de un sondeo (``None`` si no existe)."""
    ruta = proyecto.ruta_tabla(etapa, tabla, sondeo)
    if not ruta.exists():
        return None
    else:
        return pd.read_parquet(ruta, columns=columnas)


def existe_tabla(proyecto: Proyecto, etapa: str, tabla: str, sondeo: Any) -> bool:
    return proyecto.ruta_tabla(etapa, tabla, sondeo).exists()


def eliminar_tabla(proyecto: Proyecto, etapa: str, tabla: str, sondeo: Any) -> None:
    ruta = proyecto.ruta_tabla(etapa, tabla, sondeo)
    if ruta.exists():
        ruta.unlink()
    else:
        pass
    with _conectar(proyecto) as conexion:
        conexion.execute("DELETE FROM etapas WHERE sondeo=? AND etapa=? AND tabla=?",
                         (str(sondeo), etapa, tabla))


def eliminar_etapa(proyecto: Proyecto, etapa: str, sondeo: Any) -> None:
    """Borra todas las tablas de una etapa de un sondeo (resultados viejos)."""
    carpeta = proyecto.ruta_etapa(etapa) / f"{PREFIJO_PARTICION}={sanear_id(sondeo)}"
    if carpeta.exists():
        shutil.rmtree(carpeta)
    else:
        pass
    with _conectar(proyecto) as conexion:
        conexion.execute("DELETE FROM etapas WHERE sondeo=? AND etapa=?", (str(sondeo), etapa))
        conexion.execute("DELETE FROM metricas WHERE sondeo=? AND etapa=?", (str(sondeo), etapa))


def eliminar_sondeo(proyecto: Proyecto, sondeo: Any) -> None:
    """Elimina todos los datos de un sondeo."""
    for etapa in ETAPAS:
        carpeta = proyecto.ruta_etapa(etapa) / f"{PREFIJO_PARTICION}={sanear_id(sondeo)}"
        if carpeta.exists():
            shutil.rmtree(carpeta)
        else:
            pass
    with _conectar(proyecto) as conexion:
        for tabla in ("sondeos", "etapas", "metricas"):
            conexion.execute(f"DELETE FROM {tabla} WHERE sondeo = ?", (str(sondeo),))


def listar_sondeos(proyecto: Proyecto) -> pd.DataFrame:
    """Sondeos registrados (desde el catálogo), en orden de adquisición."""
    with _conectar(proyecto) as conexion:
        return pd.read_sql_query("SELECT * FROM sondeos ORDER BY inicio, sondeo", conexion)


def estado_pipeline(proyecto: Proyecto, sondeo: Optional[Any] = None) -> pd.DataFrame:
    consulta = "SELECT sondeo, etapa, tabla, n_filas, actualizado_en FROM etapas"
    parametros: tuple = ()
    if sondeo is not None:
        consulta += " WHERE sondeo = ?"
        parametros = (str(sondeo),)
    else:
        pass
    with _conectar(proyecto) as conexion:
        return pd.read_sql_query(consulta + " ORDER BY sondeo, etapa, tabla", conexion, params=parametros)


def obtener_metricas(proyecto: Proyecto, sondeo: Any, etapa: Optional[str] = None) -> pd.DataFrame:
    consulta = "SELECT etapa, clave, valor_num, valor_texto FROM metricas WHERE sondeo = ?"
    parametros: list = [str(sondeo)]
    if etapa is not None:
        consulta += " AND etapa = ?"
        parametros.append(etapa)
    else:
        pass
    with _conectar(proyecto) as conexion:
        return pd.read_sql_query(consulta, conexion, params=parametros)


def metricas_campana(proyecto: Proyecto, etapa: str) -> pd.DataFrame:
    """Métricas de una etapa de todos los sondeos, en formato ancho."""
    with _conectar(proyecto) as conexion:
        df = pd.read_sql_query(
            "SELECT sondeo, clave, valor_num, valor_texto FROM metricas WHERE etapa = ? AND sondeo != ?",
            conexion, params=(etapa, CAMPANA))
    if len(df) == 0:
        return pd.DataFrame()
    else:
        pass
    df["valor"] = df["valor_num"].where(df["valor_num"].notna(), df["valor_texto"])
    return df.pivot(index="sondeo", columns="clave", values="valor").reset_index()


# ======================================================================
# Consultas multi-sondeo (DuckDB)
# ======================================================================


def consultar_multisondeo(proyecto: Proyecto, etapa: str, tabla: str) -> pd.DataFrame:
    """Tabla de todos los sondeos (columna ``TDEM`` desde la partición)."""
    _validar_tabla(etapa, tabla)
    patron = (proyecto.ruta_etapa(etapa) / f"{PREFIJO_PARTICION}=*" / f"{tabla}.parquet").as_posix()
    if not list(proyecto.ruta_etapa(etapa).glob(f"{PREFIJO_PARTICION}=*/{tabla}.parquet")):
        return pd.DataFrame()
    else:
        pass
    try:
        with duckdb.connect() as conexion:
            df = conexion.execute(
                f"SELECT * FROM read_parquet('{patron}', hive_partitioning=true, union_by_name=true, "
                f"hive_types_autocast=false)"
            ).fetch_df()
    except (duckdb.IOException, duckdb.InvalidInputException):
        return pd.DataFrame()
    if PREFIJO_PARTICION in df.columns:
        df[PREFIJO_PARTICION] = df[PREFIJO_PARTICION].astype(str)
        df = df[df[PREFIJO_PARTICION] != CAMPANA]
    else:
        pass
    return df.reset_index(drop=True)


def tamano_proyecto(proyecto: Proyecto) -> dict[str, Any]:
    total = sum(f.stat().st_size for f in proyecto.ruta.rglob("*") if f.is_file())
    return {"total_mb": round(total / 1024 / 1024, 3)}


def rango_color_campana(proyecto: Proyecto, percentil_bajo: float = 1.5,
                        percentil_alto: float = 98.5) -> Optional[list[float]]:
    """Rango log10 de resistividad de todos los modelos multicapa."""
    df = consultar_multisondeo(proyecto, "inversion", "multicapa")
    if len(df) == 0:
        return None
    else:
        pass
    valores = np.log10(df["rho"].to_numpy(float))
    valores = valores[np.isfinite(valores)]
    if len(valores) < 2:
        return None
    else:
        pass
    bajo, alto = np.percentile(valores, [percentil_bajo, percentil_alto])
    if alto <= bajo:
        alto = bajo + 1.0
    else:
        pass
    return [float(bajo), float(alto)]
