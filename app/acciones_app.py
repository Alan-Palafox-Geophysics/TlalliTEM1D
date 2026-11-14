"""
acciones_app.py
==================================================================
**Acciones** de la aplicación: la lógica de cada botón.

Cada función devuelve un :class:`pipeline_tdem.ResultadoEtapa`
(booleano-evaluable) y no importa Streamlit, de modo que todo lo que
hace un botón se prueba desde código (ver ``test_app.py``).

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

import _ruta_src  # noqa: F401
import almacenamiento_tdem as alm
import pipeline_tdem as pl
from almacenamiento_tdem import Proyecto
from pipeline_tdem import ConfigPipeline, ResultadoEtapa, TokenCancelacion

logger = logging.getLogger("tdem.acciones")

#: Extensiones de archivo de campo que acepta la carga.
EXTENSIONES_CAMPO = ("usf", "xyz")


# ----------------------------------------------------------------------
# Proyecto
# ----------------------------------------------------------------------


def accion_crear_proyecto(ruta_base, nombre: str, cliente: Optional[str] = None,
                          anio: Optional[str] = None) -> ResultadoEtapa:
    if not str(nombre).strip():
        return ResultadoEtapa(False, "proyecto", "El nombre del proyecto no puede estar vacío.")
    else:
        pass
    try:
        proyecto = alm.crear_proyecto(ruta_base, nombre, cliente=cliente, anio=anio)
    except OSError as exc:
        return ResultadoEtapa(False, "proyecto", f"No se pudo crear el proyecto: {exc}")
    return ResultadoEtapa(True, "proyecto", f"Proyecto '{proyecto.nombre}' listo.",
                          datos={"proyecto": proyecto})


def accion_abrir_proyecto(ruta) -> ResultadoEtapa:
    try:
        proyecto = alm.abrir_proyecto(ruta)
    except alm.ProyectoNoEncontradoError as exc:
        return ResultadoEtapa(False, "proyecto", str(exc))
    return ResultadoEtapa(True, "proyecto", f"Proyecto '{proyecto.nombre}' abierto.",
                          datos={"proyecto": proyecto})


def accion_fijar_epsg(proyecto: Proyecto, epsg: int) -> ResultadoEtapa:
    """Fija el EPSG cuando ningún archivo trajo GPS.

    Solo se permite mientras no haya coordenadas: cambiarlo después
    dejaría X, Y en un sistema distinto al declarado.
    """
    from pyproj import CRS

    try:
        crs = CRS.from_epsg(int(epsg))
    except Exception:
        return ResultadoEtapa(False, "proyecto", f"EPSG {epsg} no es un código válido.")
    if not crs.is_projected:
        return ResultadoEtapa(False, "proyecto", f"EPSG {epsg} ({crs.name}) no es un sistema proyectado (en metros).")
    else:
        pass
    if proyecto.epsg is not None and int(proyecto.epsg) != int(epsg) and len(alm.coordenadas(proyecto)):
        return ResultadoEtapa(False, "proyecto", f"El proyecto ya tiene coordenadas en EPSG {proyecto.epsg}; "
                                                 "no se puede cambiar el sistema.")
    else:
        pass
    alm.fijar_epsg(proyecto, int(epsg))
    return ResultadoEtapa(True, "proyecto", f"Sistema de coordenadas: EPSG {epsg} ({crs.name}).")


def guardar_archivo_subido(proyecto: Proyecto, nombre_archivo: str, contenido: bytes) -> Path:
    """Conserva el archivo subido dentro del proyecto (``_entradas``).

    Viaja dentro del ``.itdem``: el proyecto se puede reprocesar desde
    cero sin pedir otra vez los archivos de campo.
    """
    carpeta = proyecto.ruta / alm.CARPETA_ENTRADAS
    carpeta.mkdir(exist_ok=True)
    destino = carpeta / Path(nombre_archivo).name
    destino.write_bytes(contenido)
    return destino


def accion_eliminar_sondeo(proyecto: Proyecto, sondeo: str) -> ResultadoEtapa:
    try:
        alm.eliminar_sondeo(proyecto, sondeo)
    except OSError as exc:
        return ResultadoEtapa(False, "proyecto", f"No se pudo eliminar: {exc}")
    return ResultadoEtapa(True, "proyecto", f"Sondeo '{sondeo}' eliminado del proyecto.")


# ----------------------------------------------------------------------
# Datos y coordenadas
# ----------------------------------------------------------------------


def accion_cargar_archivos(proyecto: Proyecto, rutas: list, config: ConfigPipeline,
                           callback_progreso: Optional[Callable] = None) -> ResultadoEtapa:
    """Botón «Leer y apilar»: lectura, apilado y QC de todos los sondeos."""
    salida = pl.ejecutar_pipeline_archivos(proyecto, rutas, config, "apilado", callback_progreso)
    lectura = salida["lectura"]
    if not lectura:
        return lectura
    else:
        pass
    fallidos = []
    advertencias = list(lectura.advertencias)
    for nombre, etapas in salida["sondeos"].items():
        for etapa, resultado in etapas.items():
            advertencias += [f"{nombre}/{etapa}: {a}" for a in resultado.advertencias]
            if not resultado:
                fallidos.append(f"{nombre} ({etapa}): {resultado.mensaje}")
            else:
                pass
    n = len(salida["sondeos"])
    if fallidos:
        return ResultadoEtapa(False, "lectura", f"{n - len(fallidos)} de {n} sondeos leídos y apilados. Fallos: "
                              + " | ".join(fallidos), advertencias=advertencias, datos=lectura.datos)
    else:
        return ResultadoEtapa(True, "lectura", f"{n} sondeo(s) leído(s), apilado(s) y revisado(s).",
                              advertencias=advertencias, datos=lectura.datos)


def accion_cargar_coordenadas(proyecto: Proyecto, ruta_o_bytes) -> ResultadoEtapa:
    return pl.cargar_coordenadas(proyecto, ruta_o_bytes)


def tabla_coordenadas_editable(proyecto: Proyecto) -> pd.DataFrame:
    """Todos los sondeos (con o sin coordenadas) para el editor."""
    sondeos = alm.listar_sondeos(proyecto)
    if len(sondeos) == 0:
        return pd.DataFrame(columns=["Sondeo", "X", "Y", "Z", "lado_x", "lado_y", "azimut", "origen"])
    else:
        pass
    return pd.DataFrame({
        "Sondeo": sondeos["sondeo"].astype(str), "X": sondeos["x"], "Y": sondeos["y"], "Z": sondeos["z"],
        "lado_x": sondeos["lado_x"], "lado_y": sondeos["lado_y"], "azimut": sondeos["azimut"].fillna(0.0),
        "origen": sondeos["origen_coord"].fillna("faltan"),
    })


def accion_editar_coordenadas(proyecto: Proyecto, tabla: pd.DataFrame,
                              original: Optional[pd.DataFrame] = None) -> ResultadoEtapa:
    """Guarda lo capturado en el editor.

    Solo se escriben las filas que cambiaron respecto a ``original``:
    así una coordenada que vino del archivo no pasa a «capturada» solo
    por haber guardado la tabla.
    """
    if proyecto.epsg is None:
        return ResultadoEtapa(False, "coordenadas", "Fije primero el EPSG del proyecto.")
    else:
        pass
    validas = tabla.dropna(subset=["X", "Y"]).copy()
    if original is not None and len(original):
        columnas = ["X", "Y", "Z", "lado_x", "lado_y", "azimut"]
        base = original.set_index("Sondeo")[columnas]
        cambio = []
        for _, fila in validas.iterrows():
            previa = base.loc[fila["Sondeo"]] if fila["Sondeo"] in base.index else None
            if previa is None:
                cambio.append(True)
            else:
                distintas = [not _iguales(fila[c], previa[c]) for c in columnas]
                cambio.append(any(distintas))
        validas = validas[cambio]
    else:
        pass
    if len(validas) == 0:
        return ResultadoEtapa(True, "coordenadas", "Sin cambios en las coordenadas.")
    else:
        pass
    n = alm.fijar_coordenadas(proyecto, validas)
    return ResultadoEtapa(True, "coordenadas", f"Coordenadas actualizadas en {n} sondeo(s). "
                          "Si cambió el tamaño del lazo, vuelva a correr el apilado (ρa) y la inversión.")


def _iguales(a: Any, b: Any) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    else:
        pass
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def accion_guardar_exclusiones(proyecto: Proyecto, sondeo: str, tabla: pd.DataFrame,
                               config: ConfigPipeline) -> ResultadoEtapa:
    """Guarda los gates excluidos a mano y rehace el apilado del sondeo."""
    pl.guardar_exclusiones(proyecto, sondeo, tabla)
    return pl.ejecutar_apilado(proyecto, sondeo, config)


# ----------------------------------------------------------------------
# Etapas
# ----------------------------------------------------------------------


def accion_reprocesar(proyecto: Proyecto, sondeos: list[str], config: ConfigPipeline,
                      desde_etapa: str = "apilado", hasta_etapa: str = "parametros",
                      callback_progreso: Optional[Callable] = None,
                      token_cancelacion: Optional[TokenCancelacion] = None) -> ResultadoEtapa:
    """Rehace etapas de uno o varios sondeos ya cargados."""
    fallidos, advertencias, mensajes = [], [], []
    for i, nombre in enumerate(sondeos):
        def _cb(etapa, indice, total, resultado, _i=i, _n=nombre):
            if callback_progreso is not None:
                callback_progreso(f"{_n} · {pl.ETIQUETAS_ETAPA[etapa]}", (_i + indice / total) / len(sondeos), resultado)
            else:
                pass

        etapas = pl.ejecutar_pipeline_sondeo(proyecto, nombre, config, desde_etapa, hasta_etapa, _cb, token_cancelacion)
        for etapa, resultado in etapas.items():
            advertencias += [f"{nombre}/{etapa}: {a}" for a in resultado.advertencias]
            mensajes.append(resultado.mensaje)
            if not resultado:
                fallidos.append(f"{nombre} ({etapa}): {resultado.mensaje}")
            else:
                pass
        if token_cancelacion is not None and token_cancelacion.cancelado:
            break
        else:
            pass
    if len(sondeos) > 1:
        pl.ejecutar_qc_campana(proyecto)
    else:
        pass
    if fallidos:
        return ResultadoEtapa(False, "pipeline", " | ".join(fallidos), advertencias=advertencias,
                              datos={"mensajes": mensajes})
    else:
        return ResultadoEtapa(True, "pipeline", f"{len(sondeos)} sondeo(s) procesado(s).",
                              advertencias=advertencias, datos={"mensajes": mensajes})


def accion_qc_campana(proyecto: Proyecto) -> ResultadoEtapa:
    return pl.ejecutar_qc_campana(proyecto)


def accion_unidades(proyecto: Proyecto, n_max: int = 6) -> ResultadoEtapa:
    return pl.ejecutar_unidades(proyecto, n_max=n_max)


def accion_perfil(proyecto: Proyecto, nombre: str, vertices, buffer: float) -> ResultadoEtapa:
    if not str(nombre).strip():
        return ResultadoEtapa(False, "perfil", "Dé un nombre al perfil.")
    else:
        pass
    return pl.ejecutar_perfil(proyecto, alm.sanear_id(nombre), vertices, buffer=buffer)


# ----------------------------------------------------------------------
# Paquete
# ----------------------------------------------------------------------


def accion_empaquetar_proyecto(proyecto: Proyecto, usuario: Optional[str] = None) -> ResultadoEtapa:
    import paquete_tdem as pk

    try:
        sondeos = alm.listar_sondeos(proyecto)["sondeo"].astype(str).tolist()
        contenido = pk.empaquetar_proyecto(proyecto, sondeos, usuario=usuario)
    except (OSError, ValueError) as exc:
        return ResultadoEtapa(False, "paquete", f"No se pudo empaquetar el proyecto: {exc}")
    return ResultadoEtapa(
        True, "paquete", f"Proyecto listo para descargar ({len(contenido) / 1024 / 1024:.2f} MB).",
        datos={"contenido": contenido, "nombre": pk.nombre_paquete(proyecto.nombre)},
        metricas={"n_sondeos": len(sondeos), "tamano_mb": round(len(contenido) / 1024 / 1024, 3)},
    )


def accion_restaurar_proyecto(contenido: bytes, espacio) -> ResultadoEtapa:
    import paquete_tdem as pk

    try:
        manifiesto = pk.leer_manifiesto(contenido)
        ruta = pk.restaurar_proyecto(contenido, espacio)
        proyecto = alm.abrir_proyecto(ruta)
    except (pk.PaqueteError, alm.ProyectoNoEncontradoError) as exc:
        return ResultadoEtapa(False, "paquete", str(exc))
    n = len(alm.listar_sondeos(proyecto))
    return ResultadoEtapa(True, "paquete",
                          f"Proyecto '{proyecto.nombre}' restaurado con {n} sondeo(s) "
                          f"(generado el {manifiesto.get('generado_en', '¿?')}).",
                          datos={"proyecto": proyecto})


# ----------------------------------------------------------------------
# Consultas
# ----------------------------------------------------------------------


def resumen_proyecto(proyecto: Proyecto) -> dict[str, Any]:
    sondeos = alm.listar_sondeos(proyecto)
    n_completos = 0
    n_invertidos = 0
    for nombre in sondeos["sondeo"].astype(str):
        estado = pl.estado_sondeo(proyecto, nombre)
        if all(estado.values()):
            n_completos += 1
        else:
            pass
        if estado["inversion"]:
            n_invertidos += 1
        else:
            pass
    return {
        "nombre": proyecto.nombre, "cliente": proyecto.cliente, "epsg": proyecto.epsg,
        "n_sondeos": len(sondeos), "n_completos": n_completos, "n_invertidos": n_invertidos,
        "n_con_coordenadas": int(sondeos["x"].notna().sum()) if len(sondeos) else 0,
        "tamano_mb": alm.tamano_proyecto(proyecto)["total_mb"],
    }


def metricas_sondeo(proyecto: Proyecto, sondeo: str) -> dict[str, Any]:
    df = alm.obtener_metricas(proyecto, sondeo)
    salida: dict[str, Any] = {}
    for _, fila in df.iterrows():
        clave = f"{fila['etapa']}.{fila['clave']}"
        salida[clave] = float(fila["valor_num"]) if pd.notna(fila["valor_num"]) else fila["valor_texto"]
    return salida
