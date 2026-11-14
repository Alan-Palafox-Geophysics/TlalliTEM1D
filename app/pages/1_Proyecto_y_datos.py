"""
Pantalla 1 — Proyecto y datos
==================================================================
Crear/abrir proyecto, cargar los archivos de campo (USF con prioridad y
XYZ opcional, emparejados por la marca del stb), revisar o capturar las
coordenadas y el lazo de cada sondeo, y ver el mapa de lazos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import _ruta_src  # noqa: F401
import acciones_app as acc
import almacenamiento_tdem as alm
import auth_app
import estado_app as estado
import graficos_tdem as gr
import marca_tdem
import pipeline_tdem as pl

st.set_page_config(page_title="TLALLI-RES 1D TDEM — Proyecto y datos", page_icon="📁", layout="wide")
sesion = auth_app.exigir_acceso()
auth_app.barra_lateral_sesion(sesion)
marca_tdem.encabezado()
estado.panel_descarga_proyecto(usuario=sesion["usuario"])
estado.inicializar_estado()

st.subheader("📁 Proyecto y datos")
estado.mostrar_aviso()

# ----------------------------------------------------------------------
# Proyecto
# ----------------------------------------------------------------------
with st.container(border=True):
    st.subheader("Proyecto")
    st.caption("Sus datos se procesan en un espacio temporal que **se elimina al cerrar sesión**. "
               "Descargue el archivo `.itdem` para conservar el proyecto.")
    pestana_nuevo, pestana_abrir = st.tabs(["Crear nuevo", "Abrir archivo .itdem"])
    with pestana_nuevo:
        c = st.columns([2, 1, 1])
        nombre_proyecto = c[0].text_input("Nombre del proyecto", key="nuevo_nombre")
        cliente = c[1].text_input("Cliente", key="nuevo_cliente")
        anio = c[2].text_input("Año", key="nuevo_anio")
        if st.button("Crear proyecto", type="primary", key="btn_crear"):
            resultado = acc.accion_crear_proyecto(estado.espacio_trabajo(), nombre_proyecto, cliente, anio)
            estado.registrar_resultado(resultado)
            if resultado:
                estado.fijar_proyecto(resultado.datos["proyecto"])
                st.rerun()
            else:
                st.error(resultado.mensaje)
        else:
            pass
    with pestana_abrir:
        archivo_proyecto = st.file_uploader("Archivo de proyecto (.itdem)", type=["itdem", "zip"], key="up_itdem")
        if st.button("Abrir proyecto", key="btn_abrir", disabled=archivo_proyecto is None):
            resultado = acc.accion_restaurar_proyecto(archivo_proyecto.getvalue(), estado.espacio_trabajo())
            estado.registrar_resultado(resultado)
            if resultado:
                estado.fijar_proyecto(resultado.datos["proyecto"])
                st.rerun()
            else:
                st.error(resultado.mensaje)
        else:
            pass

proyecto = estado.obtener_proyecto()
if proyecto is None:
    st.info("Cree o abra un proyecto para continuar.")
    st.stop()
else:
    pass

resumen = acc.resumen_proyecto(proyecto)
t = st.columns(5)
t[0].metric("Proyecto", resumen["nombre"])
t[1].metric("Sondeos", resumen["n_sondeos"])
t[2].metric("Invertidos", resumen["n_invertidos"])
t[3].metric("Con coordenadas", resumen["n_con_coordenadas"])
t[4].metric("Sistema", f"EPSG {resumen['epsg']}" if resumen["epsg"] else "sin fijar")

config = estado.obtener_config()

# ----------------------------------------------------------------------
# Carga de archivos
# ----------------------------------------------------------------------
with st.container(border=True):
    st.subheader("Cargar sondeos")
    st.caption(
        "Suba los **.usf** (formato universal de sondeo, fuente principal) y, si los tiene, los **.xyz** "
        "exportados con stb2xyz. Se emparejan por la marca `AAAA_MMDD_HHMMSS` del archivo stb; el XYZ "
        "aporta hora, corriente y GPS por barrido. Un XYZ sin su USF también se puede procesar. "
        "Al cargar se ejecutan la lectura, el apilado y el control de calidad; la inversión se lanza "
        "en la pantalla 3, después de revisar el QC."
    )
    archivos = st.file_uploader("Archivos de campo", type=list(acc.EXTENSIONES_CAMPO),
                                accept_multiple_files=True, key="up_campo")
    b = st.columns(2)
    if b[0].button("📥 Leer, apilar y revisar", type="primary", use_container_width=True, key="btn_cargar",
                   disabled=not archivos):
        rutas = [acc.guardar_archivo_subido(proyecto, a.name, a.getvalue()) for a in archivos]
        resultado, token = estado.ejecutar_con_progreso(
            lambda cb, tk: acc.accion_cargar_archivos(proyecto, rutas, config, cb), "Leyendo archivos…")
        estado.registrar_resultado(resultado)
        if resultado.datos.get("sondeos"):
            estado.fijar_sondeo(resultado.datos["sondeos"][0])
        else:
            pass
        estado.avisar(resultado, cancelado=token.cancelado)
        st.rerun()
    else:
        pass
    sondeos = estado.sondeos_disponibles()
    if sondeos:
        with b[1].popover("🗑️ Eliminar un sondeo", use_container_width=True):
            a_borrar = st.selectbox("Sondeo", sondeos, key="sel_borrar")
            if st.button("Eliminar definitivamente", key="btn_borrar"):
                resultado = acc.accion_eliminar_sondeo(proyecto, a_borrar)
                estado.registrar_resultado(resultado)
                estado.avisar(resultado)
                estado.fijar_sondeo(None)
                estado.limpiar_cache_datos()
                st.rerun()
            else:
                pass
    else:
        pass

sondeos = estado.sondeos_disponibles()

# ----------------------------------------------------------------------
# Coordenadas y lazos
# ----------------------------------------------------------------------
if sondeos:
    with st.container(border=True):
        st.subheader("Coordenadas y lazos")
        st.caption(
            "Cuando el archivo trae GPS, el centro del lazo se proyecta a UTM (el EPSG del proyecto se fija "
            "con el primer sondeo). Si falta, capture **X, Y** en la tabla o suba un archivo "
            "`Sondeo, X, Y, Z[, azimut]`. También puede corregir el **tamaño del lazo** y su **azimut** "
            "(grados desde el Este, sentido horario; 0 = lados alineados con los ejes). Las coordenadas "
            "capturadas a mano no se sobrescriben al volver a cargar el archivo."
        )
        if proyecto.epsg is None:
            st.warning("Ningún archivo trajo GPS válido: indique el sistema de coordenadas proyectadas "
                       "en que capturará X, Y (por ejemplo 32614 = WGS 84 / UTM 14N).", icon="🧭")
            c = st.columns([1, 3])
            epsg = c[0].number_input("EPSG", min_value=2000, max_value=999999, value=32614, step=1, key="in_epsg")
            if c[0].button("Fijar EPSG", key="btn_epsg"):
                resultado = acc.accion_fijar_epsg(proyecto, int(epsg))
                estado.registrar_resultado(resultado)
                estado.avisar(resultado)
                st.rerun()
            else:
                pass
        else:
            pass

        c = st.columns([2, 5])
        archivo_coord = c[0].file_uploader("Archivo de coordenadas", type=["txt", "csv", "dat"], key="up_coord")
        if c[0].button("📍 Cargar archivo de coordenadas", disabled=archivo_coord is None or proyecto.epsg is None,
                       key="btn_coord"):
            resultado = acc.accion_cargar_coordenadas(proyecto, archivo_coord.getvalue())
            estado.registrar_resultado(resultado)
            estado.avisar(resultado)
            st.rerun()
        else:
            pass
        original = acc.tabla_coordenadas_editable(proyecto)
        editada = c[1].data_editor(
            original, disabled=["Sondeo", "origen"], hide_index=True, key="ed_coord", use_container_width=True,
            column_config={
                "X": st.column_config.NumberColumn("X (m)", format="%.2f"),
                "Y": st.column_config.NumberColumn("Y (m)", format="%.2f"),
                "Z": st.column_config.NumberColumn("Z (m s.n.m.)", format="%.1f"),
                "lado_x": st.column_config.NumberColumn("Lado X (m)", min_value=1.0, format="%.1f"),
                "lado_y": st.column_config.NumberColumn("Lado Y (m)", min_value=1.0, format="%.1f"),
                "azimut": st.column_config.NumberColumn("Azimut (°)", min_value=-180.0, max_value=180.0,
                                                        format="%.1f"),
                "origen": st.column_config.TextColumn("Origen", help="archivo · usuario · faltan"),
            })
        if c[1].button("💾 Guardar cambios de la tabla", key="btn_guardar_coord",
                       disabled=proyecto.epsg is None):
            resultado = acc.accion_editar_coordenadas(proyecto, editada, original)
            estado.registrar_resultado(resultado)
            estado.avisar(resultado)
            st.rerun()
        else:
            pass

        coords = alm.coordenadas(proyecto)
        if len(coords):
            st.plotly_chart(gr.mapa_sondeos(coords, epsg=proyecto.epsg), use_container_width=True)
            faltan = sorted(set(sondeos) - set(coords["Sondeo"].astype(str)))
            if faltan:
                st.caption(f"Sin coordenadas (no aparecen en el mapa): {', '.join(faltan)}")
            else:
                pass
        else:
            st.info("Ningún sondeo tiene coordenadas todavía.")
else:
    pass

# ----------------------------------------------------------------------
# Datos del sondeo activo
# ----------------------------------------------------------------------
st.divider()
c = st.columns([1, 3])
with c[0]:
    sondeo = estado.selector_sondeo(key="sel_sondeo_p1")
with c[1]:
    if sondeo:
        estado.mostrar_estado_pipeline(sondeo)
    else:
        pass

if sondeo:
    barridos = estado.cargar_tabla("crudo", "barridos")
    if barridos is None:
        st.info("Este sondeo aún no tiene datos leídos.")
    else:
        meta = pl.meta_sondeo(proyecto, sondeo)
        m = st.columns(6)
        m[0].metric("Instrumento", str(meta.get("instrumento") or "—"))
        m[1].metric("Inicio", str(meta.get("inicio") or "—")[:16].replace("T", " "))
        m[2].metric("Barridos LM / HM", f"{int(meta.get('n_lm') or 0)} / {int(meta.get('n_hm') or 0)}")
        geometria = pl.geometria_sondeo(proyecto, sondeo)
        m[3].metric("Lazo", f"{geometria.lado_x:g} × {geometria.lado_y:g} m · {geometria.vueltas:g} v")
        m[4].metric("Corriente media", f"{barridos['corriente_a'].mean():.2f} A")
        m[5].metric("Fuente", str(meta.get("fuente_principal") or "—").upper())
        st.plotly_chart(gr.adquisicion(barridos), use_container_width=True)
        coords = alm.coordenadas(proyecto)
        fila = coords[coords["Sondeo"].astype(str) == sondeo]
        if len(fila):
            st.plotly_chart(gr.posicion_gps(barridos, fila.iloc[0], proyecto.epsg), use_container_width=True)
        else:
            pass
        with st.expander("Metadatos del archivo"):
            st.dataframe(pd.DataFrame({"clave": list(meta), "valor": [str(v) for v in meta.values()]}),
                         hide_index=True, use_container_width=True, height=320)
        with st.expander("Barridos (un renglón por stack)"):
            st.dataframe(barridos, use_container_width=True, height=320)
else:
    pass

with st.expander("📋 Registro de ejecución", expanded=False):
    estado.mostrar_logs()
