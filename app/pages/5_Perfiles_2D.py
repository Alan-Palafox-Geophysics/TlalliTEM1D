"""
Pantalla 5 — Perfiles 2D
==================================================================
Proyección de los sondeos sobre una ruta definida por vértices,
pseudosección de ρa tardía contra profundidad de difusión y secciones
interpoladas de los modelos 1D (multicapa, bloques y referencia).

A diferencia de la versión SEV no hay inversión lateralmente
restringida (LCI): con sondeos TDEM separados decenas o cientos de
metros, la sección se arma con los modelos 1D (ver README).
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
import perfil_tdem as pf
import pipeline_tdem as pl

st.set_page_config(page_title="TLALLI-RES 1D TDEM — Perfiles 2D", page_icon="🗺️", layout="wide")
sesion = auth_app.exigir_acceso()
auth_app.barra_lateral_sesion(sesion)
marca_tdem.encabezado()
estado.panel_descarga_proyecto(usuario=sesion["usuario"])
estado.inicializar_estado()

st.subheader("🗺️ Perfiles 2D")
estado.mostrar_aviso()
proyecto = estado.requiere_proyecto()
if proyecto is None:
    st.stop()
else:
    pass

coords = alm.coordenadas(proyecto)
if len(coords) < 2:
    st.info("Los perfiles necesitan coordenadas de al menos dos sondeos. Cárguelas en la pantalla 1.")
    st.stop()
else:
    pass

perfiles = pl.listar_perfiles(proyecto)
# Tras construir un perfil, la página se relanza y debe abrir ese perfil:
# el valor del selector solo puede fijarse antes de dibujarlo.
_pendiente = st.session_state.pop("perfil_recien_creado", None)
if _pendiente in perfiles:
    st.session_state["sel_perfil"] = _pendiente
else:
    pass

with st.container(border=True):
    st.subheader("Ruta del perfil")
    c = st.columns([2, 2])
    opcion = c[0].selectbox("Perfil", ["➕ Nuevo perfil"] + perfiles, key="sel_perfil")
    if opcion == "➕ Nuevo perfil":
        nombre = c[1].text_input("Nombre del perfil", value=f"P{len(perfiles) + 1}", key="txt_perfil")
        ruta_guardada = None
    else:
        nombre = opcion
        c[1].markdown(f"Editando **{nombre}**")
        ruta_guardada = alm.cargar_tabla(proyecto, "perfiles", "perfil_ruta", nombre)

    clave_vertices = f"vertices_{nombre}"
    if clave_vertices not in st.session_state:
        if ruta_guardada is not None:
            st.session_state[clave_vertices] = ruta_guardada[["X", "Y"]].values.tolist()
        else:
            st.session_state[clave_vertices] = pf.ruta_automatica(coords)
    else:
        pass

    c = st.columns([3, 2])
    with c[0]:
        st.caption("Vértices de la ruta (agregue filas para una ruta quebrada). Los sondeos se proyectan "
                   "sobre el segmento más cercano.")
        vertices = st.data_editor(pd.DataFrame(st.session_state[clave_vertices], columns=["X", "Y"]),
                                  num_rows="dynamic", use_container_width=True, key=f"ed_{nombre}")
        if st.button("↺ Ruta automática por el eje de los sondeos", key="btn_auto"):
            st.session_state[clave_vertices] = pf.ruta_automatica(coords)
            st.rerun()
        else:
            pass
    with c[1]:
        buffer = st.number_input("Distancia máxima a la ruta (m)", 1.0, 1e6, 250.0, key="n_buffer")
        st.caption("Los sondeos a más de esta distancia de la ruta no entran al perfil. Sondeos repetidos a "
                   "pocos metros (controles) se proyectan casi en el mismo cadenamiento: conviene dejar solo uno.")

    vert = vertices.dropna().to_numpy(float).tolist()
    proyectados_vista = None
    if len(vert) >= 2:
        proyectados_vista = pf.proyectar(coords, vert, buffer=buffer)
    else:
        pass
    st.plotly_chart(gr.mapa_sondeos(coords, ruta=np.array(vert) if len(vert) >= 2 else None,
                                    proyectados=proyectados_vista, epsg=proyecto.epsg), use_container_width=True)

    if st.button("🗺️ Construir perfil", type="primary", key="btn_perfil", disabled=len(vert) < 2):
        st.session_state[clave_vertices] = vert
        resultado = acc.accion_perfil(proyecto, nombre, vert, buffer)
        estado.registrar_resultado(resultado)
        estado.limpiar_cache_datos()
        estado.avisar(resultado)
        if resultado:
            st.session_state["perfil_recien_creado"] = alm.sanear_id(nombre)
            st.session_state[f"vertices_{alm.sanear_id(nombre)}"] = vert
        else:
            pass
        st.rerun()
    else:
        pass

# ----------------------------------------------------------------------
# Resultados del perfil
# ----------------------------------------------------------------------
nombre_id = alm.sanear_id(nombre)
proyectados = estado.cargar_tabla("perfiles", "perfil_sondeos", nombre_id)
if proyectados is None:
    st.info("Construya el perfil para ver pseudosecciones y secciones.")
    st.stop()
else:
    pass
dentro = proyectados[proyectados["dentro"].astype(bool)].copy()
dentro["dentro"] = True
nombres = dentro["Sondeo"].astype(str).tolist()

curvas = {s: estado.cargar_tabla("apilado", "curva", s) for s in nombres}
multicapas = {s: estado.cargar_tabla("inversion", "multicapa", s) for s in nombres}
bloques = {s: estado.cargar_tabla("inversion", "bloques", s) for s in nombres}
dois = {s: pl.metrica(proyecto, s, "inversion", "doi") for s in nombres}
dois = {k: v for k, v in dois.items() if v is not None and np.isfinite(v)}
referencias = {}
for s_ in nombres:
    ref_ = estado.cargar_tabla("crudo", "modelo_referencia", s_)
    if ref_ is not None and len(ref_):
        esp_ = ref_["espesor"].to_numpy(float)
        bases_ = np.cumsum(np.where(np.isfinite(esp_), esp_, np.inf))
        referencias[s_] = pd.DataFrame({"z_techo": np.concatenate([[0.0], bases_[:-1]]), "z_base": bases_,
                                        "espesor": esp_, "rho": ref_["rho"].to_numpy(float)})
    else:
        pass

st.markdown(f"**{len(dentro)} sondeos en el perfil:** " + ", ".join(
    f"{r.Sondeo} (s={r.s:.0f} m, a {r.distancia:.0f} m)" for r in dentro.itertuples()))

st.subheader("Pseudosección")
st.caption("ρa de tiempo tardío de la curva combinada LM + HM, ubicada a la profundidad de difusión "
           "d = √(2tρa/μ0). Es una vista de los datos, no un modelo: la profundidad real de cada estructura "
           "sale de la inversión.")
datos_ps = pf.datos_pseudoseccion(dentro, {k: v for k, v in curvas.items() if v is not None})
st.plotly_chart(gr.pseudoseccion(datos_ps), use_container_width=True)

st.subheader("Secciones")
OPCIONES = {"multicapa": "Multicapa 1D interpolado", "bloques": "Bloques 1D interpolado",
            "referencia": "Modelo de referencia (pozo o sintético)"}
disponibles = [k for k in OPCIONES if k != "referencia" or len(referencias) >= 2]
c = st.columns([2, 1, 1])
seleccion = c[0].multiselect("Secciones a mostrar", disponibles, default=[k for k in disponibles][:3],
                             format_func=lambda k: OPCIONES[k], key="ms_secciones")
z_max = c[1].number_input("Profundidad máxima (m)", 0.0, 5000.0,
                          float(round(np.median(list(dois.values())))) if dois else 0.0, key="n_zmax_sec",
                          help="Por defecto la DOI mediana de los sondeos; 0 = la mayor DOI.")
mostrar_doi = c[2].checkbox("Marcar DOI", value=True, key="chk_doi_sec")
escala = estado.escala_color_activa()

secciones = []
for clave in seleccion:
    fuente = {"multicapa": multicapas, "bloques": bloques, "referencia": referencias}[clave]
    modelos = {k: v for k, v in fuente.items() if v is not None}
    try:
        secciones.append((clave, pf.seccion_interpolada(dentro, modelos, "rho", z_max=z_max or None, dois=dois)))
    except pf.PerfilError as exc:
        st.warning(f"{OPCIONES[clave]}: {exc}")

# Una sola escala de color para todas las secciones, para poder
# compararlas entre sí (y con el modelo de referencia).
valores_rho = [sec["valores"][np.isfinite(sec["valores"])] for _, sec in secciones]
valores_rho = np.concatenate(valores_rho) if valores_rho else np.array([])
rango_comun = tuple(np.percentile(valores_rho, [1.5, 98.5])) if len(valores_rho) else None
for clave, sec in secciones:
    st.plotly_chart(gr.seccion(sec, OPCIONES[clave], escala=escala, mostrar_doi=mostrar_doi, rango=rango_comun),
                    use_container_width=True)

with st.expander("📋 Registro de ejecución", expanded=False):
    estado.mostrar_logs()
