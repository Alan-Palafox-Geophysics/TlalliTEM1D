"""
app.py
==================================================================
Punto de entrada de **TLALLI-RES 1D TDEM** — sondeos electromagnéticos
en el dominio del tiempo (lazo central, ABEM WalkTEM y compatibles).

Ejecutar con::

    streamlit run app/app.py

Arquitectura (idéntica a TlalliRes1D SEV)
--------------------------------------------
- Páginas separadas (``pages/``): cada una es un script independiente.
- ``st.session_state`` guarda solo punteros; los datos se releen de
  disco con caché acotado (``estado_app``).
- La lógica vive en ``acciones_app`` y ``src/``; las pantallas solo
  dibujan y delegan.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import logging

import streamlit as st

import _ruta_src  # noqa: F401
import acciones_app as acc
import auth_app
import marca_tdem
import estado_app as estado

VERSION_APP = "2026.11.13-a"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S")

st.set_page_config(page_title="TLALLI-RES 1D — TDEM", page_icon="🧲", layout="wide")

sesion = auth_app.exigir_acceso()
auth_app.barra_lateral_sesion(sesion)
marca_tdem.encabezado()
estado.panel_descarga_proyecto(usuario=sesion["usuario"])
estado.inicializar_estado()

st.subheader("🧲 Sondeos TDEM — procesamiento, control de calidad e inversión")
st.caption(
    "Flujo: lectura de USF (y XYZ) → apilado robusto por gate y control de calidad → exclusión manual "
    "de gates → inversión multicapa suave (SimPEG, forma de onda del archivo) → DOI → modelo de bloques "
    "(BIC) → parámetros de Dar Zarrouk y corrosividad → perfiles 2D."
)

proyecto = estado.obtener_proyecto()
if proyecto is None:
    st.info("**Comience en la pantalla «Proyecto y datos»** (menú lateral) para crear un proyecto "
            "nuevo o abrir un archivo `.itdem`.")
else:
    resumen = acc.resumen_proyecto(proyecto)
    st.success(f"Proyecto activo: **{resumen['nombre']}**"
               + (f" · EPSG {resumen['epsg']}" if resumen["epsg"] else ""))
    tarjetas = st.columns(5)
    tarjetas[0].metric("Sondeos", resumen["n_sondeos"])
    tarjetas[1].metric("Invertidos", resumen["n_invertidos"])
    tarjetas[2].metric("Completos", resumen["n_completos"])
    tarjetas[3].metric("Con coordenadas", resumen["n_con_coordenadas"])
    tarjetas[4].metric("Tamaño", f"{resumen['tamano_mb']:.2f} MB")

    import pipeline_tdem as pl

    tabla = pl.tabla_estado(proyecto)
    if len(tabla):
        st.subheader("Estado del proyecto")
        st.dataframe(tabla, use_container_width=True, hide_index=True)
    else:
        st.caption("El proyecto aún no tiene datos cargados.")

with st.sidebar:
    st.header("Navegación")
    st.markdown(
        "1. **Proyecto y datos** — proyecto, archivos USF/XYZ, coordenadas y mapa de lazos\n"
        "2. **Control de calidad** — apilado, QC y exclusión de gates\n"
        "3. **Inversión** — multicapa → DOI → bloques\n"
        "4. **Parámetros y unidades** — Dar Zarrouk, corrosividad, unidades\n"
        "5. **Perfiles 2D** — rutas, secciones y pseudosección\n"
        "6. **Exportar datos** — CSV y modelos"
    )
    st.divider()
    st.caption(f"Versión del código: `{VERSION_APP}`")
    if st.button("🧹 Limpiar caché de datos", use_container_width=True):
        estado.limpiar_cache_datos()
        st.toast("Caché limpiado.")
    else:
        pass

auth_app.panel_administracion(sesion)
