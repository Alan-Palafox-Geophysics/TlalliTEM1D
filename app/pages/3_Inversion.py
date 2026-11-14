"""
Pantalla 3 — Inversión
==================================================================
1. Multicapa suave (SimPEG, Gauss–Newton con λ enfriándose) → 2. DOI de
Oldenburg–Li → 3. modelo de bloques (pocas capas, BIC/AIC) con rangos
±1σ. Muestra el ajuste de cada momento, los modelos, la convergencia,
la selección del número de capas, los parámetros de Dar Zarrouk y la
corrosividad.
"""

from __future__ import annotations

import json

import numpy as np
import streamlit as st

import _ruta_src  # noqa: F401
import acciones_app as acc
import almacenamiento_tdem as alm
import auth_app
import estado_app as estado
import graficos_tdem as gr
import lectura_tdem as lec
import marca_tdem
import parametros_tdem as ps
import pipeline_tdem as pl

st.set_page_config(page_title="TLALLI-RES 1D TDEM — Inversión", page_icon="🧲", layout="wide")
sesion = auth_app.exigir_acceso()
auth_app.barra_lateral_sesion(sesion)
marca_tdem.encabezado()
estado.panel_descarga_proyecto(usuario=sesion["usuario"])
estado.inicializar_estado()

st.subheader("🧲 Inversión 1D")
estado.mostrar_aviso()
proyecto = estado.requiere_proyecto()
if proyecto is None:
    st.stop()
else:
    pass

c = st.columns([1, 3])
with c[0]:
    sondeo = estado.selector_sondeo(key="sel_sondeo_p3")
with c[1]:
    if sondeo:
        estado.mostrar_estado_pipeline(sondeo)
    else:
        pass
if sondeo is None:
    st.stop()
else:
    pass

config = estado.obtener_config()

with st.container(border=True):
    st.subheader("Configuración del proceso de inversión")
    st.caption(
        "Modelado directo con SimPEG (`Simulation1DLayered`): lazo cuadrado real (no dipolo), receptor en "
        "su posición, **forma de onda de cada momento tomada del archivo** e integración de cada gate en su "
        "ventana. Los parámetros se pueden guardar en el proyecto para toda la campaña."
    )
    b0 = st.columns([1, 1, 1, 1])
    if b0[0].button("↺ Automático (parámetros recomendados)", key="btn_auto"):
        config.inversion = pl.ConfigInversion()
        estado.fijar_config(config)
        st.rerun()
    else:
        pass
    if b0[1].button("💾 Guardar parámetros de la campaña", key="btn_guardar_par"):
        pl.guardar_parametros(proyecto, config)
        st.success("Parámetros guardados en el proyecto: se usarán al abrirlo y viajan en el .itdem.")
    else:
        pass
    b0[2].download_button("⬇️ Exportar parámetros (.json)",
                          data=json.dumps(pl.config_a_dict(config), indent=2, ensure_ascii=False, default=str),
                          file_name="parametros_tlallires1d_tdem.json", mime="application/json", key="btn_exp_par")
    archivo_par = b0[3].file_uploader("Importar parámetros (.json)", type=["json"], key="up_par",
                                      label_visibility="collapsed")
    if archivo_par is not None and st.session_state.get("par_importado") != archivo_par.name:
        try:
            config = pl.config_desde_dict(json.loads(archivo_par.getvalue().decode("utf-8")))
            estado.fijar_config(config)
            st.session_state["par_importado"] = archivo_par.name
            st.rerun()
        except Exception as exc:
            st.error(f"No se pudieron leer los parámetros: {exc}")
    else:
        pass
    ci = config.inversion

    st.markdown("**Datos**")
    cd = st.columns(4)
    MOMENTOS = {"ambos": "LM y HM juntos", "LM": "Solo LM", "HM": "Solo HM"}
    ci.momentos = cd[0].selectbox("Momentos", list(MOMENTOS), index=list(MOMENTOS).index(ci.momentos),
                                  format_func=lambda k: MOMENTOS[k], key="sel_mom",
                                  help="Invertir un solo momento ayuda a diagnosticar cuál de los dos no ajusta.")
    ci.error_relativo_minimo = cd[1].number_input(
        "Piso de error (%)", 0.5, 30.0, float(100 * ci.error_relativo_minimo), 0.5, key="n_err",
        help=("Error asignado = error estándar del apilado ⊕ este porcentaje del dato. Cubre lo que el "
              "apilado no ve: calibración, forma de onda, efectos 3D leves.")) / 100
    ci.integrar_gates = cd[2].checkbox("Integrar cada gate en su ventana", value=bool(ci.integrar_gates),
                                       key="chk_int", help="Más fiel al instrumento; desactivado usa el centro.")
    with cd[3].popover("Corrección del archivo", use_container_width=True):
        ci.aplicar_retardo = st.checkbox("Sumar TIME_DELAY a los tiempos", value=bool(ci.aplicar_retardo),
                                         key="chk_ret", help="Solo si los tiempos del USF no lo incluyen ya.")
        ci.aplicar_factor_campo = st.checkbox("Multiplicar por FIELD_SHIFT_FACTOR", value=bool(ci.aplicar_factor_campo),
                                              key="chk_fsf", help="Calibración del fabricante; por omisión 1.")

    st.markdown("**Paso 1 · Multicapa suave (regularización)**")
    st.caption(
        "**Occam**: en cada iteración se prueban varios λ con el modelado directo real y se queda el más grande "
        "(modelo más suave) que cumple el χ² objetivo. Si χ² = 1 no se alcanza, el objetivo es (1 + tolerancia) × "
        "el mínimo alcanzable: una tolerancia mayor da modelos más suaves y con menos contrastes. La resistividad "
        "queda dentro de límites: por omisión el rango de ρa de los gates invertidos ÷/× un factor."
    )
    c1 = st.columns(4)
    ci.n_capas = int(c1[0].number_input("Número de capas", 10, 60, int(ci.n_capas), key="n_capas"))
    ci.prof_max = float(c1[1].number_input("Profundidad máxima (m, 0 = automática)", 0.0, 5000.0,
                                           float(ci.prof_max), 10.0, key="n_pmax",
                                           help="Automática: factor × profundidad de difusión del último gate."))
    ci.chi2_objetivo = float(c1[2].number_input("χ² objetivo", 0.5, 5.0, float(ci.chi2_objetivo), 0.1, key="n_chi2"))
    ci.max_iter = int(c1[3].number_input("Iteraciones máximas", 5, 60, int(ci.max_iter), key="n_iter"))
    c1b = st.columns(4)
    METODOS = {"occam": "Occam (recomendado)", "enfriamiento": "Enfriamiento de λ (versión anterior)"}
    ci.metodo_lambda = c1b[0].selectbox("Selección de λ", list(METODOS), index=list(METODOS).index(ci.metodo_lambda),
                                        format_func=lambda k: METODOS[k], key="sel_metlam")
    ci.occam_tolerancia = float(c1b[1].number_input(
        "Tolerancia de χ² (%)", 0.0, 200.0, float(100 * ci.occam_tolerancia), 5.0, key="n_tol",
        help="Cuánto peor que el mínimo alcanzable se acepta el ajuste a cambio de un modelo más suave.")) / 100
    ci.alfa_cercania = float(c1b[2].number_input(
        "Peso de cercanía a la referencia (α)", 0.0, 1.0, float(ci.alfa_cercania), 0.005, format="%.3f",
        key="n_alfa", help="Atrae las celdas sin control de los datos al semiespacio de ρa tardía mediana. "
                           "0 = solo rugosidad."))
    ci.limites_automaticos = c1b[3].checkbox("Límites de ρ automáticos", value=bool(ci.limites_automaticos),
                                             key="chk_limauto")
    c1c = st.columns(4)
    ci.factor_limites = float(c1c[0].number_input(
        "Factor sobre el rango de ρa", 1.0, 100.0, float(ci.factor_limites), 0.5, key="n_flim",
        disabled=not ci.limites_automaticos, help="3 ≈ media década por debajo del mínimo y por encima del "
                                                  "máximo de la ρa tardía de los gates invertidos."))
    ci.rho_min = float(c1c[1].number_input("ρ mínima (Ω·m)", 0.0, 1e5, float(ci.rho_min), 0.5, key="n_rhomin",
                                           disabled=bool(ci.limites_automaticos)))
    ci.rho_max = float(c1c[2].number_input("ρ máxima (Ω·m)", 0.0, 1e6, float(ci.rho_max), 10.0, key="n_rhomax",
                                           disabled=bool(ci.limites_automaticos)))
    if not ci.limites_automaticos and not (0 < ci.rho_min < ci.rho_max):
        c1c[3].warning("Límites manuales inválidos: se usarán los automáticos.")
    else:
        pass

    st.markdown("**Paso 2 · DOI**")
    c2 = st.columns(3)
    ci.calcular_doi = c2[0].checkbox("Calcular DOI (Oldenburg–Li)", value=bool(ci.calcular_doi), key="chk_doi",
                                     help="Dos inversiones más con referencias ×10 y ÷10; duplica el tiempo.")
    ci.doi_umbral = float(c2[1].number_input("Umbral del índice", 0.05, 0.5, float(ci.doi_umbral), 0.05,
                                             key="n_doiu"))

    st.markdown("**Paso 3 · Modelo de bloques**")
    c3 = st.columns(4)
    ci.k_max = int(c3[0].number_input("Máximo de capas", 2, 10, int(ci.k_max), key="n_kmax"))
    ci.k_fijo = int(c3[1].number_input("Capas fijas (0 = criterio)", 0, 10, int(ci.k_fijo), key="n_kfijo"))
    ci.criterio = c3[2].selectbox("Criterio", ["BIC", "AIC"], index=0 if ci.criterio == "BIC" else 1, key="sel_crit",
                                  help="BIC penaliza más cada capa (modelos más simples); AIC es más permisivo.")
    ci.espesor_min_fraccion = float(c3[3].slider("Espesor mínimo (fracción de la prof. del techo)", 0.0, 1.0,
                                                 float(ci.espesor_min_fraccion), 0.05, key="sl_hmin"))
    c3b = st.columns(4)
    ci.desviacion_previa_bloques = float(c3b[0].number_input(
        "Rigidez de los bloques: ± décadas alrededor del multicapa", 0.0, 3.0, float(ci.desviacion_previa_bloques),
        0.1, key="n_previa",
        help=("Cada bloque se regulariza hacia la media logarítmica del multicapa en su profundidad. Menor = más "
              "rígido (0.5 ≈ factor 3). 0 = sin regularización (solo los límites).")))
    config.inversion = ci
    estado.fijar_config(config)

    b = st.columns([1, 1, 1, 1])
    if b[0].button("🧲 Invertir este sondeo", type="primary", key="btn_inv"):
        resultado, token = estado.ejecutar_con_progreso(
            lambda cb, tk: acc.accion_reprocesar(proyecto, [sondeo], config, "inversion", "parametros", cb, tk),
            "Invirtiendo (≈ 1 min por sondeo con DOI)…")
        estado.registrar_resultado(resultado)
        estado.avisar(resultado, cancelado=token.cancelado)
        st.rerun()
    else:
        pass
    if b[1].button("🧲 Invertir todos", key="btn_inv_todos"):
        resultado, token = estado.ejecutar_con_progreso(
            lambda cb, tk: acc.accion_reprocesar(proyecto, estado.sondeos_disponibles(), config,
                                                 "inversion", "parametros", cb, tk))
        estado.registrar_resultado(resultado)
        estado.avisar(resultado, cancelado=token.cancelado)
        st.rerun()
    else:
        pass
    if b[2].button("⏹️ Detener", key="btn_detener_inv",
                   help="Se detiene en la siguiente iteración; lo completado queda guardado."):
        estado.solicitar_detener()
    else:
        pass

    with st.expander("Modelo de referencia (pozo o sintético) para comparar"):
        archivo_ref = st.file_uploader("Archivo con columnas espesor, rho o z_techo, rho",
                                       type=["txt", "csv", "dat"], key="up_ref")
        cr = st.columns(2)
        if cr[0].button("Guardar referencia de este sondeo", disabled=archivo_ref is None, key="btn_ref"):
            try:
                ref = lec.leer_modelo_referencia(archivo_ref.getvalue())
                alm.guardar_tabla(proyecto, ref, "crudo", "modelo_referencia", sondeo)
                estado.limpiar_cache_datos()
                st.success(f"Modelo de referencia de {len(ref)} capas guardado.")
            except Exception as exc:
                st.error(str(exc))
        else:
            pass
        if cr[1].button("Quitar referencia", key="btn_quitar_ref"):
            alm.eliminar_tabla(proyecto, "crudo", "modelo_referencia", sondeo)
            estado.limpiar_cache_datos()
        else:
            pass

ajuste = estado.cargar_tabla("inversion", "ajuste")
multicapa = estado.cargar_tabla("inversion", "multicapa")
bloques = estado.cargar_tabla("inversion", "bloques")
if ajuste is None or bloques is None:
    st.info("Este sondeo aún no tiene inversión. Revise antes el QC y las exclusiones (pantalla 2).")
    st.stop()
else:
    pass
referencia = estado.cargar_tabla("crudo", "modelo_referencia")
metricas = acc.metricas_sondeo(proyecto, sondeo)

t = st.columns(6)
t[0].metric("Datos invertidos", int(metricas.get("inversion.n_datos", 0)))
t[1].metric("Multicapa χ²", f"{metricas.get('inversion.multicapa_chi2', np.nan):.2f}",
            help=f"RMS {metricas.get('inversion.multicapa_rms', np.nan):.1f} %")
t[2].metric("λ", f"{metricas.get('inversion.multicapa_lambda', np.nan):.3g}",
            help=f"Límites de ρ: {metricas.get('inversion.rho_limite_min', np.nan):.3g}–"
                 f"{metricas.get('inversion.rho_limite_max', np.nan):.3g} Ω·m; celdas en el límite: "
                 f"{int(metricas.get('inversion.celdas_en_limite', 0) or 0)}.")
t[3].metric("Bloques", f"{int(metricas.get('inversion.bloques_n_capas', 0))} capas")
t[4].metric("Bloques χ²", f"{metricas.get('inversion.bloques_chi2', np.nan):.2f}",
            help=f"RMS {metricas.get('inversion.bloques_rms', np.nan):.1f} %")
doi = metricas.get("inversion.doi")
doi = doi if doi is not None and np.isfinite(doi) else None
t[5].metric("DOI", f"{doi:.0f} m" if doi else "—",
            help=f"Profundidad mínima resuelta ≈ {metricas.get('inversion.prof_min_resuelta', np.nan):.0f} m "
                 "(difusión del primer gate invertido).")

proceso_inv = estado.cargar_tabla("inversion", "proceso")
with st.container(border=True):
    st.markdown("**Proceso de inversión aplicado a este sondeo**")
    if proceso_inv is not None and len(proceso_inv):
        for linea in proceso_inv["descripcion"]:
            st.markdown(f"- {linea}")
    else:
        pass

chi2 = metricas.get("inversion.multicapa_chi2", 1.0)
if chi2 is not None and chi2 > 2.0:
    st.warning(
        f"χ² del multicapa = {chi2:.2f}: el modelo 1D no reproduce los datos dentro de su error. Revise en el "
        "gráfico de residuales **qué momento y qué tramo** no ajusta: repuntes tempranos (PI/superparamagnetismo), "
        "tramo plano tardío del HM (ruido o acoplamiento) o desacuerdo LM/HM en el solape. Excluya esos gates en "
        "la pantalla 2, o invierta un solo momento para comparar."
    )
else:
    pass

c = st.columns([1, 1, 2])
vista = c[0].radio("Ajuste en", ["dbdt", "rho_a"], horizontal=True, key="rad_vista",
                   format_func=lambda k: {"dbdt": "dB/dt", "rho_a": "ρa tardía"}[k])
# Profundidad por omisión del gráfico: 1.6 × la última interfaz del modelo
# de bloques o 1.2 × la DOI, la mayor. Se acota al rango del control
# porque un modelo con una interfaz muy profunda (datos que no ven el
# fondo) daría un valor por omisión fuera de rango y Streamlit se detiene.
Z_GRAFICO_MIN, Z_GRAFICO_MAX = 5.0, 1e5
_bases = bloques["z_base"].replace(np.inf, np.nan).dropna()
_z_def = max(1.6 * float(_bases.max()) if len(_bases) else 100.0, doi * 1.2 if doi else 0.0, 50.0)
_z_def = float(np.clip(round(_z_def, 0), Z_GRAFICO_MIN, Z_GRAFICO_MAX)) if np.isfinite(_z_def) else 100.0
z_graf = c[1].number_input("Profundidad del gráfico (m)", Z_GRAFICO_MIN, Z_GRAFICO_MAX, _z_def, key="n_zgraf")
c = st.columns(2)
c[0].plotly_chart(gr.ajuste_inversion(ajuste, vista=vista), use_container_width=True)
c[1].plotly_chart(gr.modelos_resistividad(multicapa, bloques, doi=doi, real=referencia, z_max=z_graf),
                  use_container_width=True)

st.markdown("**Modelo de bloques** — rangos ±1σ de la covarianza linealizada; «resolución» indica qué tan bien "
            "determinan los datos cada capa; «bajo_doi» marca capas cuyo techo está por debajo de la DOI.")
tabla = bloques[["capa", "z_techo", "espesor", "espesor_min", "espesor_max", "rho", "rho_min", "rho_max",
                 "S", "T", "resolucion", "bajo_doi"]].copy()
st.dataframe(estado.tabla_legible(tabla), use_container_width=True, hide_index=True)

c = st.columns(2)
c[0].plotly_chart(gr.proceso_inversion(estado.cargar_tabla("inversion", "iteraciones")), use_container_width=True)
c[1].plotly_chart(gr.seleccion_capas(estado.cargar_tabla("inversion", "seleccion"),
                                     str(metricas.get("inversion.criterio", "BIC"))), use_container_width=True)

# ----------------------------------------------------------------------
# Interpretación
# ----------------------------------------------------------------------
bloques_acum = ps.acumulados(bloques)
mc_acum = ps.acumulados(multicapa)
st.subheader("Interpretación geoeléctrica")
st.plotly_chart(gr.parametros_acumulados(mc_acum, bloques_acum, doi=doi, z_max=z_graf), use_container_width=True)
c = st.columns([1, 1])
with c[0]:
    st.markdown("**Modelo de bloques: S, T y resistividad promedio** (de la superficie a la base de cada capa)")
    st.caption("ρL = H/S (promedio longitudinal), ρT = T/H (transversal), ρ promedio = √(ρL·ρT).")
    tabla_acum = bloques_acum[["capa", "z_techo", "z_base", "rho", "S", "T", "S_acum", "T_acum", "rho_L_prom",
                               "rho_T_prom", "rho_prom", "contraste_vs_arriba", "cambio"]].copy()
    st.dataframe(estado.tabla_legible(tabla_acum), use_container_width=True, hide_index=True)
with c[1]:
    equivalentes = ps.agrupar_equivalentes(multicapa)
    st.markdown(f"**Multicapa agrupado en capas equivalentes** — {len(multicapa)} celdas → "
                f"{len(equivalentes)} capas")
    st.caption(f"Celdas contiguas se unen mientras λ = √(T·S)/H del grupo no pase de {ps.LAMBDA_EQUIVALENTE}.")
    st.dataframe(estado.tabla_legible(equivalentes), use_container_width=True, hide_index=True)

st.subheader("Corrosividad del suelo")
cc = st.columns([1, 1, 2])
prof_corr = float(cc[0].number_input("Profundidad evaluada (m)", 0.5, 100.0, float(config.prof_corrosion), 0.5,
                                     key="n_prof_corr"))
z_corr = float(cc[1].number_input("Profundidad del gráfico (m)", 1.0, 1000.0, max(3.0 * prof_corr, 40.0), 1.0,
                                  key="n_z_corr"))
prof_min = metricas.get("inversion.prof_min_resuelta")
cc[2].caption(
    "Clases por resistividad (NACE/Roberge): < 5 Ω·m muy alta, 5–10 alta, 10–20 moderada, 20–100 baja, "
    "> 100 muy baja. **En TDEM** el primer gate útil rara vez ve los primeros metros"
    + (f" (aquí ≈ {prof_min:.0f} m)" if prof_min and np.isfinite(prof_min) else "")
    + ": por encima, la resistividad es una extrapolación del modelo. Para corrosividad somera, complemente "
      "con SEV o Wenner."
)
st.plotly_chart(gr.corrosividad(bloques, multicapa, z_max=z_corr, umbrales=config.umbrales_corrosion,
                                prof_corrosion=prof_corr), use_container_width=True)

with st.expander("Modelo multicapa completo (con índice DOI, S, T y acumulados)"):
    st.dataframe(estado.tabla_legible(mc_acum.merge(multicapa[["capa", "indice_doi"]], on="capa", how="left")
                                      if "indice_doi" not in mc_acum else mc_acum),
                 use_container_width=True, height=320)
with st.expander("Tabla de ajuste por gate"):
    st.dataframe(ajuste, use_container_width=True, height=320)
with st.expander("📋 Registro de ejecución", expanded=False):
    estado.mostrar_logs()
