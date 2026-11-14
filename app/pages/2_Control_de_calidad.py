"""
Pantalla 2 — Control de calidad
==================================================================
Apilado robusto por gate, revisiones de calidad del sondeo y de la
campaña, curva combinada LM + HM y exclusión manual de gates antes de
invertir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import _ruta_src  # noqa: F401
import acciones_app as acc
import auth_app
import estado_app as estado
import graficos_tdem as gr
import marca_tdem
import pipeline_tdem as pl

st.set_page_config(page_title="TLALLI-RES 1D TDEM — Control de calidad", page_icon="🔎", layout="wide")
sesion = auth_app.exigir_acceso()
auth_app.barra_lateral_sesion(sesion)
marca_tdem.encabezado()
estado.panel_descarga_proyecto(usuario=sesion["usuario"])
estado.inicializar_estado()

st.subheader("🔎 Apilado y control de calidad")
estado.mostrar_aviso()
proyecto = estado.requiere_proyecto()
if proyecto is None:
    st.stop()
else:
    pass

c = st.columns([1, 3])
with c[0]:
    sondeo = estado.selector_sondeo(key="sel_sondeo_p2")
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
cp = config.procesamiento

# ----------------------------------------------------------------------
# Configuración del apilado
# ----------------------------------------------------------------------
with st.container(border=True):
    st.subheader("Apilado y selección de gates")
    st.caption(
        "**Paso 1 · barridos completos.** Isolation Forest y LOF puntúan cada barrido como vector de "
        "desviaciones normalizadas por gate; un barrido es atípico si su puntaje pasa de mediana + k·MAD de "
        "los puntajes del sondeo (umbral adaptativo, no una fracción fija). "
        "**Paso 2 · valores sueltos.** En cada gate, mediana y MAD de **log₁₀ ρa** de los barridos que quedan "
        "y descarte a más de k MAD: el corte es relativo y no depende de la resistividad real. En gates "
        "dominados por ruido (muchos barridos negativos, sin log) el mismo corte se hace en dB/dt lineal. "
        "**Paso 3.** Media lineal de lo que queda y error estándar; la ρa se calcula sobre la curva apilada. "
        "Un gate es **útil** si es positivo, su SNR supera el mínimo y todos los anteriores lo son; un "
        "**repunte** es una subida de más de 3σ (no ocurre en un decaimiento 1D)."
    )
    cb = st.columns(4)
    DETECCION = {"if_lof": "Isolation Forest + LOF", "ninguna": "Desactivada"}
    cp.deteccion_barridos = cb[0].selectbox("Paso 1 · detección por barrido", list(DETECCION),
                                            index=list(DETECCION).index(cp.deteccion_barridos),
                                            format_func=lambda k: DETECCION[k], key="sel_det")
    CONSENSO = {"ambos": "Descartar si coinciden ambos", "cualquiera": "Descartar si lo marca cualquiera"}
    cp.consenso = cb[1].selectbox("Criterio", list(CONSENSO), index=list(CONSENSO).index(cp.consenso),
                                  format_func=lambda k: CONSENSO[k], key="sel_cons",
                                  disabled=cp.deteccion_barridos == "ninguna",
                                  help="«Cualquiera» descarta más barridos; con ruido gaussiano suele quitar "
                                       "cola normal del ruido y subestimar el error.")
    cp.k_puntaje = float(cb[2].number_input("Umbral del puntaje (k·MAD)", 1.0, 10.0, float(cp.k_puntaje), 0.5,
                                            key="n_kpunt", disabled=cp.deteccion_barridos == "ninguna"))
    cp.max_frac_descartados = float(cb[3].number_input("Máximo descartado (%)", 1.0, 50.0,
                                                       float(100 * cp.max_frac_descartados), 1.0, key="n_maxdesc",
                                                       disabled=cp.deteccion_barridos == "ninguna")) / 100
    cc = st.columns(5)
    cp.k_mad = float(cc[0].number_input("Paso 2 · umbral por gate (k·MAD)", 2.0, 20.0, float(cp.k_mad), 0.5,
                                        key="n_kmad"))
    cp.snr_regimen_log = float(cc[1].number_input("SNR por barrido para usar log ρa", 1.0, 50.0,
                                                  float(cp.snr_regimen_log), 0.5, key="n_snrlog"))
    cp.snr_min = float(cc[2].number_input("SNR mínimo del gate apilado", 1.0, 50.0, float(cp.snr_min), 0.5,
                                          key="n_snr"))
    cp.factor_rampa = float(cc[3].number_input("Efecto de rampa: t < f·rampa, f =", 0.5, 10.0,
                                               float(cp.factor_rampa), 0.5, key="n_frampa"))
    cp.excluir_repuntes = cc[4].checkbox("No invertir repuntes", value=bool(cp.excluir_repuntes), key="chk_rep")
    cp.excluir_efecto_rampa = cc[4].checkbox("No invertir gates con efecto de rampa",
                                             value=bool(cp.excluir_efecto_rampa), key="chk_rampa",
                                             help=("La forma de onda del archivo ya se modela en la inversión; "
                                                   "actívelo solo si la rampa declarada es dudosa."))
    config.procesamiento = cp
    estado.fijar_config(config)
    b = st.columns([1, 1, 2])
    if b[0].button("🔁 Rehacer apilado de este sondeo", type="primary", key="btn_apilar"):
        resultado = pl.ejecutar_apilado(proyecto, sondeo, config)
        estado.registrar_resultado(resultado)
        estado.limpiar_cache_datos()
        estado.avisar(resultado)
        st.rerun()
    else:
        pass
    if b[1].button("🔁 Rehacer apilado de todos", key="btn_apilar_todos"):
        resultado, _ = estado.ejecutar_con_progreso(
            lambda cb, tk: acc.accion_reprocesar(proyecto, estado.sondeos_disponibles(), config,
                                                 "apilado", "apilado", cb, tk))
        estado.registrar_resultado(resultado)
        estado.avisar(resultado)
        st.rerun()
    else:
        pass

apilado = estado.cargar_tabla("apilado", "apilado")
if apilado is None:
    st.info("Este sondeo aún no tiene apilado: use el botón de arriba.")
    st.stop()
else:
    pass
gates = estado.cargar_tabla("crudo", "gates")
curva = estado.cargar_tabla("apilado", "curva")
hallazgos = estado.cargar_tabla("apilado", "qc")
resumen = estado.cargar_tabla("apilado", "resumen_qc")
metricas = acc.metricas_sondeo(proyecto, sondeo)

# ----------------------------------------------------------------------
# Valoración
# ----------------------------------------------------------------------
ICONOS = {"Buena": "🟢", "Aceptable": "🟡", "Revisar": "🟠", "Deficiente": "🔴"}
valoracion = str(metricas.get("apilado.valoracion", "—"))
t = st.columns(6)
t[0].metric("Valoración", f"{ICONOS.get(valoracion, '')} {valoracion}")
t[5].metric("Barridos descartados", int(metricas.get("apilado.n_barridos_descartados", 0)),
            help=f"{100 * float(metricas.get('apilado.frac_barridos_descartados', 0) or 0):.1f} % por IF + LOF (paso 1).")
t[1].metric("Puntaje", f"{metricas.get('apilado.puntaje', np.nan):.0f} / 100",
            help="100 − 25 por cada falla − 8 por cada aviso (el peor de los momentos).")
t[2].metric("Gates para invertir", int(metricas.get("apilado.n_para_invertir", 0)))
t[3].metric("Excluidos a mano", int(metricas.get("apilado.n_excluidos_usuario", 0)))
t_cambio = metricas.get("apilado.t_cambio")
t[4].metric("Cambio LM → HM", f"{t_cambio * 1e6:.0f} µs" if t_cambio and np.isfinite(t_cambio) else "—",
            help="Desde este tiempo la curva combinada usa el HM (su error relativo ya es menor que el del LM).")

st.plotly_chart(gr.curvas_decaimiento(gates, apilado), use_container_width=True)
puntajes = estado.cargar_tabla("apilado", "barridos_qc")
with st.expander("Paso 1 · anomalía por barrido (Isolation Forest y LOF)", expanded=False):
    st.caption("Barridos descartados agrupados en el orden de adquisición indican ruido en ráfagas (tráfico, "
               "maquinaria, tormenta); puntos aislados, picos sueltos.")
    cpz = st.columns(2)
    cpz[0].plotly_chart(gr.puntajes_barridos(puntajes, "LM"), use_container_width=True)
    cpz[1].plotly_chart(gr.puntajes_barridos(puntajes, "HM"), use_container_width=True)
st.plotly_chart(gr.curva_combinada(curva, t_cambio), use_container_width=True)
st.caption("La curva combinada es para ver e interpretar (ρa, pseudosección). La inversión usa **los dos "
           "momentos por separado**, cada uno con su forma de onda y sus ventanas: en el solape aportan dos "
           "mediciones independientes del mismo subsuelo.")

# ----------------------------------------------------------------------
# Hallazgos
# ----------------------------------------------------------------------
with st.container(border=True):
    st.subheader("Hallazgos del control de calidad")
    if resumen is not None and len(resumen):
        columnas = [c for c in ("momento", "n_stacks", "corriente_media_a", "n_gates", "n_utiles", "n_para_invertir",
                                "t_primero_util", "t_ultimo_util", "snr_mediano", "piso_ruido", "rho_a_mediana",
                                "n_aviso", "n_falla", "puntaje", "valoracion") if c in resumen]
        st.dataframe(estado.tabla_legible(resumen[columnas]), use_container_width=True, hide_index=True)
    else:
        pass
    if hallazgos is not None and len(hallazgos):
        severidades = ["FAIL", "WARN", "INFO", "OK"]
        elegidas = st.multiselect("Severidad", severidades, default=["FAIL", "WARN"], key="ms_sev")
        vista = hallazgos[hallazgos["severidad"].isin(elegidas)].copy()
        vista["orden"] = vista["severidad"].map({s: i for i, s in enumerate(severidades)})
        vista = vista.sort_values(["orden", "categoria", "revision"]).drop(columns=["orden", "sondeo"])
        st.dataframe(vista, use_container_width=True, hide_index=True,
                     column_config={"mensaje": st.column_config.TextColumn("Mensaje", width="large")})
    else:
        pass

# ----------------------------------------------------------------------
# Exclusión manual de gates
# ----------------------------------------------------------------------
with st.container(border=True):
    st.subheader("Gates que entran a la inversión")
    st.caption(
        "Marque **excluir** en los gates que no quiere invertir (repuntes que no se marcaron solos, tramos "
        "planos al final del HM por ruido o acoplamiento, el primer gate de un momento contaminado por la "
        "rampa…). Al guardar se rehace el apilado de este sondeo; después vuelva a invertirlo (pantalla 3)."
    )
    excl = pl.exclusiones_sondeo(proyecto, sondeo)
    columnas_editor = [c for c in ("momento", "gate", "tiempo", "dbdt", "error_rel_se", "snr", "rho_a",
                                   "sigma_log10_rho", "regimen", "n_atipicos", "util", "repunte", "efecto_rampa",
                                   "usar") if c in apilado]
    editor = apilado[columnas_editor].copy()
    editor["excluir"] = [(m, int(g)) in set(zip(excl["momento"], excl["gate"].astype(int)))
                         for m, g in zip(editor["momento"], editor["gate"])]
    momento = st.radio("Momento", ["LM", "HM"], horizontal=True, key="rad_mom_excl")
    vista = editor[editor["momento"] == momento].reset_index(drop=True)
    editada = st.data_editor(
        vista, hide_index=True, use_container_width=True, height=360, key=f"ed_excl_{momento}",
        disabled=[c for c in vista.columns if c != "excluir"],
        column_config={
            "tiempo": st.column_config.NumberColumn("t (s)", format="%.3e"),
            "dbdt": st.column_config.NumberColumn("dB/dt", format="%.3e"),
            "error_rel_se": st.column_config.NumberColumn("error rel.", format="%.3f"),
            "snr": st.column_config.NumberColumn("SNR", format="%.0f"),
            "rho_a": st.column_config.NumberColumn("ρa (Ω·m)", format="%.3g"),
            "sigma_log10_rho": st.column_config.NumberColumn("σ log₁₀ρa", format="%.4f",
                                                             help="Incertidumbre de la ρa apilada en log₁₀."),
            "regimen": st.column_config.TextColumn("MAD en", help="log = log₁₀ ρa; lineal = dB/dt (gate ruidoso)"),
            "n_atipicos": st.column_config.NumberColumn("atíp. MAD"),
            "usar": st.column_config.CheckboxColumn("se invierte"),
            "excluir": st.column_config.CheckboxColumn("excluir"),
        })
    if st.button("💾 Guardar exclusiones y rehacer apilado", key="btn_excl"):
        otro = editor[editor["momento"] != momento][["momento", "gate", "excluir"]]
        tabla = pd.concat([otro, editada[["momento", "gate", "excluir"]]], ignore_index=True)
        resultado = acc.accion_guardar_exclusiones(proyecto, sondeo, tabla, config)
        estado.registrar_resultado(resultado)
        estado.limpiar_cache_datos()
        estado.avisar(resultado)
        st.rerun()
    else:
        pass

# ----------------------------------------------------------------------
# Campaña
# ----------------------------------------------------------------------
with st.container(border=True):
    st.subheader("Revisiones de campaña")
    st.caption("Comparan los sondeos entre sí: ventanas y parámetros de adquisición iguales, distancia entre "
               "centros (repeticiones), piso de ruido y nivel de señal.")
    if st.button("🔁 Actualizar revisiones de campaña", key="btn_qc_campana"):
        resultado = acc.accion_qc_campana(proyecto)
        estado.registrar_resultado(resultado)
        estado.limpiar_cache_datos()
        estado.avisar(resultado)
        st.rerun()
    else:
        pass
    import almacenamiento_tdem as alm

    campana = estado.cargar_tabla("parametros", "qc_campana", sondeo=alm.CAMPANA)
    if campana is not None and len(campana):
        st.dataframe(campana.drop(columns=["categoria"], errors="ignore"), use_container_width=True, hide_index=True,
                     column_config={"mensaje": st.column_config.TextColumn("Mensaje", width="large")})
    else:
        st.caption("Sin revisiones de campaña (hacen falta al menos dos sondeos).")

with st.expander("Tabla completa del apilado"):
    st.dataframe(apilado, use_container_width=True, height=320)
with st.expander("📋 Registro de ejecución", expanded=False):
    estado.mostrar_logs()
