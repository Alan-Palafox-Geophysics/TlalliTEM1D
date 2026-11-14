"""
graficos_tdem.py
==================================================================
Figuras Plotly de la aplicación TDEM.

Convenciones visuales
------------------------
- Curvas de decaimiento en **log-log** (dB/dt contra tiempo), LM en azul
  y HM en naranja en todas las figuras.
- Modelos como **escalones** de resistividad contra profundidad, con el
  eje de profundidad hacia abajo y la DOI marcada. Multicapa: naranja;
  modelo de bloques (el invertido): azul (0, 0, 255), línea gruesa.
- Mapa con el **lazo transmisor dibujado a escala** alrededor de cada
  centro de sondeo.
- Secciones con la escala oficial del producto (``generar_escala_tre``).

Las funciones de modelos, parámetros acumulados, corrosividad, secciones
y unidades vienen de TlalliRes1D (SEV) sin cambios de fondo: trabajan
sobre el mismo esquema de modelo (``capa, z_techo, z_base, espesor, rho``).

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from generar_escala_tre import generar_escala_tre

PLANTILLA = "plotly_white"

#: Colores por concepto.
COLOR = {
    "observado": "#8A8F98",
    "preprocesado": "#00838F",
    "referencia": "#5E6A75",
    "anomalo": "#D62828",
    "multicapa": "#F28C28",      # naranja, grosor normal
    "bloques": "#0000FF",   # modelo invertido: azul (0, 0, 255), fuerte y claro
    "LM": "#2A78D6",
    "HM": "#EB6834",
    "excluido": "#B0B6BD",
    "real": "#111111",
    "doi": "#7B2CBF",
}
PALETA_DIRECCIONES = ["#00838F", "#E07A1F", "#3A7D44", "#9C2C77", "#4C6EF5", "#7A5C3A"]


def _escala_plotly(invertir: bool = False, n: int = 32) -> list:
    cmap = generar_escala_tre()
    posiciones = np.linspace(0, 1, n)
    muestras = posiciones[::-1] if invertir else posiciones
    salida = []
    for p, m in zip(posiciones, muestras):
        r, g, b, _ = cmap(float(m))
        salida.append([float(p), f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"])
    return salida


ESCALA_RESISTIVIDAD = _escala_plotly(False)


def _vacia(mensaje: str, altura: int = 300) -> go.Figure:
    figura = go.Figure()
    figura.add_annotation(text=mensaje, xref="paper", yref="paper", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=13, color="#888"))
    figura.update_layout(template=PLANTILLA, height=altura, xaxis=dict(visible=False),
                         yaxis=dict(visible=False))
    return figura


def marcas_log(valores) -> dict:
    """Marcas 1-2-5 para un eje logarítmico, rotuladas en valor real.

    Plotly, en ejes log que abarcan pocas décadas, rotula las marcas
    menores con la mantisa sola ("2, 3, 5") y se lee mal; aquí se
    eligen marcas redondas de la serie 1-2-5 y se rotulan completas.
    """
    v = np.asarray(valores, float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) == 0:
        return {}
    else:
        pass
    bajo, alto = np.log10(v.min()), np.log10(v.max())
    marcas = []
    for decada in range(int(np.floor(bajo)) - 1, int(np.ceil(alto)) + 2):
        for mantisa in (1, 2, 5):
            valor = mantisa * 10.0 ** decada
            if 10 ** (bajo - 0.15) <= valor <= 10 ** (alto + 0.15):
                marcas.append(valor)
            else:
                pass
    if len(marcas) > 12:
        marcas = [m for m in marcas if str(m).lstrip("0.").startswith("1")] or marcas
    else:
        pass
    texto = [f"{m:g}" for m in marcas]
    return {"tickmode": "array", "tickvals": marcas, "ticktext": texto}


def _ejes_loglog(figura: go.Figure, fila: Optional[int] = None, col: Optional[int] = None,
                 titulo_y: str = "ρa (Ω·m)", titulo_x: str = "AB/2 (m)",
                 valores_x=None, valores_y=None) -> None:
    kwargs = {} if fila is None else {"row": fila, "col": col}
    figura.update_xaxes(type="log", title=titulo_x, showgrid=True, gridcolor="#E6E6E6",
                        **(marcas_log(valores_x) if valores_x is not None else {}), **kwargs)
    figura.update_yaxes(type="log", title=titulo_y, showgrid=True, gridcolor="#E6E6E6",
                        **(marcas_log(valores_y) if valores_y is not None else {}), **kwargs)


def _barras_error(valores: np.ndarray, error_rel: np.ndarray) -> dict:
    """Barras de error multiplicativas (simétricas en log)."""
    v = np.asarray(valores, float)
    e = np.asarray(error_rel, float)
    return dict(type="data", symmetric=False, array=v * (np.exp(e) - 1),
                arrayminus=v * (1 - np.exp(-e)), thickness=1, width=2, color="#B0B6BD")


COLORES_CORROSIVIDAD = {"Muy alta": "#B2182B", "Alta": "#EF6548", "Moderada": "#FDB863",
                        "Baja": "#A6D96A", "Muy baja": "#1A9850"}


def _escalones(modelo: pd.DataFrame, columna: str, z_fondo: float) -> tuple[np.ndarray, np.ndarray]:
    """Coordenadas (valor, profundidad) de un modelo en escalones."""
    techos = modelo["z_techo"].to_numpy(float)
    bases = np.where(np.isfinite(modelo["z_base"]), modelo["z_base"], z_fondo).astype(float)
    valores = modelo[columna].to_numpy(float)
    x, z = [], []
    for t, b, v in zip(techos, bases, valores):
        x += [v, v]
        z += [t, min(b, z_fondo)]
    return np.array(x), np.array(z)


def modelos_resistividad(
    multicapa: Optional[pd.DataFrame], bloques: Optional[pd.DataFrame],
    doi: Optional[float] = None, real: Optional[pd.DataFrame] = None,
    z_max: Optional[float] = None, titulo: str = "Modelos de resistividad",
) -> go.Figure:
    """Multicapa y bloques en escalones (con incertidumbre y DOI)."""
    if multicapa is None and bloques is None:
        return _vacia("Sin modelos")
    else:
        pass
    if z_max is None:
        candidatos = [doi * 1.3 if doi else np.nan]
        if bloques is not None:
            finitas = bloques["z_base"].replace(np.inf, np.nan).dropna()
            candidatos.append(float(finitas.max()) * 1.6 if len(finitas) else np.nan)
        else:
            pass
        z_max = float(np.nanmax(candidatos)) if np.isfinite(np.nanmax(candidatos)) else 100.0
    else:
        pass

    figura = go.Figure()
    barras = None
    if bloques is not None and "rho_min" in bloques:
        # Incertidumbre ±1σ de ρ como barra horizontal en el centro de cada
        # capa (acotada a ×/÷10 para no deformar el eje cuando una capa está
        # en equivalencia y su ρ por separado no está determinada).
        techos = bloques["z_techo"].to_numpy(float)
        bases = np.where(np.isfinite(bloques["z_base"]), bloques["z_base"], z_max).astype(float)
        medios = np.sqrt(np.maximum(techos, 0.05 * bases) * np.minimum(bases, z_max))
        rho_s = bloques["rho"].to_numpy(float)
        arriba = np.minimum(bloques["rho_max"].to_numpy(float), rho_s * 10) - rho_s
        abajo = rho_s - np.maximum(bloques["rho_min"].to_numpy(float), rho_s / 10)
        visibles = techos < z_max
        barras = go.Scatter(
            x=rho_s[visibles], y=medios[visibles], mode="markers", name="Bloques ±1σ (ρ)",
            marker=dict(color=COLOR["bloques"], size=5),
            error_x=dict(type="data", symmetric=False, array=arriba[visibles], arrayminus=abajo[visibles],
                         color="rgba(0,0,255,0.45)", thickness=1.5, width=4),
            hovertemplate="ρ=%{x:.3g} Ω·m<extra>±1σ</extra>",
        )
    else:
        pass
    if real is not None:
        x, z = _escalones(real, "rho", z_max)
        figura.add_trace(go.Scatter(x=x, y=z, mode="lines", name="Modelo de referencia",
                                    line=dict(color="rgba(17,17,17,0.35)", width=5)))
    else:
        pass
    if multicapa is not None:
        x, z = _escalones(multicapa, "rho", z_max)
        figura.add_trace(go.Scatter(x=x, y=z, mode="lines", name="Multicapa",
                                    line=dict(color=COLOR["multicapa"], width=2)))
    else:
        pass
    if bloques is not None:
        x, z = _escalones(bloques, "rho", z_max)
        texto = [f"Capa {c}: {r:.3g} Ω·m · {h:.3g} m · {res}" for c, r, h, res in zip(
            bloques["capa"], bloques["rho"], bloques["espesor"],
            bloques.get("resolucion", [""] * len(bloques)))]
        figura.add_trace(go.Scatter(x=x, y=z, mode="lines", name="Invertido (bloques)",
                                    line=dict(color=COLOR["bloques"], width=3),
                                    text=np.repeat(texto, 2), hovertemplate="%{text}<br>z=%{y:.3g} m<extra></extra>"))
    else:
        pass
    valores_eje = []
    for tr in figura.data:
        if tr.x is not None:
            valores_eje += [v for v in tr.x if v is not None]
        else:
            pass
    if barras is not None:
        figura.add_trace(barras)
    else:
        pass
    if doi and doi <= z_max:
        figura.add_hline(y=doi, line=dict(color=COLOR["doi"], dash="dash", width=1.5),
                         annotation_text=f"DOI ≈ {doi:.1f} m", annotation_position="bottom right")
        figura.add_hrect(y0=doi, y1=z_max, fillcolor="rgba(120,120,120,0.08)", line_width=0)
    else:
        pass
    figura.update_xaxes(type="log", title="Resistividad (Ω·m)", showgrid=True, gridcolor="#E6E6E6",
                        **marcas_log(valores_eje))
    figura.update_yaxes(title="Profundidad (m)", range=[z_max, 0], showgrid=True, gridcolor="#E6E6E6")
    figura.update_layout(template=PLANTILLA, height=600, title=titulo,
                         legend=dict(orientation="h", y=-0.12), margin=dict(l=60, r=20, t=50, b=40))
    return figura


def parametros_acumulados(
    multicapa: Optional[pd.DataFrame], bloques: Optional[pd.DataFrame],
    doi: Optional[float] = None, real: Optional[pd.DataFrame] = None, z_max: float = 100.0,
) -> go.Figure:
    """Parámetros de Dar Zarrouk acumulados desde la superficie hasta z.

    1. T(z) = ∫₀ᶻ ρ dz: pendiente = ρ (tramo empinado → capa resistiva).
    2. S(z) = ∫₀ᶻ dz/ρ: pendiente = 1/ρ (tramo empinado → capa conductora).
    3. ρm(z) = √(ρL·ρT) = √(T/S), con ρL = z/S y ρT = T/z: resistividad
       promedio del paquete 0–z. Es un promedio **corrido**: sube mientras
       se atraviesa una capa más resistiva que el promedio de lo de arriba
       y baja en una más conductora; sus picos y valles marcan las
       interfaces donde cambia esa tendencia.
    4. λ(z) = √(ρT/ρL) = √(T·S)/z: pseudo-anisotropía del paquete 0–z
       (1 = homogéneo). Crece con cualquier contraste, sin distinguir si la
       resistividad sube o baja.
    5. Contraste de cada capa del modelo de bloques: ρ_capa / ρm del paquete que
       tiene encima, en escala log₂ centrada en 1: a la derecha más
       resistiva, a la izquierda más conductora; ≥ ×2 o ≤ ×0.5 = cambio
       fuerte.
    """
    import parametros_tdem as ps

    if multicapa is None and bloques is None:
        return _vacia("Sin modelos")
    else:
        pass
    z = np.linspace(0.0, float(z_max), 400)[1:]
    figura = make_subplots(rows=1, cols=6, shared_yaxes=True, horizontal_spacing=0.025,
                           column_widths=[1.1, 1.0, 1.0, 1.0, 0.6, 0.7],
                           subplot_titles=("Modelos de inversión", "T acumulada", "S acumulada", "ρ promedio 0–z",
                                           "Pseudo-anisotropía λ", "Contraste por capa"))
    series = [(real, "Modelo de referencia", dict(color="rgba(17,17,17,0.35)", width=5)),
              (multicapa, "Multicapa", dict(color=COLOR["multicapa"], width=2)),
              (bloques, "Invertido (bloques)", dict(color=COLOR["bloques"], width=3))]
    # Panel 1: los modelos de inversión (referencia, multicapa, bloques).
    for modelo, nombre, linea in series:
        if modelo is None or not len(modelo):
            continue
        else:
            pass
        x, zz = _escalones(modelo, "rho", z_max)
        figura.add_trace(go.Scatter(x=x, y=zz, mode="lines", name=nombre, line=linea, legendgroup=nombre,
                                    hovertemplate="z=%{y:.1f} m<br>ρ=%{x:.3g} Ω·m<extra>" + nombre + "</extra>"),
                         row=1, col=1)
    ejes = {1: [], 2: [], 3: [], 4: []}
    for modelo, nombre, linea in series:
        if modelo is None or not len(modelo):
            continue
        else:
            pass
        T = ps.curva_acumulada(modelo, z, "T")
        S = ps.curva_acumulada(modelo, z, "S")
        rm = np.sqrt(T / S)
        lam = np.sqrt(T * S) / z
        for col, valores, unidad in ((1, T, "Ω·m²"), (2, S, "S"), (3, rm, "Ω·m"), (4, lam, "")):
            ejes[col].append(valores)
            figura.add_trace(go.Scatter(
                x=valores, y=z, mode="lines", name=nombre, line=linea, legendgroup=nombre, showlegend=False,
                hovertemplate="z=%{y:.1f} m<br>%{x:.4g} " + unidad + "<extra>" + nombre + "</extra>",
            ), row=1, col=col + 1)
    # Contraste por capa (bloques; el de referencia como contorno).
    for modelo, nombre, relleno in ((real, "Referencia", False), (bloques, "Bloques", True)):
        if modelo is None or not len(modelo):
            continue
        else:
            pass
        m = ps.acumulados(modelo)
        m = m[np.isfinite(m["contraste_vs_arriba"]) & (m["z_techo"] < z_max)]
        if not len(m):
            continue
        else:
            pass
        techos = m["z_techo"].to_numpy(float)
        bases = np.minimum(np.where(np.isfinite(m["z_base"]), m["z_base"], z_max), z_max)
        c = m["contraste_vs_arriba"].to_numpy(float)
        x = np.log2(c)
        colores = np.where(c >= 1.0, COLOR["multicapa"], COLOR["bloques"])
        figura.add_trace(go.Bar(
            x=x, y=(techos + bases) / 2, width=(bases - techos) * 0.92, orientation="h", name=f"Contraste · {nombre}",
            marker=dict(color=colores if relleno else "rgba(0,0,0,0)",
                        line=dict(color="rgba(17,17,17,0.55)" if not relleno else colores, width=2 if not relleno else 0)),
            customdata=np.stack([c, techos, bases], axis=-1), showlegend=False,
            hovertemplate=("capa %{customdata[1]:.1f}–%{customdata[2]:.1f} m<br>ρ capa / ρ promedio de arriba = "
                           "×%{customdata[0]:.2f}<extra>" + nombre + "</extra>"),
        ), row=1, col=6)
    for umbral in (-1.0, 1.0):
        figura.add_vline(x=umbral, line=dict(color="#999", dash="dot", width=1), row=1, col=6)
    figura.add_vline(x=0.0, line=dict(color="#666", width=1), row=1, col=6)
    figura.update_xaxes(title="ρ capa / ρ prom. arriba", tickvals=[-3, -2, -1, 0, 1, 2, 3],
                        ticktext=["×⅛", "×¼", "×½", "×1", "×2", "×4", "×8"], range=[-3.6, 3.6],
                        showgrid=True, gridcolor="#E6E6E6", row=1, col=6)
    valores_rho = [m["rho"].to_numpy(float) for m in (real, multicapa, bloques)
                   if m is not None and len(m)]
    vals_rho = np.concatenate(valores_rho) if valores_rho else np.array([1.0, 100.0])
    figura.update_xaxes(type="log", title="ρ (Ω·m)", showgrid=True, gridcolor="#E6E6E6", row=1, col=1,
                        **marcas_log(vals_rho[np.isfinite(vals_rho) & (vals_rho > 0)]))
    for col, titulo in ((1, "T (Ω·m²)"), (2, "S (siemens)"), (3, "ρ promedio (Ω·m)")):
        vals = np.concatenate(ejes[col]) if ejes[col] else np.array([1.0, 10.0])
        vals = vals[np.isfinite(vals) & (vals > 0)]
        figura.update_xaxes(type="log", title=titulo, showgrid=True, gridcolor="#E6E6E6", row=1, col=col + 1,
                            **marcas_log(vals))
    lam_max = float(np.nanmax(np.concatenate(ejes[4]))) if ejes[4] else 1.5
    figura.update_xaxes(title="λ = √(ρT/ρL)", range=[0.98, max(1.1, lam_max * 1.05)], showgrid=True,
                        gridcolor="#E6E6E6", row=1, col=5)
    for col in range(1, 7):
        if doi and doi <= z_max:
            figura.add_hline(y=doi, line=dict(color=COLOR["doi"], dash="dash", width=1.2), row=1, col=col)
        else:
            pass
        figura.update_yaxes(range=[z_max, 0], showgrid=True, gridcolor="#E6E6E6", row=1, col=col)
    figura.update_yaxes(title="Profundidad (m)", row=1, col=1)
    figura.update_layout(template=PLANTILLA, height=600,
                         title="Interpretación geoeléctrica: modelos y parámetros de Dar Zarrouk", bargap=0,
                         legend=dict(orientation="h", y=-0.15), margin=dict(l=60, r=20, t=70, b=40))
    return figura


COLORES_CORROSIVIDAD = {"Muy alta": "#B2182B", "Alta": "#EF6548", "Moderada": "#FDB863",
                        "Baja": "#A6D96A", "Muy baja": "#1A9850"}


def corrosividad(
    bloques: Optional[pd.DataFrame], multicapa: Optional[pd.DataFrame] = None, z_max: float = 20.0,
    umbrales=(5.0, 10.0, 20.0, 100.0), prof_corrosion: float = 3.0,
) -> go.Figure:
    """Corrosividad del suelo estimada con el modelo invertido.

    1. Modelo de resistividad sobre las franjas de clase (ρ < 5: muy alta,
       5–10: alta, 10–20: moderada, 20–100: baja, > 100 Ω·m: muy baja; escala
       de protección catódica NACE/Roberge).
    2. Clase de cada capa del modelo invertido.
    3. Clase del suelo de 0 a z con la resistividad longitudinal ρL(z) =
       z / S(z), la que ve una corriente de corrosión (dominada por las
       capas conductoras); se marca la profundidad de evaluación.
    """
    import parametros_tdem as ps

    if bloques is None or not len(bloques):
        return _vacia("Sin modelo invertido")
    else:
        pass
    clases = list(COLORES_CORROSIVIDAD)
    limites = [1e-3] + list(umbrales) + [1e6]
    figura = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.05,
                           column_widths=[1.2, 0.6, 1.2],
                           subplot_titles=("Modelo invertido y clases de corrosividad", "Clase por capa",
                                           "Suelo de 0 a z (ρL acumulada)"))
    for col in (1, 3):
        for clase, x0, x1 in zip(clases, limites[:-1], limites[1:]):
            # plotly 6+ (plotly.js 3): las figuras en ejes log usan valores de datos.
            figura.add_vrect(x0=x0, x1=x1, fillcolor=COLORES_CORROSIVIDAD[clase],
                             opacity=0.18, line_width=0, layer="below", row=1, col=col,
                             exclude_empty_subplots=False)
    if multicapa is not None and len(multicapa):
        x, zz = _escalones(multicapa, "rho", z_max)
        figura.add_trace(go.Scatter(x=x, y=zz, mode="lines", name="Multicapa",
                                    line=dict(color=COLOR["multicapa"], width=2)), row=1, col=1)
    else:
        pass
    x, zz = _escalones(bloques, "rho", z_max)
    figura.add_trace(go.Scatter(x=x, y=zz, mode="lines", name="Invertido (bloques)",
                                line=dict(color=COLOR["bloques"], width=3)), row=1, col=1)
    # Clase por capa
    for fila in bloques.itertuples():
        if fila.z_techo >= z_max:
            continue
        else:
            pass
        clase = ps.clase_corrosividad(float(fila.rho), umbrales)
        base = min(float(fila.z_base) if np.isfinite(fila.z_base) else z_max, z_max)
        figura.add_trace(go.Bar(
            x=[1.0], y=[(fila.z_techo + base) / 2], width=[(base - fila.z_techo) * 0.96], orientation="h",
            marker=dict(color=COLORES_CORROSIVIDAD.get(clase, "#CCC"), line=dict(color="#FFFFFF", width=1)),
            text=[f"{clase}<br>{fila.rho:.3g} Ω·m"], textposition="inside", insidetextanchor="middle",
            showlegend=False, hovertemplate=f"{fila.z_techo:.2f}–{base:.2f} m · {clase}<extra></extra>",
        ), row=1, col=2)
    # ρL acumulada
    z = np.linspace(0.0, float(z_max), 300)[1:]
    S = ps.curva_acumulada(bloques, z, "S")
    rho_l = z / S
    figura.add_trace(go.Scatter(x=rho_l, y=z, mode="lines", name="ρL de 0 a z",
                                line=dict(color="#111111", width=2.5)), row=1, col=3)
    if prof_corrosion and prof_corrosion <= z_max:
        rl = float(prof_corrosion / ps.curva_acumulada(bloques, [prof_corrosion], "S")[0])
        clase = ps.clase_corrosividad(rl, umbrales)
        figura.add_trace(go.Scatter(x=[rl], y=[prof_corrosion], mode="markers+text", name="Evaluación",
                                    marker=dict(size=11, color="#111111", symbol="diamond"),
                                    text=[f"  0–{prof_corrosion:g} m: {clase} ({rl:.3g} Ω·m)"],
                                    textposition="middle right", showlegend=False), row=1, col=3)
    else:
        pass
    todos = np.concatenate([bloques["rho"].to_numpy(float), rho_l])
    todos = todos[np.isfinite(todos) & (todos > 0)]
    rango = [np.log10(max(min(todos.min() / 2, 2.0), 1e-2)), np.log10(max(todos.max() * 2, 200.0))]
    for col in (1, 3):
        figura.update_xaxes(type="log", range=rango, title="ρ (Ω·m)", showgrid=True, gridcolor="#E6E6E6",
                            row=1, col=col)
    figura.update_xaxes(visible=False, range=[0, 1], row=1, col=2)
    for col in (1, 2, 3):
        figura.update_yaxes(range=[z_max, 0], showgrid=True, gridcolor="#E6E6E6", row=1, col=col)
    figura.update_yaxes(title="Profundidad (m)", row=1, col=1)
    for clase in clases:
        figura.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"Corrosividad {clase.lower()}",
                                    marker=dict(size=12, symbol="square", color=COLORES_CORROSIVIDAD[clase])),
                         row=1, col=1)
    figura.update_layout(template=PLANTILLA, height=520, bargap=0,
                         title="Corrosividad del suelo estimada con el modelo invertido",
                         legend=dict(orientation="h", y=-0.15), margin=dict(l=60, r=20, t=70, b=40))
    return figura


def seccion(sec: dict[str, Any], titulo: str, escala=None, invertir: bool = False,
            unidad: str = "Ω·m", mostrar_doi: bool = True, rango: Optional[tuple[float, float]] = None) -> go.Figure:
    """Sección 2D (salida de ``perfil_tdem.seccion_interpolada``).

    ``rango`` fija los límites de color (en log10 si la sección es
    logarítmica) para comparar varias secciones con la misma escala.
    """
    valores = sec["valores"]
    usar_log = sec.get("usar_log", True)
    datos = valores
    if escala is not None:
        datos = escala.recortar(valores)
        colorscale = escala.colorscale()
        zmin, zmax = escala.rango_dibujo
        mostrar_barra = False
    else:
        colorscale = _escala_plotly(True) if invertir else ESCALA_RESISTIVIDAD
        finitos = valores[np.isfinite(valores)]
        if rango is not None:
            zmin, zmax = rango
        else:
            zmin, zmax = (np.percentile(finitos, [1.5, 98.5]) if len(finitos) else (0, 1))
        mostrar_barra = True
    texto = np.where(np.isfinite(valores), 10 ** valores if usar_log else valores, np.nan)
    figura = go.Figure(go.Heatmap(
        x=sec["s"], y=sec["elev"], z=datos, colorscale=colorscale, zmin=zmin, zmax=zmax,
        showscale=mostrar_barra, customdata=texto, zsmooth="best",
        colorbar=dict(title=f"log₁₀ {unidad}" if usar_log else unidad),
        hovertemplate="s=%{x:.1f} m<br>cota=%{y:.1f} m<br>valor=%{customdata:.4g} " + unidad + "<extra></extra>",
    ))
    if escala is not None:
        figura.add_trace(escala.traza_barra(unidad))
    else:
        pass
    figura.add_trace(go.Scatter(x=sec["s"], y=sec["topo"], mode="lines", name="Terreno",
                                line=dict(color="#333", width=2), showlegend=False))
    s_sondeos = sec["sondeos"]["s"].to_numpy(float)
    z_sondeos = sec["sondeos"]["Z"].to_numpy(float)
    figura.add_trace(go.Scatter(x=s_sondeos, y=z_sondeos, mode="markers+text", text=sec["sondeos"]["Sondeo"],
                                textposition="top center", marker=dict(symbol="triangle-down", size=12, color="#111"),
                                showlegend=False, hoverinfo="text"))
    for s_i, z_i in zip(s_sondeos, z_sondeos):
        figura.add_trace(go.Scatter(x=[s_i, s_i], y=[z_i, z_i - sec["z_max"]], mode="lines",
                                    line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dot"),
                                    showlegend=False, hoverinfo="skip"))
    if mostrar_doi and sec.get("doi") is not None:
        # Solo el tramo de la DOI que cae dentro de la sección dibujada.
        doi = np.where(np.asarray(sec["doi"], float) <= sec["z_max"], sec["doi"], np.nan)
        figura.add_trace(go.Scatter(x=sec["s"], y=sec["topo"] - doi, mode="lines", name="DOI",
                                    line=dict(color="#FFFFFF", width=2, dash="dash"), showlegend=False,
                                    hovertemplate="DOI<extra></extra>"))
    else:
        pass
    figura.update_yaxes(range=[float(np.min(sec["topo"])) - sec["z_max"],
                               float(np.max(sec["topo"])) + 0.06 * sec["z_max"]])
    figura.update_xaxes(title="Distancia sobre la ruta (m)")
    figura.update_yaxes(title="Cota (m)" if np.ptp(sec["topo"]) > 0 or sec["topo"][0] != 0 else "Profundidad (m)")
    figura.update_layout(template=PLANTILLA, height=460, title=titulo,
                         margin=dict(l=60, r=20, t=50, b=40), legend=dict(orientation="h", y=-0.2))
    return figura


def columnas_unidades(unidades: pd.DataFrame, orden: Optional[list[str]] = None,
                      z_max: Optional[float] = None) -> go.Figure:
    """Columnas estratigráficas por sondeo coloreadas por unidad."""
    if unidades is None or len(unidades) == 0:
        return _vacia("Sin unidades geoeléctricas")
    else:
        pass
    nombres = orden or sorted(unidades["Sondeo"].astype(str).unique())
    finitas = unidades["z_base"].replace(np.inf, np.nan)
    z_max = z_max or float(np.nanmax(finitas) * 1.4) if finitas.notna().any() else 50.0
    etiquetas = sorted(unidades["unidad"].unique(), key=lambda u: int(str(u)[1:]))
    paleta = [ESCALA_RESISTIVIDAD[int(i)][1] for i in np.linspace(2, len(ESCALA_RESISTIVIDAD) - 3, max(len(etiquetas), 2))]
    color_de = {u: paleta[i] for i, u in enumerate(etiquetas)}
    figura = go.Figure()
    vistos = set()
    for i, nombre in enumerate(nombres):
        capas = unidades[unidades["Sondeo"].astype(str) == nombre]
        for _, c in capas.iterrows():
            base = c["z_base"] if np.isfinite(c["z_base"]) else z_max
            figura.add_trace(go.Bar(
                x=[nombre], y=[min(base, z_max) - c["z_techo"]], base=[c["z_techo"]],
                marker_color=color_de[c["unidad"]], marker_line=dict(color="#333", width=0.5),
                name=c["unidad"], legendgroup=c["unidad"], showlegend=c["unidad"] not in vistos,
                text=f"{c['rho']:.3g}", textposition="inside",
                hovertemplate=(f"{nombre} · {c['unidad']}<br>ρ={c['rho']:.4g} Ω·m<br>"
                               f"{c['z_techo']:.3g}–{base:.3g} m<extra></extra>"),
            ))
            vistos.add(c["unidad"])
    figura.update_yaxes(range=[z_max, 0], title="Profundidad (m)")
    figura.update_layout(template=PLANTILLA, barmode="overlay", height=520, bargap=0.45,
                         title="Unidades geoeléctricas por sondeo (modelo de bloques)",
                         margin=dict(l=60, r=20, t=50, b=40))
    return figura


# ======================================================================
# Datos de campo y apilado
# ======================================================================


def curvas_decaimiento(gates: Optional[pd.DataFrame], apilado: pd.DataFrame,
                       titulo: str = "Curvas de decaimiento: barridos y curva apilada",
                       max_puntos: int = 6000) -> go.Figure:
    """Nube de barridos individuales y curva apilada (±1 error estándar).

    Marcadores: lleno = gate usado en la inversión; vacío = bajo el piso de
    ruido o excluido; ✕ = valor negativo; rombo rojo = repunte.
    """
    if apilado is None or len(apilado) == 0:
        return _vacia("Sin curva apilada")
    else:
        pass
    figura = go.Figure()
    for momento in ("LM", "HM"):
        color = COLOR[momento]
        if gates is not None and len(gates):
            g = gates[gates["momento"] == momento]
            muestra = g.sample(min(len(g), max_puntos // 2), random_state=0) if len(g) else g
            figura.add_trace(go.Scattergl(
                x=muestra["tiempo"], y=muestra["dbdt"].abs(), mode="markers", name=f"{momento} · barridos",
                marker=dict(color=color, size=3, opacity=0.08), hoverinfo="skip", legendgroup=momento))
        else:
            pass
        ap = apilado[apilado["momento"] == momento].sort_values("tiempo")
        if ap.empty:
            continue
        else:
            pass
        figura.add_trace(go.Scatter(x=ap["tiempo"], y=ap["dbdt"].abs(), mode="lines", showlegend=False,
                                    line=dict(color=color, width=1.2), opacity=0.5, legendgroup=momento,
                                    hoverinfo="skip"))
        usados = ap[ap["usar"].astype(bool)]
        figura.add_trace(go.Scatter(
            x=usados["tiempo"], y=usados["dbdt"], mode="markers", name=f"{momento} · usado ({len(usados)})",
            marker=dict(color=color, size=8), legendgroup=momento,
            error_y=dict(type="data", array=usados["error_std"], thickness=1, width=3, color=color),
            customdata=np.stack([usados["gate"], usados["snr"]], axis=-1),
            hovertemplate="gate %{customdata[0]}<br>t=%{x:.3e} s<br>dB/dt=%{y:.3e}<br>SNR=%{customdata[1]:.0f}"
                          "<extra>" + momento + "</extra>"))
        otros = ap[~ap["usar"].astype(bool) & (ap["dbdt"] > 0)]
        figura.add_trace(go.Scatter(
            x=otros["tiempo"], y=otros["dbdt"], mode="markers", name=f"{momento} · no usado",
            marker=dict(color="white", size=8, line=dict(color=color, width=1.5)), legendgroup=momento,
            customdata=otros["gate"], hovertemplate="gate %{customdata} (no usado)<extra></extra>"))
        negativos = ap[ap["dbdt"] <= 0]
        if len(negativos):
            figura.add_trace(go.Scatter(x=negativos["tiempo"], y=negativos["dbdt"].abs(), mode="markers",
                                        name=f"{momento} · negativo", legendgroup=momento,
                                        marker=dict(color=color, size=8, symbol="x")))
        else:
            pass
        repuntes = ap[ap["repunte"].astype(bool)]
        if len(repuntes):
            figura.add_trace(go.Scatter(x=repuntes["tiempo"], y=repuntes["dbdt"], mode="markers",
                                        name=f"{momento} · repunte", legendgroup=momento,
                                        marker=dict(color=COLOR["anomalo"], size=12, symbol="diamond-open",
                                                    line=dict(width=2))))
        else:
            pass
        figura.add_trace(go.Scatter(x=ap["tiempo"], y=ap["error_std"], mode="lines", name=f"{momento} · piso de ruido",
                                    line=dict(color=color, dash="dot", width=1), legendgroup=momento))
    valores = apilado["dbdt"].abs()
    _ejes_loglog(figura, titulo_y="dB/dt normalizado (V/(A·m²))", titulo_x="Tiempo después del apagado (s)",
                 valores_x=apilado["tiempo"], valores_y=pd.concat([valores[valores > 0], apilado["error_std"]]))
    figura.update_layout(template=PLANTILLA, height=560, title=titulo, legend=dict(orientation="h", y=-0.18),
                         margin=dict(l=60, r=20, t=50, b=40))
    return figura


def curva_combinada(curva: Optional[pd.DataFrame], t_cambio: Optional[float] = None,
                    titulo: str = "Curva combinada LM + HM y resistividad aparente") -> go.Figure:
    """dB/dt y ρa de tiempo tardío de la curva combinada."""
    if curva is None or len(curva) == 0:
        return _vacia("Sin curva combinada")
    else:
        pass
    figura = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                           subplot_titles=("dB/dt (gates útiles)", "ρa de tiempo tardío"))
    for momento in ("LM", "HM"):
        c = curva[curva["momento"] == momento]
        dentro = c[c["en_curva"].astype(bool)]
        fuera = c[~c["en_curva"].astype(bool)]
        for parte, simbolo, sufijo in ((dentro, "circle", "en la curva"), (fuera, "circle-open", "reemplazado")):
            if parte.empty:
                continue
            else:
                pass
            figura.add_trace(go.Scatter(x=parte["tiempo"], y=parte["dbdt"], mode="markers",
                                        name=f"{momento} · {sufijo}", legendgroup=momento,
                                        marker=dict(color=COLOR[momento], size=7, symbol=simbolo)), row=1, col=1)
            figura.add_trace(go.Scatter(x=parte["tiempo"], y=parte["rho_a"], mode="markers", showlegend=False,
                                        legendgroup=momento, marker=dict(color=COLOR[momento], size=7, symbol=simbolo),
                                        hovertemplate="t=%{x:.3e} s<br>ρa=%{y:.3g} Ω·m<extra>" + momento + "</extra>"),
                             row=1, col=2)
    unida = curva[curva["en_curva"].astype(bool)].sort_values("tiempo")
    figura.add_trace(go.Scatter(x=unida["tiempo"], y=unida["dbdt"], mode="lines", name="Curva combinada",
                                line=dict(color="#111", width=1.5)), row=1, col=1)
    figura.add_trace(go.Scatter(x=unida["tiempo"], y=unida["rho_a"], mode="lines", showlegend=False,
                                line=dict(color="#111", width=1.5)), row=1, col=2)
    if t_cambio is not None and np.isfinite(t_cambio):
        for col in (1, 2):
            figura.add_vline(x=t_cambio, line=dict(color="#999", dash="dash", width=1), row=1, col=col)
    else:
        pass
    _ejes_loglog(figura, 1, 1, titulo_y="dB/dt (V/(A·m²))", titulo_x="Tiempo (s)", valores_x=curva["tiempo"],
                 valores_y=curva["dbdt"][curva["dbdt"] > 0])
    _ejes_loglog(figura, 1, 2, titulo_y="ρa (Ω·m)", titulo_x="Tiempo (s)", valores_x=curva["tiempo"],
                 valores_y=curva["rho_a"].dropna())
    figura.update_layout(template=PLANTILLA, height=430, title=titulo, legend=dict(orientation="h", y=-0.22),
                         margin=dict(l=60, r=20, t=60, b=40))
    return figura


def adquisicion(barridos: pd.DataFrame, titulo: str = "Corriente de transmisión por barrido") -> go.Figure:
    """Corriente de cada barrido (desviación respecto a su media)."""
    if barridos is None or len(barridos) == 0:
        return _vacia("Sin barridos")
    else:
        pass
    usa_hora = "hora" in barridos and barridos["hora"].notna().any()
    figura = go.Figure()
    for momento in ("LM", "HM"):
        b = barridos[barridos["momento"] == momento].sort_values("indice_momento")
        if b.empty:
            continue
        else:
            pass
        media = float(b["corriente_a"].mean())
        x = pd.to_datetime(b["hora"]) if usa_hora else b["indice_momento"]
        figura.add_trace(go.Scatter(x=x, y=100 * (b["corriente_a"] / media - 1), mode="lines",
                                    name=f"{momento} (media {media:.2f} A)", line=dict(color=COLOR[momento], width=1)))
    figura.add_hline(y=0, line=dict(color="#666", width=1))
    figura.update_xaxes(title="Hora" if usa_hora else "Barrido")
    figura.update_yaxes(title="Corriente vs su media (%)")
    figura.update_layout(template=PLANTILLA, height=320, title=titulo, legend=dict(orientation="h", y=-0.3),
                         margin=dict(l=60, r=20, t=50, b=40))
    return figura


def posicion_gps(barridos: pd.DataFrame, sondeo: pd.Series, epsg: Optional[int]) -> go.Figure:
    """Fijos GPS por barrido respecto al lazo dibujado en el mapa (m)."""
    if barridos is None or "gps_lat" not in barridos or barridos["gps_lat"].isna().all() or epsg is None:
        return _vacia("Sin GPS por barrido (hace falta el XYZ)", 320)
    else:
        pass
    import geodesia_tdem as geo

    x, y = geo.geograficas_a_proyectadas(barridos["gps_lon"], barridos["gps_lat"], epsg)
    figura = go.Figure()
    lazo = geo.esquinas_lazo(sondeo["X"], sondeo["Y"], sondeo["lado_x"], sondeo["lado_y"], sondeo.get("azimut", 0.0))
    figura.add_trace(go.Scatter(x=lazo[:, 0], y=lazo[:, 1], mode="lines", name="Lazo",
                                line=dict(color="#333", dash="dash", width=1.5)))
    figura.add_trace(go.Scatter(x=[sondeo["X"]], y=[sondeo["Y"]], mode="markers", name="Centro",
                                marker=dict(symbol="cross", size=12, color="#111")))
    figura.add_trace(go.Scattergl(x=x, y=y, mode="markers", name="Fijos GPS",
                                  marker=dict(size=4, color="#1BAF7A", opacity=0.5)))
    figura.update_yaxes(scaleanchor="x", scaleratio=1, title="Y (m)", tickformat=",.0f")
    figura.update_xaxes(title="X (m)", tickformat=",.0f")
    figura.update_layout(template=PLANTILLA, height=320, title="Posición GPS durante el sondeo",
                         legend=dict(orientation="h", y=-0.3), margin=dict(l=60, r=20, t=50, b=40))
    return figura


# ======================================================================
# Inversión
# ======================================================================


def ajuste_inversion(ajuste: pd.DataFrame, vista: str = "dbdt",
                     titulo: str = "Observado contra calculado") -> go.Figure:
    """Datos observados y respuesta de los modelos, con residuales.

    ``vista``: ``"dbdt"`` o ``"rho_a"`` (ρa de tiempo tardío).
    """
    if ajuste is None or len(ajuste) == 0:
        return _vacia("Sin inversión")
    else:
        pass
    figura = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26], vertical_spacing=0.04)
    columna_obs = "dbdt_obs" if vista == "dbdt" else "rho_a_obs"
    for momento in ("LM", "HM"):
        a = ajuste[ajuste["momento"] == momento].sort_values("tiempo")
        if a.empty:
            continue
        else:
            pass
        usados = a[a["usado"].astype(bool)]
        no_usados = a[~a["usado"].astype(bool)]
        error = dict(type="data", array=usados["error"], thickness=1, width=3, color=COLOR[momento]) \
            if vista == "dbdt" else None
        figura.add_trace(go.Scatter(x=usados["tiempo"], y=usados[columna_obs], mode="markers",
                                    name=f"{momento} · observado", legendgroup=momento, error_y=error,
                                    marker=dict(color=COLOR[momento], size=8)), row=1, col=1)
        if len(no_usados):
            figura.add_trace(go.Scatter(x=no_usados["tiempo"], y=no_usados[columna_obs], mode="markers",
                                        name=f"{momento} · no invertido", legendgroup=momento,
                                        marker=dict(color="white", size=8, line=dict(color=COLOR[momento], width=1.5))),
                             row=1, col=1)
        else:
            pass
        for modelo, estilo in (("multicapa", dict(color=COLOR["multicapa"], width=2)),
                               ("bloques", dict(color=COLOR["bloques"], width=2.5))):
            columna = f"dbdt_{modelo}" if vista == "dbdt" else f"rho_a_{modelo}"
            figura.add_trace(go.Scatter(x=a["tiempo"], y=a[columna], mode="lines",
                                        name=f"{momento} · {modelo}", legendgroup=momento,
                                        line=dict(dash="solid" if momento == "HM" else "dot", **estilo)), row=1, col=1)
            figura.add_trace(go.Scatter(x=a["tiempo"], y=a[f"error_{modelo}_pct"], mode="markers+lines",
                                        showlegend=False, line=dict(width=1, **{k: v for k, v in estilo.items() if k == "color"}),
                                        marker=dict(size=5, symbol="circle" if momento == "HM" else "square"),
                                        name=f"{momento} · error {modelo}"), row=2, col=1)
    figura.update_xaxes(type="log", title=None, showgrid=True, gridcolor="#E6E6E6", row=1, col=1,
                        **marcas_log(ajuste["tiempo"]))
    figura.update_xaxes(type="log", title="Tiempo (s)", showgrid=True, gridcolor="#E6E6E6", row=2, col=1,
                        **marcas_log(ajuste["tiempo"]))
    valores = ajuste[columna_obs]
    figura.update_yaxes(type="log", title="dB/dt (V/(A·m²))" if vista == "dbdt" else "ρa (Ω·m)",
                        showgrid=True, gridcolor="#E6E6E6", row=1, col=1, **marcas_log(valores[valores > 0]))
    figura.update_yaxes(title="Calc/obs − 1 (%)", zeroline=True, row=2, col=1)
    figura.update_layout(template=PLANTILLA, height=620, title=titulo, legend=dict(orientation="h", y=-0.14),
                         margin=dict(l=60, r=20, t=50, b=40))
    return figura


def proceso_inversion(iteraciones: Optional[pd.DataFrame]) -> go.Figure:
    """χ² y λ por iteración del multicapa (estrella = modelo elegido)."""
    if iteraciones is None or len(iteraciones) == 0:
        return _vacia("Sin iteraciones", 300)
    else:
        pass
    it = iteraciones.sort_values("iteracion")
    figura = make_subplots(specs=[[{"secondary_y": True}]])
    figura.add_trace(go.Scatter(x=it["iteracion"], y=it["chi2"], mode="lines+markers", name="χ²",
                                line=dict(color=COLOR["multicapa"])), secondary_y=False)
    figura.add_trace(go.Scatter(x=it["iteracion"], y=it["lambda"], mode="lines", name="λ",
                                line=dict(color="#666", dash="dot")), secondary_y=True)
    if "elegido" in it:
        e = it[it["elegido"].astype(bool)]
        figura.add_trace(go.Scatter(x=e["iteracion"], y=e["chi2"], mode="markers", name="Modelo elegido",
                                    marker=dict(size=15, symbol="star", color=COLOR["anomalo"])), secondary_y=False)
    else:
        pass
    figura.add_hline(y=1.0, line=dict(color="#999", dash="dash"), secondary_y=False)
    figura.update_yaxes(type="log", title="χ²", secondary_y=False)
    figura.update_yaxes(type="log", title="λ", secondary_y=True)
    figura.update_xaxes(title="Iteración", dtick=2)
    figura.update_layout(template=PLANTILLA, height=320, title="Convergencia del multicapa",
                         legend=dict(orientation="h", y=-0.3), margin=dict(l=60, r=60, t=50, b=40))
    return figura


def seleccion_capas(seleccion: Optional[pd.DataFrame], criterio: str = "BIC") -> go.Figure:
    """Criterio de información y χ² contra número de capas."""
    if seleccion is None or len(seleccion) == 0:
        return _vacia("Sin selección", 300)
    else:
        pass
    s = seleccion.sort_values("k")
    columna = criterio if criterio in s else "BIC"
    figura = make_subplots(specs=[[{"secondary_y": True}]])
    figura.add_trace(go.Bar(x=s["k"], y=s[columna], name=columna,
                            marker_color=np.where(s["elegido"].astype(bool), COLOR["bloques"], "#C7CDD6")),
                     secondary_y=False)
    figura.add_trace(go.Scatter(x=s["k"], y=s["chi2"], name="χ²", mode="lines+markers",
                                line=dict(color=COLOR["multicapa"])), secondary_y=True)
    figura.add_hline(y=1.0, line=dict(color=COLOR["multicapa"], dash="dot", width=1), secondary_y=True)
    figura.update_xaxes(title="Número de capas", dtick=1)
    figura.update_yaxes(title=columna, secondary_y=False)
    figura.update_yaxes(title="χ²", type="log", secondary_y=True)
    figura.update_layout(template=PLANTILLA, height=320, title=f"Selección del número de capas ({columna})",
                         legend=dict(orientation="h", y=-0.3), margin=dict(l=60, r=60, t=50, b=40))
    return figura


# ======================================================================
# Campaña: mapa y pseudosección
# ======================================================================


def mapa_sondeos(coords: pd.DataFrame, valores: Optional[pd.Series] = None, etiqueta_valor: str = "",
                 ruta: Optional[np.ndarray] = None, usar_log: bool = True,
                 proyectados: Optional[pd.DataFrame] = None, epsg: Optional[int] = None) -> go.Figure:
    """Ubicación de los sondeos con su lazo dibujado a escala.

    ``coords`` necesita ``Sondeo, X, Y`` y, para el lazo, ``lado_x,
    lado_y`` (y ``azimut`` opcional, grados desde el Este en sentido
    horario; 0 = lados alineados con los ejes).
    """
    import geodesia_tdem as geo

    if coords is None or len(coords) == 0:
        return _vacia("Sin coordenadas de sondeos")
    else:
        pass
    figura = go.Figure()
    for i, fila in coords.reset_index(drop=True).iterrows():
        lado_x = float(fila.get("lado_x") or 0.0) if pd.notna(fila.get("lado_x")) else 0.0
        lado_y = float(fila.get("lado_y") or 0.0) if pd.notna(fila.get("lado_y")) else 0.0
        if lado_x > 0 and lado_y > 0:
            lazo = geo.esquinas_lazo(fila["X"], fila["Y"], lado_x, lado_y, fila.get("azimut", 0.0) or 0.0)
            figura.add_trace(go.Scatter(
                x=lazo[:, 0], y=lazo[:, 1], mode="lines", fill="toself", fillcolor="rgba(0,131,143,0.08)",
                line=dict(color=COLOR["preprocesado"], width=2), name="Lazo transmisor", legendgroup="lazo",
                showlegend=i == 0, hovertemplate=f"{fila['Sondeo']}: lazo {lado_x:g} × {lado_y:g} m<extra></extra>"))
        else:
            pass
    # Etiquetas alternadas arriba/abajo: sondeos repetidos a pocos metros
    # (p. ej. una repetición de control) no se enciman.
    posiciones = ["top center" if i % 2 == 0 else "bottom center" for i in range(len(coords))]
    if valores is not None:
        v = np.asarray(valores, float)
        color = np.log10(v) if usar_log else v
        figura.add_trace(go.Scatter(
            x=coords["X"], y=coords["Y"], mode="markers+text", text=coords["Sondeo"], textposition=posiciones,
            name="Sondeos", marker=dict(size=14, color=color, colorscale=ESCALA_RESISTIVIDAD, showscale=True,
                                        line=dict(color="#333", width=1),
                                        colorbar=dict(title=etiqueta_valor + (" (log₁₀)" if usar_log else ""))),
            customdata=v, hovertemplate="%{text}<br>" + etiqueta_valor + "=%{customdata:.4g}<extra></extra>"))
    else:
        figura.add_trace(go.Scatter(
            x=coords["X"], y=coords["Y"], mode="markers+text", text=coords["Sondeo"], textposition=posiciones,
            name="Centro del lazo", marker=dict(size=9, color=COLOR["preprocesado"], line=dict(color="#333", width=1)),
            hovertemplate="%{text}<br>X=%{x:,.1f}<br>Y=%{y:,.1f}<extra></extra>"))
    if ruta is not None and len(ruta) >= 2:
        r = np.asarray(ruta, float)
        figura.add_trace(go.Scatter(x=r[:, 0], y=r[:, 1], mode="lines+markers", name="Ruta",
                                    line=dict(color=COLOR["anomalo"], width=2), marker=dict(symbol="diamond", size=9)))
    else:
        pass
    if proyectados is not None and len(proyectados):
        for _, f in proyectados[proyectados["dentro"]].iterrows():
            figura.add_trace(go.Scatter(x=[f["X"], f["Xp"]], y=[f["Y"], f["Yp"]], mode="lines", showlegend=False,
                                        line=dict(color="#888", dash="dot", width=1), hoverinfo="skip"))
    else:
        pass
    figura.update_yaxes(scaleanchor="x", scaleratio=1, title="Y (m)", tickformat=",.0f")
    figura.update_xaxes(title="X (m)", tickformat=",.0f")
    subtitulo = f" · EPSG:{epsg}" if epsg else ""
    figura.update_layout(template=PLANTILLA, height=540, title=f"Ubicación de sondeos y lazos{subtitulo}",
                         margin=dict(l=60, r=20, t=50, b=40), legend=dict(orientation="h", y=-0.15))
    return figura


def pseudoseccion(datos: pd.DataFrame, titulo: str = "Pseudosección: ρa tardía contra profundidad de difusión",
                  rango: Optional[tuple[float, float]] = None) -> go.Figure:
    """ρa de la curva combinada en (cadenamiento, profundidad de difusión)."""
    if datos is None or len(datos) == 0:
        return _vacia("Sin curvas para la pseudosección")
    else:
        pass
    log_v = np.log10(datos["rho_a"].to_numpy(float))
    zmin, zmax = rango if rango else (np.percentile(log_v, 2), np.percentile(log_v, 98))
    figura = go.Figure(go.Scatter(
        x=datos["s"], y=datos["z"], mode="markers",
        marker=dict(size=14, symbol="square", color=log_v, colorscale=ESCALA_RESISTIVIDAD, cmin=zmin, cmax=zmax,
                    colorbar=dict(title="log₁₀ ρa")),
        customdata=np.column_stack([datos["rho_a"], datos["tiempo"], datos["Sondeo"]]),
        hovertemplate="%{customdata[2]}<br>t=%{customdata[1]:.3e} s<br>ρa=%{customdata[0]:.4g} Ω·m<extra></extra>"))
    figura.update_xaxes(title="Distancia sobre la ruta (m)")
    figura.update_yaxes(title="Profundidad de difusión (m)", type="log", autorange="reversed",
                        **marcas_log(datos["z"]))
    figura.update_layout(template=PLANTILLA, height=380, title=titulo, margin=dict(l=60, r=20, t=50, b=40))
    return figura


def puntajes_barridos(puntajes: Optional[pd.DataFrame], momento: str) -> go.Figure:
    """Puntaje de anomalía de cada barrido (IF arriba, LOF abajo) con su umbral.

    Los barridos descartados (ambos detectores, o el que indique la
    configuración) se marcan en rojo. Dos paneles con su propia escala, en
    vez de un eje doble: los puntajes de IF y LOF no son comparables.
    """
    if puntajes is None or len(puntajes) == 0 or puntajes["puntaje_if"].isna().all():
        return _vacia("Sin detección por barrido (desactivada o con menos de 20 barridos)", 300)
    else:
        pass
    p = puntajes[puntajes["momento"] == momento].sort_values("indice")
    if p.empty:
        return _vacia(f"Sin barridos {momento}", 300)
    else:
        pass
    figura = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                           subplot_titles=("Isolation Forest", "Local Outlier Factor"))
    for fila, (col, umbral, marca) in enumerate((("puntaje_if", "umbral_if", "atipico_if"),
                                                 ("puntaje_lof", "umbral_lof", "atipico_lof")), start=1):
        normales = p[~p["descartado"].astype(bool)]
        descartados = p[p["descartado"].astype(bool)]
        figura.add_trace(go.Scattergl(
            x=normales["indice"], y=normales[col], mode="markers", name="Conservado", showlegend=fila == 1,
            legendgroup="c", marker=dict(size=5, color=COLOR[momento], opacity=0.55,
                                         symbol=np.where(normales[marca], "circle-open", "circle")),
            hovertemplate="barrido %{x}<br>puntaje %{y:.3f}<extra></extra>"), row=fila, col=1)
        figura.add_trace(go.Scattergl(
            x=descartados["indice"], y=descartados[col], mode="markers", name="Descartado",
            showlegend=fila == 1, legendgroup="d", marker=dict(size=9, color=COLOR["anomalo"], symbol="x"),
            hovertemplate="barrido %{x} (descartado)<br>puntaje %{y:.3f}<extra></extra>"), row=fila, col=1)
        figura.add_hline(y=float(p[umbral].iloc[0]), line=dict(color="#666", dash="dash", width=1), row=fila, col=1)
    figura.update_xaxes(title="Barrido (orden de adquisición)", row=2, col=1)
    figura.update_yaxes(title="puntaje", row=1, col=1)
    figura.update_yaxes(title="LOF", row=2, col=1)
    n_desc = int(p["descartado"].sum())
    figura.update_layout(template=PLANTILLA, height=420,
                         title=f"{momento}: anomalía por barrido · {n_desc} de {len(p)} descartados "
                               "(línea = mediana + k·MAD de los puntajes)",
                         legend=dict(orientation="h", y=-0.2), margin=dict(l=60, r=20, t=70, b=40))
    return figura
