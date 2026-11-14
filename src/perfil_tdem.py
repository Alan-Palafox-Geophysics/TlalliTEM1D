"""
perfil_tdem.py
==================================================================
Perfiles 2D a partir de sondeos TDEM: proyección sobre una ruta y
secciones interpoladas de los modelos 1D.

Proyección
-------------
La ruta es una polilínea de vértices (X, Y) en el EPSG del proyecto.
Cada sondeo se proyecta sobre el segmento más cercano: su
**cadenamiento** (distancia a lo largo de la ruta) es la abscisa del
perfil, y la distancia perpendicular decide si entra (``≤ buffer``).

Sección interpolada (modelos 1D "cosidos")
---------------------------------------------
Cada modelo 1D se muestrea en una rejilla de profundidad y se interpola
**en log ρ** a lo largo del cadenamiento. Es la presentación estándar de
TDEM terrestre con sondeos separados. No usa los datos: si dos sondeos
vecinos llegaron a modelos equivalentes distintos, la sección muestra un
cambio lateral que puede ser solo equivalencia. La inversión
lateralmente restringida (LCI) de la versión SEV no está incluida aquí.

Pseudosección
----------------
ρa de tiempo tardío de la curva combinada LM + HM, graficada contra la
profundidad de difusión √(2tρa/μ0) de cada gate.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

import forward_tdem as fw

logger = logging.getLogger("tdem.perfil")


class PerfilError(Exception):
    """El perfil no se pudo construir."""


def proyectar(sondeos: pd.DataFrame, vertices: Sequence[Sequence[float]], buffer: float = np.inf) -> pd.DataFrame:
    """Proyecta sondeos (``Sondeo, X, Y, Z``) sobre una polilínea.

    Returns
    -------
    DataFrame
        Entrada más ``s`` (cadenamiento), ``distancia``, ``Xp``, ``Yp``,
        ``dentro``; ordenado por ``s``.
    """
    v = np.asarray(vertices, float)
    if v.ndim != 2 or len(v) < 2:
        raise PerfilError("La ruta necesita al menos dos vértices (X, Y).")
    else:
        pass
    segmentos = v[1:] - v[:-1]
    longitudes = np.hypot(segmentos[:, 0], segmentos[:, 1])
    acumulado = np.concatenate([[0.0], np.cumsum(longitudes)])
    filas = []
    for _, fila in sondeos.iterrows():
        p = np.array([fila["X"], fila["Y"]], float)
        mejor = None
        for i, (a, d, largo) in enumerate(zip(v[:-1], segmentos, longitudes)):
            if largo <= 0:
                continue
            else:
                pass
            t = float(np.clip(np.dot(p - a, d) / largo ** 2, 0.0, 1.0))
            q = a + t * d
            distancia = float(np.hypot(*(p - q)))
            if mejor is None or distancia < mejor[0]:
                mejor = (distancia, acumulado[i] + t * largo, q)
            else:
                pass
        filas.append({**fila.to_dict(), "distancia": mejor[0], "s": mejor[1],
                      "Xp": mejor[2][0], "Yp": mejor[2][1], "dentro": mejor[0] <= buffer})
    salida = pd.DataFrame(filas).sort_values("s").reset_index(drop=True)
    salida.attrs["longitud"] = float(acumulado[-1])
    return salida


def seccion_interpolada(proyectados: pd.DataFrame, modelos: dict[str, pd.DataFrame], columna: str = "rho",
                        z_max: Optional[float] = None, n_s: int = 160, n_z: int = 120, usar_log: bool = True,
                        dois: Optional[dict[str, float]] = None) -> dict[str, Any]:
    """Sección en una rejilla regular (cadenamiento × elevación).

    Returns
    -------
    dict
        ``s``, ``elev``, ``valores`` (n_z × n_s, NaN fuera del terreno o
        bajo z_max), ``topo``, ``doi`` y los sondeos usados.
    """
    usados = proyectados[proyectados["Sondeo"].astype(str).isin(modelos)].reset_index(drop=True)
    if len(usados) < 2:
        raise PerfilError("Hacen falta al menos dos sondeos invertidos sobre la ruta.")
    else:
        pass
    if z_max is None:
        if dois:
            z_max = float(np.nanmax([dois.get(str(s), np.nan) for s in usados["Sondeo"]]))
        else:
            z_max = float(np.nanmax([modelos[str(s)]["z_techo"].max() * 1.3 for s in usados["Sondeo"]]))
    else:
        pass
    if not np.isfinite(z_max) or z_max <= 0:
        z_max = 100.0
    else:
        pass

    z = np.linspace(0.0, z_max, n_z)
    columnas = []
    for _, fila in usados.iterrows():
        modelo = modelos[str(fila["Sondeo"])]
        espesores = modelo["espesor"].to_numpy(float)[:-1]
        valores = fw.muestrear_modelo(espesores, modelo[columna].to_numpy(float), z)
        columnas.append(np.log10(np.maximum(valores, 1e-6)) if usar_log else valores)
    matriz = np.column_stack(columnas)

    s_sondeos = usados["s"].to_numpy(float)
    s = np.linspace(s_sondeos.min(), s_sondeos.max(), n_s)
    topo = np.interp(s, s_sondeos, usados["Z"].to_numpy(float))
    en_profundidad = np.vstack([np.interp(s, s_sondeos, matriz[i]) for i in range(n_z)])
    elev = np.linspace(topo.max(), topo.min() - z_max, n_z)
    valores = np.full((n_z, n_s), np.nan)
    for j in range(n_s):
        profundidad = topo[j] - elev
        dentro = (profundidad >= 0) & (profundidad <= z_max)
        valores[dentro, j] = np.interp(profundidad[dentro], z, en_profundidad[:, j])
    doi_s = None
    if dois:
        doi_s = np.interp(s, s_sondeos, [dois.get(str(x), np.nan) for x in usados["Sondeo"]])
    else:
        pass
    return {"s": s, "elev": elev, "valores": valores, "topo": topo, "doi": doi_s,
            "sondeos": usados, "z_max": z_max, "usar_log": usar_log}


def ruta_automatica(coords: pd.DataFrame) -> list[list[float]]:
    """Ruta recta por el eje principal de los sondeos (PCA)."""
    xy = coords[["X", "Y"]].to_numpy(float)
    centro = xy.mean(axis=0)
    if len(xy) < 2:
        return [[float(centro[0]) - 50, float(centro[1])], [float(centro[0]) + 50, float(centro[1])]]
    else:
        pass
    _, _, vt = np.linalg.svd(xy - centro, full_matrices=False)
    eje = vt[0]
    t = (xy - centro) @ eje
    margen = 0.05 * (t.max() - t.min() + 1.0) + 20.0
    a = centro + (t.min() - margen) * eje
    b = centro + (t.max() + margen) * eje
    return [[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]]


def datos_pseudoseccion(proyectados: pd.DataFrame, curvas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """ρa tardía de la curva combinada contra profundidad de difusión."""
    filas = []
    for _, fila in proyectados[proyectados["dentro"]].iterrows():
        curva = curvas.get(str(fila["Sondeo"]))
        if curva is None or len(curva) == 0:
            continue
        else:
            pass
        c = curva[curva["en_curva"].astype(bool)].dropna(subset=["rho_a", "prof_difusion"])
        filas.append(pd.DataFrame({"Sondeo": fila["Sondeo"], "s": fila["s"], "tiempo": c["tiempo"],
                                   "momento": c["momento"], "rho_a": c["rho_a"], "z": c["prof_difusion"]}))
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()
