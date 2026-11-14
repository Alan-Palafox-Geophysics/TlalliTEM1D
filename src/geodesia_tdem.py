"""
geodesia_tdem.py
==================================================================
Conversión de coordenadas geográficas (WGS84) a proyectadas (UTM).

Los archivos USF y XYZ traen la ubicación en grados decimales
(``EPSG:4326``). El mapa, las distancias entre sondeos y los perfiles
necesitan metros, así que el proyecto trabaja en UTM:

- La zona UTM se elige con la longitud del **primer** sondeo que trae
  coordenadas y queda fija para todo el proyecto (``EPSG:326zz`` en el
  hemisferio norte, ``EPSG:327zz`` en el sur). Mezclar zonas dentro de
  un proyecto rompería las distancias.
- Si el usuario captura X, Y a mano, se asume que están en el EPSG del
  proyecto.

Se usa ``pyproj``: es la referencia para transformaciones y evita
reimplementar las fórmulas de Krüger.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

EPSG_GEOGRAFICAS = 4326


def epsg_utm(lon: float, lat: float) -> int:
    """Código EPSG de la zona UTM WGS84 que contiene el punto."""
    zona = int(math.floor((float(lon) + 180.0) / 6.0)) + 1
    zona = min(max(zona, 1), 60)
    if float(lat) >= 0:
        return 32600 + zona
    else:
        return 32700 + zona


@lru_cache(maxsize=16)
def _transformador(origen: int, destino: int):
    from pyproj import Transformer

    return Transformer.from_crs(origen, destino, always_xy=True)


def geograficas_a_proyectadas(lon, lat, epsg: int) -> tuple[np.ndarray, np.ndarray]:
    """(lon, lat) en grados → (X, Y) en metros del ``epsg`` indicado."""
    transformador = _transformador(EPSG_GEOGRAFICAS, int(epsg))
    x, y = transformador.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x, float), np.asarray(y, float)


def proyectadas_a_geograficas(x, y, epsg: int) -> tuple[np.ndarray, np.ndarray]:
    """(X, Y) en metros del ``epsg`` → (lon, lat) en grados."""
    transformador = _transformador(int(epsg), EPSG_GEOGRAFICAS)
    lon, lat = transformador.transform(np.asarray(x, float), np.asarray(y, float))
    return np.asarray(lon, float), np.asarray(lat, float)


def coordenadas_validas(lon, lat) -> bool:
    """``True`` si (lon, lat) es una ubicación real (no dummy ni 0, 0)."""
    try:
        lon_f, lat_f = float(lon), float(lat)
    except (TypeError, ValueError):
        return False
    if not (np.isfinite(lon_f) and np.isfinite(lat_f)):
        return False
    else:
        pass
    if abs(lon_f) > 180 or abs(lat_f) > 90:
        return False
    else:
        pass
    return not (abs(lon_f) < 1e-9 and abs(lat_f) < 1e-9)


def esquinas_lazo(x: float, y: float, lado_x: float, lado_y: float,
                  azimut_grados: float = 0.0) -> np.ndarray:
    """Polígono cerrado (5 × 2) del lazo centrado en (x, y).

    ``azimut_grados`` rota el lado X del lazo desde el Este en sentido
    horario (0 = lados alineados con los ejes). Los archivos no traen la
    orientación del lazo; por omisión se dibuja alineado a los ejes.
    """
    hx, hy = 0.5 * float(lado_x), 0.5 * float(lado_y)
    locales = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy], [-hx, -hy]], float)
    angulo = -math.radians(float(azimut_grados or 0.0))
    rotacion = np.array([[math.cos(angulo), -math.sin(angulo)],
                         [math.sin(angulo), math.cos(angulo)]])
    return locales @ rotacion.T + np.array([float(x), float(y)])
