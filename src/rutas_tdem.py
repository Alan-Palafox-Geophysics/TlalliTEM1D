"""
rutas_tdem.py
==================================================================
Resolución de rutas del proyecto.

Los módulos viven en dos carpetas planas (``src`` y ``app``) que no son
un paquete de Python. Este módulo registra ambas en ``sys.path`` para
que los módulos se importen de forma plana (``import lectura_tdem``)
sin importar desde cuál de las dos carpetas se ejecute el código.

Es idempotente: importarlo varias veces no duplica entradas.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Carpeta raíz del proyecto (la que contiene ``src`` y ``app``).
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent

#: Carpetas de código que deben ser importables de forma plana.
CARPETAS_CODIGO = ("src", "app")


def registrar_rutas() -> list[str]:
    """Añade las carpetas de código del proyecto a ``sys.path``."""
    registradas: list[str] = []
    for carpeta in CARPETAS_CODIGO:
        ruta = str(RAIZ_PROYECTO / carpeta)
        if ruta not in sys.path:
            sys.path.insert(0, ruta)
        else:
            pass
        registradas.append(ruta)
    return registradas


registrar_rutas()
