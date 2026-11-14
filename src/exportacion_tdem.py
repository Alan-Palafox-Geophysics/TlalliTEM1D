"""
exportacion_tdem.py
==================================================================
Exportación a CSV de las tablas del pipeline TDEM con selección de
columnas, y exportación de modelos en formato de texto para otros
programas.

Lógica pura: recibe DataFrames y devuelve bytes o DataFrames.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("tdem.exportacion")


@dataclass(frozen=True)
class TablaExportable:
    """Tabla que el usuario puede exportar."""

    clave: str
    etiqueta: str
    etapa: str
    tabla: str
    descripcion: str
    grupo: str


TABLAS_EXPORTABLES: tuple[TablaExportable, ...] = (
    TablaExportable("barridos", "Barridos (metadatos por stack)", "crudo", "barridos",
                    "Un renglón por barrido: momento, corriente, frecuencia, rampa, filtros y, si hubo "
                    "XYZ, hora y GPS de cada stack.", "procesamiento"),
    TablaExportable("gates", "Gates de cada barrido (crudo)", "crudo", "gates",
                    "dB/dt normalizado, error del equipo y bandera de calidad de cada gate de cada "
                    "barrido, más el valor del XYZ cuando existe.", "procesamiento"),
    TablaExportable("ventanas", "Ventanas de tiempo", "crudo", "ventanas",
                    "Inicio, centro y fin de cada gate por momento.", "procesamiento"),
    TablaExportable("forma_onda", "Forma de onda del transmisor", "crudo", "forma_onda",
                    "Puntos de la rampa declarados en el archivo, usados en el modelado directo.",
                    "procesamiento"),
    TablaExportable("apilado", "Curva apilada por momento", "apilado", "apilado",
                    "Apilado robusto (mediana/MAD + media), error estándar, SNR, banderas de gate útil, "
                    "repunte, exclusión manual y ρa tardía.", "procesamiento"),
    TablaExportable("curva", "Curva combinada LM + HM", "apilado", "curva",
                    "Gates útiles de ambos momentos con la marca de los que forman la curva única.",
                    "procesamiento"),
    TablaExportable("qc", "Hallazgos del control de calidad", "apilado", "qc",
                    "Cada revisión con su severidad (OK/INFO/WARN/FAIL), valor, umbral y mensaje.",
                    "procesamiento"),
    TablaExportable("exclusiones", "Gates excluidos por el usuario", "apilado", "exclusiones",
                    "Momento y gate que el usuario retiró de la inversión.", "procesamiento"),
    TablaExportable("multicapa", "Modelo multicapa", "inversion", "multicapa",
                    "Capas del modelo suave (SimPEG): profundidades, ρ, índice DOI, S y T por capa.",
                    "resultados"),
    TablaExportable("bloques", "Modelo de bloques", "inversion", "bloques",
                    "Capas del modelo de pocas capas (BIC/AIC) con rangos ±1σ y resolución.",
                    "resultados"),
    TablaExportable("ajuste", "Ajuste observado vs calculado", "inversion", "ajuste",
                    "dB/dt observado y calculado por ambos modelos, error porcentual y ρa tardía.",
                    "resultados"),
    TablaExportable("seleccion", "Selección del número de capas", "inversion", "seleccion",
                    "χ², BIC y AIC de cada número de capas evaluado.", "resultados"),
    TablaExportable("iteraciones", "Iteraciones del multicapa", "inversion", "iteraciones",
                    "χ², λ y rugosidad de cada iteración de Gauss–Newton.", "resultados"),
    TablaExportable("parametros", "Parámetros de Dar Zarrouk y corrosividad", "parametros", "parametros",
                    "S, T, ρL, ρT, λ, ρm y clase de corrosividad para ambos modelos.", "resultados"),
)

GRUPOS_EXPORTACION: dict[str, str] = {
    "procesamiento": "Datos de campo y procesamiento",
    "resultados": "Resultados (inversión y parámetros)",
}


def modelo_a_texto(modelo: pd.DataFrame, sondeo: str, tipo: str) -> bytes:
    """Modelo de capas en texto plano (espesor, ρ) para otros programas."""
    lineas = [f"# Sondeo: {sondeo}", f"# Modelo: {tipo}",
              "# capa\tz_techo_m\tespesor_m\trho_ohm_m"]
    for _, c in modelo.iterrows():
        espesor = "inf" if not np.isfinite(c["espesor"]) else f"{c['espesor']:.4f}"
        lineas.append(f"{int(c['capa'])}\t{c['z_techo']:.4f}\t{espesor}\t{c['rho']:.4f}")
    return ("\n".join(lineas) + "\n").encode("utf-8")


def tablas_del_grupo(grupo: str) -> tuple[TablaExportable, ...]:
    """Tablas exportables de un grupo, en orden."""
    return tuple(t for t in TABLAS_EXPORTABLES if t.grupo == grupo)


def buscar_tabla(clave: str) -> Optional[TablaExportable]:
    """Descripción de una tabla exportable por su clave."""
    for tabla in TABLAS_EXPORTABLES:
        if tabla.clave == clave:
            return tabla
        else:
            pass
    return None


def aplanar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Aplana un ``MultiIndex`` de columnas a nombres simples.

    El resumen de facies llega con columnas de dos niveles
    (``('Rho', 'mean')``); un CSV no puede representarlas, así que se
    unen con doble guion bajo (``Rho__mean``) — legible y reversible.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    else:
        pass

    plano = df.copy()
    plano.columns = [
        "__".join(str(parte) for parte in columna if str(parte) != "").strip("__")
        for columna in plano.columns
    ]
    return plano.reset_index()


def columnas_disponibles(df: pd.DataFrame) -> list[str]:
    """Nombres de columna de una tabla, ya aplanados."""
    return list(aplanar_columnas(df).columns)


def preparar_exportacion(
    df: pd.DataFrame, columnas: Optional[list[str]] = None
) -> pd.DataFrame:
    """Recorta un DataFrame a las columnas seleccionadas, listo para
    exportar.

    Parameters
    ----------
    columnas : list[str], optional
        Columnas a conservar, en el orden pedido. Si es ``None``, se
        conservan todas.

    Raises
    ------
    ValueError
        Si la selección queda vacía: exportar un CSV sin columnas no es
        un resultado útil y es preferible avisar que entregar un archivo
        inservible.
    """
    plano = aplanar_columnas(df)

    if columnas is None:
        return plano
    else:
        pass

    seleccion = [c for c in columnas if c in plano.columns]
    if not seleccion:
        raise ValueError(
            "La selección de columnas quedó vacía: elija al menos una columna "
            "para exportar."
        )
    else:
        pass

    return plano[seleccion]


def a_csv(
    df: pd.DataFrame,
    columnas: Optional[list[str]] = None,
    separador: str = ",",
    decimales: Optional[int] = None,
) -> bytes:
    """Serializa un DataFrame a CSV en memoria.

    Se devuelve en **bytes** (codificados en UTF-8 con BOM) porque es lo
    que consume el botón de descarga, y el BOM hace que Excel abra
    correctamente los acentos y símbolos como ``Ω`` sin pedir un paso de
    importación manual.

    Parameters
    ----------
    decimales : int, optional
        Redondeo de las columnas numéricas antes de exportar.
    """
    exportable = preparar_exportacion(df, columnas)

    if decimales is not None:
        columnas_numericas = exportable.select_dtypes(include="number").columns
        exportable = exportable.copy()
        exportable[columnas_numericas] = exportable[columnas_numericas].round(decimales)
    else:
        pass

    memoria = io.StringIO()
    exportable.to_csv(memoria, index=False, sep=separador)
    return memoria.getvalue().encode("utf-8-sig")


def vista_previa(
    df: pd.DataFrame, columnas: Optional[list[str]] = None, n_filas: int = 8
) -> pd.DataFrame:
    """Primeras filas de lo que se exportará, con las columnas ya
    filtradas — para que el usuario confirme el contenido antes de
    descargar."""
    exportable = preparar_exportacion(df, columnas)
    return exportable.head(n_filas)


def resumen_exportacion(
    df: pd.DataFrame, columnas: Optional[list[str]] = None
) -> dict[str, Any]:
    """Métricas de lo que se va a exportar (filas, columnas y tamaño
    estimado), para mostrarlas junto a la vista previa."""
    exportable = preparar_exportacion(df, columnas)
    contenido = a_csv(df, columnas)
    return {
        "n_filas": len(exportable),
        "n_columnas": len(exportable.columns),
        "tamano_kb": round(len(contenido) / 1024, 1),
    }
