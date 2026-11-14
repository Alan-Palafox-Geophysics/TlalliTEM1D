"""
pipeline_tdem.py
==================================================================
Orquestador del pipeline TDEM: funciones puras y probables desde
código, sin Streamlit (mismo contrato que TlalliRes1D SEV).

Etapas por sondeo
--------------------
==============  ==========================================================
Etapa           Qué hace
==============  ==========================================================
``crudo``       Lectura del USF (prioridad) y del XYZ, tablas de barridos,
                gates, ventanas y forma de onda; registro en el catálogo
                con coordenadas del archivo (si las trae).
``apilado``     Barridos atípicos (IF + LOF), MAD por gate en log₁₀ ρa,
                media lineal; gates útiles, exclusiones del
                usuario, ρa tardía, curva combinada LM+HM y control de
                calidad (18 revisiones).
``inversion``   Multicapa suave (SimPEG) → DOI → modelo de bloques.
``parametros``  S, T, ρL, ρT, λ, ρm y corrosividad.
==============  ==========================================================

A nivel de campaña: QC entre sondeos, unidades geoeléctricas (GMM) y
perfiles 2D (sección interpolada de modelos 1D y pseudosección).

Cada ``ejecutar_*`` devuelve un :class:`ResultadoEtapa`
booleano-evaluable.

Regla de estilo del proyecto
-------------------------------
Este módulo no usa la sentencia ``elif``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import warnings
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd

import rutas_tdem  # noqa: F401
import almacenamiento_tdem as alm
from almacenamiento_tdem import Proyecto
from inversion_tdem import ConfigInversion
from procesamiento_tdem import ConfigProcesamiento
from qc_tdem import ConfigQC

logger = logging.getLogger("tdem.pipeline")


class CancelacionSolicitada(Exception):
    """Se solicitó detener la ejecución."""


class TokenCancelacion:
    """Señal de cancelación cooperativa.

    La inversión la consulta entre iteraciones, así que la parada surte
    efecto en pocos segundos.
    """

    def __init__(self) -> None:
        self._evento = threading.Event()

    def solicitar_cancelacion(self) -> None:
        self._evento.set()

    def reiniciar(self) -> None:
        self._evento.clear()

    @property
    def cancelado(self) -> bool:
        return self._evento.is_set()

    def verificar(self) -> None:
        if self._evento.is_set():
            raise CancelacionSolicitada("Ejecución detenida por el usuario.")
        else:
            pass


@dataclass
class ResultadoEtapa:
    """Resultado de una etapa (``bool(resultado)`` = éxito)."""

    exito: bool
    etapa: str
    mensaje: str = ""
    datos: dict[str, Any] = field(default_factory=dict)
    metricas: dict[str, Any] = field(default_factory=dict)
    duracion_s: float = 0.0
    advertencias: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.exito


@dataclass
class ConfigPipeline:
    """Parámetros de ejecución de todo el pipeline.

    Attributes
    ----------
    procesamiento : ConfigProcesamiento
        Apilado y selección de gates.
    qc : ConfigQC
        Umbrales del control de calidad.
    inversion : ConfigInversion
    autoguardado : bool
    prof_parametros : float, optional
        Profundidad de cálculo de Dar Zarrouk (``None`` = DOI).
    prof_corrosion : float
        Profundidad del suelo evaluado para corrosividad. En TDEM el
        primer gate útil rara vez ve los primeros metros, así que el
        valor por omisión es mayor que en SEV y el resultado se debe
        leer con cautela (ver README).
    umbrales_corrosion : tuple
        Umbrales de clase de corrosividad (Ω·m).
    """

    procesamiento: ConfigProcesamiento = field(default_factory=ConfigProcesamiento)
    qc: ConfigQC = field(default_factory=ConfigQC)
    inversion: ConfigInversion = field(default_factory=ConfigInversion)
    autoguardado: bool = True
    prof_parametros: Optional[float] = None
    prof_corrosion: float = 10.0
    umbrales_corrosion: tuple = (5.0, 10.0, 20.0, 100.0)


#: Archivo (en la carpeta del proyecto, viaja dentro del .itdem) con los
#: parámetros de procesamiento e inversión de la campaña.
NOMBRE_PARAMETROS = "parametros_campana.json"


def config_a_dict(config: ConfigPipeline) -> dict[str, Any]:
    """Configuración completa como diccionario serializable (JSON)."""
    datos = asdict(config)
    datos["umbrales_corrosion"] = list(config.umbrales_corrosion)
    return datos


def config_desde_dict(datos: dict[str, Any]) -> ConfigPipeline:
    """Reconstruye la configuración; ignora claves desconocidas y completa
    las que falten con los valores por omisión (archivos de versiones
    anteriores siguen sirviendo)."""

    def filtrar(clase, valores):
        validos = {f.name for f in fields(clase)}
        return clase(**{k: v for k, v in (valores or {}).items() if k in validos})

    cfg = ConfigPipeline()
    cfg.procesamiento = filtrar(ConfigProcesamiento, datos.get("procesamiento"))
    cfg.qc = filtrar(ConfigQC, datos.get("qc"))
    cfg.inversion = filtrar(ConfigInversion, datos.get("inversion"))
    for clave in ("autoguardado", "prof_parametros", "prof_corrosion"):
        if clave in datos:
            setattr(cfg, clave, datos[clave])
        else:
            pass
    if "umbrales_corrosion" in datos:
        cfg.umbrales_corrosion = tuple(datos["umbrales_corrosion"])
    else:
        pass
    return cfg


def guardar_parametros(proyecto: Proyecto, config: ConfigPipeline) -> Path:
    """Guarda los parámetros de la campaña en el proyecto."""
    ruta = proyecto.ruta / NOMBRE_PARAMETROS
    ruta.write_text(json.dumps(config_a_dict(config), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return ruta


def cargar_parametros(proyecto: Proyecto) -> Optional[ConfigPipeline]:
    """Parámetros guardados de la campaña (None si no hay)."""
    ruta = proyecto.ruta / NOMBRE_PARAMETROS
    if not ruta.exists():
        return None
    else:
        pass
    try:
        return config_desde_dict(json.loads(ruta.read_text(encoding="utf-8")))
    except Exception as exc:
        logger.warning("No se pudieron leer los parámetros guardados: %s", exc)
        return None


ORDEN_ETAPAS = ("crudo", "apilado", "inversion", "parametros")

ETIQUETAS_ETAPA = {
    "crudo": "1 · Lectura",
    "apilado": "2 · Apilado y QC",
    "inversion": "3 · Inversión",
    "parametros": "4 · Parámetros",
}


def _cronometrar(funcion: Callable[[], ResultadoEtapa], etapa: str) -> ResultadoEtapa:
    """Ejecuta una etapa capturando tiempo, advertencias y errores."""
    inicio = time.time()
    with warnings.catch_warnings(record=True) as registro:
        warnings.simplefilter("always")
        try:
            resultado = funcion()
        except CancelacionSolicitada as exc:
            resultado = ResultadoEtapa(exito=False, etapa=etapa, mensaje=str(exc))
        except Exception as exc:  # la interfaz nunca debe caerse por una etapa
            logger.exception("Etapa %s falló", etapa)
            resultado = ResultadoEtapa(exito=False, etapa=etapa, mensaje=f"{type(exc).__name__}: {exc}")
    resultado.duracion_s = time.time() - inicio
    ruidosas = ("Degrees of freedom", "Mean of empty slice", "invalid value", "divide by zero",
                "overflow", "FutureWarning", "DeprecationWarning")
    resultado.advertencias += [
        str(w.message) for w in registro
        if not any(r in str(w.message) for r in ruidosas)
    ][:5]
    return resultado


def _requerir(proyecto: Proyecto, etapa: str, tabla: str, sondeo: str) -> pd.DataFrame:
    df = alm.cargar_tabla(proyecto, etapa, tabla, sondeo)
    if df is None:
        raise ValueError(f"El sondeo '{sondeo}' no tiene la tabla '{etapa}/{tabla}': ejecute la etapa antes.")
    else:
        return df


# ======================================================================
# Metadatos y geometría
# ======================================================================


def _meta_a_tabla(meta: dict[str, Any]) -> pd.DataFrame:
    """Metadatos como tabla de dos columnas (clave, valor en texto).

    Se guardan como texto porque mezclan números, fechas y cadenas; al
    leerlos se convierten de vuelta con :func:`meta_sondeo`.
    """
    filas = [{"clave": str(k), "valor": "" if v is None else str(v)} for k, v in meta.items()]
    return pd.DataFrame(filas)


def _texto_a_valor(texto: str) -> Any:
    if texto in ("", "None", "nan"):
        return None
    else:
        pass
    try:
        return float(texto)
    except ValueError:
        return texto


def meta_sondeo(proyecto: Proyecto, sondeo: str) -> dict[str, Any]:
    """Metadatos del sondeo leídos del archivo (diccionario)."""
    tabla = alm.cargar_tabla(proyecto, "crudo", "meta", sondeo)
    if tabla is None:
        return {}
    else:
        return {str(c): _texto_a_valor(str(v)) for c, v in zip(tabla["clave"], tabla["valor"])}


def geometria_sondeo(proyecto: Proyecto, sondeo: str):
    """Geometría del lazo para el modelado: catálogo (editable) + archivo.

    Los lados del lazo salen del catálogo (el usuario los puede corregir
    en la pantalla 1); la posición del receptor, de los metadatos.
    """
    import forward_tdem as fw

    catalogo = alm.listar_sondeos(proyecto)
    fila = catalogo[catalogo["sondeo"] == str(sondeo)]
    meta = meta_sondeo(proyecto, sondeo)

    def numero(valor, defecto):
        try:
            v = float(valor)
        except (TypeError, ValueError):
            return defecto
        return v if np.isfinite(v) else defecto

    lado_x = numero(fila["lado_x"].iloc[0] if len(fila) else meta.get("lado_x"), 40.0)
    lado_y = numero(fila["lado_y"].iloc[0] if len(fila) else meta.get("lado_y"), lado_x)
    vueltas = numero(fila["vueltas"].iloc[0] if len(fila) else meta.get("vueltas"), 1.0)
    return fw.GeometriaLazo(lado_x=lado_x, lado_y=lado_y, vueltas=vueltas,
                            rx_x=numero(meta.get("rx_x"), 0.0), rx_y=numero(meta.get("rx_y"), 0.0))


# ======================================================================
# Etapa 1: lectura
# ======================================================================


def ejecutar_lectura(proyecto: Proyecto, rutas: list[Union[str, Path]], guardar: bool = True) -> ResultadoEtapa:
    """Lee USF y XYZ (emparejados por la marca del stb) y guarda ``crudo``.

    Returns
    -------
    ResultadoEtapa
        ``datos['sondeos']``: lista de sondeos leídos.
    """

    def _ejecutar() -> ResultadoEtapa:
        import lectura_tdem as lec

        pares = lec.emparejar([Path(r) for r in rutas])
        if not pares:
            raise ValueError("No se encontraron archivos .usf ni .xyz entre los subidos.")
        else:
            pass
        nombres: list[str] = []
        advertencias: list[str] = []
        sin_coordenadas: list[str] = []

        # Un mismo sondeo puede llegar repetido (el usuario sube el USF dos
        # veces, o un USF suelto y otro emparejado con su XYZ). Se conserva
        # una sola lectura por nombre, prefiriendo la que trae XYZ (GPS y
        # hora por barrido).
        por_nombre: dict[str, Any] = {}
        for ruta_usf, ruta_xyz in pares:
            for tablas in lec.leer_par(ruta_usf, ruta_xyz):
                nombre = alm.sanear_id(tablas.nombre)
                previa = por_nombre.get(nombre)
                if previa is None:
                    por_nombre[nombre] = tablas
                else:
                    advertencias.append(f"{nombre}: archivo repetido; se usa una sola lectura.")
                    if previa.meta.get("archivo_xyz") is None and tablas.meta.get("archivo_xyz") is not None:
                        por_nombre[nombre] = tablas
                    else:
                        pass

        for nombre, tablas in por_nombre.items():
            advertencias += [f"{nombre}: {a}" for a in tablas.advertencias]
            if guardar:
                alm.eliminar_etapa(proyecto, "crudo", nombre)
                alm.registrar_sondeo(proyecto, nombre, tablas.meta)
                alm.guardar_tabla(proyecto, _meta_a_tabla(tablas.meta), "crudo", "meta", nombre)
                alm.guardar_tabla(proyecto, tablas.barridos, "crudo", "barridos", nombre,
                                  metricas={"n_barridos": len(tablas.barridos),
                                            "fuente": tablas.meta.get("fuente_principal", "")})
                alm.guardar_tabla(proyecto, tablas.gates, "crudo", "gates", nombre)
                alm.guardar_tabla(proyecto, tablas.ventanas, "crudo", "ventanas", nombre)
                if tablas.forma_onda is not None and len(tablas.forma_onda):
                    alm.guardar_tabla(proyecto, tablas.forma_onda, "crudo", "forma_onda", nombre)
                else:
                    pass
            else:
                pass
            nombres.append(nombre)
        if guardar:
            con_xy = set(alm.coordenadas(proyecto)["Sondeo"].astype(str))
            sin_coordenadas = [n for n in nombres if n not in con_xy]
        else:
            pass
        if sin_coordenadas:
            advertencias.append(f"Sin coordenadas en el archivo: {', '.join(sin_coordenadas)}. "
                                "Captúrelas en la tabla de coordenadas o súbalas en un archivo.")
        else:
            pass
        return ResultadoEtapa(
            exito=True, etapa="crudo",
            mensaje=f"{len(nombres)} sondeo(s) leído(s): {', '.join(nombres)}.",
            datos={"sondeos": nombres, "sin_coordenadas": sin_coordenadas},
            metricas={"n_sondeos": len(nombres)}, advertencias=advertencias,
        )

    return _cronometrar(_ejecutar, "crudo")


def cargar_coordenadas(proyecto: Proyecto, fuente: Union[str, Path, bytes, pd.DataFrame]) -> ResultadoEtapa:
    """Asigna coordenadas desde un archivo ``Sondeo, X, Y, Z[, azimut]`` o
    desde la tabla editada en la aplicación.

    X, Y van en el EPSG del proyecto. Si el proyecto aún no tiene EPSG
    (ningún archivo traía GPS) hay que fijarlo antes en la pantalla 1.
    """

    def _ejecutar() -> ResultadoEtapa:
        import lectura_tdem as lec

        if proyecto.epsg is None:
            raise ValueError("Fije primero el sistema de coordenadas (EPSG) del proyecto.")
        else:
            pass
        coords = fuente if isinstance(fuente, pd.DataFrame) else lec.leer_coordenadas(fuente)
        coords = coords.dropna(subset=["X", "Y"])
        n = alm.fijar_coordenadas(proyecto, coords)
        faltan = sorted(set(coords["Sondeo"].astype(str)) - set(alm.listar_sondeos(proyecto)["sondeo"]))
        adv = [f"Sondeos del archivo de coordenadas que no están en el proyecto: {faltan}"] if faltan else []
        return ResultadoEtapa(True, "coordenadas", f"Coordenadas asignadas a {n} sondeo(s).", advertencias=adv)

    return _cronometrar(_ejecutar, "coordenadas")


# ======================================================================
# Etapa 2: apilado y control de calidad
# ======================================================================


def exclusiones_sondeo(proyecto: Proyecto, sondeo: str) -> pd.DataFrame:
    """Gates excluidos a mano (``momento, gate, excluir``)."""
    tabla = alm.cargar_tabla(proyecto, "apilado", "exclusiones", sondeo)
    if tabla is None:
        return pd.DataFrame({"momento": pd.Series(dtype=str), "gate": pd.Series(dtype=int),
                             "excluir": pd.Series(dtype=bool)})
    else:
        return tabla


def guardar_exclusiones(proyecto: Proyecto, sondeo: str, exclusiones: pd.DataFrame) -> None:
    """Guarda la selección manual (solo las filas con ``excluir = True``)."""
    tabla = exclusiones[exclusiones["excluir"].astype(bool)][["momento", "gate", "excluir"]].copy()
    tabla["gate"] = tabla["gate"].astype(int)
    alm.guardar_tabla(proyecto, tabla.reset_index(drop=True), "apilado", "exclusiones", sondeo)


def ejecutar_apilado(proyecto: Proyecto, sondeo: str, config: Optional[ConfigPipeline] = None,
                     guardar: bool = True) -> ResultadoEtapa:
    """Apilado robusto, ρa, curva combinada y QC del sondeo."""
    cfg = config or ConfigPipeline()

    def _ejecutar() -> ResultadoEtapa:
        import procesamiento_tdem as pr
        import qc_tdem as qc

        gates = _requerir(proyecto, "crudo", "gates", sondeo)
        barridos = _requerir(proyecto, "crudo", "barridos", sondeo)
        ventanas = _requerir(proyecto, "crudo", "ventanas", sondeo)
        meta = meta_sondeo(proyecto, sondeo)
        geometria = geometria_sondeo(proyecto, sondeo)

        puntajes = pr.puntuar_barridos(gates, barridos, cfg.procesamiento, geometria.area, geometria.vueltas)
        apilado = pr.apilar(gates, barridos, cfg.procesamiento, geometria.area, geometria.vueltas, puntajes)
        apilado = pr.aplicar_exclusiones(apilado, exclusiones_sondeo(proyecto, sondeo), cfg.procesamiento)
        apilado = pr.agregar_rho_a(apilado, geometria.area, geometria.vueltas)
        curva, t_cambio = pr.combinar_momentos(apilado, cfg.procesamiento.factor_rampa)

        config_qc = cfg.qc
        config_qc.k_mad = cfg.procesamiento.k_mad
        hallazgos = qc.evaluar_sondeo(sondeo, meta, barridos, gates, ventanas, apilado, config_qc)
        resumen = qc.resumen(sondeo, hallazgos, apilado, barridos)
        total = resumen[resumen["momento"] == "TODOS"].iloc[0] if len(resumen) else None

        metricas = {
            "t_cambio": t_cambio,
            "n_para_invertir": int(apilado["usar"].sum()),
            "n_excluidos_usuario": int(apilado["excluido_usuario"].sum()),
            "puntaje": float(total["puntaje"]) if total is not None else np.nan,
            "n_barridos_descartados": int(puntajes["descartado"].sum()) if len(puntajes) else 0,
            "frac_barridos_descartados": float(puntajes["descartado"].mean()) if len(puntajes) else 0.0,
            "valoracion": str(total["valoracion"]) if total is not None else "",
            "config": json.dumps(cfg.procesamiento.as_dict(), default=str),
        }
        if guardar:
            alm.guardar_tabla(proyecto, apilado, "apilado", "apilado", sondeo, metricas=metricas)
            alm.guardar_tabla(proyecto, curva, "apilado", "curva", sondeo)
            alm.guardar_tabla(proyecto, hallazgos, "apilado", "qc", sondeo)
            alm.guardar_tabla(proyecto, resumen, "apilado", "resumen_qc", sondeo)
            alm.guardar_tabla(proyecto, puntajes, "apilado", "barridos_qc", sondeo)
        else:
            pass
        n_aviso = int((hallazgos["severidad"] == "WARN").sum())
        n_falla = int((hallazgos["severidad"] == "FAIL").sum())
        mensaje = (f"{metricas['n_barridos_descartados']} barrido(s) descartado(s) por IF+LOF "
                   f"({100 * metricas['frac_barridos_descartados']:.1f} %); "
                   f"{metricas['n_para_invertir']} gates para invertir "
                   f"({metricas['n_excluidos_usuario']} excluidos a mano); QC: {metricas['valoracion']} "
                   f"({n_falla} FAIL, {n_aviso} WARN).")
        return ResultadoEtapa(True, "apilado", mensaje, metricas=metricas,
                              datos={"apilado": apilado, "curva": curva, "qc": hallazgos, "resumen": resumen})

    return _cronometrar(_ejecutar, "apilado")


# ======================================================================
# Etapas 3 y 4
# ======================================================================


def ejecutar_inversion(proyecto: Proyecto, sondeo: str, config: Optional[ConfigPipeline] = None,
                       guardar: bool = True, token_cancelacion: Optional[TokenCancelacion] = None) -> ResultadoEtapa:
    """Inversión completa de un sondeo (multicapa, DOI y bloques)."""
    cfg = config or ConfigPipeline()

    def _ejecutar() -> ResultadoEtapa:
        import inversion_tdem as inv

        apilado = _requerir(proyecto, "apilado", "apilado", sondeo)
        barridos = _requerir(proyecto, "crudo", "barridos", sondeo)
        forma_onda = alm.cargar_tabla(proyecto, "crudo", "forma_onda", sondeo)
        if forma_onda is None:
            forma_onda = pd.DataFrame(columns=["momento", "punto", "tiempo", "amplitud"])
        else:
            pass
        geometria = geometria_sondeo(proyecto, sondeo)
        verificar = token_cancelacion.verificar if token_cancelacion is not None else None
        res = inv.invertir_sondeo(apilado, forma_onda, barridos, geometria, cfg.inversion,
                                  verificar_cancelacion=verificar)
        metricas = dict(res.metricas)
        # Profundidad de difusión del primer gate invertido: por encima de
        # ella el modelo no tiene control de los datos (útil para avisar de
        # la corrosividad del suelo, que en TDEM casi siempre se extrapola).
        usados = apilado[apilado["usar"].astype(bool)]
        metricas["prof_min_resuelta"] = float(np.nanmin(usados["prof_difusion"])) \
            if "prof_difusion" in usados and usados["prof_difusion"].notna().any() else np.nan
        metricas["config"] = json.dumps(cfg.inversion.as_dict(), default=str)
        if guardar:
            alm.eliminar_etapa(proyecto, "inversion", sondeo)
            for tabla, df in (("multicapa", res.multicapa), ("bloques", res.bloques), ("ajuste", res.ajuste),
                              ("seleccion", res.seleccion), ("iteraciones", res.iteraciones),
                              ("proceso", res.proceso)):
                if df is not None:
                    alm.guardar_tabla(proyecto, df, "inversion", tabla, sondeo)
                else:
                    pass
            alm.registrar_metricas(proyecto, sondeo, "inversion", metricas)
        else:
            pass
        m = res.metricas
        doi = f"{m['doi']:.0f} m" if np.isfinite(m.get("doi", np.nan)) else "sin calcular"
        mensaje = (f"Multicapa χ²={m['multicapa_chi2']:.2f} (λ={m['multicapa_lambda']:.3g}); "
                   f"bloques {m['bloques_n_capas']} capas χ²={m['bloques_chi2']:.2f}; DOI≈{doi}.")
        return ResultadoEtapa(True, "inversion", mensaje, metricas=metricas,
                              advertencias=res.advertencias, datos={"resultado": res})

    return _cronometrar(_ejecutar, "inversion")


def ejecutar_parametros(proyecto: Proyecto, sondeo: str, config: Optional[ConfigPipeline] = None,
                        guardar: bool = True) -> ResultadoEtapa:
    """Parámetros de Dar Zarrouk y corrosividad."""
    cfg = config or ConfigPipeline()

    def _ejecutar() -> ResultadoEtapa:
        import parametros_tdem as pa

        multicapa = _requerir(proyecto, "inversion", "multicapa", sondeo)
        bloques = _requerir(proyecto, "inversion", "bloques", sondeo)
        doi = metrica(proyecto, sondeo, "inversion", "doi")
        tabla, resumen = pa.parametros_sondeo(
            multicapa, bloques, profundidad=cfg.prof_parametros, doi=doi,
            prof_corrosion=cfg.prof_corrosion, umbrales_corrosion=cfg.umbrales_corrosion,
        )
        if guardar:
            alm.guardar_tabla(proyecto, tabla, "parametros", "parametros", sondeo, metricas=resumen)
        else:
            pass
        adv = []
        if abs(resumen["diferencia_S_pct"]) > 25 or abs(resumen["diferencia_T_pct"]) > 25:
            adv.append(
                f"S o T difieren más de un 25 % entre el multicapa y el modelo de bloques "
                f"(ΔS={resumen['diferencia_S_pct']:.0f} %, ΔT={resumen['diferencia_T_pct']:.0f} %): "
                "el modelo de bloques puede estar sobre-simplificado."
            )
        else:
            pass
        primer = metrica(proyecto, sondeo, "inversion", "prof_min_resuelta")
        if primer is not None and cfg.prof_corrosion < primer:
            adv.append(f"La profundidad de corrosividad ({cfg.prof_corrosion:g} m) es menor que la "
                       f"profundidad mínima resuelta (~{primer:.0f} m): la clase es una extrapolación.")
        else:
            pass
        mensaje = (f"S={resumen['S']:.3g} S, T={resumen['T']:.3g} Ω·m², ρm={resumen['rho_m']:.3g} Ω·m "
                   f"hasta {resumen['profundidad']:.0f} m; corrosividad {resumen['corrosividad'].lower()}.")
        return ResultadoEtapa(True, "parametros", mensaje, metricas=resumen, advertencias=adv,
                              datos={"tabla": tabla})

    return _cronometrar(_ejecutar, "parametros")


# ======================================================================
# Campaña
# ======================================================================


def ejecutar_qc_campana(proyecto: Proyecto) -> ResultadoEtapa:
    """Revisiones que comparan los sondeos entre sí."""

    def _ejecutar() -> ResultadoEtapa:
        import qc_tdem as qc

        coords = alm.coordenadas(proyecto)
        ventanas = alm.consultar_multisondeo(proyecto, "crudo", "ventanas")
        barridos = alm.consultar_multisondeo(proyecto, "crudo", "barridos")
        apilados = alm.consultar_multisondeo(proyecto, "apilado", "apilado")
        hallazgos = qc.revisiones_campana(coords, ventanas, barridos, apilados)
        alm.guardar_tabla(proyecto, hallazgos, "parametros", "qc_campana", alm.CAMPANA)
        n = int(hallazgos["severidad"].isin(["WARN", "FAIL"]).sum()) if len(hallazgos) else 0
        return ResultadoEtapa(True, "qc_campana", f"{len(hallazgos)} revisión(es) de campaña; {n} con aviso o falla.",
                              datos={"hallazgos": hallazgos})

    return _cronometrar(_ejecutar, "qc_campana")


def ejecutar_unidades(proyecto: Proyecto, n_max: int = 6) -> ResultadoEtapa:
    """Unidades geoeléctricas de toda la campaña (mezcla gaussiana)."""

    def _ejecutar() -> ResultadoEtapa:
        import parametros_tdem as pa

        capas = alm.consultar_multisondeo(proyecto, "inversion", "bloques")
        if len(capas) == 0:
            raise ValueError("Ningún sondeo tiene modelo de bloques todavía.")
        else:
            pass
        capas = capas.rename(columns={alm.PREFIJO_PARTICION: "Sondeo"})
        tabla, resumen = pa.unidades_geoelectricas(capas, n_max=n_max)
        alm.guardar_tabla(proyecto, tabla, "parametros", "unidades", alm.CAMPANA)
        alm.guardar_tabla(proyecto, resumen, "parametros", "resumen_unidades", alm.CAMPANA)
        return ResultadoEtapa(True, "unidades",
                              f"{tabla['unidad'].nunique()} unidad(es) geoeléctrica(s) en "
                              f"{tabla['Sondeo'].nunique()} sondeo(s).", datos={"resumen": resumen})

    return _cronometrar(_ejecutar, "unidades")


def ejecutar_perfil(proyecto: Proyecto, nombre: str, vertices: list[list[float]],
                    buffer: float = 100.0) -> ResultadoEtapa:
    """Proyecta los sondeos sobre la ruta y guarda el perfil.

    A diferencia de SEV no hay inversión lateralmente acoplada (LCI): la
    sección se arma interpolando los modelos 1D (ver README).
    """

    def _ejecutar() -> ResultadoEtapa:
        import perfil_tdem as pf

        coords = alm.coordenadas(proyecto)
        if len(coords) < 2:
            raise ValueError("Hacen falta coordenadas de al menos dos sondeos (pantalla 1).")
        else:
            pass
        proyectados = pf.proyectar(coords, vertices, buffer=buffer)
        dentro = proyectados[proyectados["dentro"]].reset_index(drop=True)
        alm.guardar_tabla(proyecto, proyectados, "perfiles", "perfil_sondeos", nombre)
        alm.guardar_tabla(proyecto, pd.DataFrame(vertices, columns=["X", "Y"]), "perfiles", "perfil_ruta", nombre)
        if len(dentro) < 2:
            raise ValueError(f"Solo {len(dentro)} sondeo(s) caen dentro del buffer de {buffer} m.")
        else:
            pass
        metricas = {"n_sondeos": len(dentro), "buffer": buffer, "longitud": proyectados.attrs.get("longitud")}
        alm.registrar_metricas(proyecto, nombre, "perfiles", metricas)
        return ResultadoEtapa(True, "perfil", f"{len(dentro)} sondeos proyectados sobre la ruta '{nombre}'.",
                              metricas=metricas)

    return _cronometrar(_ejecutar, "perfil")


def listar_perfiles(proyecto: Proyecto) -> list[str]:
    carpeta = proyecto.ruta_etapa("perfiles")
    prefijo = f"{alm.PREFIJO_PARTICION}="
    return sorted(p.name.split("=", 1)[1] for p in carpeta.glob(f"{prefijo}*") if (p / "perfil_ruta.parquet").exists())


# ======================================================================
# Pipeline
# ======================================================================


def ejecutar_pipeline_sondeo(
    proyecto: Proyecto, sondeo: str, config: Optional[ConfigPipeline] = None,
    desde_etapa: str = "apilado", hasta_etapa: str = "parametros",
    callback_progreso: Optional[Callable] = None, token_cancelacion: Optional[TokenCancelacion] = None,
) -> dict[str, ResultadoEtapa]:
    """Ejecuta las etapas ``desde..hasta`` (posteriores a la lectura)."""
    cfg = config or ConfigPipeline()
    orden = list(ORDEN_ETAPAS)
    a_ejecutar = orden[orden.index(desde_etapa): orden.index(hasta_etapa) + 1]
    a_ejecutar = [e for e in a_ejecutar if e != "crudo"]
    resultados: dict[str, ResultadoEtapa] = {}
    for indice, etapa in enumerate(a_ejecutar, start=1):
        if token_cancelacion is not None and token_cancelacion.cancelado:
            resultados[etapa] = ResultadoEtapa(False, etapa, "Ejecución detenida por el usuario.")
            break
        else:
            pass
        if etapa == "apilado":
            resultado = ejecutar_apilado(proyecto, sondeo, cfg, guardar=cfg.autoguardado)
        else:
            if etapa == "inversion":
                resultado = ejecutar_inversion(proyecto, sondeo, cfg, guardar=cfg.autoguardado,
                                               token_cancelacion=token_cancelacion)
            else:
                resultado = ejecutar_parametros(proyecto, sondeo, cfg, guardar=cfg.autoguardado)
        resultado.mensaje = f"[{sondeo}] {resultado.mensaje}"
        resultados[etapa] = resultado
        if callback_progreso is not None:
            callback_progreso(etapa, indice, len(a_ejecutar), resultado)
        else:
            pass
        if not resultado:
            break
        else:
            pass
    return resultados


def ejecutar_pipeline_archivos(
    proyecto: Proyecto, rutas: list[Union[str, Path]], config: Optional[ConfigPipeline] = None,
    hasta_etapa: str = "apilado", callback_progreso: Optional[Callable] = None,
    token_cancelacion: Optional[TokenCancelacion] = None,
) -> dict[str, Any]:
    """Lee archivos y procesa todos sus sondeos hasta ``hasta_etapa``.

    Por omisión se detiene en el apilado: la inversión conviene lanzarla
    después de revisar el QC y excluir gates (pantalla 2).

    Returns
    -------
    dict
        ``{'lectura': ResultadoEtapa, 'sondeos': {nombre: {etapa: ResultadoEtapa}}}``
    """
    cfg = config or ConfigPipeline()
    lectura = ejecutar_lectura(proyecto, rutas, guardar=True)
    salida: dict[str, Any] = {"lectura": lectura, "sondeos": {}}
    if callback_progreso is not None:
        callback_progreso("Lectura", 0.1, lectura)
    else:
        pass
    if not lectura or hasta_etapa == "crudo":
        return salida
    else:
        pass
    nombres = lectura.datos["sondeos"]
    for i, nombre in enumerate(nombres):
        def _cb(etapa, indice, total, resultado, _i=i, _nombre=nombre):
            if callback_progreso is not None:
                fraccion = 0.1 + 0.9 * (_i + indice / total) / len(nombres)
                callback_progreso(f"{_nombre} · {ETIQUETAS_ETAPA[etapa]}", fraccion, resultado)
            else:
                pass

        salida["sondeos"][nombre] = ejecutar_pipeline_sondeo(proyecto, nombre, cfg, "apilado", hasta_etapa,
                                                             _cb, token_cancelacion)
        if token_cancelacion is not None and token_cancelacion.cancelado:
            break
        else:
            pass
    if len(nombres) > 1:
        ejecutar_qc_campana(proyecto)
    else:
        pass
    return salida


# ======================================================================
# Consultas
# ======================================================================


def estado_sondeo(proyecto: Proyecto, sondeo: str) -> dict[str, bool]:
    return {
        "crudo": alm.existe_tabla(proyecto, "crudo", "gates", sondeo),
        "apilado": alm.existe_tabla(proyecto, "apilado", "apilado", sondeo),
        "inversion": alm.existe_tabla(proyecto, "inversion", "bloques", sondeo),
        "parametros": alm.existe_tabla(proyecto, "parametros", "parametros", sondeo),
    }


def metrica(proyecto: Proyecto, sondeo: str, etapa: str, clave: str) -> Optional[float]:
    df = alm.obtener_metricas(proyecto, sondeo, etapa)
    fila = df[df["clave"] == clave]
    if len(fila) == 0 or pd.isna(fila["valor_num"].iloc[0]):
        return None
    else:
        return float(fila["valor_num"].iloc[0])


def metrica_texto(proyecto: Proyecto, sondeo: str, etapa: str, clave: str) -> Optional[str]:
    df = alm.obtener_metricas(proyecto, sondeo, etapa)
    fila = df[df["clave"] == clave]
    if len(fila) == 0:
        return None
    else:
        valor = fila["valor_texto"].iloc[0]
        return None if pd.isna(valor) else str(valor)


def tabla_estado(proyecto: Proyecto) -> pd.DataFrame:
    """Estado de todos los sondeos (una fila por sondeo) para las pantallas."""
    filas = []
    for _, s in alm.listar_sondeos(proyecto).iterrows():
        nombre = str(s["sondeo"])
        estado = estado_sondeo(proyecto, nombre)
        filas.append({
            "Sondeo": nombre, "Inicio": s["inicio"], "Barridos": s["n_barridos"],
            "Lazo (m)": f"{s['lado_x']:g} × {s['lado_y']:g}" if pd.notna(s["lado_x"]) and pd.notna(s["lado_y"]) else "—",
            "Coordenadas": {"archivo": "archivo", "usuario": "capturadas"}.get(s["origen_coord"], "faltan"),
            "QC": metrica_texto(proyecto, nombre, "apilado", "valoracion") or "—",
            **{ETIQUETAS_ETAPA[e]: "✓" if v else "" for e, v in estado.items()},
        })
    return pd.DataFrame(filas)
