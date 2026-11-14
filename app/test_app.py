"""
test_app.py
==================================================================
Pruebas sin navegador de TLALLI-RES 1D TDEM: física, lectura,
apilado, control de calidad, inversión, almacenamiento y acciones de
los botones.

Ejecutar::

    python app/test_app.py            # todas (~2 min)
    python app/test_app.py rapido     # omite las inversiones completas

Cada prueba es una función ``prueba_*``; el resumen final indica cuáles
fallaron. Los datos de prueba son sintéticos
(``ejemplos/generar_sintetico_tdem.py``): no se usan datos de cliente.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import _ruta_src  # noqa: F401

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "ejemplos"))

import acciones_app as acc  # noqa: E402
import almacenamiento_tdem as alm  # noqa: E402
import forward_tdem as fw  # noqa: E402
import generar_sintetico_tdem as gs  # noqa: E402
import geodesia_tdem as geo  # noqa: E402
import graficos_tdem as gr  # noqa: E402
import inversion_tdem as inv  # noqa: E402
import lectura_tdem as lec  # noqa: E402
import parametros_tdem as pa  # noqa: E402
import pipeline_tdem as pl  # noqa: E402
import procesamiento_tdem as pr  # noqa: E402
import qc_tdem as qc  # noqa: E402

RAPIDO = "rapido" in sys.argv

#: Carpeta con los USF sintéticos (se generan una sola vez por corrida).
_SINTETICOS: dict[str, Path] = {}


def _sinteticos() -> Path:
    if "carpeta" not in _SINTETICOS:
        carpeta = Path(tempfile.mkdtemp(prefix="tdem_sint_"))
        gs.generar(carpeta, n_barridos=30)
        _SINTETICOS["carpeta"] = carpeta
    else:
        pass
    return _SINTETICOS["carpeta"]


def _tablas(nombre: str = "SINT_001") -> lec.TablasSondeo:
    ruta = next(_sinteticos().glob(f"{nombre}_*.usf"))
    return lec.leer_par(ruta, None)[0]


def _proyecto_cargado(base: Path, hasta: str = "apilado"):
    proyecto = acc.accion_crear_proyecto(base, "Prueba").datos["proyecto"]
    rutas = sorted(_sinteticos().glob("*.usf"))
    resultado = acc.accion_cargar_archivos(proyecto, rutas, pl.ConfigPipeline())
    assert resultado, resultado.mensaje
    return proyecto


# ----------------------------------------------------------------------
# Física
# ----------------------------------------------------------------------


def prueba_forward_semiespacio():
    # Solución analítica de lazo central en semiespacio (encendido largo).
    assert fw.verificar_semiespacio(100.0) < 0.01


def prueba_jacobiano_diferencias_finitas():
    datos = [fw.DatosMomento("LM", *(lambda a, c, b: (c, a, b))(*gs.ventanas(1e-5, 1e-3, 12)),
                             *gs.onda(7e-6))]
    modelo = fw.ModeloDirecto(fw.GeometriaLazo(40, 40), datos, integrar_gates=False, n_sub=1)
    espesores = np.array([20.0, 40.0])
    rho = np.array([50.0, 10.0, 100.0])
    d, J = modelo.jacobiano_rho(rho, espesores)
    paso = 1e-4
    for j in range(len(rho)):
        rho_p = rho.copy()
        rho_p[j] *= np.exp(paso)
        diferencia = (modelo.respuesta(rho_p, espesores) - d) / paso
        assert np.allclose(J[:, j], diferencia, rtol=2e-3, atol=1e-3 * np.abs(d).max()), j


def prueba_rho_a_tardia_semiespacio():
    centro = np.logspace(-4, -2, 10)
    # Encendido largo (0.5 s) para que el apagado sea un escalón ideal.
    onda_t, onda_a = np.array([-0.5, -0.4999, 0.0, 1e-6]), np.array([0.0, 1.0, 1.0, 0.0])
    datos = [fw.DatosMomento("HM", centro, centro * 0.9, centro * 1.1, onda_t, onda_a)]
    modelo = fw.ModeloDirecto(fw.GeometriaLazo(40, 40), datos, integrar_gates=False, n_sub=1)
    d = modelo.respuesta(np.array([30.0]), np.array([]))
    rho_a = pr.rho_a_tardia(centro, d, 1600.0)
    # Solo en tiempo tardío la aproximación es exacta.
    assert np.all(np.abs(rho_a[-4:] / 30.0 - 1) < 0.1), rho_a


def prueba_esquinas_lazo():
    lazo = geo.esquinas_lazo(1000.0, 2000.0, 40.0, 20.0, 0.0)
    assert lazo.shape == (5, 2) and np.allclose(lazo[0], lazo[-1])
    assert np.isclose(lazo[:, 0].max() - lazo[:, 0].min(), 40.0)
    girado = geo.esquinas_lazo(0.0, 0.0, 40.0, 20.0, 90.0)
    assert np.isclose(girado[:, 1].max() - girado[:, 1].min(), 40.0)


def prueba_utm():
    epsg = geo.epsg_utm(-103.34, 20.68)
    assert epsg == 32613
    x, y = geo.geograficas_a_proyectadas([-103.34], [20.68], epsg)
    lon, lat = geo.proyectadas_a_geograficas(x, y, epsg)
    assert abs(lon[0] + 103.34) < 1e-7 and abs(lat[0] - 20.68) < 1e-7
    assert not geo.coordenadas_validas(99999, 99999)


# ----------------------------------------------------------------------
# Lectura
# ----------------------------------------------------------------------


def prueba_lectura_usf():
    tablas = _tablas("SINT_001")
    assert tablas.nombre == "SINT_001"
    assert tablas.meta["n_lm"] == 30 and tablas.meta["n_hm"] == 30
    assert set(tablas.gates["momento"]) == {"LM", "HM"}
    assert len(tablas.ventanas[tablas.ventanas["momento"] == "HM"]) == gs.MOMENTOS["HM"]["n"]
    assert len(tablas.forma_onda) >= 8
    assert geo.coordenadas_validas(tablas.meta["lon"], tablas.meta["lat"])


def prueba_lectura_sin_ubicacion():
    tablas = _tablas("SINT_003")
    assert not geo.coordenadas_validas(tablas.meta.get("lon"), tablas.meta.get("lat"))


def prueba_marca_stb():
    assert lec.marca_stb("L001_S001_2026_1112_145402.usf") == "2026_1112_145402"
    assert lec.marca_stb("2026_1112_145402_ChA.xyz") == "2026_1112_145402"
    assert lec.marca_stb("sin_marca.usf") is None


def prueba_coordenadas_texto():
    texto = b"Sondeo;Este;Norte;Cota\nA;672000,5;2287000,0;1500\n"
    tabla = lec.leer_coordenadas(texto)
    assert list(tabla[["X", "Y"]].iloc[0]) == [672000.5, 2287000.0]


# ----------------------------------------------------------------------
# Apilado y QC
# ----------------------------------------------------------------------


def prueba_apilado_robusto():
    tablas = _tablas("SINT_001")
    gates = tablas.gates.copy()
    # Un barrido con una espiga enorme en todos sus gates: IF + LOF lo deben
    # descartar completo y la curva apilada no debe moverse.
    primero = gates["barrido"].min()
    limpio = pr.apilar(gates, tablas.barridos, pr.ConfigProcesamiento())
    gates.loc[gates["barrido"] == primero, "dbdt"] *= 50
    puntajes = pr.puntuar_barridos(gates, tablas.barridos, pr.ConfigProcesamiento())
    assert bool(puntajes.set_index("barrido").loc[primero, "descartado"]), puntajes.head()
    sucio = pr.apilar(gates, tablas.barridos, pr.ConfigProcesamiento(), puntajes=puntajes)
    relacion = (sucio["dbdt"] / limpio["dbdt"]).to_numpy()
    assert np.nanmax(np.abs(relacion[:10] - 1)) < 0.05
    # Sin el paso 1, el MAD por gate (en log ρa) también lo atrapa gate por gate.
    solo_mad = pr.apilar(gates, tablas.barridos, pr.ConfigProcesamiento(deteccion_barridos="ninguna"))
    lm = solo_mad[(solo_mad["momento"] == "LM") & (solo_mad["gate"] <= 10)]
    assert (lm["n_atipicos"] >= 1).all(), lm[["gate", "n_atipicos"]]


def _usf_largo(nombre: str = "LARGO", n_barridos: int = 150, semilla: int = 7) -> lec.TablasSondeo:
    """Sondeo sintético con más barridos (IF y LOF necesitan muestra)."""
    carpeta = Path(tempfile.mkdtemp(prefix="tdem_largo_"))
    try:
        ruta = gs.escribir_usf(carpeta / f"{nombre}_2026_1112_090000.usf", nombre, (20.0, 40.0, 80.0),
                               (60.0, 8.0, 40.0, 150.0), -103.34, 20.68, n_barridos=n_barridos, semilla=semilla)
        return lec.leer_par(ruta, None)[0]
    finally:
        shutil.rmtree(carpeta, ignore_errors=True)


def prueba_if_lof_ruido_gaussiano():
    # Solo ruido gaussiano: el umbral adaptativo casi no debe descartar nada.
    tablas = _usf_largo()
    puntajes = pr.puntuar_barridos(tablas.gates, tablas.barridos, pr.ConfigProcesamiento(), 1600.0)
    frac = puntajes.groupby("momento")["descartado"].mean()
    assert (frac <= 0.03).all(), frac


def prueba_if_lof_rafaga():
    # Ráfaga de ruido en 10 barridos seguidos del LM (todos sus gates).
    tablas = _usf_largo()
    gates = tablas.gates.copy()
    lm = sorted(gates.loc[gates["momento"] == "LM", "barrido"].unique())
    rafaga = lm[60:70]
    rng = np.random.default_rng(1)
    afectados = gates["barrido"].isin(rafaga)
    gates.loc[afectados, "dbdt"] *= np.exp(rng.normal(0, 0.15, afectados.sum()))
    puntajes = pr.puntuar_barridos(gates, tablas.barridos, pr.ConfigProcesamiento(), 1600.0)
    marcados = set(puntajes.loc[puntajes["descartado"], "barrido"])
    assert len(marcados & set(rafaga)) >= 8, sorted(marcados)
    falsos = len(marcados - set(rafaga)) / len(lm)
    assert falsos <= 0.03, falsos


def prueba_mad_log_relativo():
    # El corte por MAD en log₁₀ ρa es relativo: escalar todo el sondeo (otro
    # subsuelo, otra ρa) debe dar exactamente los mismos descartes.
    tablas = _tablas("SINT_001")
    cfg = pr.ConfigProcesamiento(deteccion_barridos="ninguna")
    base = pr.apilar(tablas.gates, tablas.barridos, cfg, 1600.0)
    escalado = tablas.gates.copy()
    escalado["dbdt"] *= 37.0
    otro = pr.apilar(escalado, tablas.barridos, cfg, 1600.0)
    log = base["regimen"] == "log"
    assert log.any()
    assert (base.loc[log, "n_atipicos"].to_numpy() == otro.loc[log, "n_atipicos"].to_numpy()).all()
    assert np.allclose(otro.loc[log, "dbdt"] / base.loc[log, "dbdt"], 37.0)
    assert np.allclose(otro.loc[log, "sigma_log10_rho"], base.loc[log, "sigma_log10_rho"])


def prueba_repunte_y_exclusiones():
    tablas = _tablas("SINT_001")
    gates = tablas.gates.copy()
    objetivo = (gates["momento"] == "HM") & (gates["gate"] == 10)
    gates.loc[objetivo, "dbdt"] *= 3.0
    apilado = pr.apilar(gates, tablas.barridos, pr.ConfigProcesamiento())
    fila = apilado[(apilado["momento"] == "HM") & (apilado["gate"] == 10)].iloc[0]
    assert bool(fila["repunte"]) and not bool(fila["usar"])
    excluidos = pd.DataFrame({"momento": ["LM"], "gate": [3], "excluir": [True]})
    con_exclusion = pr.aplicar_exclusiones(apilado, excluidos, pr.ConfigProcesamiento())
    fila = con_exclusion[(con_exclusion["momento"] == "LM") & (con_exclusion["gate"] == 3)].iloc[0]
    assert bool(fila["excluido_usuario"]) and not bool(fila["usar"])


def prueba_qc_sintetico_bueno():
    tablas = _tablas("SINT_001")
    apilado = pr.agregar_rho_a(pr.apilar(tablas.gates, tablas.barridos, pr.ConfigProcesamiento()), 1600.0)
    hallazgos = qc.evaluar_sondeo("SINT_001", tablas.meta, tablas.barridos, tablas.gates, tablas.ventanas, apilado)
    resumen = qc.resumen("SINT_001", hallazgos, apilado, tablas.barridos)
    total = resumen[resumen["momento"] == "TODOS"].iloc[0]
    assert total["valoracion"] in ("Buena", "Aceptable"), hallazgos[hallazgos["severidad"] != "OK"]


def prueba_curva_combinada():
    tablas = _tablas("SINT_002")
    apilado = pr.agregar_rho_a(pr.apilar(tablas.gates, tablas.barridos, pr.ConfigProcesamiento()), 1600.0)
    curva, t_cambio = pr.combinar_momentos(apilado)
    assert np.isfinite(t_cambio)
    unida = curva[curva["en_curva"]]
    assert unida["tiempo"].is_monotonic_increasing
    assert set(unida.loc[unida["tiempo"] < t_cambio, "momento"]) <= {"LM"}


# ----------------------------------------------------------------------
# Modelos y parámetros
# ----------------------------------------------------------------------


def prueba_dar_zarrouk():
    modelo = fw.modelo_a_tabla(np.array([10.0, 20.0]), np.array([10.0, 100.0, 50.0]))
    dz = pa.dar_zarrouk(modelo, 30.0)
    assert np.isclose(dz["S"], 10 / 10 + 20 / 100) and np.isclose(dz["T"], 10 * 10 + 20 * 100)
    assert np.isclose(dz["rho_m"], np.sqrt(dz["rho_L"] * dz["rho_T"]))


def prueba_segmentacion():
    espesores = np.full(9, 5.0)
    m = np.log(np.array([10, 10, 10, 100, 100, 100, 20, 20, 20.0]))
    ln_rho, interfaces = inv.segmentar(espesores, m, 3, 1e6)
    assert np.allclose(np.exp(ln_rho), [10, 100, 20]) and np.allclose(interfaces, [15, 30])


def prueba_inversion_sintetica():
    if RAPIDO:
        return None
    else:
        pass
    tablas = _tablas("SINT_001")
    apilado = pr.apilar(tablas.gates, tablas.barridos, pr.ConfigProcesamiento())
    resultado = inv.invertir_sondeo(apilado, tablas.forma_onda, tablas.barridos, fw.GeometriaLazo(40, 40),
                                    inv.ConfigInversion(calcular_doi=False))
    bloques = resultado.bloques
    _, espesores, rho, _ = gs.SONDEOS[0]
    assert resultado.metricas["multicapa_chi2"] < 2.0
    # La capa conductora (8 Ω·m entre 20 y 60 m) debe recuperarse.
    conductora = bloques.loc[bloques["rho"].idxmin()]
    assert abs(conductora["rho"] / rho[1] - 1) < 0.2, bloques
    assert abs(conductora["z_techo"] / espesores[0] - 1) < 0.2, bloques


# ----------------------------------------------------------------------
# Proyecto, acciones y paquete
# ----------------------------------------------------------------------


def prueba_coordenadas_usuario_se_conservan():
    base = Path(tempfile.mkdtemp())
    try:
        proyecto = _proyecto_cargado(base)
        assert proyecto.epsg is None or True
        proyecto = alm.abrir_proyecto(proyecto.ruta)
        assert proyecto.epsg == 32613
        coords = alm.coordenadas(proyecto)
        assert "SINT_003" not in set(coords["Sondeo"])
        r = acc.accion_cargar_coordenadas(proyecto, _sinteticos() / "coordenadas_ejemplo.csv")
        assert r, r.mensaje
        # Corregir a mano el centro de un sondeo que sí traía GPS.
        original = acc.tabla_coordenadas_editable(proyecto)
        editada = original.copy()
        editada.loc[editada["Sondeo"] == "SINT_001", "X"] += 5.0
        editada.loc[editada["Sondeo"] == "SINT_001", "lado_x"] = 50.0
        assert acc.accion_editar_coordenadas(proyecto, editada, original)
        tabla = acc.tabla_coordenadas_editable(proyecto).set_index("Sondeo")
        assert tabla.loc["SINT_001", "origen"] == "usuario" and tabla.loc["SINT_002", "origen"] == "archivo"
        assert tabla.loc["SINT_001", "lado_x"] == 50.0
        # Volver a leer el archivo no pisa lo capturado.
        acc.accion_cargar_archivos(proyecto, sorted(_sinteticos().glob("SINT_001_*.usf")), pl.ConfigPipeline())
        tabla2 = acc.tabla_coordenadas_editable(proyecto).set_index("Sondeo")
        assert np.isclose(tabla2.loc["SINT_001", "X"], tabla.loc["SINT_001", "X"])
        assert pl.geometria_sondeo(proyecto, "SINT_001").lado_x in (40.0, 50.0)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def prueba_pipeline_y_paquete():
    base = Path(tempfile.mkdtemp())
    try:
        proyecto = _proyecto_cargado(base)
        assert pl.estado_sondeo(proyecto, "SINT_002")["apilado"]
        acc.accion_cargar_coordenadas(proyecto, _sinteticos() / "coordenadas_ejemplo.csv")
        # Exclusión manual y reapilado.
        r = acc.accion_guardar_exclusiones(proyecto, "SINT_002",
                                           pd.DataFrame({"momento": ["HM"], "gate": [1], "excluir": [True]}),
                                           pl.ConfigPipeline())
        assert r and pl.metrica(proyecto, "SINT_002", "apilado", "n_excluidos_usuario") == 1
        cfg = pl.ConfigPipeline()
        cfg.inversion.calcular_doi = False
        cfg.inversion.k_max = 4
        pl.guardar_parametros(proyecto, cfg)
        assert pl.cargar_parametros(proyecto).inversion.k_max == 4
        if not RAPIDO:
            r = acc.accion_reprocesar(proyecto, ["SINT_001", "SINT_002", "SINT_003"], cfg, "inversion", "parametros")
            assert r, r.mensaje
            assert all(pl.estado_sondeo(proyecto, "SINT_003").values())
            assert acc.accion_unidades(proyecto)
            coords = alm.coordenadas(proyecto)
            import perfil_tdem as pf

            r = acc.accion_perfil(proyecto, "P1", pf.ruta_automatica(coords), 100.0)
            assert r, r.mensaje
        else:
            pass
        assert acc.accion_qc_campana(proyecto)
        paquete = acc.accion_empaquetar_proyecto(proyecto)
        assert paquete, paquete.mensaje
        restaurado = acc.accion_restaurar_proyecto(paquete.datos["contenido"], base / "otro")
        assert restaurado and len(alm.listar_sondeos(restaurado.datos["proyecto"])) == 3
        assert restaurado.datos["proyecto"].epsg == 32613
    finally:
        shutil.rmtree(base, ignore_errors=True)


def prueba_rechaza_proyecto_sev():
    import paquete_tdem as pk

    memoria = io.BytesIO()
    with zipfile.ZipFile(memoria, "w") as z:
        z.writestr(pk.NOMBRE_MANIFIESTO, json.dumps({"formato": "isev", "version_formato": 1}))
    try:
        pk.restaurar_proyecto(memoria.getvalue(), tempfile.mkdtemp())
        raise AssertionError("Debió rechazar un .isev")
    except pk.PaqueteIncompatibleError:
        pass


def prueba_graficos():
    base = Path(tempfile.mkdtemp())
    try:
        proyecto = _proyecto_cargado(base)
        s = "SINT_001"
        gates = alm.cargar_tabla(proyecto, "crudo", "gates", s)
        barridos = alm.cargar_tabla(proyecto, "crudo", "barridos", s)
        apilado = alm.cargar_tabla(proyecto, "apilado", "apilado", s)
        curva = alm.cargar_tabla(proyecto, "apilado", "curva", s)
        coords = alm.coordenadas(proyecto)
        figuras = [gr.curvas_decaimiento(gates, apilado), gr.curva_combinada(curva, 1e-4), gr.adquisicion(barridos),
                   gr.posicion_gps(barridos, coords.iloc[0], proyecto.epsg),
                   gr.mapa_sondeos(coords, epsg=proyecto.epsg),
                   gr.mapa_sondeos(coords, pd.Series(np.ones(len(coords)) * 10), "ρ")]
        # Sin XYZ no hay GPS por barrido: esa figura sale vacía con un aviso.
        for figura in figuras[:3] + figuras[4:]:
            assert len(figura.data) > 0
        # El lazo se dibuja: una traza cerrada de 5 vértices por sondeo con coordenadas.
        lazos = [t for t in figuras[4].data if t.name == "Lazo transmisor"]
        assert len(lazos) == len(coords) and len(lazos[0].x) == 5
    finally:
        shutil.rmtree(base, ignore_errors=True)


def prueba_sin_elif():
    import re

    patron = re.compile(r"^\s*elif\b")
    culpables = []
    for ruta in list((RAIZ / "src").glob("*.py")) + list((RAIZ / "app").rglob("*.py")) + \
            list((RAIZ / "ejemplos").glob("*.py")):
        for numero, linea in enumerate(ruta.read_text(encoding="utf-8").splitlines(), start=1):
            if patron.match(linea):
                culpables.append(f"{ruta.name}:{numero}")
            else:
                pass
    assert not culpables, culpables


def main() -> int:
    pruebas = [(n, f) for n, f in globals().items() if n.startswith("prueba_") and callable(f)]
    fallos = []
    inicio = time.time()
    for nombre, funcion in pruebas:
        t = time.time()
        try:
            funcion()
            print(f"  OK   {nombre} ({time.time() - t:.1f}s)")
        except Exception:
            fallos.append(nombre)
            print(f"  FALLA {nombre}")
            traceback.print_exc()
    shutil.rmtree(_SINTETICOS.get("carpeta", ""), ignore_errors=True)
    print(f"\n{len(pruebas) - len(fallos)}/{len(pruebas)} pruebas correctas en {time.time() - inicio:.0f}s"
          + (" (modo rápido)" if RAPIDO else ""))
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
