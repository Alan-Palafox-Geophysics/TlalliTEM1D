# TLALLI-RES 1D TDEM: Sondeos electromagnéticos en el dominio del tiempo

Esta aplicación Streamlit procesa e invierte sondeos TDEM de lazo central (ABEM WalkTEM y
compatibles). Es hermana de TlalliRes1D (SEV/IP) y comparte con ella la arquitectura, la
autenticación y las licencias, las imágenes, el proyecto descargable y la regla de estilo. Lo
que cambia es la física: la lectura sale del proyecto *temdb* (USF y XYZ) y la inversión se
hace con **SimPEG**.

```
USF (prioridad) + XYZ opcional, emparejados por la marca del stb
→ barridos atípicos (Isolation Forest + LOF, umbral adaptativo)
→ MAD por gate en log₁₀ ρa (k = 3) → media lineal, error estándar, SNR
→ control de calidad: 18 revisiones por sondeo + revisiones de campaña
→ exclusión manual de gates → curva combinada LM + HM y ρa tardía
→ 1. multicapa suave (SimPEG, Gauss–Newton con λ enfriándose)
→ 2. DOI de Oldenburg–Li
→ 3. modelo de bloques (k = 2…6, BIC/AIC, rangos ±1σ)
→ S, T, ρL, ρT, λ, ρm, corrosividad → unidades geoeléctricas (GMM)
→ perfiles 2D: pseudosección y secciones interpoladas
```

## Instalación y ejecución

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Las credenciales se leen igual que en TlalliRes1D. Las fuentes son las variables
`TRE_CREDENCIALES_YAML` o `TRE_CREDENCIALES_RUTA`, los secretos de Streamlit o
`app/credenciales.yaml`. El archivo incluido es la copia de la versión SEV, así que
sirven las mismas cuentas.

Pruebas sin navegador:

```bash
python app/test_app.py          # 26 pruebas, ~3 min
python app/test_app.py rapido   # sin las inversiones completas, ~6 s
```

Datos de ejemplo: `python ejemplos/generar_sintetico_tdem.py` escribe en
`ejemplos/sinteticos/` tres USF sintéticos (40 barridos LM + 40 HM, con ruido) y el archivo
`coordenadas_ejemplo.csv`. El tercer sondeo no trae GPS, para probar la captura manual de
coordenadas.

## ¿SimPEG o pyGIMLi?

El **modelado directo y el jacobiano** se calculan con **SimPEG**: `Simulation1DLayered` del módulo
`electromagnetics.time_domain`. El ciclo de inversión (Occam, DOI, bloques) es código propio en numpy
y scipy. Las razones:

- **La física del equipo.** SimPEG modela de forma nativa un lazo cuadrado real (`LineCurrent`), el
  receptor en su posición, una forma de onda arbitraria (`PiecewiseLinearWaveform` con la `TX_RAMP` de
  cada momento) e integra cada gate en su ventana. Con un WalkTEM, la rampa de apagado y el ancho de
  los gates pesan mucho en los tiempos tempranos.
- **El soporte de pyGIMLi para TDEM es más limitado.** Su fuerte es la inversión (marcos de
  regularización, ERT 2D/3D, inversión acoplada), y por eso es lo que usa TlalliRes1D SEV. Hasta donde
  conozco, su modelado TDEM 1D es más básico (dipolo, ρa) y para formas de onda reales habría que
  acoplarlo con empymod.
- **Instalación.** SimPEG se instala con pip; pyGIMLi se instala más cómodamente con conda.
- **Auditoría.** Se usa solo la física de SimPEG, no su marco de inversión por *directivas*. El ciclo
  propio deja explícita cada decisión (elección de λ, límites, DOI, bloques) en unas decenas de líneas
  revisables, y la pantalla 3 la reporta en «Proceso de inversión aplicado».

## Pantallas

| # | Pantalla | Qué hace |
|---|---|---|
| 1 | Proyecto y datos | Crea o abre el `.itdem` y sube varios USF/XYZ. Edita X, Y, Z, el tamaño del lazo y su azimut; fija el EPSG si ningún archivo trae GPS. Muestra el **mapa con el lazo de cada sondeo dibujado a escala**, la corriente por barrido y el GPS contra el lazo. |
| 2 | Control de calidad | Ajustes del apilado, valoración y puntaje, curvas de decaimiento (nube de barridos + apilado), curva combinada, hallazgos filtrables, **editor de exclusión de gates** y revisiones de campaña. |
| 3 | Inversión | Configuración de los tres pasos. Muestra el ajuste por momento (dB/dt o ρa) con residuales, los modelos con DOI, la convergencia, la selección de capas, Dar Zarrouk acumulado y la corrosividad. |
| 4 | Parámetros y unidades | Resumen y mapas de campaña, capas con clase de corrosividad y unidades geoeléctricas. |
| 5 | Perfiles 2D | Ruta por vértices y mapa con lazos, pseudosección y secciones de los modelos multicapa, de bloques y de referencia. |
| 6 | Exportar datos | CSV por tabla con selección de columnas, y modelos en texto. |

## Datos de entrada

- **USF** (`stb2UFS`). Cada barrido es un stack crudo: el canal 1 es el LM y el canal 2 el
  HM. Se leen `CURRENT`, `RAMP_TIME`, `TX_RAMP` (la forma de onda), `TIME_DELAY`,
  `FIELD_SHIFT_FACTOR`, `LOOP_SIZE`, `COIL_LOCATION`, `LOCATION` y las ventanas
  `TIMEOPEN/TIMECLOSE`. El valor dummy (`//DUMMY`) se convierte en vacío.
- **XYZ** (`stb2xyz`, opcional). Aporta hora, corriente y GPS por barrido. Sus
  `dbdtDat` están normalizados solo por el área del receptor, así que se dividen entre la
  corriente antes de compararlos con el USF. Un XYZ sin USF también se procesa.
- **Duplicados.** Si el mismo sondeo llega dos veces, se conserva una sola lectura y se
  prefiere la que trae XYZ.
- **Coordenadas.** Si el archivo trae `LOCATION` válida, se proyecta a UTM y el EPSG del
  proyecto queda fijado por el primer sondeo. Si no la trae, el usuario captura X y Y en la
  tabla o sube un archivo `Sondeo, X, Y, Z[, azimut]`, con coma o punto y coma (también con
  coma decimal). Lo capturado a mano queda marcado como `usuario` y **no se sobrescribe**
  al volver a cargar el archivo.

## Decisiones de método

**Apilado (tres pasos).**

1. **Barridos completos: Isolation Forest + LOF.** Cada barrido se trata como un vector de desviaciones
   normalizadas por gate (en log₁₀ ρa donde el gate tiene señal). Cada detector da un puntaje y marca el
   barrido si supera **mediana + 3·MAD de los puntajes del sondeo**. Por omisión se descarta cuando
   coinciden ambos detectores, con un tope de seguridad del 20 %. No se usa la fracción de
   contaminación automática de Isolation Forest: en los datos de campo marcaba el 5–14 % de los
   barridos, que eran solo la cola normal del ruido gaussiano, y quitarlos subestimaba el error. Con el
   umbral adaptativo se descarta el 0.6–5 %, y lo descartado tiene el doble de dispersión que el resto
   y llega en ráfagas.
2. **Valores sueltos: MAD por gate, k = 3, en log₁₀ ρa.** Se calcula la ρa de cada barrido, y luego su
   mediana y su MAD en log. El corte es un factor de ρa, independiente del valor real de resistividad.
   Si cada barrido individual tiene SNR < 3, muchos valores son negativos y el log no existe. En esos
   gates el mismo corte se hace en dB/dt lineal: promediar en log ahí sesgaría la curva (en campo, la
   ρa tardía salía 2 a 10 veces menor).
3. **Valor apilado: media lineal de dB/dt** de lo que queda, con su error estándar. La ρa se calcula
   sobre la curva apilada, con su banda ±1σ en log₁₀ ρa (`sigma_log10_rho`, `rho_a_min/max`).

Después, un gate es *útil* si es positivo, su SNR es ≥ 3 y todos los gates anteriores del mismo
momento también lo son. Un *repunte* es un gate cuya señal sube más de 3σ respecto al anterior; por
omisión no se invierte. Los detectores solo ven ruido que **cambia entre barridos**. El ruido
coherente (acoplamiento, meseta tardía) está en todos los barridos por igual y se maneja excluyendo
gates.

**Error para invertir.** Es el error estándar del apilado ⊕ un piso del 3 % del dato
(editable). El piso cubre lo que el apilado no ve: calibración, forma de onda y efectos
3D leves.

**Modelado directo** (`forward_tdem.py`). Usa `simpeg.electromagnetics.time_domain.Simulation1DLayered`
con estos elementos:

- un lazo cuadrado real (`LineCurrent`), no un dipolo equivalente;
- el receptor en su posición (`COIL_LOCATION`);
- `PiecewiseLinearWaveform` con la `TX_RAMP` de cada momento;
- cada gate integrado en su ventana con 3 subtiempos logarítmicos.

Los coeficientes de Hankel se calculan una sola vez por firma de survey y se comparten entre
simulaciones y sondeos con la misma configuración, lo que baja el tiempo de ~100 s a ~30 s por
sondeo. El modelo se validó así:

- contra la solución analítica de semiespacio (error < 0.3 % a 100 Ω·m);
- el jacobiano, contra diferencias finitas.

**Paso 1: multicapa (Occam con límites).** Usa 30 capas logarítmicas hasta 1.5 × la profundidad de
difusión del último gate. La regularización es la rugosidad (primera diferencia de ln ρ) más un término
de cercanía (α = 0.01) al semiespacio de referencia, que es la ρa tardía mediana. λ se elige al estilo
Occam (Constable et al., 1987): en cada iteración se resuelve el modelo completo para 9 valores de λ,
se calcula el χ² real de cada uno con el modelado directo y se queda el **λ más grande que cumple el
objetivo**. Hay dos fases:

1. Bajar el desajuste hasta χ² = 1 o hasta que deje de mejorar. Ese es el χ² mínimo alcanzable.
2. Si χ² = 1 no se alcanza, tomar el modelo más suave con χ² ≤ 1.2 × ese mínimo (la tolerancia es
   editable).

**Límites de resistividad.** Por omisión van del mínimo al máximo de la ρa tardía de los gates
invertidos, ÷ y × 3; también se pueden fijar a mano. Se imponen por conjunto activo: las celdas que
violan un límite se fijan en él y se re-resuelve el resto, que es más correcto que recortar. Si una
celda queda en el límite, la app avisa: casi siempre es un tramo de datos que el modelo 1D no explica.

En S001 esto pasó de un multicapa de 0.47–499 Ω·m (rugosidad 39) a uno de 1–149 Ω·m (rugosidad 13).
La versión anterior (enfriamiento de λ) sigue disponible como opción.

**Paso 2: DOI.** Sigue a Oldenburg–Li: se hacen dos inversiones más (λ fijo, sin límites, para no falsear la sensibilidad) con referencias ×10 y
÷10 del semiespacio. La DOI es la profundidad desde la que el índice supera 0.1, buscada
desde el fondo.

**Paso 3: bloques.** Para k = 2…6:

- El arranque sale de la segmentación óptima del multicapa (programación dinámica) por
  encima de la DOI.
- El ajuste usa `scipy.optimize.least_squares` (trf) con la parametrización
  `h_i = h_min,i + exp(w_i)`. El espesor mínimo es el 20 % de la profundidad del techo, y la
  primera capa no puede ser más delgada que la primera celda del multicapa.
- ρ queda dentro de los mismos límites del multicapa y se regulariza hacia la media logarítmica del
  multicapa en la profundidad de cada bloque (previa de ±0.5 décadas, editable). Un bloque solo se aleja
  del modelo suave si los datos lo piden. La previa también entra en la covarianza de los rangos ±1σ.
- La última interfaz no puede pasar del fondo de la malla del multicapa (penalización tipo bisagra).
  Sin ese tope, cuando los datos no ven el fondo, la interfaz podía irse a cientos de km.

El número de capas se elige por BIC (o AIC). Si dos modelos difieren en menos de 2, gana el
más simple. Los errores se escalan por max(1, χ²_multicapa) para que un ajuste imperfecto no
infle k. Los rangos ±1σ salen de la covarianza linealizada.

**Curva combinada LM + HM.** Sirve para ver e interpretar (ρa y pseudosección). El cambio al
HM ocurre en el primer gate HM sin efecto de rampa cuyo error relativo ya es menor que el
del LM. **La inversión usa los dos momentos por separado**, cada uno con su forma de onda.

**Corrosividad.** En TDEM el primer gate útil rara vez ve los primeros metros. Por eso la
inversión guarda la *profundidad mínima resuelta* (difusión del primer gate invertido), y
la aplicación avisa cuando la profundidad de corrosividad es menor: en ese caso la clase es
una extrapolación.

## Limitaciones conocidas

- **No hay inversión lateralmente acoplada (LCI).** Los perfiles interpolan los modelos 1D.
  Con sondeos separados decenas o cientos de metros, la LCI aportaría poco y duplicaría el
  costo.
- **Modelo 1D isótropo sin polarización inducida.** Los repuntes tempranos (PI o
  superparamagnetismo) y el acoplamiento con infraestructura no se modelan: se detectan en el
  QC y se excluyen.
- **`TIME_DELAY` y `FIELD_SHIFT_FACTOR` vienen desactivados por omisión.** Los tiempos de
  los USF probados ya traen el retardo aplicado. Ambas correcciones se activan en la
  pantalla 3, en «Corrección del archivo».
- **El azimut del lazo no viene en los archivos.** Por omisión el lazo se dibuja alineado a
  los ejes. Solo afecta al dibujo y al QC de GPS, no a la inversión (lazo central).

## Validación

- **Sintético** (60/8/40/150 Ω·m; espesores 20/40/80 m; 30 barridos con 5 % de ruido). El
  multicapa da χ² = 0.78 y el modelo de bloques recupera 59.5 Ω·m hasta 20.1 m, 8.0 Ω·m
  hasta 61.4 m y 45.6 Ω·m debajo. La DOI es ≈ 109 m.
- **Datos de campo** (`Line001_001…003`). El QC detecta lo siguiente:
  - el HM de 001 tiene un tramo plano tardío (ruido o acoplamiento): con LM solo χ² ≈ 1, y
    con HM, ≈ 7;
  - 002 y 003 tienen repuntes tempranos en los mismos tiempos en LM y HM;
  - 003 está a ~4 m de 002 y es una repetición que reproduce lo mismo;
  - el GPS cae en una esquina del lazo, con un desfase (28.3, 28.3) m que parece la diagonal
    capturada en cada eje.

  Con todos los gates el multicapa no baja de χ² ≈ 8–18. **Antes de interpretar esos
  sondeos hay que excluir los gates señalados** en la pantalla 2.

## Estructura

```
app/    app.py, pages/1…6, acciones_app.py (lógica de botones), estado_app.py (sesión y caché),
        auth_app.py + licencias (de la versión SEV), marca_tdem.py, assets/ (mismas imágenes)
src/    lectura_tdem, procesamiento_tdem, qc_tdem, forward_tdem, inversion_tdem,
        parametros_tdem, perfil_tdem, geodesia_tdem, almacenamiento_tdem, paquete_tdem,
        pipeline_tdem, exportacion_tdem, graficos_tdem, escalas_color_tre, licencias_tre
ejemplos/  generar_sintetico_tdem.py y sinteticos/
```

**Almacenamiento.** Cada tabla se guarda en Parquet, en la ruta
`<etapa>/TDEM=<sondeo>/<tabla>.parquet`; las etapas son crudo, apilado, inversión,
parámetros y perfiles. El catálogo SQLite guarda sondeos, coordenadas, etapas y métricas;
DuckDB se usa para las consultas de campaña. Todo viaja en el `.itdem` (ZIP con manifiesto),
junto con los archivos de campo subidos (`_entradas/`) y los parámetros de la campaña.

**Regla de estilo.** El código no usa `elif`: cada condición adicional va en un
`else:` anidado y las ramas vacías llevan `else: pass` explícito. La prueba `prueba_sin_elif`
lo verifica.
