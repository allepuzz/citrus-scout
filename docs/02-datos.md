# 2. Los datos: qué son exactamente, y qué no son

Este es el documento más importante del conjunto. El modelo, la arquitectura y las
métricas son decisiones reversibles; la naturaleza de los datos determina el techo de
todo lo demás.

Todas las cifras aquí están medidas sobre los ficheros reales del repositorio, no
estimadas.

---

## 2.1 Qué tenemos, literalmente

Dos datasets públicos descargados de Kaggle:

| Clave local | Referencia Kaggle | Tamaño | Ficheros | Licencia |
|---|---|---|---|---|
| `citrus_leaf_pathology` | `chayanmondalabir/citrus-leaf-pathology-multi-class-image-dataset` | 3.5 GB | 12.520 | CC0: Dominio público |
| `citrus_diseases` | `superlord/citrus-diseases` | 2.0 GB | 1.704 | CC BY 4.0 |

Total en disco: **5.5 GB, 14.224 imágenes**.

### Por qué solo estos dos

El catálogo (`data/sources.py`) tiene un campo `licence` y una propiedad:

```python
@property
def is_commercially_usable(self) -> bool:
    permissive = ("CC0", "CC BY", "Public Domain", "MIT", "Apache")
    return any(token.lower() in self.licence.lower() for token in permissive)
```

**La decisión**: este proyecto pretende convertirse en un producto comercial para
cooperativas y ATRIAs. Un dataset con licencia no comercial, o con licencia
desconocida, es una bomba de relojería legal: entrenas con él, vendes el modelo, y
tienes un problema.

El criterio es **conservador por diseño**: lo que no es explícitamente permisivo se
trata como inutilizable. Hay más datasets de cítricos en Kaggle que se ven útiles y
están excluidos por esto. Es una pérdida de datos aceptada a cambio de poder
vender el resultado.

CC BY 4.0 exige atribución, que está registrada en `docs/data_sources.md`. Eso es
una obligación legal real, no una formalidad.

---

## 2.2 La naturaleza física de estas imágenes

Esto es lo que determina todo lo demás, así que lo digo sin rodeos.

**Son fotografías de primer plano de hojas individuales, sobre fondos controlados.**

Características típicas:
- Una sola hoja, centrada, ocupando la mayor parte del encuadre
- Fondo uniforme o poco informativo: papel blanco, mesa, mano, suelo
- Distancia de disparo: centímetros
- Iluminación: variable, a menudo de interior o sombra
- Resolución de la hoja: miles de píxeles de ancho

**Lo que el producto final va a ver es otra cosa completamente distinta.**

El README del proyecto define dos pasadas:

| Pasada | Altitud | GSD | Qué detecta |
|---|---|---|---|
| **Screening** | 15-25 m | 0.3-0.7 cm/px | Declive de copa, pérdida de vigor, árboles muertos |
| **Inspección** | < 2 m | < 0.05 cm/px | Síntomas a nivel de órgano |

**GSD** = *Ground Sample Distance*, cuántos centímetros de suelo real cubre un píxel.
A 0.5 cm/px, una lesión de 2 mm **no ocupa ni medio píxel**. Es invisible, no
"difícil de ver": físicamente no está en la imagen.

### La consecuencia honesta

| | Nuestras fotos | Pasada de screening | Pasada de inspección |
|---|---|---|---|
| Punto de vista | lateral, a mano | nadir (desde arriba) | cercano, desde dron |
| Escala | una hoja | una copa entera | un órgano |
| Fondo | controlado | otros árboles, suelo, sombra | hojas vecinas |
| Iluminación | variable, interior | sol directo cambiante | variable |

De las tres columnas, nuestras fotos se parecen **algo** a la tercera y **nada** a la
segunda.

Conclusión, que está escrita en el docstring de `sources.py`:

> They are useless for the nadir screening pass, where a canopy is tens of
> centimetres across and a leaf lesion sits below one pixel.

Lo que estos datos compran no es un modelo de producción. Es:
1. Una tubería que funciona de punta a punta
2. Un backbone con características de textura de hoja de cítrico
3. Una idea honesta de qué arquitecturas se comportan bien

---

## 2.3 Las 17 clases, y cuáles sirven de algo en Murcia

El código clasifica cada etiqueta en una de cuatro categorías de relevancia local
(`data/taxonomy.py`). Esta es la tabla completa, con las cuentas reales del archivo:

| Clase | Imágenes | Relevancia | Por qué |
|---|---|---|---|
| `healthy` | 998 | **healthy** | Hoja sana |
| `sooty mould` | 478 | **present** | Negrilla. Sigue a la melaza de mosca blanca o cochinilla |
| `gummosis` | 462 | **present** | *Phytophthora*. El objetivo de mayor valor del proyecto |
| `leaf minnor` | 459 | **present** | Minador de los cítricos (el dataset escribe "minnor") |
| `aphids` | 416 | **present** | Pulgones. Presentes en la brotación de primavera |
| `curl leaf` | 377 | nonspecific | Enrollamiento. Muchas causas posibles |
| `citrus canker` | 360 | **absent** | Cancro. Patógeno de cuarentena, ausente de España |
| `dry leaf` | 304 | nonspecific | Hoja seca. Síntoma, no diagnóstico |
| `greening` | 204 | **absent** | HLB. Ausente de España |
| `deficiency leaf` | 191 | nonspecific | Deficiencia nutricional |
| `black spot` | 169 | **absent** | Mancha negra. Ausente de los cítricos españoles |
| `melanose` | 130 | **absent** | Melanosis. Ausente |
| `spider mite` | 114 | **present** | *Tetranychus urticae*. Presente y dañina en verano |
| `anthracnose` | 107 | **absent** | Antracnosis. Ausente |
| `curl virus` | 106 | **absent** | Ausente |
| `bacterial blight` | 105 | **absent** | Ausente |
| `citrus mite` | 89 | **present** | Ácaro. Presente; el daño es lo visible, no el ácaro |

### El desglose por relevancia

| Categoría | Imágenes | Porcentaje |
|---|---|---|
| **present** (accionable en Murcia) | 2.018 | 39.8% |
| **absent** (ausente de España) | 1.181 | 23.3% |
| **healthy** | 998 | 19.7% |
| **nonspecific** (síntoma ambiguo) | 872 | 17.2% |

**Casi una cuarta parte de los datos son enfermedades que no existen en España.**

### Por qué esto no es un detalle administrativo

El caso del HLB lo ilustra. *Huanglongbing*, también llamado *greening*, es la
enfermedad más devastadora de los cítricos del mundo. Hay literatura abundante sobre
detectarla con drones, y 204 imágenes de ella en nuestro dataset.

Y es **irrelevante para este proyecto**, por hechos verificables:

- España está libre de *Candidatus Liberibacter* (la bacteria).
- España está libre de *Diaphorina citri* (el vector principal).
- *Trioza erytreae* (el otro vector) está en Canarias y la cornisa cantábrica, pero
  **no** en el Levante mediterráneo.

Consecuencia: **cualquier alerta de HLB en Murcia es, por definición, un falso
positivo.** No hay forma de obtener ground truth local, porque no hay casos locales.

Y eso significa que **toda la literatura de detección de HLB por UAV no es
reproducible aquí.** Es una de las afirmaciones más fuertes del README y es correcta.

### Y el caso inverso: lo que sí importa y no se puede ver

Estas plagas **sí** están en Murcia y **no** son detectables desde visión nadir:

| Plaga | Tamaño del síntoma |
|---|---|
| Piojo rojo de California | ~2 mm |
| *Delottococcus aberiae* | milímetros |
| Minador de los cítricos | galerías de <1 mm de ancho |
| Pulgones | en brotes jóvenes |
| *Ceratitis capitata* (mosca) | en fruto |

A 0.5 cm/px esto no se ve. Necesitan la pasada de inspección cercana, y es
exactamente para eso para lo que sirven estos datos de primer plano.

**La lógica del proyecto, entonces**: la pasada de screening busca declive de copa
(*Phytophthora*, tristeza, estrés hídrico), que sí tiene firma a escala de copa. La
pasada de inspección, sobre los árboles marcados, busca los síntomas milimétricos.
Los datos públicos preentrenan la segunda.

---

## 2.4 El 80.3% que lo cambia todo

Colapsando las 17 clases a binario (`healthy` vs todo lo demás):

| | Imágenes | Porcentaje |
|---|---|---|
| Afectada | 4.071 | **80.3%** |
| Sana | 998 | 19.7% |

Un limonar comercial bien gestionado tiene entre el 2% y el 5% de árboles afectados.

**Nuestro conjunto de datos tiene una prevalencia 16 a 40 veces superior a la
realidad.**

### Por qué

No es un error de nadie. Es un sesgo de recolección inevitable: los datasets públicos
de enfermedades son colecciones de *fotos de enfermedades*. La gente fotografía lo
interesante. Nadie sube 50.000 fotos de hojas sanas.

### Las consecuencias concretas

1. **Accuracy queda inutilizable** (§1.8). Decir "afectada" siempre da un 80.3%.

2. **El PPV medido directamente sobre este test es una fantasía.** El mismo modelo da
   99.7% de PPV aquí y 64.7% en campo (§1.10). Por eso el código lo *proyecta* con
   Bayes en vez de medirlo.

3. **Hay que pesar las clases durante el entrenamiento.** Sin eso el modelo aprende
   que "afectada" casi siempre acierta, que es lo contrario de lo que necesita. Los
   pesos reales que usa tu entrenamiento son `[1.606, 0.394]` (§3.5).

4. **El PR-AUC no es comparable entre conjuntos.** Un modelo aleatorio aquí da
   PR-AUC ≈ 0.80.

El campo `field_prevalence: 0.02` de los configs existe precisamente para esto. Es la
prevalencia real, usada para proyectar. Cambiarlo cambia todos los PPV reportados sin
tocar el modelo.

---

## 2.5 La calidad de las etiquetas

Algo que hay que decir claramente: **estas etiquetas no están verificadas por nadie.**

Son etiquetas aportadas por la comunidad de Kaggle. No hay:
- Confirmación por un agrónomo
- Confirmación por qPCR o cualquier prueba de laboratorio
- Protocolo documentado de etiquetado
- Medida de acuerdo entre etiquetadores

Problemas específicos observables:

**Ortografía que revela el proceso**: la clase es `leaf minnor`. Lo correcto es *leaf
miner* (minador). El código la normaliza y la documenta como "the dataset's spelling
of leaf miner", pero una errata así indica un etiquetado manual sin revisión.

**Clases que son síntomas, no diagnósticos**: `curl leaf`, `dry leaf`,
`deficiency leaf`. Una hoja seca puede ser sequía, salinidad, *Phytophthora* en
raíz, o un herbicida. Etiquetarla como una sola clase asume una causa que nadie
verificó. Por eso el código las marca `nonspecific`.

**Posible solapamiento**: la negrilla (`sooty mould`) *sigue* a una infestación de
cochinilla o mosca blanca. Una hoja puede tener las dos cosas y estar etiquetada con
una sola.

### Qué hacemos con esto

No lo arreglamos, porque no se puede sin re-etiquetar con un agrónomo. Lo que hace el
código es **no fingir que no existe**:

- La taxonomía marca explícitamente las clases no específicas
- El README dice que son etiquetas "that no agronomist or qPCR ever confirmed"
- El notebook de Colab lo repite en su sección final

La honestidad aquí no es una virtud moral, es ingeniería: si crees tus propias
etiquetas, optimizas contra ruido y te convences de tener un producto.

---

## 2.6 El embudo: de 14.224 ficheros a 5.069 imágenes

Este es el recorrido real, con las cifras medidas:

```
14.224  ficheros de imagen en disco
   │
   │  filtro de duplicados aumentados
   ▼
 5.951  imágenes originales
   │
   │  filtro min_samples_per_class (20)
   ▼
 5.951  (ninguna clase cae por debajo de 20)
   │
   │  deduplicación por hash SHA-256
   ▼
 5.069  imágenes únicas  ←  esto es lo que entra al modelo
```

Se descarta el **64% de los ficheros**. Cada paso tiene una razón.

### Paso 1: el filtro de aumentadas — el más importante

**8.273 de las 12.520 imágenes de `citrus_leaf_pathology` son copias pre-aumentadas.**
El 66% de ese dataset.

El dataset incluye una carpeta con versiones ya aumentadas (rotadas, volteadas, con
brillo cambiado) de las mismas fotografías originales.

**Por qué hay que quitarlas, y es un fallo catastrófico si no lo haces**: imagina la
foto original `hoja_042.jpg` y su versión rotada `hoja_042_aug3.jpg`. Si el reparto
aleatorio manda la original a entrenamiento y la rotada a validación, estás
**evaluando el modelo sobre una foto que ya ha visto**. El modelo reconoce esa hoja
concreta, y tu PR-AUC de validación es ficción.

Esto se llama **fuga de datos** (*data leakage*), y es la forma más común de
autoengañarse en visión por computador.

La detección:

```python
AUGMENTED_DIR_PATTERN = re.compile(r"(^|[-_ ])aug(mented)?([-_ ]|$)", re.IGNORECASE)
```

Busca "aug" o "augmented" como palabra completa en nombres de directorio.

**Un detalle que parece paranoia y no lo es.** La función tiene este diseño:

```python
def looks_augmented(path: Path, root: Path | None = None) -> bool:
    parts = path.parts
    if root is not None:
        with contextlib.suppress(ValueError):
            parts = path.relative_to(root).parts
    return any(AUGMENTED_DIR_PATTERN.search(part) for part in parts[:-1])
```

Solo inspecciona la parte de la ruta **por debajo de `root`**. Si mirara la ruta
absoluta, un directorio llamado `augmented-data` o incluso un directorio temporal de
pytest en cualquier punto superior del árbol haría que **todas** las imágenes se
descartaran silenciosamente. El pipeline diría "0 imágenes encontradas" y tendrías que
depurar por qué.

Y `parts[:-1]` excluye el nombre del fichero: solo los *directorios* indican una copia
aumentada. Un fichero llamado `augusto.jpg` no debe descartarse.

`citrus_diseases` tiene **0** aumentadas. Los dos datasets se comportan distinto y el
código no asume nada sobre la estructura.

### Paso 2: el mínimo por clase

`min_samples_per_class: 20`. Con menos de 20 imágenes:
- El 15% de validación es **3 imágenes**. Esa métrica es ruido puro.
- Los promedios macro se distorsionan: una clase de 5 imágenes pesa igual que una de 500.

En la práctica ninguna clase cae aquí (la más pequeña, `citrus mite`, tiene 89). El
filtro está por robustez ante descargas parciales.

### Paso 3: la deduplicación por contenido

**882 imágenes eliminadas.** El 14.8%.

El método es hash criptográfico SHA-256 del contenido del fichero:

```python
def file_digest(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
```

Lee en trozos de 1 MB en vez de cargar el fichero entero: acota el uso de memoria
independientemente del tamaño del fichero.

**Por qué hash y no comparar nombres**: los duplicados aquí tienen nombres distintos,
y a veces **están en carpetas de clases diferentes** — la misma fotografía etiquetada
de dos maneras. Eso es ruido de etiqueta además de duplicación.

**La limitación, que el código documenta en vez de esconder**:

> Only exact duplicates are caught; a resized or re-encoded copy of the same
> photograph survives this and remains a source of optimistic bias.

SHA-256 detecta ficheros **byte a byte idénticos**. Si alguien guardó la misma foto
con calidad JPEG distinta, o redimensionada, el hash cambia por completo y el
duplicado sobrevive.

La solución real sería **hashing perceptual** (pHash, dHash) o embeddings. No está
implementado, y es una de las razones candidatas de que el PR-AUC salga 1.0.

### Un manejo de errores deliberado

```python
try:
    digest = file_digest(sample.path)
except OSError:
    removed += 1
    continue
```

Un fichero ilegible se cuenta como eliminado y se sigue. **La decisión**: este
pipeline procesa 14.000 ficheros y se ejecuta durante minutos. Abortar todo porque un
JPEG está truncado es peor que saltárselo y reportarlo.

---

## 2.7 El reparto: train / val / test

De las 5.069 imágenes:

| Split | Imágenes | Fracción |
|---|---|---|
| Train | 3.545 | 69.9% |
| Validación | 762 | 15.0% |
| Test | 762 | 15.0% |

Coincide exactamente con lo que imprimió tu entrenamiento:
`data: train=3545 val=762 test=762`.

### Decisión: 15% / 15%

Compromiso entre dos presiones opuestas:

- **Más grande** = métricas más estables. Con 762 imágenes, el error estándar de una
  proporción cerca del 50% es aproximadamente `sqrt(0.25/762) ≈ 1.8%`. Aceptable.
- **Más pequeño** = más datos para entrenar.

Con 5.069 imágenes en total, 15% da 762, suficiente para que las clases minoritarias
tengan presencia (la más pequeña, `citrus mite`, aporta 13 imágenes a cada split).
Bajar al 10% dejaría esa clase con 9.

### Decisión: reparto estratificado

**Estratificado** significa que el reparto se hace **por clase**, no sobre el montón
entero. El 15% de cada clase va a validación.

```python
for label in sorted(by_class):
    group = by_class[label]
    group.sort(key=lambda s: str(s.path))
    indices = rng.permutation(len(group))
    n_val = round(len(group) * val_fraction)
    n_test = round(len(group) * test_fraction)
```

**Por qué**: con un reparto puramente aleatorio, `citrus mite` (89 imágenes) podría
quedarse con 2 en validación por puro azar. Su recall de validación sería 0%, 50% o
100% según acertara o no dos fotos. Ruido sin información.

Así se garantiza que las 17 clases aparecen en los 3 splits, con proporciones iguales.
La tabla de §2.3 lo confirma: las cuentas de val y test son idénticas clase a clase.

### El detalle del orden determinista

```python
group.sort(key=lambda s: str(s.path))   # ANTES de permutar
indices = rng.permutation(len(group))
```

Ordenar por ruta antes de mezclar parece redundante, pero no lo es.
`Path.rglob` devuelve ficheros en el orden que da el sistema de ficheros, que **varía
entre sistemas operativos y entre sistemas de ficheros**. Sin el `sort`, la misma
semilla daría repartos distintos en Windows y en el Linux de Colab.

Eso rompería la reproducibilidad de forma silenciosa y muy difícil de diagnosticar.

### La protección para clases diminutas

```python
while n_val + n_test >= len(group) and (n_val > 0 or n_test > 0):
    if n_test >= n_val:
        n_test -= 1
    else:
        n_val -= 1
```

Con una clase de 3 imágenes: `round(3*0.15) = 0` para cada una, así que no hay
problema. Pero con otras fracciones, `round()` podría asignar tanto a val y test que
train se quedara vacío. Este bucle garantiza **al menos una imagen en train** para
cualquier clase que se conserve.

Alterna entre quitar de test y de val para no vaciar uno de los dos.

### La verificación de fuga

```python
def verify_no_leakage(split: SplitResult) -> None:
    for left, right in (("train","val"), ("train","test"), ("val","test")):
        overlap = paths[left] & paths[right]
        if overlap:
            raise AssertionError(...)
```

Comprueba que ninguna **ruta** aparezca en dos splits. Se ejecuta en cada
`build_splits`.

Es un invariante barato que detecta errores de indexación en el propio código de
reparto. El docstring es explícito sobre su límite:

> It cannot detect two different files holding the same photograph.

Protege contra bugs nuestros, no contra duplicados semánticos.

---

## 2.8 El empaquetado para Colab

### El problema

Subir 5.951 ficheros y 5.5 GB a Google Drive tarda horas. No por el ancho de banda:
por la **latencia por fichero**. Cada fichero es una transacción HTTP.

Y en Colab es peor: Drive se monta como sistema de ficheros de red, así que leer
14.000 ficheros pequeños durante el entrenamiento es lentísimo.

### La solución y sus números

Un único zip con solo las imágenes que sobreviven al filtrado, reescaladas:

| | Antes | Después |
|---|---|---|
| Ficheros | 14.224 | 1 |
| Tamaño | 5.5 GB | **174 MB** |
| Tiempo de subida | horas | minutos |

Reducción de **32 veces**.

### Decisión: lado mayor 512 píxeles

```python
DEFAULT_MAX_SIDE = 512
```

El entrenamiento recorta a 224×224. ¿Por qué guardar 512 y no 224?

Porque el aumento de datos usa `RandomResizedCrop` con `scale=(0.7, 1.0)`: recorta
entre el 70% y el 100% del área y reescala a 224. Si guardáramos a 224 exactos, cada
recorte sería un *upscale* de una imagen ya pequeña, perdiendo nitidez.

512 deja margen para que el recorte aleatorio tenga píxeles reales con los que
trabajar, sin cargar la resolución completa de cámara (que puede ser 4000×3000 y no
aporta nada a 224).

Nunca se hace upscale:

```python
longest = max(image.size)
if longest <= max_side:
    return image      # ya es pequeña: devolverla tal cual
```

Escalar hacia arriba no añade información y cuesta espacio.

### Decisión: LANCZOS para reescalar

```python
image.resize(new_size, Image.Resampling.LANCZOS)
```

Hay varios algoritmos de remuestreo. LANCZOS es el de mayor calidad disponible en
PIL: usa una ventana sinc que preserva el detalle de alta frecuencia mejor que
bilinear o bicúbico.

**Por qué importa aquí**: los síntomas que queremos detectar **son** detalle de alta
frecuencia — bordes de lesión, textura de necrosis, galerías de minador. Un remuestreo
que los difumina destruye exactamente la señal.

Es más lento, y se ejecuta una vez por imagen al empaquetar. Compensa.

### Decisión: JPEG calidad 90

```python
DEFAULT_QUALITY = 90
```

JPEG con pérdida, calidad 90 sobre 100. Visualmente casi indistinguible del original
para este contenido, y aproximadamente una cuarta parte del tamaño de calidad 100.

El riesgo teórico es que los artefactos JPEG destruyan detalle diagnóstico. A calidad
90 los artefactos están muy por debajo de la escala de las lesiones visibles. A
calidad 70 empezaría a preocuparme.

### Decisión: ZIP_STORED, no DEFLATE

```python
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
```

`ZIP_STORED` significa **sin comprimir**: el zip solo agrupa, no comprime.

Contraintuitivo, y correcto: **JPEG ya está comprimido**. Intentar comprimirlo otra
vez con DEFLATE gasta CPU al empaquetar y al desempaquetar para ganar una fracción de
porcentaje. El comentario del código lo dice:

> ZIP_STORED, not DEFLATE: JPEG is already compressed, so deflating it burns CPU on
> both ends for a fraction of a percent of size.

Esto es lo que hace que Colab descomprima los 174 MB en segundos.

### Cómo se preserva el reparto

Esto es la parte más importante del diseño del archivo. La estructura interna es:

```
leaf_dataset.zip
├── manifest.json
├── train/
│   ├── healthy/000000.jpg ... (698 imágenes)
│   ├── gummosis/...          (324)
│   └── ... 17 clases
├── val/
│   ├── healthy/...           (150)
│   └── ...
└── test/
    └── ...
```

**El reparto está codificado en la estructura de carpetas.** No es un metadato que
haya que interpretar: `train/`, `val/` y `test/` son directorios.

**Por qué esto es crucial**: si Colab recalculara el reparto, usaría la misma semilla
(42) y el mismo algoritmo... pero si *cualquier* entrada cambiara — una imagen más, un
duplicado distinto detectado, otra versión de numpy — el reparto sería distinto. Y
entonces tu entrenamiento en Colab y tu evaluación local estarían midiendo
particiones diferentes, **sin avisar**.

El docstring lo explica:

> Recomputing it remotely would risk a different split if any input changed, quietly
> invalidating the comparison.

El manifest real de tu zip:

```json
{
  "max_side": 512,
  "quality": 90,
  "classes": [17 clases ordenadas],
  "counts": { "train": 3545, "val": 762, "test": 762 },
  "duplicates_removed": 882,
  "images_written": 5069,
  "images_skipped": 0
}
```

`images_skipped: 0` confirma que ninguna imagen estaba corrupta.

La lista `classes` está **ordenada alfabéticamente**, y eso fija el mapeo
clase→índice. Un checkpoint entrenado con un orden y evaluado con otro produciría
predicciones confidentemente erróneas sin dar error. Por eso `save_checkpoint` guarda
la lista de clases junto a los pesos.

---

## 2.9 Cómo se leen las imágenes

### Conversión a RGB, siempre

```python
image = Image.open(sample.path).convert("RGB")
```

Parece redundante en un JPEG. No lo es:

- Algunos PNG son **paletizados** (modo `P`): un canal de índices a una paleta.
- Algunos JPEG son **CMYK** (4 canales, de origen de imprenta).
- Algunos PNG tienen **alfa** (modo `RGBA`, 4 canales).
- Algunos son **escala de grises** (modo `L`, 1 canal).

Sin el `convert("RGB")`, el tensor tendría 1 o 4 canales en vez de 3, y el modelo
fallaría con un error de dimensiones a mitad de entrenamiento — o peor, en un caso
raro que solo aparece tras 10 minutos.

Normalizar la entrada en el punto de lectura es más barato que defenderse después.

### El orden de clases estable

```python
self.classes = list(classes) if classes else sorted({s.label for s in self.samples})
self.class_to_index = {name: i for i, name in enumerate(self.classes)}

unknown = {s.label for s in self.samples} - set(self.class_to_index)
if unknown:
    raise ValueError(f"samples carry labels missing from `classes`: {sorted(unknown)}")
```

Dos cosas:

1. **`sorted()`**: el orden viene de ordenar alfabéticamente, no del orden de
   aparición. Estable entre ejecuciones y máquinas.

2. **La validación de etiquetas desconocidas**: si un sample trae una etiqueta que no
   está en la lista de clases, falla inmediatamente. Sin esto, el `KeyError`
   aparecería a mitad del entrenamiento, desde dentro de un worker del DataLoader,
   con una traza ilegible.

---

## 2.10 El aumento de datos, y qué transformaciones están prohibidas

El **aumento de datos** (*data augmentation*) aplica transformaciones aleatorias a
cada imagen de entrenamiento. El modelo ve una variante distinta cada época, lo que
combate el sobreajuste y enseña invarianzas útiles.

El principio de diseño aquí está en el docstring de `transforms.py`, y es el correcto:
**aumentar solo lo que el sistema desplegado va a ver de verdad.**

### Lo que se aplica, y por qué

```python
A.RandomResizedCrop(size=(224,224), scale=(0.7,1.0), ratio=(0.85,1.18))
```
El caballo de batalla. Recorta entre el 70% y el 100% del área y reescala. Imita un
dron a distancias distintas y con encuadre descentrado. `ratio` permite una ligera
distorsión de aspecto (±15%).

```python
A.HorizontalFlip(p=0.5)
A.VerticalFlip(p=0.2)
A.Rotate(limit=30, p=0.5, border_mode=0)
```
Volteos y rotación. **Justificación**: una lesión en una hoja no tiene orientación
preferente respecto a la gravedad. Un dron fotografía hojas en orientación arbitraria.
Estas transformaciones enseñan una invarianza que es cierta en el mundo real.

El volteo vertical tiene probabilidad menor (0.2 vs 0.5) porque las hojas sí tienen
una cara y un dorso con aspecto distinto, aunque la rotación ya cubre buena parte.

```python
A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7)
```
Brillo y contraste, ±25%, con probabilidad alta (0.7). **Justificación**: la
iluminación de campo varía enormemente. Un mismo vuelo pasa de sol directo a sombra
de copa en segundos. Esta es la variación más real de todas, y por eso tiene la
probabilidad más alta.

```python
A.OneOf([A.MotionBlur(blur_limit=5), A.GaussianBlur(blur_limit=(3,5))], p=0.2)
A.GaussNoise(p=0.15)
```
Desenfoque de movimiento y ruido. **Justificación**: un dron en movimiento produce
desenfoque real, y sensores con poca luz producen ruido. Son modos de fallo
auténticos, no hipotéticos.

### Lo que NO se aplica, y es la parte interesante

```python
A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=20, val_shift_limit=10, p=0.3)
```

Fíjate en `hue_shift_limit=6`. Es un desplazamiento de tono diminuto, casi testimonial.
La rotación de tono es una de las aumentaciones más comunes en visión por computador,
y aquí está deliberadamente casi desactivada.

**La razón, del docstring**:

> **No aggressive hue shift.** Colour is the signal. Chlorosis is a yellow shift and
> sooty mould is a darkening; rotating hue teaches the model to ignore exactly what
> distinguishes them.

El color **es** el síntoma. La clorosis es un amarilleamiento. La negrilla es un
oscurecimiento. Si rotas el tono agresivamente, le enseñas al modelo a ignorar el
color — es decir, a ignorar la señal diagnóstica.

Los 6 grados que quedan cubren variación real de balance de blancos entre cámaras.
Más que eso destruiría información.

Igualmente, **no hay deformaciones de perspectiva fuertes**: distorsionan la *forma*
de la lesión, que también es diagnóstica (una mancha circular de mancha negra frente a
una galería serpenteante de minador).

Esto es el principio general: **una aumentación que enseña una invarianza falsa es
peor que ninguna aumentación.** Le está diciendo al modelo que ignore algo que
importa.

### La transformación de evaluación

```python
def eval_transform(image_size=224):
    return A.Compose([
        A.SmallestMaxSize(max_size=int(image_size * 1.14)),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
```

**Completamente determinista.** Cero aleatoriedad.

El `1.14` es el protocolo estándar de evaluación en ImageNet: reescala el lado menor a
256 (224 × 1.14 ≈ 256), luego recorta el centro a 224. Preserva la proporción y coge
la región central.

**Por qué nada aleatorio**: si la evaluación tuviera aleatoriedad, dos ejecuciones del
mismo modelo sobre los mismos datos darían números distintos. Sería imposible saber si
una mejora es real o es ruido.

### La normalización con estadísticas de ImageNet

```python
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
```

Cada canal se transforma como `(pixel - media) / desviación`.

Estos números concretos son la media y desviación de los 1,2 millones de imágenes de
ImageNet. **No son nuestros datos**, y eso es deliberado:

> The backbones are pre-trained with these, and normalising differently would discard
> part of what the pre-training learned.

El backbone preentrenado aprendió sus filtros esperando entradas normalizadas con
*estas* estadísticas. Sus primeras capas están afinadas a ese rango. Normalizar con
las estadísticas de nuestro dataset desplazaría la distribución de entrada respecto a
lo que el modelo espera, y parte del valor del preentrenamiento se perdería.

Es contraintuitivo (lo normal sería usar tus propias estadísticas) pero es lo correcto
al hacer transferencia.

---

## 2.11 Resumen honesto de los datos

**Lo que estos datos permiten hacer:**
- Validar que la tubería completa funciona
- Preentrenar un backbone con textura de hoja de cítrico
- Comparar arquitecturas y configuraciones de entrenamiento
- Construir y probar todo el aparato de evaluación

**Lo que estos datos NO permiten:**
- Afirmar nada sobre rendimiento en campo
- Entrenar para la pasada de screening (escala incompatible)
- Detectar las plagas que realmente importan en Murcia (son milimétricas)
- Confiar en las etiquetas como verdad agronómica

**La restricción real del proyecto**, y esto no lo cambia ninguna línea de código:
imagen aérea real de limonares de Murcia, con ground truth verificado por un
agrónomo, a las dos escalas de vuelo.

Hasta que exista eso, cualquier mejora de modelo es optimizar contra el dataset
equivocado.

---

Siguiente: [03-entrenamiento.md](03-entrenamiento.md), cada decisión del entrenamiento.
