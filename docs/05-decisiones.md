# 5. Registro de decisiones, bugs y limitaciones

Este documento contiene tres cosas que normalmente no se escriben:

1. Todas las decisiones de diseño, con la alternativa que se rechazó y por qué.
2. Todos los bugs encontrados durante el desarrollo, incluidos los míos.
3. Las limitaciones reales, sin suavizar.

---

## 5.1 Tabla maestra de decisiones

### Datos

| Decisión | Alternativa rechazada | Por qué |
|---|---|---|
| Solo datasets con licencia permisiva | Usar todos los de Kaggle | El proyecto será comercial. Licencia desconocida = bomba legal |
| Excluir directorios `aug*` | Usarlo todo | 8.273 de 12.520 son copias aumentadas. Fuga garantizada |
| Deduplicar por SHA-256 | Comparar nombres | Los duplicados tienen nombres distintos, y algunos cruzan clases |
| Inspeccionar ruta relativa a `root` | Ruta absoluta | Un directorio `augmented` superior descartaría todo silenciosamente |
| Reparto estratificado | Aleatorio simple | `citrus mite` (89 img) podría quedarse con 2 en val |
| Ordenar por ruta antes de mezclar | Confiar en `rglob` | El orden del sistema de ficheros varía entre Windows y Linux |
| Semilla fija (42) | Sin semilla | Reproducibilidad entre Colab y local |
| 15% / 15% | 20%/20% o 10%/10% | 762 da error estándar ~1.8%; con 10% las clases raras quedan con 9 |
| `min_samples_per_class: 20` | Sin mínimo | Con <20, el 15% de val son 3 imágenes: ruido puro |
| Binario por defecto | 17 clases | El técnico decide si visitar, no diagnostica el patógeno |
| Envolver el dataset, no reetiquetar | Reetiquetar directo | Preserva la clase original para el análisis de errores |
| Lado mayor 512 px | 224 px (lo que usa el modelo) | `RandomResizedCrop` necesita margen para recortar |
| LANCZOS | Bilinear / bicúbico | Los síntomas son detalle de alta frecuencia |
| JPEG calidad 90 | 100 (sin pérdida) / 70 | 90 es casi indistinguible y ocupa 1/4 de 100 |
| `ZIP_STORED` | `ZIP_DEFLATED` | JPEG ya está comprimido; deflate gasta CPU por ~0% |
| Split en estructura de carpetas | Recalcular en Colab | Un cambio en las entradas daría particiones distintas, sin avisar |
| `convert("RGB")` siempre | Asumir 3 canales | Hay PNG paletizados, JPEG CMYK, y grises |
| Orden de clases `sorted()` | Orden de aparición | Un orden distinto invierte predicciones sin dar error |

### Aumento de datos

| Decisión | Alternativa rechazada | Por qué |
|---|---|---|
| `hue_shift_limit=6` (casi nada) | Rotación de tono estándar | **El color es la señal.** Clorosis = amarilleo, negrilla = oscurecimiento |
| Sin deformación de perspectiva fuerte | Incluirla | Distorsiona la forma de la lesión, que es diagnóstica |
| Brillo/contraste ±25%, p=0.7 | Probabilidad menor | La iluminación es la variación más real en campo |
| Desenfoque y ruido, p baja | No incluirlos | Un dron en movimiento los produce de verdad |
| Normalizar con estadísticas de ImageNet | Usar las de nuestro dataset | El backbone preentrenado espera ese rango exacto |
| Evaluación 100% determinista | Aumentar también en test | Sin eso, dos ejecuciones dan números distintos |

### Modelo y entrenamiento

| Decisión | Alternativa rechazada | Por qué |
|---|---|---|
| EfficientNet-B0 (4M) | ResNet-50 (25M), ConvNeXt (88M) | El techo son los datos, no la capacidad. Y cabe en 4 GB |
| `pretrained=True` | Desde cero | 3.545 imágenes no bastan. ImageNet ya enseñó textura y color |
| `dropout=0.3` | 0 | Regulariza **y** es lo que MC Dropout muestrea |
| Congelar backbone 1 época | Sin warmup | Una cabeza aleatoria produce gradientes que destruyen el preentrenamiento |
| AdamW | SGD, Adam | Decaimiento de pesos desacoplado. Estándar para ajuste fino |
| LR 3e-4 | 1e-2, 1e-5 | ~10× menor que desde cero, la regla del ajuste fino |
| Coseno | Escalones, constante | Sin hiperparámetros y empíricamente mejor |
| `T_max = epochs - warmup` | `T_max = epochs` | El coseno no debe consumir recorrido con el backbone congelado |
| Pesado por frecuencia inversa | Sin pesar, o sobremuestrear | 80.3% de afectadas invierte el sesgo respecto al campo |
| Normalizar pesos a media 1 | Usarlos crudos | Así activar el pesado no cambia la escala de la pérdida ni exige retocar el LR |
| `label_smoothing=0.05` | 0 | Sin él, el objetivo P=1.0 exige logits infinitos |
| `grad_clip=1.0` | Sin recorte | Un lote patológico puede lanzar la pérdida a NaN |
| AMP activado | FP32 puro | ~2× velocidad, ~½ memoria en T4 |
| Desescalar antes de recortar | Recortar directo | Recortaría gradientes escalados: el modelo no aprendería, sin error |
| `cudnn.benchmark = False` | `True` (por defecto) | 5-15% de velocidad a cambio de no poder reproducir |
| Seleccionar por PR-AUC | Por pérdida de validación | La pérdida puede bajar mientras empeora el ranking de la minoría |
| `drop_last=True` solo en train | Siempre o nunca | En val/test descartaría muestras y cambiaría las métricas |
| `batch_size=32` base, 64 en Colab | Uno solo | 32 cabe en 4 GB; 64 estabiliza el pesado en T4 |
| `num_workers` automático | Fijo en 4 | Colab da 2 CPUs; 4 workers avisaba en cada ejecución |
| Config en YAML | Argumentos de CLI o constantes | Un resultado debe reconstruirse del config solo |
| Notebook como lanzador | Lógica en celdas | El estado de celdas es invisible e irreproducible |

### Métricas y evaluación

| Decisión | Alternativa rechazada | Por qué |
|---|---|---|
| **No reportar accuracy** | Reportarla | Al 2% de prevalencia, "siempre sana" da 98% |
| PR-AUC como principal | ROC-AUC | ROC no tiene VN en el eje Y; es optimista con desbalance |
| PPV proyectado con Bayes | PPV medido en test | El test tiene 80.3%; el campo 2%. Factor 16-40× |
| Puntos de operación por especificidad | Umbrales fijos, o maximizar F1 | F1 asume que FP y FN cuestan igual. No es cierto en campo |
| Columna "alertas por caso" | Solo PPV | "6 caminatas por hallazgo" es accionable; "16.9%" es abstracto |
| Mostrar prevalencia de test | Ocultarla | Sin ella, PR-AUC 1.0 se lee como problema resuelto |
| Interpretación impresa automáticamente | Dejarla al lector | Un informe malinterpretable se malinterpretará |
| Matriz de confusión escrita a mano | `sklearn.metrics` | sklearn deduce el nº de clases y desplaza índices si falta una |
| Confusiones ordenadas por tasa | Por recuento | Una clase pequeña perdiendo el 22% importa más que una grande perdiendo el 2% |
| Coste de confusión desde la taxonomía | Tratar todas igual | Cancro↔HLB es gratis en España; gummosis↔pulgón no |
| Clases sin muestras dan 0.0 | `nan` | Un `nan` propaga y convierte toda la tabla en `nan` |
| Información mutua para el triaje | Entropía total | Solo la epistémica se reduce yendo a mirar |
| 20 pases de MC Dropout | 5, 50, 100 | El error cae como 1/√N; para ordenar, 20 sobran |
| Verificar que el dropout está activo | Confiar | timm no tiene módulos `nn.Dropout`: fallaría en silencio |
| BatchNorm en eval durante MC Dropout | Todo en train | Sus estadísticas derivarían durante la inferencia |
| `separates: None` si no hay errores | `True` o excepción | Con 0 errores no hay nada que medir. Afirmarlo sería inventar |
| Temperatura optimizando log(T) | T directa | T≤0 invertiría el ranking o dividiría por cero |
| LBFGS | Adam | Un parámetro, objetivo suave: segundo orden converge antes |
| Ajustar T en val, medir en test | Ambos en test | Ajustar y medir en lo mismo reporta una calibración ficticia |
| Rechazar `--calibrate-on == --split` | Avisar | Un aviso se ignora; un error de argumentos no |
| Imprimir ECE antes y después | Solo el mejor | El ECE puede empeorar en datos fáciles. Ocultarlo sería deshonesto |
| Reportar Brier además de ECE | Solo ECE | El ECE se puede engañar prediciendo la tasa base |
| `border_excess` normalizado por área | Umbral absoluto | El anillo es 49% del área a 7×7 y 26% a 14×14 |
| Grad-CAM como gestor de contexto | Hooks manuales | Un hook filtrado acumula memoria y no apunta a su causa |
| `register_full_backward_hook` | `register_backward_hook` | El segundo es poco fiable si el forward tiene varias operaciones |
| Rechazar arquitectura desconocida | Adivinar la capa | Un mapa de la capa equivocada se ve plausible y no significa nada |
| Ampliar el mapa a cuadros | Interpolación suave | Sugeriría una precisión espacial que 7×7 no tiene |
| Guardar las 12 peores | 12 al azar | Si solo miras 12 de 762, que sean las sospechosas |
| ReLU sobre el mapa | Sin ReLU | Lo negativo es evidencia de *otra* clase y ensucia el mapa |
| Normalizar mapas por imagen | Por lote | La magnitud del gradiente varía con la confianza |

---

## 5.2 Los bugs, uno por uno

### Bug 1: `requires-python = "<3.13"` — PR #5

**Síntoma** (lo viste tú en Colab):

```
ERROR: Package 'citrus-scout' requires a different Python: 3.13.15 not in '<3.13,>=3.11'
```

**Causa**: el techo `<3.13` era un valor por defecto del andamiaje inicial, no una
restricción medida. Colab actualizó sus runtimes a Python 3.13.15.

**Por qué era grave y no cosmético**: `pip install -e .` fallaba, así que el
*console script* `citrus-scout` nunca se creaba. Las celdas 12, 14 y 16 del notebook
habrían fallado todas con `command not found`. El notebook era inservible.

**Arreglo**: subir a `<3.14`. Ninguna dependencia del proyecto necesitaba el techo
antiguo; todas publican ruedas para 3.13.

**Lección**: los valores por defecto del andamiaje caducan. Un techo de versión que
nadie midió es deuda esperando a que el entorno avance.

---

### Bug 2: early stopping con métrica saturada — PR #6

**Síntoma**:

```
epoch  14 | train 0.2498 | val 0.2902 | PR-AUC 1.0000 | F1 0.9951   ← mejor val loss
epoch  15 | train 0.2434 | val 0.2910 | PR-AUC 1.0000 | F1 0.9951
early stopping: no improvement for 5 epochs
```

El mensaje decía "sin mejora durante 5 épocas" mientras la pérdida de validación
alcanzaba su mínimo en la época 14.

**Causa**:

```python
if report["pr_auc"] > best_pr_auc:     # comparación estricta
    ...
else:
    epochs_without_improvement += 1
```

PR-AUC está **acotado en 1.0**. Al alcanzarlo, `>` no puede volver a ser cierto. Nunca.
Cada época posterior cuenta como regresión.

**Por qué importaba de verdad**: en este dataset solo se desperdició la cola del
entrenamiento. Pero en un dataset difícil, donde un pico del 0.85 sea una meseta real de
la que el modelo sale, el contador correría durante la meseta y **cortaría un modelo
que estaba mejorando**, informando mal del motivo.

**Arreglo**: separar "¿es el mejor?" (sin umbral, decide el checkpoint) de "¿nos
rendimos?" (con `min_delta`, decide la parada), más una parada explícita por saturación
con un mensaje honesto.

**Cómo verifiqué que los tests valían**: escribí los tests, luego **revertí
`should_stop` a la lógica vieja**. Cuatro fallaron. Un test que pasa con el bug presente
no demuestra nada.

---

### Bug 3: `num_workers` fijo en 4 — PR #6

**Síntoma**, en cada ejecución de Colab:

```
UserWarning: This DataLoader will create 4 worker processes in total. Our suggested
max number of worker in current system is 2 ...
```

**Causa**: `num_workers: int = 4` fijo. Colab da 2 CPUs.

**Gravedad**: baja. No rompía nada, pero cuatro procesos peleando por dos núcleos puede
ser más lento que dos, y un aviso que aparece siempre entrena a la gente a ignorar los
avisos.

**Arreglo**: `num_workers: int | None = None`, resuelto desde `os.cpu_count()`, dejando
un núcleo libre y con tope en 4.

**El detalle que importa**: usar `if self.num_workers is not None` y no
`if self.num_workers`. Con la comprobación de veracidad, un `0` explícito (que significa
"cargar en el proceso principal", una petición válida) se trataría como "sin
configurar".

---

### Bug 4: gradientes filtrados en Grad-CAM — PR #11

**Síntoma**: el test `test_leaves_no_gradients_on_the_parameters` que yo mismo escribí
**falló**.

**Causa**, en mi propio código:

```python
self.model.zero_grad(set_to_none=True)   # ANTES de backward
score.backward()
```

Limpiaba los gradientes antes de generarlos, así que quedaban en los parámetros al
salir del método.

**Por qué era grave**: si alguien ejecutara Grad-CAM a mitad de un entrenamiento, esos
gradientes se sumarían al siguiente `optimizer.step()` y corromperían la actualización.
PyTorch acumula gradientes por defecto; no se queja.

**Arreglo**: mover el `zero_grad` después del `backward`.

**Lección**: es el ejemplo más limpio del repositorio de un test que encuentra un bug
real en vez de confirmar lo que el autor ya creía.

---

### Bug 5 (mío, en el diseño de la métrica): `border_fraction` dependía de la resolución

**Síntoma**: mi primera versión marcaba como "sospechoso de usar el fondo" a un modelo
con **pesos completamente aleatorios**:

```
summary: border=45.3% centre=54.7% concentration=28.9% [suspect]
```

**Causa**: umbral absoluto de 0.35 sobre la fracción de atención en el borde. Pero el
área del borde depende del tamaño del mapa:

| Tamaño | Borde = ...% del área |
|---|---|
| 7×7 | 49.0% |
| 14×14 | 26.5% |
| 28×28 | 38.3% |
| 224×224 | 43.8% |

A 7×7 (el tamaño real de EfficientNet-B0) un mapa uniforme pone el 49% en el borde, por
encima del umbral. El mismo patrón saldría limpio a 14×14.

**Arreglo**: `border_excess = border_fraction / border_area_fraction`. Un mapa uniforme
da exactamente 1.000 a cualquier resolución.

**Lección**: una métrica espacial sobre mapas pequeños necesita normalizarse por
geometría. Lo pillé porque probé con pesos aleatorios, que es un control que suele
saltarse.

---

### Bug 6 (mío, en los tests): saturación de float32

**Síntoma**: dos tests de monotonía de la calibración fallaban.

**Diagnóstico**:

```
T=1.0: unique=315/400, at 1.0=54, min=9.095e-16
T=2.7: unique=400/400, at 1.0=0,  min=2.686e-06
```

Con mi fixture a `scale=6.0`, **54 de 400 probabilidades se redondeaban a exactamente
1.0** en float32. `argsort` desempata por índice, así que el orden difería por razones
numéricas, no matemáticas.

**La matemática era correcta.** Mi test era demasiado agresivo.

**Arreglo**: usar escalas no saturantes en los tests de monotonía, y añadir un test
aparte que **fija el límite de saturación** como comportamiento conocido:

```python
def test_saturation_is_a_float_limit_not_a_maths_error(self):
    logits, _ = overconfident_logits(seed=4, scale=8.0)
    saturated = apply_temperature(logits, 1.0)
    assert np.any(saturated == 1.0)
```

**Lección**: cuando un test falla, la primera pregunta es si el código está mal o el
test está mal. Aquí era el test. Documentar el límite numérico es más útil que esconderlo.

---

### Bug 7 (mío, en los tests): afirmé una mejora de ECE que no siempre ocurre

**Síntoma**: el test que afirmaba `report.improved()` falló:

```
temperature=1.6437, ece_before=0.01665, ece_after=0.01898,
nll_before=0.08230, nll_after=0.06747
```

La temperatura se ajustó bien (detectó el exceso de confianza), el NLL mejoró
claramente, y el **ECE empeoró ligeramente**.

**Causa**: el ECE es un estadístico con agrupamiento en cubos. Con separación casi
perfecta ya es diminuto (0.017), y el ruido del agrupamiento domina.

**No era un bug del código.** Mi test afirmaba algo que no se cumple universalmente.

**Arreglo**: dividir en dos tests. Uno verifica que la temperatura detecta el exceso de
confianza (siempre cierto). Otro verifica la mejora de ECE **en datos donde la
descalibración es el error dominante** (clases solapadas), donde sí se cumple.

Y el informe imprime los dos valores de ECE en vez de afirmar una mejora.

**Lección**: una métrica que puede empeorar cuando el procedimiento funciona bien no
debe usarse como criterio de éxito sin matizar.

---

### El bug que tú arreglaste antes de esta sesión: MC Dropout sobre timm

Commit `d26341c`. Lo incluyo porque es el más instructivo de todos.

**Causa**: la receta estándar de MC Dropout es recorrer los módulos y poner los
`nn.Dropout` en modo entrenamiento. Lo verifiqué en el modelo real:

```
nn.Dropout modules found by walking the tree: 0
model.drop_rate attribute: 0.3
```

**Cero módulos.** timm guarda `drop_rate` como atributo y aplica el dropout
funcionalmente dentro de `forward_head`.

**Por qué es el peor tipo de bug**: el código no falla. Hace 20 pases idénticos, reporta
varianza cero, y el sistema afirma **confianza total en cada árbol**. La salida tiene el
formato correcto y es completamente falsa.

De tu mensaje de commit:

> uncertainty is what routes trees to human inspection, so a model that always claims
> certainty sends the technician nowhere useful.

**Arreglo**: `model.train()` completo (para que el dropout funcional dispare) con
BatchNorm devuelto a `eval()` (para que sus estadísticas no deriven), más
`has_active_dropout()` para que quien llame pueda verificarlo en vez de confiar.

---

### La trampa de diagnóstico que casi me comí

No es un bug del código, es un error de interpretación que estuve a punto de cometer.

Al probar MC Dropout con pesos aleatorios, la dispersión salía ~1e-5. Parecía que el
dropout no funcionaba y que la corrección de `d26341c` no bastaba.

Lo comprobé directamente, comparando logits en vez de probabilidades:

```
identical logits: False
max abs logit difference: 0.000115
logit scale (max abs value): 0.000126
```

La diferencia es el **91% de la escala total de los logits**. El dropout perturba
enormemente en términos relativos.

**Lo que pasaba**: un modelo sin entrenar produce logits cercanos a cero. Ahí el softmax
es casi plano, así que una perturbación grande del logit mueve poquísimo la
probabilidad. Con pesos preentrenados la dispersión sube a 0.33.

**Ahora está fijado como dos tests** (`test_dropout_perturbs_the_logits` y
`test_untrained_weights_give_a_tiny_spread`) con un comentario explicando que se
diagnostica con los logits, nunca con la dispersión.

---

## 5.3 Limitaciones reales

### 5.3.1 La limitación que invalida cualquier afirmación de producto

**Los datos no son del dominio de destino.**

| | Nuestras fotos | Lo que verá el producto |
|---|---|---|
| Punto de vista | lateral, a mano | nadir desde 15-25 m, o cercano desde dron |
| Escala | una hoja llena el cuadro | una copa entera, o un órgano |
| Fondo | controlado, uniforme | otros árboles, suelo, sombras |
| Enfermedades | mayoría ausentes de España | las que hay en Murcia |
| Etiquetas | sin verificar | requerirían un agrónomo |

Un PR-AUC de 1.0 aquí **no predice nada** sobre un dron en un limonar. Todas las
dimensiones que importan son distintas.

### 5.3.2 La deduplicación solo atrapa duplicados exactos

SHA-256 detecta ficheros byte a byte idénticos. **No** detecta:
- La misma foto guardada con otra calidad JPEG
- La misma foto redimensionada
- Dos fotos de la misma hoja desde ángulos ligeramente distintos

Las tres inflan las métricas si cruzan la frontera train/val.

La solución sería **hashing perceptual** (pHash, dHash) o comparar *embeddings* del
propio modelo. **No está implementado.** Es una de las explicaciones candidatas del
PR-AUC de 1.0 y la que más me preocupa.

### 5.3.3 Las etiquetas no están verificadas

Nadie con credenciales agronómicas revisó estas etiquetas. No hay qPCR, ni protocolo
documentado, ni medida de acuerdo entre etiquetadores.

La errata `leaf minnor` (por *leaf miner*) indica etiquetado manual sin revisión. Y
clases como `dry leaf` o `curl leaf` son síntomas con muchas causas posibles,
etiquetados como si fueran diagnósticos.

### 5.3.4 El conjunto de test está agotado

Con PR-AUC y ROC-AUC exactamente 1.0000, el test **ya no tiene resolución para medir
nada**. No puede distinguir un modelo bueno de uno excelente.

Consecuencias prácticas:
- No se pueden comparar dos configuraciones
- El early stopping necesitó un caso especial (§5.2, bug 2)
- Cualquier mejora futura será invisible

Para volver a medir haría falta un conjunto más difícil. En la práctica, imagen real.

### 5.3.5 La pasada de screening no tiene ningún dato

El proyecto define dos pasadas. Los datos cubren (parcialmente) la de inspección. Para
la de screening — detectar declive de copa desde 15-25 m — hay **cero** datos.

Y es la pasada que hace viable el coste por hectárea, porque es la que cubre la parcela
entera rápido.

### 5.3.6 La incertidumbre no está calibrada

MC Dropout es una aproximación variacional. La dispersión sirve para **ordenar**, no
es un intervalo de confianza.

Concretamente: si el sistema dice "incertidumbre 0.3", eso **no** significa "hay un 30%
de probabilidad de que me equivoque". Solo significa "más incierto que un 0.1".

El docstring lo dice:

> It is an approximation, not a calibrated posterior. [...] the spread it yields is
> informative for ranking, not a probability interval to quote at a grower.

### 5.3.7 Grad-CAM tiene resolución de 7×7

Cada celda del mapa cubre 32×32 píxeles de la imagen de 224×224. Es suficiente para
distinguir "mira el centro" de "mira el borde". **No** es suficiente para "mira la
lesión concreta".

Para eso harían falta métodos de mayor resolución (Grad-CAM++, Score-CAM) o capas
intermedias. No está implementado.

### 5.3.8 Lo que el detector de atajos no puede ver

`border_excess` detecta atajos **espaciales**: el modelo mira los bordes. No detecta:

- **Atajos de color global**: si una clase se fotografió con otra temperatura de color,
  el modelo puede usar el tono medio. La atención estaría repartida y el mapa saldría
  limpio.
- **Atajos de textura de fondo**: si el fondo es distinto pero ocupa el centro.
- **Atajos de nitidez**: si una clase se fotografió con mejor cámara.

El detector cubre el caso más probable en fotos de hoja centrada, no todos.

### 5.3.9 El umbral se elige en un conjunto con prevalencia equivocada

`threshold_for_specificity` calcula el cuantil sobre los negativos del split. Hay 150
muestras negativas en test.

Dos problemas:

1. **Granularidad**: con 150 negativos, los cuantiles caen entre valores discretos. Por
   eso pediste 90/95/99 y obtuviste 90.0/94.7/98.7.

2. **Distribución**: esos 150 negativos son fotos de hoja sana de primer plano. Los
   negativos de campo serán árboles sanos vistos desde un dron. El umbral calibrado en
   unos puede no transferir a los otros.

### 5.3.10 No hay validación cruzada

El reparto es uno solo, con semilla 42. No hay *k-fold*.

Consecuencia: todas las métricas tienen la varianza de una sola partición, y no se
puede estimar. Con PR-AUC 1.0 da igual; con un resultado intermedio, no sabrías si una
diferencia del 2% entre dos configuraciones es real.

No está implementado porque multiplicaría por k el coste de entrenamiento y, con este
dataset, mediría mejor un número que ya sabemos que no significa nada.

### 5.3.11 El pesado no sustituye a tener datos

`class_weights` arregla el sesgo del gradiente. **No** arregla que solo haya 89
imágenes de `citrus mite`. El pesado hace que el modelo preste más atención a la clase
minoritaria, no que tenga más información sobre ella.

### 5.3.12 Una sola semilla, una sola ejecución

Todo lo reportado es de una ejecución. Las redes neuronales tienen varianza entre
semillas, a veces de varios puntos de métrica.

Lo correcto sería ejecutar 3-5 semillas y reportar media y desviación. No se hizo.

---

## 5.4 Lo que haría a continuación, por orden

### 1. Conseguir imagen real (no es código)

Todo lo demás está bloqueado por esto. Concretamente hace falta:

- Vuelos nadir a 15-25 m sobre limonares de Murcia
- Vuelos cercanos (<2 m) sobre los árboles marcados
- Ground truth por árbol, verificado por un agrónomo
- Varias fechas, para capturar progresión estacional

Si tienes acceso a una cooperativa o ATRIA, esa gestión vale más que cualquier commit.

### 2. Hashing perceptual

El bloqueador técnico más probable del PR-AUC de 1.0. Implementar pHash y medir cuántos
duplicados no exactos hay cruzando splits. Si son muchos, el resultado actual es
ficción y hay que rehacer el reparto.

Es barato y podría invalidar el resultado, que es exactamente por lo que hay que
hacerlo.

### 3. Ejecutar `attention` sobre tu checkpoint real

Ya está implementado. Una ejecución te diría si ese 1.0 viene de leer la hoja o el
fondo. Es la información de mayor valor por minuto de cómputo disponible ahora mismo.

### 4. Entrenar con `binary: false`

Activaría `per-class` de verdad. Te diría si el modelo funciona en las clases presentes
en Murcia o si su puntuación la llevan las ausentes. Con los datos actuales, esa es la
pregunta más informativa que se puede responder.

### 5. Varias semillas

Tres ejecuciones con semillas distintas darían una idea de la varianza. Barato, y
necesario antes de comparar cualquier configuración.

---

## 5.5 Cómo leer un informe de este sistema

Por orden de importancia:

1. **Mira primero `attention`.** Si el modelo usa el fondo, nada más importa.
2. **Luego la prevalencia del test.** Dice cuánto vale todo lo demás.
3. **Luego el PPV proyectado**, no el PR-AUC.
4. **Luego la columna de alertas por caso**, que es la carga de trabajo real.
5. **Luego `per-class`**, si el checkpoint es multiclase: ¿funciona en lo que hay aquí?
6. **Luego la separación de errores de `uncertainty`**: ¿sirve la cola?
7. **El PR-AUC, al final, y con desconfianza si es alto.**

Y la regla general: **cuando un número sale demasiado bien, es una señal de alarma, no
una celebración.** Este proyecto entero está construido alrededor de esa idea.

---

Índice: [00-indice.md](00-indice.md)
