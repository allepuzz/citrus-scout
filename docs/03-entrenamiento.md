# 3. El entrenamiento: cada decisión, una por una

Este documento recorre el entrenamiento en el orden en que ocurre, explicando cada
decisión y por qué la alternativa era peor.

La referencia es tu ejecución real en Colab:

```
device: cuda
classes: ['healthy', 'affected']
model: efficientnet_b0, 4.0M parameters
data: train=3545 val=762 test=762
class weights: [1.606, 0.394]
warmup: backbone frozen for 1 epoch(s)
epoch   1 | train 3.2493 | val 2.4754 | PR-AUC 0.8244 | F1 0.6562 | 30s
backbone unfrozen: 4.0M trainable parameters
epoch   2 | train 1.4042 | val 0.9135 | PR-AUC 0.9951 | F1 0.9244 | 32s
...
epoch  15 | train 0.2434 | val 0.2910 | PR-AUC 1.0000 | F1 0.9951 | 25s
early stopping: no improvement for 5 epochs
training finished in 6m 45s
```

Cada línea de esa salida se explica aquí.

---

## 3.1 La configuración vive en YAML, no en código

El primer paso es cargar un fichero de configuración:

```yaml
name: leaf_baseline
device: auto
data:
  image_size: 224
  batch_size: 32
  seed: 42
  binary: true
  balance_classes: true
model:
  backbone: efficientnet_b0
  pretrained: true
  dropout: 0.3
optim:
  epochs: 15
  learning_rate: 0.0003
  weight_decay: 0.0001
  warmup_epochs: 1
  label_smoothing: 0.05
  grad_clip: 1.0
  amp: true
  early_stopping_patience: 5
field_prevalence: 0.02
```

**La decisión**: nada que afecte a un resultado vive en código ni en una celda de
notebook. El docstring de `config.py`:

> Nothing that affects a result lives in code, so a run can be reproduced from its
> config file alone, and a Colab run and a local run differ only in the file they
> were handed.

**Por qué importa tanto**: el modo de fallo que esto previene es el más corrosivo en
investigación. Entrenas, obtienes un número bueno, cambias tres cosas, entrenas otra
vez, y un mes después no puedes reconstruir qué produjo el número bueno. Los
notebooks son especialmente malos en esto porque el estado de las celdas es invisible.

Y hay una consecuencia práctica: `run_training` escribe el config **junto a la
salida**:

```python
config.to_yaml(run_dir / "config.yaml")
```

Así que `runs/leaf_baseline/config.yaml` es la receta exacta de ese checkpoint. Por
eso existe `runs/leaf_smoke/config.yaml` en tu repositorio aunque el smoke test local
no terminara: el config se escribe antes de entrenar.

### Validación con Pydantic

```python
@model_validator(mode="after")
def check_fractions(self) -> TrainingConfig:
    total = self.data.val_fraction + self.data.test_fraction
    if not 0.0 < total < 1.0:
        raise ValueError(f"val + test fractions must be in (0, 1), got {total}")
    if self.optim.warmup_epochs >= self.optim.epochs:
        raise ValueError(...)
```

**La decisión**: fallar al cargar, no a mitad de entrenamiento.

Si `warmup_epochs = 15` y `epochs = 15`, el backbone nunca se descongela y el
entrenamiento ajusta solo la cabeza durante 15 épocas. No da error, simplemente
produce un modelo malo. Detectarlo requeriría darse cuenta de que el número es peor
de lo esperado.

Estas validaciones convierten un fallo silencioso en un error inmediato.

---

## 3.2 La reproducibilidad

```python
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
```

Hay **cuatro** generadores de números aleatorios distintos en juego:

1. `random` — el de Python, usado por albumentations internamente
2. `np.random` — numpy, usado en el reparto de datos
3. `torch.manual_seed` — CPU
4. `torch.cuda.manual_seed_all` — todas las GPUs

Sembrar solo uno deja los otros tres libres. Es un error muy común.

### Las dos líneas de cudnn, que son un compromiso real

```python
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

**`benchmark = True`** (el valor por defecto de PyTorch) hace que cuDNN pruebe varios
algoritmos de convolución al empezar y se quede con el más rápido para tu hardware.
Puede dar entre un 5% y un 15% de velocidad.

Pero la elección depende de **medir tiempos**, que varía con la carga de la máquina.
Dos ejecuciones idénticas pueden elegir algoritmos distintos y, como los algoritmos
acumulan en coma flotante en orden distinto, producir resultados ligeramente
diferentes.

**`deterministic = True`** fuerza algoritmos con orden de operaciones fijo.

**El compromiso, explícito en el docstring**:

> The throughput cost is small next to being unable to reproduce a result.

Con épocas de 25-32 segundos, un 10% son 3 segundos. Perder la capacidad de
reproducir un resultado cuesta horas de depuración. Es una compra fácil.

---

## 3.3 El backbone: EfficientNet-B0 y la transferencia

```python
return timm.create_model(
    backbone,            # "efficientnet_b0"
    pretrained=pretrained,   # True
    num_classes=num_classes, # 2
    drop_rate=dropout,       # 0.3
)
```

### Qué es la transferencia de aprendizaje

Entrenar una red convolucional desde cero necesita cientos de miles de imágenes.
Tenemos 3.545.

**Transferencia**: coges un modelo ya entrenado en ImageNet (1,2 millones de imágenes,
1000 categorías de objetos cotidianos), le quitas la última capa, le pones una nueva
de 2 salidas, y entrenas eso.

**Por qué funciona**: las primeras capas de una red convolucional aprenden cosas
genéricas — detectores de bordes, gradientes de color, texturas. Eso es útil para
cualquier tarea visual. El docstring de `classifier.py`:

> the pre-trained features already encode edges, texture and colour gradients, which
> is most of what separates a necrotic lesion from a healthy lamina.

Una lesión necrótica se distingue de una lámina sana por textura y color. ImageNet ya
enseñó a la red a ver textura y color.

### Por qué EfficientNet-B0 y no algo más grande

EfficientNet-B0 es el más pequeño de su familia: **4,0 millones de parámetros**. Para
comparar, ResNet-50 tiene 25M y ConvNeXt-Base tiene 88M.

**La razón, del docstring**:

> on this problem the ceiling is data quality rather than model capacity, so a larger
> backbone mostly buys longer epochs.

Esto es un juicio que conviene desmenuzar. El argumento es: con 3.545 imágenes,
etiquetas no verificadas, y un dominio que no coincide con el destino final, **el
cuello de botella no es la capacidad del modelo**. Un modelo 20 veces más grande
aprendería las mismas correlaciones espurias, más rápido y con más sobreajuste.

Y tu resultado lo confirma empíricamente: PR-AUC 1.0 con 4M de parámetros. No hay
margen arriba que un modelo mayor pudiera ocupar.

La segunda razón es práctica: cabe en una GPU de portátil de 4 GB con un batch útil.
Eso permite desarrollar localmente en vez de depender de Colab para cada prueba.

### El dropout del 30%

```python
dropout: 0.3
```

El dropout apaga aleatoriamente el 30% de las activaciones antes de la capa final
durante el entrenamiento. Es regularización: impide que el modelo dependa de ninguna
neurona concreta.

**Pero aquí tiene un segundo propósito que es igual de importante**, y está en el
docstring:

> Dropout is enabled by default because it is what MC Dropout later samples over to
> estimate uncertainty. Uncertainty is not an afterthought here: the product routes
> uncertain trees to human inspection, so the model must be able to say it does not
> know.

Si el dropout fuera 0, MC Dropout no tendría nada que muestrear y la incertidumbre
sería cero en todas partes. La decisión de arquitectura está acoplada a la decisión de
producto.

---

## 3.4 Binario: cómo y por qué

```python
class BinaryLabelWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: LeafDataset, binary: list[int]) -> None:
        if len(dataset) != len(binary):
            raise ValueError(...)
        self.dataset = dataset
        self.binary = binary
        self.classes = ["healthy", "affected"]
```

### Por qué binario y no 17 clases

```python
def binary_labels(samples):
    return [0 if relevance_of(s.label) is LocalRelevance.HEALTHY else 1 for s in samples]
```

**La razón de producto**, del docstring:

> This is the task the product actually performs: the technician needs to know which
> trees to visit, not the exact pathogen. Naming the pathogen is a separate, harder
> problem, and for classes absent from Spain it is one we cannot honestly claim to
> solve.

Tres argumentos encadenados:

1. **El técnico necesita saber a dónde ir**, no el nombre del patógeno. La decisión
   operativa es binaria: ¿visito este árbol?

2. **Nombrar el patógeno es un problema más difícil** y separado. Requiere más datos
   por clase y etiquetas fiables.

3. **Para las clases ausentes de España no podemos afirmar honestamente que lo
   resolvemos.** Si el modelo dice "cancro", y el cancro no existe en España, ¿qué
   hace el técnico con esa información?

### La decisión técnica: envolver en vez de reetiquetar

Esto es más astuto de lo que parece. El wrapper **mantiene el dataset original
intacto** y solo traduce las etiquetas al pedirlas.

**Por qué**, del docstring:

> Wrapping rather than relabelling keeps the original class of every sample available
> for error analysis, which is where the interesting failures show up: knowing the
> model confuses sooty mould with healthy is actionable, knowing it got a "1" wrong
> is not.

"Se equivocó en un 1" no te dice nada. "Confunde negrilla con hoja sana" es
accionable: la negrilla es un oscurecimiento difuso, quizás necesita otra aumentación
o más ejemplos.

Esa información es exactamente lo que consume el módulo `per_class.py` (§7).

---

## 3.5 El pesado de clases: `[1.606, 0.394]`

Esta línea de tu salida merece su propia sección.

```python
def class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = Counter(labels)
    total = len(labels)
    weights = torch.tensor(
        [total / (num_classes * max(counts.get(c, 0), 1)) for c in range(num_classes)],
        dtype=torch.float32,
    )
    return weights / weights.mean()
```

### El cálculo con tus números reales

Entrenamiento: 3.545 imágenes, de las cuales 698 sanas y 2.847 afectadas.

```
peso_sana     = 3545 / (2 × 698)  = 2.539
peso_afectada = 3545 / (2 × 2847) = 0.6226

media = (2.539 + 0.6226) / 2 = 1.581

peso_sana     = 2.539  / 1.581 = 1.606   ✓
peso_afectada = 0.6226 / 1.581 = 0.394   ✓
```

Coincide exactamente con lo que imprimió tu entrenamiento.

### Qué hace

La pérdida de cada muestra se multiplica por el peso de su clase. Una imagen sana
contribuye 1.606 veces su pérdida; una afectada, 0.394.

**Por qué es necesario**: con el 80.3% de afectadas, un modelo que siempre dice
"afectada" tiene una pérdida bastante baja. El gradiente apunta hacia esa estrategia.
El docstring de `loop.py`:

> The datasets run 80% affected. Left alone, the model learns that predicting
> "affected" is usually right, which is the exact opposite of the field situation it
> will be deployed into.

"Lo contrario de la situación de campo" es la clave. En campo el 98% están sanos. Un
modelo sesgado hacia "afectada" es inútil justo al revés.

### El detalle de la normalización a media 1

```python
return weights / weights.mean()
```

Sin esto, los pesos serían 2.539 y 0.6226, cuya media es 1.58. La pérdida total sería
un 58% mayor que sin pesar.

**Por qué importa**: la magnitud de la pérdida determina la magnitud del gradiente, y
eso interactúa con la tasa de aprendizaje. Si activar el pesado multiplicara la
pérdida por 1.58, habría que retocar el learning rate para compensar.

Normalizando a media 1, **activar o desactivar `balance_classes` no cambia la escala
de la pérdida**, y el learning rate sigue siendo válido. Es un detalle pequeño que
evita una interacción sutil entre dos hiperparámetros.

### La guarda contra división por cero

```python
max(counts.get(c, 0), 1)
```

Si una clase no aparece en el entrenamiento, `counts.get(c, 0)` da 0 y dividiríamos
por cero. El `max(..., 1)` lo evita. No debería pasar con reparto estratificado, pero
el coste de la guarda es nulo.

---

## 3.6 El optimizador y la pérdida

### Entropía cruzada con suavizado de etiquetas

```python
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=config.optim.label_smoothing)
```

**Entropía cruzada** es la pérdida estándar de clasificación. Para una muestra cuya
clase correcta es `c`:

```
pérdida = −ln( P(clase c) )
```

Si el modelo da P(correcta) = 0.99, la pérdida es 0.01. Si da 0.01, la pérdida es 4.6.
Castiga fuertemente estar confiadamente equivocado.

**Suavizado de etiquetas** (`label_smoothing: 0.05`): en vez de entrenar contra el
objetivo `[0, 1]`, se entrena contra `[0.025, 0.975]`.

**Por qué**: sin suavizado, el modelo persigue P = 1.0 exactamente, lo que solo se
consigue con logits infinitos. El resultado es exceso de confianza extremo. El
suavizado le da un objetivo alcanzable.

Y hay una conexión directa con §5: el suavizado **reduce** el problema de calibración,
aunque no lo elimina. Tus umbrales de 0.009 muestran que con 0.05 de suavizado el
modelo sigue siendo muy confiado.

### AdamW

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.optim.learning_rate,      # 0.0003
    weight_decay=config.optim.weight_decay,  # 0.0001
)
```

**Adam** adapta la tasa de aprendizaje por parámetro, usando estimaciones del primer y
segundo momento del gradiente. En la práctica converge mucho más rápido que el
descenso de gradiente estocástico simple y necesita menos ajuste.

**La "W"** es *decoupled weight decay*. El **decaimiento de pesos** es regularización:
empuja los pesos hacia cero, penalizando modelos complejos. En el Adam original se
implementaba sumándolo al gradiente, lo que interactuaba mal con la adaptación por
parámetro. AdamW lo aplica por separado.

Es la elección por defecto correcta para ajuste fino de modelos preentrenados.

### Learning rate 3e-4

`0.0003`. Es el valor convencional para ajuste fino con Adam.

**El razonamiento**: estamos ajustando un modelo que ya sabe cosas útiles. Una tasa
alta (por ejemplo 1e-2) haría pasos grandes que destruirían las características
preentrenadas. Una tasa muy baja (1e-6) tardaría eternidades.

3e-4 es aproximadamente diez veces menor que lo que usarías entrenando desde cero, que
es la regla de oro para ajuste fino.

### Programador cosenoidal

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(config.optim.epochs - config.optim.warmup_epochs, 1)
)
```

La tasa de aprendizaje desciende siguiendo un cuarto de coseno, desde 3e-4 hasta casi
cero a lo largo del entrenamiento.

**Por qué**: al principio quieres pasos grandes para moverte hacia una buena región.
Al final quieres pasos pequeños para asentarte en un mínimo en vez de rebotar
alrededor.

El coseno empieza a bajar despacio, acelera en medio, y se aplana al final. Empíricamente
funciona mejor que reducciones escalonadas y no tiene hiperparámetros que ajustar.

**El detalle del `T_max`**: `epochs - warmup_epochs` = 15 − 1 = 14. El programador solo
cuenta las épocas en las que realmente se entrena el backbone. Y el código solo lo
avanza después del warmup:

```python
if epoch > config.optim.warmup_epochs:
    scheduler.step()
```

Sin eso, el coseno habría consumido parte de su recorrido durante el warmup, cuando el
backbone está congelado y la tasa no le afecta.

---

## 3.7 El warmup: congelar el backbone una época

Esta es la decisión que explica el salto más grande de tu salida.

```python
if config.optim.warmup_epochs > 0:
    freeze_backbone(model)
    print(f"warmup: backbone frozen for {config.optim.warmup_epochs} epoch(s)")

for epoch in range(1, config.optim.epochs + 1):
    if epoch == config.optim.warmup_epochs + 1:
        unfreeze_all(model)
```

### Qué hace `freeze_backbone`

```python
def freeze_backbone(model: nn.Module) -> None:
    classifier = model.get_classifier()
    classifier_params = {id(p) for p in classifier.parameters()}
    for param in model.parameters():
        param.requires_grad = id(param) in classifier_params
```

`requires_grad = False` significa que ese parámetro no recibe gradientes y no se
actualiza. Durante la época 1, solo se entrena la capa final.

Nota el uso de `id(p)`: compara identidad de objetos, no valores. Comparar tensores
con `==` daría un tensor de booleanos, no un booleano.

### Por qué es necesario

La cabeza clasificadora es **nueva y aleatoria**. El backbone lleva 1,2 millones de
imágenes de entrenamiento.

En los primeros pasos, una cabeza aleatoria produce predicciones malísimas, y por tanto
**gradientes enormes**. Esos gradientes se retropropagan al backbone y pueden destruir
características que tardaron días de GPU en aprenderse.

El docstring lo dice:

> a randomly initialised head produces large gradients that can wreck pre-trained
> features in the first few hundred steps.

Congelando una época, la cabeza se asienta en algo razonable antes de que el backbone
empiece a moverse.

### La evidencia en tus números

```
epoch   1 | train 3.2493 | val 2.4754 | PR-AUC 0.8244 | F1 0.6562   ← backbone congelado
backbone unfrozen: 4.0M trainable parameters
epoch   2 | train 1.4042 | val 0.9135 | PR-AUC 0.9951 | F1 0.9244   ← descongelado
```

La pérdida de validación cae de 2.4754 a 0.9135 y el PR-AUC salta de 0.8244 a 0.9951
**en una sola época**. Eso es el backbone empezando a adaptarse.

Y fíjate que el PR-AUC de la época 1 (0.8244) es apenas mejor que la prevalencia
(0.803), que es lo que daría un modelo aleatorio. Con el backbone congelado y una
cabeza recién entrenada, el modelo casi no discrimina.

---

## 3.8 La selección del modelo y el early stopping

Aquí está el bug que arreglamos en el PR #6, y merece la explicación completa.

### Selección sobre PR-AUC, no sobre la pérdida

```python
if pr_auc > best_pr_auc:
    best_pr_auc = pr_auc
    save_checkpoint(run_dir / "best.pt", model, config, classes, epoch, report)
```

El checkpoint que se guarda es el de mejor PR-AUC de validación, **no** el de menor
pérdida.

**Por qué**, del comentario del código:

> Select on PR-AUC: validation loss can fall while minority-class ranking gets worse,
> and ranking is what the operating threshold depends on.

La pérdida premia predicciones confiadas y correctas. Con el 80% de una clase, el
modelo puede reducir la pérdida volviéndose más confiado en la mayoría mientras
empeora su ordenación de la minoría. El PR-AUC mide precisamente lo segundo.

Y lo segundo es lo que importa, porque el umbral de operación se elige del ranking.

### El bug original

El código antes del PR #6 era:

```python
if report["pr_auc"] > best_pr_auc:
    best_pr_auc = report["pr_auc"]
    epochs_without_improvement = 0
    save_checkpoint(...)
else:
    epochs_without_improvement += 1
    if patience is not None and epochs_without_improvement >= patience:
        print(f"early stopping: no improvement for {patience} epochs")
        break
```

El problema: **PR-AUC está acotado en 1.0**. Cuando una ejecución lo alcanza, la
comparación `>` no puede volver a ser cierta. Nunca. Cada época posterior cae en el
`else` e incrementa el contador.

En tu ejecución:

```
epoch  10 | ... | PR-AUC 1.0000 |    ← llega al máximo
epoch  11 | ... | PR-AUC 0.9999 |
epoch  12 | ... | PR-AUC 1.0000 |
epoch  13 | ... | PR-AUC 1.0000 |
epoch  14 | val 0.2902 | PR-AUC 1.0000 |   ← mejor pérdida de validación de todas
epoch  15 | ... | PR-AUC 1.0000 |
early stopping: no improvement for 5 epochs
```

Paró en la 15 diciendo "sin mejora durante 5 épocas" mientras **la pérdida de
validación seguía bajando** y alcanzó su mínimo en la época 14.

**El mensaje era falso.** No era una meseta, era una métrica agotada.

### Por qué importa más allá de este caso

En este dataset solo se desperdició la cola de un entrenamiento. Pero considera un
dataset difícil donde un pico del 0.85 *sí* sea una meseta real de la que el modelo
sale tras unas épocas. El contador correría durante esa meseta y **cortaría un modelo
que estaba mejorando de verdad**, informando mal del motivo.

### El arreglo

```python
def should_stop(*, pr_auc, best_pr_auc, epochs_without_improvement, patience, min_delta):
    improved = pr_auc > best_pr_auc + min_delta
    stalled = 0 if improved else epochs_without_improvement + 1

    if patience is not None and stalled >= patience:
        return True, stalled, f"early stopping: no PR-AUC gain for {patience} epochs"

    if pr_auc >= 1.0:
        return True, stalled, "PR-AUC saturated at 1.0000: the task is too easy to rank on"

    return False, stalled, None
```

Tres cambios, cada uno con su razón:

**1. Separar dos preguntas que estaban confundidas.**

| Pregunta | ¿Umbral? | Decide |
|---|---|---|
| ¿Es el mejor modelo hasta ahora? | ninguno, cualquier ganancia vale | el checkpoint |
| ¿Nos rendimos? | `min_delta` | el early stopping |

El checkpoint debe seguir cualquier mejora, por pequeña que sea. La decisión de
rendirse necesita un umbral.

**2. `min_delta = 1e-4`.**

Sin un mínimo, el ruido del cuarto decimal se lee como progreso y la paciencia nunca
expira en una meseta. Con él, una ganancia de 0.00001 no cuenta como mejora.

**3. Parada explícita por saturación.**

```python
if pr_auc >= 1.0:
    return True, stalled, "PR-AUC saturated at 1.0000: the task is too easy to rank on"
```

Cuando la métrica llega a su máximo, no queda resolución con la que guiar. Decirlo
así es honesto; dejar que expire la paciencia e informar de una meseta es mentir.

Ahora tu ejecución pararía en la época 10 con:

```
stopping at epoch 10: PR-AUC saturated at 1.0000: the task is too easy to rank on
```

Mismo modelo, 2 minutos menos, y un mensaje que dice la verdad sobre el dataset.

### Cómo se verificó que los tests sirven

Escribí los tests y luego **revertí `should_stop` a la lógica vieja**. Cuatro de ellos
fallaron. Eso demuestra que fijan la regresión real y no mi implementación concreta. Un
test que pasa con el bug presente no vale nada.

---

## 3.9 AMP: precisión mixta

```python
scaler = torch.amp.GradScaler("cuda") if config.optim.amp and device.type == "cuda" else None

if scaler is not None:
    with torch.amp.autocast("cuda"):
        loss = criterion(model(images), labels)
    scaler.scale(loss).backward()
    if grad_clip is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
```

**AMP** = *Automatic Mixed Precision*. Los cálculos se hacen en coma flotante de 16
bits en vez de 32 donde es seguro.

**Ventajas**: aproximadamente la mitad de memoria y hasta el doble de velocidad en GPUs
con núcleos tensoriales (la T4 los tiene).

**El problema que introduce**: float16 tiene un rango mucho más estrecho. Los
gradientes pequeños se redondean a cero — *underflow*. Y un gradiente cero significa
que ese parámetro no aprende.

**La solución, el `GradScaler`**: multiplica la pérdida por un número grande antes de
retropropagar, así los gradientes quedan dentro del rango representable. Luego divide
los gradientes por ese mismo número antes de actualizar. Y ajusta el factor
dinámicamente: si detecta infinitos, lo reduce.

### El detalle crítico del recorte de gradientes

```python
scaler.unscale_(optimizer)                  # PRIMERO deshacer la escala
torch.nn.utils.clip_grad_norm_(...)         # LUEGO recortar
```

El comentario del código:

> Unscale before clipping, or the threshold applies to scaled gradients.

Esto es un error fácil de cometer y difícil de detectar. El recorte limita la norma del
gradiente a 1.0. Pero si los gradientes están escalados por, digamos, 65536, su norma
es 65536 veces mayor, y recortar a 1.0 los aplastaría a casi nada. El modelo no
aprendería y no habría ningún error.

### `zero_grad(set_to_none=True)`

```python
optimizer.zero_grad(set_to_none=True)
```

PyTorch **acumula** gradientes: si no los borras, el gradiente del lote 2 se suma al
del lote 1.

`set_to_none=True` pone los gradientes a `None` en vez de a un tensor de ceros. Es
ligeramente más rápido (no hay que escribir ceros) y ahorra memoria.

---

## 3.10 Recorte de gradientes

```python
grad_clip: 1.0
```

Si la norma L2 del vector de todos los gradientes supera 1.0, se reescala todo el
vector para que valga exactamente 1.0.

**Contra qué protege**: un lote patológico (imágenes raras, o una combinación
desafortunada) puede producir un gradiente enorme. Un paso del optimizador con ese
gradiente puede lanzar los pesos a una región de la que no se recuperan. Se ve como una
pérdida que salta a `NaN` y nunca vuelve.

El recorte limita el daño de cualquier lote individual. Es un seguro baratísimo.

---

## 3.11 Tamaño de lote, y por qué Colab usa 64

Config: `batch_size: 32`. Pero el notebook de Colab ejecuta:

```python
cmd = "citrus-scout train --config configs/leaf_baseline.yaml --batch-size 64"
```

**Por qué 32 por defecto**: cabe en una GPU de portátil de 4 GB.

**Por qué 64 en Colab**, del notebook:

> A T4 has 16 GB, four times the laptop card this was written on, so the batch size
> is raised accordingly. Larger batches also make the class weighting behave more
> predictably, since each batch is more likely to contain minority-class examples.

El segundo argumento es el interesante. Con el 19.7% de imágenes sanas y lotes de 32,
un lote tiene unas 6 imágenes sanas de media — pero por azar puede tener 1, o ninguna.
En un lote sin ninguna imagen sana, el peso de 1.606 no se aplica a nada y el gradiente
de ese paso está sesgado.

Con lotes de 64, la media sube a 12 y la varianza relativa baja. El pesado se comporta
de forma más estable.

Y nota que el `--batch-size` de la CLI **sobreescribe el config**:

```python
if batch_size is not None:
    settings.data.batch_size = batch_size
```

Eso es una excepción deliberada al principio de "todo en el config": es un parámetro
de hardware, no de experimento. No cambia el resultado esperado, solo cómo de rápido
se llega.

---

## 3.12 Los workers del DataLoader

```python
def resolve_num_workers(self) -> int:
    if self.num_workers is not None:
        return self.num_workers
    available = os.cpu_count() or 1
    return max(0, min(available - 1, 4))
```

Los **workers** son procesos separados que cargan y transforman imágenes en paralelo
mientras la GPU calcula. Sin ellos, la GPU espera a que la CPU decodifique JPEGs.

### El bug que arreglamos en el PR #6

El valor era `4` fijo. Colab da **2 CPUs**. Resultado, en cada ejecución:

```
UserWarning: This DataLoader will create 4 worker processes in total. Our suggested
max number of worker in current system is 2 ...
```

Cuatro procesos peleando por dos núcleos es contraproducente: el coste de cambio de
contexto puede superar el beneficio del paralelismo.

### La lógica del arreglo

- **`available - 1`**: deja un núcleo al proceso principal, que tiene que alimentar la
  GPU y ejecutar el bucle de entrenamiento.
- **`min(..., 4)`**: tope en 4. Más allá, la decodificación de JPEG deja de ser el
  cuello de botella y solo se añade memoria y sobrecarga.
- **`max(0, ...)`**: nunca negativo. Con 1 CPU, `0` significa "cargar en el proceso
  principal", que es válido.
- **`is not None`** y no `if self.num_workers`: así un `0` explícito se respeta. Con
  la comprobación de veracidad, `0` se trataría como "sin configurar".

Resultado: tu escritorio de 20 núcleos resuelve a 4 (igual que antes), Colab resuelve
a 1, y ya no hay aviso.

### `persistent_workers`

```python
"persistent_workers": workers > 0,
```

Mantiene los procesos vivos entre épocas. Sin esto, se crean y destruyen 15 veces (una
por época), y cada creación cuesta segundos.

La condición `workers > 0` es necesaria: PyTorch lanza un error si pides workers
persistentes con cero workers.

### `pin_memory`

```python
"pin_memory": torch.cuda.is_available(),
```

Memoria "anclada" (*pinned*) es memoria de CPU que el sistema operativo no puede
mover. Permite transferencias a GPU por DMA, que son más rápidas.

Solo tiene sentido con GPU, de ahí la condición.

### `drop_last` solo en entrenamiento

```python
train_loader = DataLoader(train_ds, ..., shuffle=True, drop_last=True, **common)
val_loader = DataLoader(val_ds, ..., shuffle=False, **common)
```

3.545 imágenes con lotes de 32 dan 110 lotes completos y uno final de 25.

**En entrenamiento se descarta** ese último lote incompleto. Razón: un lote más pequeño
produce una estimación de gradiente más ruidosa, y con BatchNorm también estadísticas
menos fiables.

**En validación y test no se descarta**, porque descartar muestras cambiaría las
métricas. Sería evaluar sobre 737 imágenes en vez de 762 por un detalle de
implementación.

Y `shuffle=True` solo en entrenamiento: mezclar el orden evita que el modelo aprenda
patrones del orden de los datos. En evaluación el orden es irrelevante para el
resultado, y mantenerlo fijo hace que las predicciones se alineen con las etiquetas
(lo cual MC Dropout aprovecha, §6).

---

## 3.13 Qué se guarda en un checkpoint

```python
torch.save({
    "model_state": model.state_dict(),
    "config": config.model_dump(mode="json"),
    "classes": classes,
    "epoch": epoch,
    "metrics": {k: v for k, v in metrics.items() if isinstance(v, int | float)},
}, path)
```

No solo los pesos. **Todo lo necesario para interpretarlos.**

**`classes`** es el campo crítico, y el docstring lo explica:

> The class list is stored with the weights: a checkpoint whose class order is unknown
> produces confidently wrong predictions rather than an error.

Si el modelo se entrenó con `['healthy', 'affected']` y lo cargas asumiendo
`['affected', 'healthy']`, todas las predicciones salen invertidas. **Sin ningún
error.** El modelo funciona, da números plausibles, y son exactamente lo contrario de
la verdad.

**`config`** permite que `load_checkpoint` reconstruya la arquitectura correcta:

```python
model = build_model(
    backbone=config.model.backbone,
    num_classes=len(classes),
    pretrained=False,      # los pesos vienen del checkpoint
    dropout=config.model.dropout,
)
```

Nota `pretrained=False`: no hace falta descargar pesos de ImageNet si los vamos a
sobreescribir inmediatamente. Ahorra una descarga.

**El filtro de métricas**:

```python
{k: v for k, v in metrics.items() if isinstance(v, int | float)}
```

El diccionario de métricas contiene `operating_points`, que es una lista de objetos
`OperatingPoint`. Guardarlos requeriría pickle de clases propias, lo que rompe el
checkpoint si la clase cambia de nombre o de módulo. Filtrar a solo números mantiene
el checkpoint portable.

---

## 3.14 El historial

```python
def write_history(path: Path, history: list[EpochResult]) -> None:
    payload = [{"epoch": r.epoch, "train_loss": r.train_loss, ...} for r in history]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
```

Se escribe `history.json` con las métricas de cada época, incluida la tasa de
aprendizaje (en `extra`).

**Por qué**: permite dibujar curvas de aprendizaje después, sin reentrenar. Las curvas
son la forma más rápida de diagnosticar problemas: si la pérdida de entrenamiento baja
y la de validación sube, es sobreajuste; si ninguna baja, la tasa de aprendizaje está
mal.

JSON y no pickle, para que se pueda leer desde cualquier herramienta.

---

## 3.15 Por qué Colab y no el portátil

Del README:

> Training on a laptop GPU is slow and heats the machine, so the intended path is
> Colab.

Pero la decisión de diseño interesante es **cómo** se usa Colab. Del notebook:

> This notebook is deliberately thin. All the logic lives in the repository, version
> controlled and tested; the notebook only clones it, installs dependencies, and calls
> the same CLI you would run locally.
>
> That is the point: a result produced here is reproducible on any machine, because
> nothing that affects it is defined in a notebook cell.

El notebook tiene 10 celdas de código y ninguna define lógica. Solo:
1. Comprueba la GPU
2. Clona el repo
3. Instala
4. Monta Drive y localiza el archivo
5. Llama a `citrus-scout train`
6. Llama a `citrus-scout evaluate`
7. Copia los resultados a Drive

**Por qué esto importa**: los notebooks son el enemigo de la reproducibilidad. El
estado de las celdas es invisible, el orden de ejecución es arbitrario, y el control de
versiones de un `.ipynb` es ilegible. Un resultado producido en una celda no se puede
reconstruir de forma fiable.

Manteniendo el notebook como un lanzador, el resultado de Colab y el local son el mismo
código con el mismo config.

---

Siguiente: [04-evaluacion.md](04-evaluacion.md), las cuatro herramientas de evaluación.
