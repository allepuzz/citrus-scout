# 4. La evaluación: cuatro herramientas, cuatro preguntas

Hay cuatro comandos de evaluación. No son redundantes: cada uno responde algo que los
otros no pueden.

| Comando | Pregunta | Qué mirar |
|---|---|---|
| `evaluate` | ¿Qué tan bueno es? | PPV a prevalencia de campo |
| `uncertainty` | ¿Dónde debe mirar un humano? | Si la incertidumbre es mayor en los errores |
| `per-class` | ¿Qué confunde? | Recall en presentes vs ausentes |
| `attention` | ¿Está haciendo trampas? | Fracción de imágenes explicadas por el borde |

El último es el que audita a los otros tres. Volveré a eso.

---

## 4.1 `evaluate`: el informe principal

```bash
citrus-scout evaluate --checkpoint runs/leaf_baseline/best.pt --archive leaf_dataset.zip
```

### Lo que imprimió tu ejecución

```
checkpoint: runs/leaf_baseline/best.pt
split: test, n=762, classes=['healthy', 'affected']
loss: 0.0609

      Ranking quality
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ metric          ┃  value ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ PR-AUC          │ 1.0000 │
│ ROC-AUC         │ 1.0000 │
│ F1 at 0.5       │ 0.9967 │
│ test prevalence │  80.3% │
└─────────────────┴────────┘

        Operating points, PPV projected to 2.0% field prevalence
┏━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ threshold ┃ sensitivity ┃ specificity ┃   PPV ┃ alerts per true case ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│     0.009 │      100.0% │       90.0% │ 16.9% │                  5.9 │
│     0.013 │      100.0% │       94.7% │ 27.7% │                  3.6 │
│     0.046 │      100.0% │       98.7% │ 60.5% │                  1.7 │
└───────────┴─────────────┴─────────────┴───────┴──────────────────────┘
```

### La estructura en dos tablas, que es deliberada

**Tabla 1, "calidad de ranking"**: métricas que no dependen de un umbral. Miden si el
modelo ordena bien.

**Tabla 2, "puntos de operación"**: métricas que sí dependen de un umbral. Miden qué
pasa cuando tomas decisiones.

La separación existe porque son preguntas distintas y se responden en fases distintas
del desarrollo. El ranking es propiedad del modelo entrenado; el punto de operación es
una decisión de producto posterior.

### Por qué la prevalencia de test está en la tabla

La fila `test prevalence: 80.3%` no es una métrica de rendimiento. Está ahí para que
**no puedas leer las otras cifras sin verla**.

Si la quitaras, un lector vería PR-AUC 1.0000 y concluiría que el problema está
resuelto. Con el 80.3% a la vista, la interpretación correcta está disponible.

### El mensaje final, que es parte del diseño

```python
console.print(
    "\n[yellow]Read the PPV column, not PR-AUC.[/yellow] This test set is "
    f"{report['test_prevalence']:.0%} affected; a real grove runs "
    f"{report['field_prevalence']:.0%}. The last column is how many trees a "
    "technician would visit per genuine case found."
)
```

**La decisión**: la herramienta interpreta sus propios números en vez de dejarlo al
lector.

El razonamiento: un informe que se puede malinterpretar se malinterpretará,
especialmente meses después o por otra persona. Escribir la interpretación correcta al
lado del número cuesta tres líneas y previene la conclusión errónea más probable.

### Cómo se eligen los tres umbrales

```python
target_specificities: tuple[float, ...] = (0.90, 0.95, 0.99)
```

No son umbrales fijos. Son **especificidades objetivo**, y el umbral que las consigue
se calcula de los datos:

```python
def threshold_for_specificity(y_true, y_score, target_specificity):
    negative_scores = y_score[y_true == 0]
    if negative_scores.size == 0:
        raise ValueError("no negative samples available to compute specificity")
    return float(np.quantile(negative_scores, target_specificity))
```

**Cómo funciona**: coge solo las puntuaciones de las muestras **negativas** (sanas) y
toma el cuantil. El cuantil 0.90 de los negativos es el valor por debajo del cual está
el 90% de ellos. Usarlo como umbral significa que el 90% de los sanos se clasifican
correctamente: especificidad del 90%, por construcción.

Lo verifiqué con datos sintéticos: pidiendo 90% se obtiene exactamente 90.0%.

**Por qué especificar por especificidad y no por umbral**: un umbral de 0.5 no
significa nada comparable entre modelos. "99% de especificidad" sí. Es el lenguaje en
el que se puede discutir una decisión operativa: *¿cuántas falsas alarmas tolera el
técnico?*

Y fíjate que tus especificidades reales salieron 90.0%, 94.7% y 98.7% en vez de
90/95/99 exactos. Con 150 muestras negativas en el test, el cuantil cae entre dos
valores discretos. Es granularidad de muestra finita, no un error.

**Por qué esta es la elección de diseño central del proyecto**, del docstring:

> This is how the system's operating point is tuned. The cost of skipping a healthy
> tree is low, but flooding the technician with false positives makes them abandon the
> tool. Hence the operating point is set by specificity rather than by maximising F1.

Maximizar F1 trataría falsos positivos y falsos negativos como igual de costosos. En
campo no lo son. Esta asimetría es la razón de ser de toda la tabla.

### La columna de "alertas por caso real"

```python
alerts = f"{1 / point.ppv:.1f}" if point.ppv > 0 else "inf"
```

Es simplemente `1/PPV`. Si el PPV es 16.9%, hacen falta 5.9 alertas para encontrar un
caso real.

**Por qué añadir una columna que es el inverso de otra**: porque "16.9% de PPV" es
abstracto y "6 caminatas por cada árbol enfermo" no lo es. Es la misma información en
las unidades en las que se toma la decisión.

### Validación de entradas

```python
unique = np.unique(y_true)
if not np.all(np.isin(unique, [0, 1])):
    raise ValueError(f"y_true must be binary (0/1), found {unique}")
```

Si alguien pasa etiquetas multiclase a una función binaria, falla inmediatamente en vez
de calcular un número sin sentido. Las métricas silenciosamente erróneas son el peor
tipo de bug en un proyecto de ML, porque el resultado parece válido.

---

## 4.2 `uncertainty`: el triaje

```bash
citrus-scout uncertainty --checkpoint runs/leaf_baseline/best.pt --passes 20
```

### Qué problema resuelve

El informe de `evaluate` te da un número: a 98.7% de especificidad, 1.7 alertas por
caso real. Pero **todas esas alertas llegan con el mismo peso**. El técnico no tiene
forma de priorizar.

La incertidumbre convierte una lista plana en una cola ordenada.

### Las tres categorías

```python
def triage(self, *, affected_threshold, healthy_threshold, uncertainty_quantile=0.80):
    cutoff = float(np.quantile(self.mutual_information, uncertainty_quantile))
    uncertain = self.mutual_information > cutoff

    confident_affected = (self.mean_probability >= affected_threshold) & ~uncertain
    confident_healthy = (self.mean_probability <= healthy_threshold) & ~uncertain

    return {
        "act": confident_affected,
        "ignore": confident_healthy,
        "inspect": ~(confident_affected | confident_healthy),
    }
```

| Categoría | Condición | Qué pasa |
|---|---|---|
| `act` | confiadamente afectada | va directo a la pasada de inspección |
| `ignore` | confiadamente sana | se salta |
| `inspect` | todo lo demás | cola para un humano |

### El caso que justifica todo el módulo

Fíjate en `& ~uncertain`. Una predicción de 0.97 **no** va a `act` si su incertidumbre
está en el quintil superior. Va a `inspect`.

**Ese es el grupo que un umbral solo no puede encontrar.** El modelo dice "afectada con
97% de confianza" mientras internamente se contradice a sí mismo entre pases. Ninguna
cantidad de ajuste del umbral sobre la probabilidad media distingue esa predicción de
una 0.97 robusta.

Del docstring:

> `inspect` is the queue a technician works through: anything in the undecided band,
> plus anything the model called confidently while being internally unsure about it.
> That second group is the reason uncertainty earns its cost; a threshold alone cannot
> find it.

### Por qué información mutua y no entropía total

```python
mutual_information: np.ndarray
"""Entropy of the mean minus mean entropy: the epistemic part alone.

This is the component that more data could reduce, which makes it the better
signal for deciding what to go and look at. Aleatoric uncertainty, a genuinely
ambiguous image, does not get better by visiting the tree again.
"""
```

Esta es la decisión conceptual más fina del módulo.

La entropía total mezcla dos cosas:
- **Aleatórica**: la foto es mala o la hoja es ambigua. Irreducible.
- **Epistémica**: el modelo no ha visto nada parecido. Reducible.

Si ordenas la cola por entropía total, mandas al técnico a mirar fotos borrosas. Eso no
genera información: la foto seguirá siendo borrosa.

Si ordenas por incertidumbre epistémica, mandas al técnico a los casos donde **su
visita aporta conocimiento nuevo**.

La matemática está en §1.17. Lo importante aquí es que la elección se deriva de qué
acción va a tomar una persona con el resultado.

### 20 pases: de dónde sale el número

```python
passes: int = 20
```

Del docstring:

> 20 passes is the usual default and is enough to rank by; the standard error of the
> mean falls as 1/sqrt(passes), so doubling it buys about 40% tighter estimates for
> double the inference cost.

El error estándar de la media de N muestras escala como `1/sqrt(N)`. Pasar de 20 a 40
pases reduce el error en `1 - 1/sqrt(2) ≈ 29%`... duplicando el coste.

Y el objetivo no es un intervalo de confianza preciso, es **ordenar**. Para ordenar,
20 sobran.

### La guarda que existe por un bug real

```python
enable_mc_dropout(model)

if verify_dropout and not has_active_dropout(model):
    raise RuntimeError(
        "dropout is not active, so every pass would be identical and the "
        "uncertainty would read as zero everywhere. Build the model with "
        "dropout > 0."
    )
```

Esto existe por el bug que arreglaste en el commit `d26341c`, y es el ejemplo más
instructivo del repositorio.

**La receta estándar de MC Dropout** es: recorre el árbol de módulos, encuentra los
`nn.Dropout`, y ponlos en modo entrenamiento.

**En el EfficientNet de timm eso no hace nada.** Lo verifiqué:

```
nn.Dropout modules found by walking the tree: 0
model.drop_rate attribute: 0.3
```

timm guarda `drop_rate` como un **atributo** y aplica el dropout **funcionalmente**
dentro de `forward_head`. No hay ningún módulo que conmutar.

**La consecuencia de no darse cuenta**: el código "funciona". No da error. Hace 20
pases idénticos, reporta varianza cero, y el sistema afirma confianza total en cada
árbol. Un fallo silencioso que produce una salida con el formato correcto.

Del mensaje de commit original:

> That matters more than a normal bug here: uncertainty is what routes trees to human
> inspection, so a model that always claims certainty sends the technician nowhere
> useful.

**La solución**:

```python
def enable_mc_dropout(model: nn.Module) -> None:
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
```

Pone **todo el modelo** en modo entrenamiento (necesario, porque el dropout funcional
consulta `self.training`), pero devuelve el **BatchNorm** a modo evaluación.

**Por qué ese segundo paso es imprescindible**: BatchNorm en modo entrenamiento
*actualiza sus estadísticas móviles* con cada lote que ve. Si lo dejaras en modo
entrenamiento durante 20 pases de inferencia, estarías modificando el modelo mientras
lo evalúas. Las estadísticas de normalización derivarían, y los pases posteriores
usarían un modelo distinto del de los primeros.

El docstring lo dice:

> Get this wrong and the uncertainty estimates are computed by a model whose
> normalisation statistics are drifting with every forward pass.

### La trampa que encontré al verificar esto

Al probar con pesos aleatorios, la varianza salía ~1e-5. Parecía que el dropout no
funcionaba.

No era eso. Lo comprobé directamente:

```
después de enable_mc_dropout:
  identical logits: False                        ← el dropout SÍ dispara
  max abs logit difference: 0.000115
  logit scale (max abs value): 0.000126
```

La diferencia (1.15e-4) es el **91% de la escala total de los logits** (1.26e-4). El
dropout perturba muchísimo en términos relativos.

**Lo que pasaba**: un modelo sin entrenar produce logits cercanos a cero, y ahí el
softmax es casi plano. Una perturbación grande del logit mueve poquísimo la
probabilidad.

Con pesos preentrenados:

```
std across passes: min=0.096126 max=0.329582
epistemic (MI)   : min=0.072843 max=0.346279
```

Dispersión de hasta 0.33. Funciona perfectamente.

**La lección, ahora fijada como test**: diagnostica el dropout mirando los **logits**,
nunca la dispersión de probabilidades. La dispersión pequeña puede ser un modelo sin
señal, no un dropout roto. Estuve a punto de diagnosticarlo mal y el repositorio ahora
contiene los dos casos como tests para que nadie repita el error.

### La medida que decide si el módulo vale su coste

```python
def uncertainty_separates_errors(result, *, threshold=0.5) -> dict:
    predictions = (result.mean_probability >= threshold).astype(int)
    wrong = predictions != result.labels
    ...
```

Del docstring:

> This is the check that decides if the signal is worth routing on. If wrong
> predictions are not measurably more uncertain than right ones, the queue ordering
> carries no information and the extra inference cost buys nothing.

Medido sobre un modelo deliberadamente mediocre (54.7% de accuracy, para tener errores
reales):

```
n_errors: 29
mean_uncertainty_correct: 0.0225
mean_uncertainty_wrong:   0.1102
ratio: 4.91
separates: True
```

**4.9 veces más incertidumbre en los errores.** Eso es lo que hace que ordenar la cola
funcione.

**El caso del modelo perfecto** está manejado explícitamente:

```python
if n_wrong == 0 or n_wrong == len(result):
    return { ..., "separates": None }
```

Devuelve `None`, no `True`. Con cero errores no hay nada contra lo que comparar, y
afirmar que la incertidumbre funciona sería inventarse un resultado. Del comentario:

> Reported rather than raised: a perfect run on an easy test set is exactly the
> situation this project keeps hitting.

Y la salida por consola lo dice sin rodeos:

> No errors on this split (0 wrong), so whether uncertainty tracks mistakes cannot be
> measured here. That is a property of the test set being easy, not evidence the signal
> works.

### Por qué el loader no puede mezclar

```python
for pass_index in range(passes):
    for images, batch_labels in loader:
        ...
        if pass_index == 0:
            pass_labels.append(batch_labels.numpy())
```

Las etiquetas se capturan **solo en el primer pase**. Eso asume que el loader devuelve
las muestras en el mismo orden cada vez.

Es cierto porque `build_eval_loader` usa `shuffle=False`. Pero es una dependencia
implícita, así que hay un test que la fija:

```python
def test_labels_survive_unpermuted(self, loader, device):
    result = mc_dropout_predict(TinyDropoutNet(), loader, device, passes=3)
    expected = torch.cat([labels for _, labels in loader]).numpy()
    assert np.array_equal(result.labels, expected)
```

Si alguien activara el shuffle en el loader de evaluación, este test fallaría en vez de
producir incertidumbres silenciosamente desalineadas con sus etiquetas.

### Y la restauración del modo

```python
model.eval()   # al final de mc_dropout_predict
```

`enable_mc_dropout` dejó el modelo en modo entrenamiento. Si no se restaurara, una
llamada posterior a `evaluate()` sería **estocástica sin que nadie lo supiera**: cada
ejecución daría métricas distintas.

Hay un test para esto también (`test_model_is_left_in_eval_mode`).

---

## 4.3 `per-class`: qué confunde, y si importa

```bash
citrus-scout per-class --checkpoint runs/leaf_baseline/best.pt
```

### El problema que resuelve

El informe binario responde "¿separa sanas de afectadas?". No puede responder "¿qué
confunde?".

Y en este dataset esa es la pregunta más útil, por la razón de §2.3: **el 23.3% de los
datos son enfermedades ausentes de España**.

### El escenario que ninguna métrica agregada distingue

Construí dos modelos y es la demostración más clara de por qué existe este módulo:

```
Model A accuracy: 72.0%
Model B accuracy: 72.0%
```

Accuracy idéntico. Ahora la vista por clase:

```
--- Model A ---
  healthy          recall=100.0%  (healthy)
  gummosis         recall=100.0%  (present, actionable)
  aphids           recall=100.0%  (present, actionable)
  citrus canker    recall= 30.0%  (absent)
  greening         recall= 30.0%  (absent)
  mean recall, locally relevant classes : 100.0%
  mean recall, classes absent from Spain:  53.3%
  -> USABLE: works on what Murcia actually has
  confusions: 0 costly, 2 free

--- Model B ---
  healthy          recall=100.0%  (healthy)
  gummosis         recall= 30.0%  (present, actionable)
  aphids           recall= 30.0%  (present, actionable)
  citrus canker    recall=100.0%  (absent)
  greening         recall=100.0%  (absent)
  mean recall, locally relevant classes :  30.0%
  mean recall, classes absent from Spain: 100.0%
  -> MISLEADING: the score is carried by diseases nobody here can act on
  confusions: 2 costly, 0 free
```

**Mismo accuracy, veredictos opuestos.** El modelo B es inútil en Murcia y su métrica
agregada es idéntica a la del modelo A.

Esto está fijado como test de regresión (`TestAggregateMetricBlindSpot`).

### El coste de confusión

```python
@property
def costly(self) -> bool:
    """Whether this confusion would matter in the field.

    Mistaking one absent disease for another is free: neither occurs in Spain,
    so neither triggers an action. Mistaking a present disease for a healthy
    tree is a missed detection, and the reverse is a wasted visit. Only
    confusions touching a locally relevant class can cost anything.
    """
    return _locally_relevant(self.true_label) or _locally_relevant(self.predicted_label)
```

Confundir cancro con HLB es **gratis**: ninguna de las dos existe en España, así que
ninguna dispara una acción. El modelo se equivoca y no pasa nada.

Confundir gummosis con pulgones es **caro**: las dos existen, y el tratamiento es
completamente distinto. Mandas al técnico con el producto equivocado.

Esta distinción no se puede derivar de los datos. Viene de la taxonomía, que codifica
conocimiento agronómico del dominio.

### Ordenar por tasa, no por recuento

```python
pairs.sort(key=lambda p: (p.rate, p.count), reverse=True)
```

Del docstring:

> Ordered by rate rather than raw count, so a small class losing most of its samples
> outranks a large class losing a few. The small class is usually the more interesting
> failure.

Ejemplo: `healthy` (998 imágenes) perdiendo 20 es una tasa del 2%. `citrus mite` (89
imágenes) perdiendo 20 es una tasa del 22%. El recuento es el mismo; el segundo es un
problema mucho más serio.

### Por qué la matriz de confusión está escrita a mano

```python
def confusion_matrix(y_true, y_pred, *, n_classes=None):
    """...
    Written out rather than taken from sklearn so that the class count is explicit:
    sklearn infers it from the labels present, which silently drops a class that
    never appears in either array and shifts every subsequent index.
    """
```

sklearn tiene `confusion_matrix`. No la usamos.

**La razón**: sklearn deduce el número de clases de las etiquetas presentes. Si la clase
7 de 17 no aparece ni en las verdaderas ni en las predichas de ese split, sklearn
devuelve una matriz de 16×16, y **todos los índices del 7 en adelante se desplazan en
uno**.

El resultado es una matriz que parece correcta y cuyas filas están asignadas a las
clases equivocadas. Con 17 clases y clases minoritarias de 13 imágenes en test, ese
caso es perfectamente posible.

Pasar `n_classes` explícitamente lo hace imposible.

### Clases sin muestras: ceros, no NaN

```python
recall = correct / support if support else 0.0
precision = correct / predicted_as if predicted_as else 0.0
f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
```

Una clase sin muestras en el split daría división por cero. Devolver `0.0` en vez de
`nan` tiene una razón práctica: un solo `nan` propaga a través de cualquier promedio
posterior y convierte toda la tabla en `nan`.

Y el `support` queda visible en su propia columna, así que la ausencia de datos no se
confunde con un rendimiento del 0%.

### El aviso en checkpoints binarios

```python
if len(classes) == 2:
    console.print(
        "[yellow]This is a binary checkpoint.[/yellow] Per-class structure only "
        "becomes informative with binary: false in the config, which keeps the "
        "original disease labels."
    )
```

Tu checkpoint es binario, así que este comando te dirá esto. Es honesto: con dos clases
la única confusión posible es sana↔afectada, que la tabla de puntos de operación ya
cubre.

Para usarlo de verdad hay que entrenar con `binary: false`.

---

## 4.4 `attention`: la auditoría

```bash
citrus-scout attention --checkpoint runs/leaf_baseline/best.pt --save-to runs/cam
```

### Por qué este comando audita a los otros tres

Los tres comandos anteriores puntúan **qué** responde el modelo. Ninguno comprueba
**por qué**.

Con tu PR-AUC de 1.0000 hay tres explicaciones (§1.14): tarea trivial, fuga de datos, o
atajo. **Las tres predicen números idénticos en `evaluate`, `uncertainty` y
`per-class`.**

Grad-CAM es la única herramienta del repositorio que puede distinguirlas.

### La capa objetivo

```python
DEFAULT_TARGET_LAYERS: dict[str, str] = {
    "efficientnet": "bn2",
    "convnext": "norm_pre",
    "resnet": "layer4",
    "mobilenet": "bn2",
}
```

Grad-CAM necesita una capa que **todavía tenga extensión espacial**. Lo verifiqué en el
modelo real:

```
forward_features shape: torch.Size([1, 1280, 7, 7])
```

1280 mapas de 7×7. Después del *global pooling* eso se reduce a 1280×1×1 y no queda
ninguna información de localización que leer.

`bn2` es la última capa antes del pooling en EfficientNet.

### Rechazar en vez de adivinar

```python
raise ValueError(
    f"no default Grad-CAM layer known for backbone {backbone!r}. Pass "
    f"target_layer explicitly: it must be the last layer with spatial extent, "
    f"before global pooling."
)
```

Del docstring:

> Raises rather than guessing when the family is unknown. A silently wrong layer
> produces a plausible-looking heatmap of nothing in particular, which is worse than
> an error because it invites a confident wrong conclusion.

Un mapa de calor de la capa equivocada **se ve bien**. Tiene colores, tiene estructura,
y no significa nada. Alguien lo mira y concluye "el modelo mira la hoja" sobre la base
de ruido. Un error explícito es estrictamente mejor.

### Los hooks y por qué un gestor de contexto

```python
def __enter__(self) -> GradCAM:
    self._handles.append(self._module.register_forward_hook(self._save_activations))
    self._handles.append(self._module.register_full_backward_hook(self._save_gradients))
    return self

def __exit__(self, *args: object) -> None:
    for handle in self._handles:
        handle.remove()
    self._handles.clear()
    self._activations = None
    self._gradients = None
```

Un **hook** es una función que PyTorch llama automáticamente cuando un módulo procesa
datos. Los necesitamos para capturar las activaciones intermedias y sus gradientes, que
no están accesibles de otro modo.

**Por qué gestor de contexto**, del docstring:

> A leaked hook keeps every activation tensor alive, which on a full split leaks memory
> steadily and is easy to mistake for something else.

Un hook no eliminado mantiene vivo el tensor de activaciones. Recorriendo 762 imágenes,
eso acumula memoria sin parar. Y el síntoma (memoria creciente) no apunta a su causa.

Hay un test que verifica la eliminación **incluso cuando el cuerpo lanza una
excepción**:

```python
with pytest.raises(RuntimeError, match="deliberate"), GradCAM(model, "final_conv"):
    raise RuntimeError("deliberate")
assert len(model.final_conv._forward_hooks) == before
```

### `register_full_backward_hook` y no `register_backward_hook`

```python
# full_backward_hook rather than backward_hook: the latter is documented as
# unreliable for modules whose forward is not a single operation.
```

PyTorch documenta `register_backward_hook` como poco fiable cuando el `forward` de un
módulo hace más de una operación. `bn2` incluye normalización, escala, desplazamiento y
activación. La versión `full` es la correcta.

### Por qué hay gradientes en una operación de inferencia

```python
# Gradients are required even though this is inference: Grad-CAM's weights
# *are* gradients. A no_grad block here yields an empty map.
images = images.clone().requires_grad_(False)
logits = self.model(images)
```

Esto es contraintuitivo: normalmente la inferencia va dentro de `torch.no_grad()` para
ahorrar memoria.

Aquí no se puede. Grad-CAM **pondera los mapas por sus gradientes**, así que necesita
que el grafo de retropropagación exista. Dentro de `no_grad()` el mapa saldría vacío.

Nota que `requires_grad_(False)` se aplica a las **imágenes**: no necesitamos el
gradiente respecto a la entrada, solo respecto a las activaciones intermedias (que se
capturan por hook).

### El bug que su propio test encontró

Escribí el test `test_leaves_no_gradients_on_the_parameters` y **falló**. Mi código
hacía:

```python
self.model.zero_grad(set_to_none=True)   # ANTES
score.backward()
```

Limpiaba los gradientes antes de generarlos, así que quedaban en los parámetros al
salir. Si alguien ejecutara Grad-CAM a mitad de un entrenamiento, esos gradientes se
sumarían al siguiente paso del optimizador y corromperían la actualización.

El arreglo:

```python
score.backward()
self.model.zero_grad(set_to_none=True)   # DESPUÉS
```

Es un ejemplo limpio de un test que encuentra un bug real en vez de confirmar lo que ya
creía.

### El ReLU sobre el mapa

```python
weights = self._gradients.mean(dim=(2, 3), keepdim=True)
maps = (weights * self._activations).sum(dim=1)
maps = torch.relu(maps)
```

`weights` es el gradiente medio por mapa de características: cuánto contribuye cada uno
de los 1280 mapas a la puntuación de la clase.

El **ReLU** (que deja pasar solo lo positivo) es deliberado:

> ReLU: only evidence *for* the class is wanted. Negative contributions are evidence
> for some other class and would muddy the map.

Una contribución negativa es evidencia *contra* la clase, que es información distinta.
Mezclarla confundiría el mapa.

### Normalización por imagen, no por lote

```python
def _normalise_per_image(maps: np.ndarray) -> np.ndarray:
    """...
    Per image, not per batch: the absolute gradient magnitude varies with how
    confident the prediction was, and normalising across a batch would make
    confident images look like they have more attention everywhere.
    """
```

La magnitud absoluta del gradiente depende de lo confiada que fuera la predicción.
Normalizando por lote, una imagen muy confiada parecería tener más atención en todas
partes, lo que es un artefacto y no información.

### El `border_excess` y el bug de resolución

Esta es la parte con más historia. El objetivo es una cifra automática que detecte
atajos sin que un humano mire 762 mapas.

**Primera versión**: medir qué fracción de la atención cae en el anillo exterior, y
marcar si supera 0.35.

**Problema**: los mapas son de 7×7. Un anillo de una celda en 7×7 son 24 de 49 celdas:
**el 49% del área**.

| Tamaño | El borde es... del área |
|---|---|
| 7×7 | 49.0% |
| 14×14 | 26.5% |
| 28×28 | 38.3% |
| 224×224 | 43.8% |

Un mapa uniforme pondría el 49% de su atención en el borde a 7×7, y el umbral de 0.35
lo marcaría. **Mi primera versión marcaba como sospechoso un modelo con pesos
aleatorios.** Y el mismo patrón saldría limpio a 14×14.

**El arreglo**: comparar contra la cuota de área en vez de una constante.

```python
@property
def border_excess(self) -> float:
    if self.border_area_fraction <= 0.0:
        return 0.0
    return self.border_fraction / self.border_area_fraction

def looks_like_background_shortcut(self, *, excess_threshold: float = 1.15) -> bool:
    return self.border_excess > excess_threshold
```

- **1.0** = proporcional. Un mapa uniforme da exactamente esto a **cualquier**
  resolución.
- **> 1.15** = el modelo prefiere los bordes.

Verificado:

```
7x7:     border=49.0% area=49.0% excess=1.000 suspect=False
14x14:   border=26.5% area=26.5% excess=1.000 suspect=False
28x28:   border=38.3% area=38.3% excess=1.000 suspect=False
224x224: border=43.8% area=43.8% excess=1.000 suspect=False
```

El margen del 15% sobre proporcional es un juicio: en fotos de hoja centrada, un
modelo que lee la planta debería poner **menos** que su cuota en los bordes, nunca
claramente más.

### La validación con un modelo entrenado para hacer trampas

Una métrica de detección de atajos que no se prueba contra un atajo real no vale nada.
Construí dos datasets donde la clase está codificada solo en una región:

```
señal en el borde:  accuracy 100.0%, border excess 1.51x, 50% marcadas
señal en el centro: accuracy 100.0%, border excess 0.41x,  0% marcadas
```

**Los dos ajustan perfectamente.** Ninguna otra métrica los distingue.

El 50% tiene explicación exacta: la señal solo está en la clase positiva, así que la
mitad de las imágenes no tiene nada en el borde. Marca precisamente las tramposas.

**El control del centro importa tanto como el caso positivo.** Sin él, un detector que
marcara todo pasaría la prueba. Los dos casos son tests de regresión.

### La interpretación automática

```python
if share > 0.25:
    console.print(
        "[red]A quarter or more of the images are explained by their edges.[/red] "
        "On close-range leaf photographs the edges hold no diagnostic information, "
        "so this is the signature of a model keying on the background. Expect the "
        "score not to survive a change of viewpoint."
    )
elif share > 0.10:
    console.print("[yellow]A minority of images lean on the border.[/yellow] ...")
else:
    console.print(
        "[green]Attention sits mostly away from the frame edges.[/green] Consistent "
        "with the model reading the leaf, though it does not prove the features "
        "transfer to aerial imagery."
    )
```

Fíjate en la cláusula final del caso verde: *"though it does not prove the features
transfer to aerial imagery"*. Incluso el mejor resultado posible aquí no demuestra que
funcione en campo. Atención en la hoja es condición necesaria, no suficiente.

### El guardado de imágenes: las peores primero

```python
worst.sort(key=lambda row: row[0], reverse=True)
for rank, (excess, image, heatmap, label) in enumerate(worst[:save_count]):
    path = save_to / f"{rank:02d}_excess{excess:.2f}_{name}.png"
```

Guarda 12 por defecto, ordenadas por peor `border_excess`. El nombre del fichero lleva
el rango, la cifra y la clase.

**La decisión**: si solo vas a mirar 12 de 762 imágenes, que sean las más sospechosas.
Guardar 12 al azar sería casi inútil.

### El sobreimpresionado y su deliberada tosquedad

```python
# Nearest-neighbour upsample from the feature map's resolution, typically 7x7.
# Deliberately blocky: a smooth interpolation implies spatial precision the
# underlying map does not have.
y_index = (np.arange(height) * heatmap.shape[0] // height).clip(0, heatmap.shape[0]-1)
x_index = (np.arange(width) * heatmap.shape[1] // width).clip(0, heatmap.shape[1]-1)
upsampled = heatmap[np.ix_(y_index, x_index)]
```

Ampliar de 7×7 a 224×224 con interpolación suave produce un mapa bonito. Y **miente**:
sugiere una precisión espacial que el mapa de 7×7 no tiene. Cada celda cubre 32×32
píxeles de la imagen original; fingir un gradiente suave dentro de ella es inventarse
información.

El resultado a cuadros es fiel a la resolución real.

Y el canal rojo plano en vez de un mapa de color perceptual:

> Uses a plain red channel rather than a perceptual colourmap so the module stays free
> of a matplotlib dependency. The point here is localisation, which a single channel
> conveys adequately.

---

## 4.5 La calibración dentro de `evaluate`

Se ejecuta automáticamente al evaluar, con `--calibrate-on val` por defecto.

### La restricción que el código impone

```python
fit_split = None if calibrate_on in (None, "none", "") else calibrate_on
if fit_split == split:
    raise typer.BadParameter(
        f"--calibrate-on ({fit_split}) must differ from --split ({split}): "
        "fitting and measuring on the same data reports a calibration that "
        "does not hold anywhere else."
    )
```

**No se puede ajustar y medir en el mismo split.** Es un error de argumentos, no un
aviso.

**Por qué es tan estricto**: la temperatura se ajusta minimizando la pérdida. Si la
ajustas en test y mides en test, por construcción sale bien. Reportarías una
calibración que no se cumple en ningún otro sitio. Es una de las formas más fáciles de
engañarse sin mala intención.

### Por qué hace falta `collect_logits`

```python
@torch.no_grad()
def collect_logits(model, loader, device):
    """...
    Calibration needs logits, not probabilities: temperature scaling divides them
    before the softmax, and a softmax output cannot be inverted back without
    knowing the temperature that produced it.
    """
```

`evaluate()` devuelve probabilidades post-softmax. Una vez aplicado el softmax, la
información del logit original se ha perdido (salvo por una constante aditiva). No se
puede recuperar.

Por eso hay un pase aparte que guarda los logits crudos.

### La optimización de log(T), no de T

```python
log_t = nn.Parameter(torch.zeros(1))  # T = exp(0) = 1, la identidad
optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

def closure() -> torch.Tensor:
    optimizer.zero_grad()
    loss = criterion(logits_t / torch.exp(log_t), labels_t)
    loss.backward()
    return loss
```

Se optimiza `log(T)` y se usa `exp(log_t)` como temperatura.

**Por qué**: `exp` de cualquier real es siempre positivo. Así la temperatura **no
puede** volverse cero o negativa sin necesitar un optimizador con restricciones.

**Por qué importa**: con T negativa, dividir invertiría el signo de los logits y por
tanto **invertiría el ranking**. El modelo predeciría exactamente lo contrario. Con
T = 0 habría división por cero.

E inicializar `log_t = 0` significa `T = 1`, la identidad. El punto de partida es "no
cambiar nada", así que el óptimo nunca puede ser peor que no calibrar.

**LBFGS** en vez de Adam: es un optimizador de segundo orden, ideal para problemas de
pocos parámetros (aquí, uno) con una función objetivo suave. Converge en muchas menos
iteraciones.

### La propiedad que hace la calibración segura

Verificado end-to-end:

```
PR-AUC raw        : 0.999157
PR-AUC calibrated : 0.999157
ranking preserved: PR-AUC identical to 9 decimal places

thresholds for the same specificities, raw -> calibrated:
  spec 89.5%: 0.013329 -> 0.033813
  spec 94.7%: 0.034960 -> 0.069743
  spec 98.7%: 0.826973 -> 0.781311
```

PR-AUC idéntico a nueve decimales. Los umbrales se mueven. El PPV no cambia, porque es
función de sensibilidad y especificidad, no del valor del umbral.

Calibrar no cambia lo que el modelo sabe ni cómo ordena. Cambia la escala en la que
expresa su confianza.

### La cosa incómoda que el código no esconde

En datos con separación casi perfecta, el **ECE puede empeorar** después de calibrar:

```
fitted on val: T=1.284 (overconfident) | ECE 0.0205 -> 0.0237 | Brier 0.0180 -> 0.0177

test ECE raw        : 0.0257
test ECE calibrated : 0.0277
```

La temperatura sale 1.284 (detecta exceso de confianza correctamente), el Brier mejora
(0.0180 → 0.0177), el NLL mejora mucho... y el ECE sube ligeramente.

**No es un bug.** El ECE es un estadístico con agrupamiento en cubos. Cuando ya es
diminuto (0.02), el ruido del agrupamiento domina cualquier mejora real.

**La decisión de diseño**: imprimir los dos números en vez de reportar solo el que
queda bien. El informe muestra ECE antes y después, y el lector ve lo que pasó.

Esto está también en los tests: el test que afirma mejora de ECE usa datos con clases
solapadas, donde la descalibración es lo bastante grande para dominar el ruido. Mi
primera versión del test afirmaba mejora en el caso fácil y **falló correctamente**.

---

Siguiente: [05-decisiones.md](05-decisiones.md), el registro de todas las decisiones y
los bugs encontrados.
