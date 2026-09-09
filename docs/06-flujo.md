# 6. El flujo completo: una imagen de principio a fin

Los documentos anteriores explican las piezas. Este las ata, siguiendo el recorrido
real de los datos y señalando en qué fichero y función ocurre cada paso.

---

## 6.1 El mapa general

```
KAGGLE
  │  citrus-scout data download
  ▼
data/raw/                                      14.224 ficheros, 5.5 GB
  │  discover_samples()         leaf_dataset.py
  │    ├── descarta directorios aug*           -8.273
  │    ├── lee la etiqueta del directorio padre
  │    └── normaliza la etiqueta
  ▼                                            5.951 muestras
  │  build_splits()             build.py
  │    ├── descarta clases con <20 muestras
  │    ├── deduplicate()        splits.py       -882 duplicados exactos
  │    ├── stratified_split()   splits.py
  │    └── verify_no_leakage()  splits.py
  ▼                                            5.069: 3.545 / 762 / 762
  │  package_splits()           package.py
  │    ├── reescala a 512 px (LANCZOS)
  │    ├── recodifica a JPEG q90
  │    └── escribe zip SIN comprimir
  ▼
data/processed/leaf_dataset.zip                174 MB, un fichero
  │  [ subida manual a Google Drive ]
  ▼
COLAB
  │  load_packaged_splits()     archive_dataset.py
  │    ├── extract_archive()
  │    ├── read_manifest()      lee la lista de clases
  │    └── ArchiveDataset × 3   el split viene de las carpetas
  ▼
  │  build_loaders()            train.py
  │    ├── train_transform()    transforms.py    aumento aleatorio
  │    ├── eval_transform()     transforms.py    determinista
  │    ├── BinaryLabelWrapper   train.py         17 clases -> 2
  │    └── DataLoader × 3       workers, pin_memory, drop_last
  ▼
  │  run_training()             train.py
  │    ├── seed_everything()
  │    ├── build_model()        classifier.py    EfficientNet-B0 + ImageNet
  │    ├── class_weights()      loop.py          [1.606, 0.394]
  │    ├── freeze_backbone()    classifier.py    warmup
  │    └── bucle de 15 épocas:
  │         ├── train_one_epoch()    loop.py     AMP, recorte, AdamW
  │         ├── evaluate()           loop.py     -> classification_report()
  │         ├── should_stop()        loop.py     paciencia + saturación
  │         └── save_checkpoint()    loop.py     si mejora el PR-AUC
  ▼
runs/leaf_baseline/
  ├── best.pt          pesos + config + lista de clases
  ├── config.yaml      la receta exacta
  └── history.json     métricas por época
  │
  │  cuatro caminos de evaluación
  ▼
┌──────────────┬──────────────┬──────────────┬──────────────┐
│  evaluate    │ uncertainty  │  per-class   │  attention   │
│ metrics.py   │uncertainty.py│ per_class.py │  gradcam.py  │
│calibration.py│              │              │              │
│ ¿qué tan     │ ¿dónde debe  │ ¿qué         │ ¿está        │
│  bueno?      │ mirar un     │ confunde?    │ haciendo     │
│              │ humano?      │              │ trampas?     │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

---

## 6.2 El recorrido de una imagen concreta

Sigamos `data/raw/citrus_diseases/Citrus-Diseases/aphids/aphids_001.jpg`.

### Paso 1 — Descubrimiento

`discover_samples()` en `data/leaf_dataset.py`:

```python
for path in sorted(root.rglob("*")):
    if path.suffix.lower() not in IMAGE_SUFFIXES:   # .jpg pasa
        continue
    if not include_augmented and looks_augmented(path, root):
        continue                                    # no hay "aug" en la ruta
    label = infer_label(path, root)                 # -> "aphids"
    samples.append(LeafSample(path=path, label=normalise_label(label), source=source_key))
```

`infer_label` sube desde el fichero buscando el primer directorio que no sea
estructural:

```python
for part in reversed(relative.parts[:-1]):
    if part.lower() in SPLIT_DIR_NAMES:   # train/test/val/valid/validation/eval
        continue
    return part
```

Para `Citrus-Diseases/aphids/aphids_001.jpg`, el primer directorio desde el final es
`aphids`. Devuelve eso.

`normalise_label("aphids")` → `"aphids"`. (Para `"Citrus Canker"` o `"citrus_canker"`
daría `"citrus canker"`, que es cómo dos datasets con convenciones distintas acaban
coincidiendo.)

Resultado: `LeafSample(path=..., label="aphids", source="citrus_diseases")`.

### Paso 2 — Filtrado por tamaño de clase

`build_splits()` en `data/build.py`:

```python
counts = Counter(s.label for s in samples)
kept = [s for s in samples if counts[s.label] >= min_samples_per_class]
```

`aphids` tiene 416 muestras. Sobrevive. (Ninguna clase cae aquí; la menor, `citrus
mite`, tiene 89.)

### Paso 3 — Deduplicación

`deduplicate()` en `data/splits.py` calcula el SHA-256 del contenido:

```python
seen: dict[str, LeafSample] = {}
for sample in samples:
    digest = file_digest(sample.path)
    if digest in seen:
        removed += 1
        continue
    seen[digest] = sample
```

Si `aphids_001.jpg` es byte a byte idéntico a otro fichero ya visto, se descarta. En
total se descartan 882 de 5.951.

### Paso 4 — Asignación de split

`stratified_split()` agrupa por clase y reparte dentro de cada grupo:

```python
group = by_class["aphids"]                    # 416 muestras
group.sort(key=lambda s: str(s.path))         # orden determinista
indices = rng.permutation(len(group))         # mezcla con semilla 42

n_val  = round(416 * 0.15) = 62
n_test = round(416 * 0.15) = 62

val.extend(group[i] for i in indices[:62])
test.extend(group[i] for i in indices[62:124])
train.extend(group[i] for i in indices[124:])   # 292
```

Coincide con las cuentas reales del archivo: `aphids` tiene 292/62/62.

Supongamos que nuestra imagen cae en `train`.

### Paso 5 — Empaquetado

`package_splits()` en `data/package.py`:

```python
with Image.open(sample.path) as source:
    image = _resized(source.convert("RGB"), 512)     # lado mayor <= 512, LANCZOS
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90, optimize=True)

name = f"train/aphids/000042.jpg"
archive.writestr(name, buffer.getvalue())
```

El nombre dentro del zip **codifica el split y la clase**. El índice `000042` es su
posición en la lista, no tiene significado.

### Paso 6 — Lectura en Colab

`ArchiveDataset.__getitem__` en `data/archive_dataset.py`:

```python
image = Image.open(self.paths[index]).convert("RGB")
if self.transform is not None:
    image = self.transform(image=np.array(image))["image"]
return image, self._target(self.labels[index])
```

Y `_target` aplica la conversión binaria:

```python
def _target(self, label: str) -> int:
    if self.binary:
        return 0 if is_healthy(label) else 1
    return self.class_to_index[label]
```

`is_healthy("aphids")` es `False` (su relevancia es `PRESENT`, no `HEALTHY`), así que el
objetivo es **1** = afectada.

### Paso 7 — Transformación

`train_transform()` aplica, en orden:

```
imagen original (hasta 512 px de lado)
  │ RandomResizedCrop(224, scale=0.7-1.0)    recorta el 70-100% y reescala
  │ HorizontalFlip(p=0.5)                    volteo horizontal
  │ VerticalFlip(p=0.2)                      volteo vertical
  │ Rotate(limit=30, p=0.5)                  rotación hasta 30 grados
  │ RandomBrightnessContrast(±25%, p=0.7)    luz
  │ HueSaturationValue(hue=±6, p=0.3)        tono CASI intacto
  │ OneOf([MotionBlur, GaussianBlur], p=0.2) desenfoque
  │ GaussNoise(p=0.15)                       ruido
  │ Normalize(ImageNet mean/std)             (x - media) / desviación
  │ ToTensorV2()                             HWC uint8 -> CHW float32
  ▼
tensor de forma (3, 224, 224)
```

Cada época ve una variante distinta. Con `hue_shift_limit=6`, el color — que **es** la
señal diagnóstica — se preserva.

### Paso 8 — El paso de entrenamiento

`train_one_epoch()` en `training/loop.py`:

```python
images = images.to(device, non_blocking=True)       # CPU -> GPU
labels = labels.to(device, non_blocking=True)
optimizer.zero_grad(set_to_none=True)               # borrar gradientes previos

with torch.amp.autocast("cuda"):                    # float16 donde es seguro
    loss = criterion(model(images), labels)         # forward + pérdida

scaler.scale(loss).backward()                       # escalar y retropropagar
scaler.unscale_(optimizer)                          # desescalar ANTES de recortar
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)                              # actualizar pesos
scaler.update()                                     # ajustar el factor de escala
```

La pérdida de nuestra imagen se multiplica por el peso de su clase (0.394, por ser
afectada), porque `criterion` se construyó con `weight=weights`.

### Paso 9 — El paso forward dentro del modelo

```
tensor (3, 224, 224)
  │ stem: conv 3x3, 32 canales
  │ blocks 0-6: bloques MBConv de EfficientNet
  │   (convolución separable en profundidad + squeeze-excitation)
  │ conv_head: 1x1, 1280 canales
  │ bn2 + activación          <-- AQUÍ engancha Grad-CAM
  ▼ mapas de características (1280, 7, 7)
  │ global_pool: media espacial
  ▼ vector (1280,)
  │ dropout(0.3)              <-- AQUÍ actúa MC Dropout (funcional, no módulo)
  │ classifier: lineal 1280 -> 2
  ▼ logits (2,)
```

Dos puntos de este diagrama son los que usan los módulos de evaluación:

- **`bn2`** es la última capa con extensión espacial (7×7). Grad-CAM la engancha porque
  después del pooling no queda localización.
- **El dropout** es funcional, aplicado dentro de `forward_head` desde el atributo
  `drop_rate`. Por eso la receta estándar de MC Dropout (buscar módulos `nn.Dropout`) no
  encuentra nada.

### Paso 10 — Evaluación al final de la época

`evaluate()` en `training/loop.py` recorre el loader de validación:

```python
logits = model(images)
probabilities = torch.softmax(logits.float(), dim=1)[:, 1]   # P(afectada)
```

y pasa las probabilidades a `classification_report()`:

```python
operating_points = [
    operating_point_at_threshold(
        y_true, y_score,
        threshold_for_specificity(y_true, y_score, spec),   # cuantil de los negativos
        field_prevalence,                                    # 0.02
    )
    for spec in (0.90, 0.95, 0.99)
]
```

Para cada especificidad objetivo se calcula el umbral que la consigue, se mide
sensibilidad y especificidad reales, y se proyecta el PPV a la prevalencia de campo con
Bayes.

### Paso 11 — Decisión de parada

`should_stop()` en `training/loop.py`:

```python
improved = pr_auc > best_pr_auc + min_delta          # min_delta = 1e-4
stalled = 0 if improved else epochs_without_improvement + 1

if patience is not None and stalled >= patience:
    return True, stalled, f"early stopping: no PR-AUC gain for {patience} epochs"

if pr_auc >= 1.0:
    return True, stalled, "PR-AUC saturated at 1.0000: the task is too easy to rank on"

return False, stalled, None
```

Y el checkpoint se guarda por separado, sin umbral:

```python
if pr_auc > best_pr_auc:
    best_pr_auc = pr_auc
    save_checkpoint(run_dir / "best.pt", model, config, classes, epoch, report)
```

Con tus números, la época 10 alcanza 1.0000 y ahora pararía ahí con el mensaje de
saturación.

---

## 6.3 El recorrido de la evaluación

Después del entrenamiento hay cuatro caminos. Todos empiezan igual:

```python
model, config, classes = load_checkpoint(checkpoint, device)   # pesos + config + clases
loader = build_eval_loader(config, split, archive=archive)      # mismo preproceso
```

`load_checkpoint` reconstruye la arquitectura desde el config guardado y carga los
pesos. Nota `pretrained=False`: no hace falta descargar ImageNet si los pesos se van a
sobreescribir.

### Camino A: `evaluate`

```
collect probabilities    ->  classification_report()
                              ├── PR-AUC, ROC-AUC, F1@0.5
                              └── 3 puntos de operación con PPV proyectado

collect_logits()         ->  calibrate() sobre validación
                              ├── fit_temperature()   LBFGS sobre log(T)
                              └── aplicar T a test, medir ECE antes/después
                         ->  classification_report() con las probabilidades calibradas
```

La clave: la temperatura se ajusta en **validación** y se aplica a **test**. El CLI
rechaza que sean el mismo split.

### Camino B: `uncertainty`

```
enable_mc_dropout(model)        model.train() + BatchNorm en eval
has_active_dropout(model)       verificar o lanzar excepción

20 pases sobre el loader  ->  matriz (20, n) de probabilidades
                              ├── media                  -> mean_probability
                              ├── desviación             -> std_probability
                              ├── H(media)               -> entropía total
                              ├── media de H             -> aleatórica
                              └── total - aleatórica     -> epistémica (info mutua)

uncertainty_separates_errors()  ¿es mayor la incertidumbre en los errores?
triage()                        act / inspect / ignore
model.eval()                    restaurar el modo
```

### Camino C: `per-class`

```
collect_logits()  ->  argmax  ->  predicciones de clase

confusion_matrix(n_classes explícito)
per_class_metrics()   -> recall, precisión, F1, soporte, relevancia
actionable_recall()   -> media en presentes vs media en ausentes
top_confusions()      -> ordenadas por TASA, etiquetadas costly/free
```

### Camino D: `attention`

```
resolve_target_layer(model, backbone)   -> "bn2" para EfficientNet

with GradCAM(model, "bn2") as cam:      hooks forward + full_backward
    for lote in loader:
        logits = model(images)          CON gradientes (no no_grad)
        score = logits[clase].sum()
        score.backward()
        model.zero_grad()               DESPUÉS del backward

        weights = gradientes.mean(dim=(2,3))
        maps = relu((weights * activaciones).sum(dim=1))
        normalizar cada mapa a [0,1]

        summarise_attention(mapa)  -> border_excess, centre, concentración
# hooks eliminados al salir del with

fracción marcada > 0.25  ->  aviso en rojo
guardar las 12 peores como PNG
```

---

## 6.4 Los puntos donde el sistema puede fallar en silencio

Esta es la lista que yo revisaría primero si algo diera un resultado sospechoso.
Ninguno de estos fallos produce una excepción.

| Punto | Fallo silencioso | Qué lo previene |
|---|---|---|
| `looks_augmented` con ruta absoluta | Un directorio `augmented` superior descarta todo | Se inspecciona la ruta relativa a `root` |
| Deduplicación | Duplicados no exactos sobreviven | **Nada.** Limitación conocida (§5.3.2) |
| Reparto sin `sort` previo | Particiones distintas entre SO | `group.sort(key=...)` antes de permutar |
| Orden de clases | Predicciones invertidas | `sorted()` + lista guardada en el checkpoint |
| Recalcular el split en Colab | Particiones distintas sin avisar | El split va en la estructura de carpetas |
| Recortar gradientes escalados | El modelo no aprende | `scaler.unscale_()` antes de recortar |
| `zero_grad` olvidado | Gradientes acumulados | `zero_grad(set_to_none=True)` cada paso |
| MC Dropout sobre timm | Varianza cero, confianza total | `has_active_dropout()` lanza excepción |
| BatchNorm en train durante MC Dropout | Estadísticas derivando | Se devuelve a `eval()` explícitamente |
| Modelo dejado en modo train | `evaluate()` se vuelve estocástico | `model.eval()` al final, con test |
| Hook de Grad-CAM filtrado | Memoria creciente sin causa aparente | Gestor de contexto, con test de excepción |
| Gradientes de Grad-CAM filtrados | Corrompe el siguiente paso del optimizador | `zero_grad()` después del backward |
| Capa de Grad-CAM equivocada | Mapa plausible y sin sentido | Se rechazan arquitecturas desconocidas |
| `confusion_matrix` de sklearn | Índices desplazados si falta una clase | Implementación propia con `n_classes` |
| Calibrar y medir en el mismo split | Calibración ficticia | `BadParameter` en el CLI |
| Temperatura ≤ 0 | Ranking invertido | Se optimiza `log(T)`, nunca T |
| `num_workers` truthy check | Un `0` explícito se ignoraría | `is not None` |
| Early stopping con métrica saturada | Se corta un modelo que mejora | `min_delta` + parada por saturación |

---

## 6.5 Los ficheros que produce una ejecución

```
runs/leaf_baseline/
├── best.pt            el checkpoint del mejor PR-AUC de validación
│                        ├── model_state   los 4M de pesos
│                        ├── config        la receta completa
│                        ├── classes       ['healthy', 'affected']  <- crítico
│                        ├── epoch         en qué época se guardó
│                        └── metrics       solo valores numéricos (portable)
├── config.yaml        escrito ANTES de entrenar, por si la ejecución muere
└── history.json       métricas y LR de cada época, para dibujar curvas
```

`config.yaml` se escribe antes de empezar. Por eso existe
`runs/leaf_smoke/config.yaml` en tu repositorio aunque ese smoke test local no llegara
a entrenar.

---

## 6.6 Qué falta entre esto y un producto

El flujo documentado va de fotos de hoja a una alerta binaria. Un producto real
necesita:

| Pieza | Estado |
|---|---|
| Clasificación de imagen suelta | **hecho** |
| Evaluación honesta a prevalencia de campo | **hecho** |
| Incertidumbre y triaje | **hecho** |
| Auditoría de atajos | **hecho** |
| Datos de dron nadir | **no existe** |
| Detección a escala de copa | **no existe** |
| Segmentación de árboles individuales en un ortomosaico | **no existe** |
| Georreferenciación de las alertas | **no existe** |
| Agregación de varias fotos por árbol | **no existe** |
| Seguimiento temporal entre vuelos | **no existe** |
| Informe para el técnico | **no existe** |
| Ground truth agronómico | **no existe** |

Lo que está hecho es la mitad de clasificación. Y la mitad de clasificación está
bloqueada por la última fila, no por nada de este código.

---

Índice: [00-indice.md](00-indice.md)
