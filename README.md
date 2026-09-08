# citrus-scout

Detección de plagas y enfermedades en cítricos mediante imagen aérea (UAV) y visión por computador.

## Contexto

El *scouting* fitosanitario en cítricos se hace hoy de forma manual y muestral: un técnico
recorre la parcela siguiendo protocolos de muestreo (GIP-IVIA) y extrapola. Es caro, lento y
no da un censo árbol a árbol.

`citrus-scout` busca automatizar esa inspección con dron + visión por computador, orientado a
**cooperativas y ATRIAs** de la Región de Murcia (28.442 ha de limonero, ~53 % del limón nacional).

### Enfoque en dos pasadas

| Pasada | Altura | GSD | Qué detecta |
|---|---|---|---|
| **Criba** | 15-25 m | 0,3-0,7 cm/px | Decaimiento de copa, pérdida de vigor, árboles muertos |
| **Inspección** | < 2 m | < 0,05 cm/px | Síntomas de órgano (hoja, fruto) sobre árboles marcados |

La criba cubre la parcela entera rápido; la inspección solo desciende sobre los árboles
sospechosos. Es lo que hace el coste por hectárea viable.

## Dianas realistas (Región de Murcia)

**Detectables por UAV cenital** — firma a escala de copa:
- Decaimiento por *Phytophthora* (gomosis / podredumbre de cuello)
- Tristeza (CTV)
- Estrés hídrico y nutricional

**NO detectables desde vista cenital** — síntoma milimétrico en órgano:
piojo rojo de California (~2 mm), cotonet, minador, pulgones, *Ceratitis*.
Requieren la pasada de inspección a corta distancia.

**Ausentes en España** — sin *ground truth* local posible:
HLB (*Candidatus* Liberibacter spp.) y sus vectores. España está libre de la bacteria y de
*Diaphorina citri*; *Trioza erytreae* está en Canarias y la cornisa cantábrica, no en el
Levante mediterráneo. **Toda la literatura de detección de HLB por UAV no es replicable aquí.**

## Estado

🚧 Fase 0 — montaje del pipeline de clasificación sobre datasets públicos de hoja.

## Instalación

Requiere [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/allepuzz/citrus-scout.git
cd citrus-scout
uv sync --extra dev
```

### Credenciales de Kaggle

Los datasets públicos se descargan con la API de Kaggle. Coloca tu token en:

```
~/.kaggle/kaggle.json          # Linux / macOS
C:\Users\<usuario>\.kaggle\kaggle.json   # Windows
```

Nunca añadas ese archivo al repositorio (está en `.gitignore`).

## Uso

```bash
# Descargar datasets públicos de hoja
uv run citrus-scout data download

# Entrenar
uv run citrus-scout train --config configs/leaf_baseline.yaml

# Evaluar
uv run citrus-scout evaluate --checkpoint runs/<id>/best.pt
```

### Entrenar en Colab

El código está diseñado para correr igual en local y en Colab: el notebook clona el repo,
instala y lanza el mismo script. Ver `notebooks/colab_train.ipynb`.

## Estructura

```
src/citrus_scout/
├── data/          # datasets, descarga, transformaciones
├── models/        # arquitecturas y factory de backbones
├── training/      # bucle de entrenamiento, callbacks
├── evaluation/    # métricas, incertidumbre, calibración
└── utils/         # configuración, semillas, logging
configs/           # configuraciones de experimento (YAML)
scripts/           # utilidades sueltas
notebooks/         # exploración y Colab
```

## Métricas

Este problema tiene **fuerte desbalanceo de clases** (prevalencia de árboles afectados
típicamente del 2-5 %). La *accuracy* es engañosa y no se usa.

Métricas de referencia:
- **PR-AUC** (área bajo precisión-recall)
- **F1** y sensibilidad a especificidad fija
- **VPP a prevalencia real** — con prevalencia 2 %, sensibilidad 90 % y especificidad 90 %,
  el valor predictivo positivo es del 15,5 %: ~6 de cada 7 alertas serían falsas.
  Subir la especificidad al 99 % lo lleva al ~65 %.

El punto de operación se ajusta hacia **alta especificidad**: el coste de no inspeccionar un
árbol sano es bajo, pero saturar al técnico de falsos positivos hace el sistema inútil.

## Datos

Los datos **no se versionan en git**. Se gestionan con DVC.

Datasets públicos usados en Fase 0 (hoja a corta distancia, fondo controlado):
PlantVillage (naranjo), colecciones Kaggle de cítricos, dataset MDPI de 649 hojas.

⚠️ Son de enfermedades mayoritariamente exóticas (canker, HLB) y de hoja cercana, **no de
vista aérea**. Sirven para preentrenar el clasificador de la pasada de inspección y para
validar el pipeline, no como datos finales de producción.

## Normativa

Operación bajo Reglamentos UE 2019/947 y 2019/945 + RD 517/2024:
- Registro de operador UAS en AESA (obligatorio, gratuito)
- Formación A1/A3 (online, gratuita)
- Consulta de zonas geográficas en **ENAIRE Drones** antes de cada vuelo
- Altura máxima 120 m en categoría Abierta
- Seguro no obligatorio en A1/A3 con < 20 kg (RD 517/2024 art. 8), pero recomendable

## Licencia

MIT. Ver [LICENSE](LICENSE).
