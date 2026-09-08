# Data sources

Attribution and licence record for every dataset used. `citrus_diseases` is CC BY 4.0,
which requires attribution wherever the work is distributed, including in a commercial
product built on it.

## citrus_leaf_pathology

- **Kaggle**: `chayanmondalabir/citrus-leaf-pathology-multi-class-image-dataset`
- **Licence**: CC0 (public domain)
- **Size**: about 3.7 GB
- **Content**: 14 classes of citrus leaf pathology, with train/test folders and a
  pre-augmented copy of the same photographs

## citrus_diseases

- **Kaggle**: `superlord/citrus-diseases`
- **Licence**: CC BY 4.0, attribution required
- **Size**: about 2.1 GB
- **Content**: 4 classes, gummosis, leaf miner, aphids and healthy

## What the combined set looks like

After excluding pre-augmented copies and removing exact duplicates:

| | |
|---|---|
| Images | 5,069 |
| Classes | 17 |
| Exact duplicates removed | 882 |
| Split | train 3,545 / val 762 / test 762 |

By relevance to the Region of Murcia:

| Relevance | Images | Meaning |
|---|---|---|
| present | 2,018 | Occurs in Murcia, detections are actionable |
| absent | 1,181 | Not present in Spain, pre-training value only |
| healthy | 998 | |
| nonspecific | 872 | Real symptom, many possible causes |

## Classes that matter locally

| Class | Images | Note |
|---|---|---|
| gummosis | 462 | Phytophthora. The highest-value target: present in Murcia and its canopy decline shows in the screening pass |
| sooty mould | 478 | Follows honeydew from whitefly or scale, visible as canopy darkening |
| leaf minnor | 459 | Citrus leafminer. Galleries under 1 mm wide, close-range only |
| aphids | 416 | Present in spring flush, symptom on young shoots |
| spider mite | 114 | *Tetranychus urticae*, damaging in late summer |
| citrus mite | 89 | Present. Individual mites are sub-millimetre, the damage is what shows |

## Classes that do not occur in Spain

Kept for pre-training, never surfaced as alerts: `citrus canker`, `greening` (HLB),
`black spot`, `melanose`, `anthracnose`, `bacterial blight`, `curl virus`.

Greening is the important one. Spain is free of *Candidatus* Liberibacter and of
*Diaphorina citri*, and *Trioza erytreae* has not reached the Mediterranean Levante.
A greening alert over a Murcian grove is guaranteed to be a false positive, so the
deployed system must not report it even though the model is trained on it.

## Known limitations

**These are not aerial images.** Every source is close-range leaf photography against
a controlled background. They are a proxy for the inspection pass, at under 2 m over a
flagged tree, and carry no information about the nadir screening pass.

**Prevalence is inverted.** The combined set is 80% affected; a real grove is 2-5%.
Metrics computed here must be projected to field prevalence before they mean anything,
which is what `citrus_scout.evaluation.metrics` does.

**Deduplication is exact-match only.** Byte-identical files are removed, but a resized
or re-encoded copy of the same photograph survives and would leak across splits. This
is the main remaining source of optimistic bias.

**Label quality is unverified.** These are community-uploaded datasets with no stated
diagnostic protocol. No qPCR or agronomist confirmation backs the labels, so a
proportion are likely wrong, particularly among the visually similar leaf spots.
