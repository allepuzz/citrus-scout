# Documentación completa de citrus-scout

Explicación de cada parte del sistema, cada decisión de diseño, y cada limitación.

Escrita asumiendo cero conocimiento previo de aprendizaje automático, sin simplificar
las matemáticas hasta volverlas falsas.

---

## Documentos

| # | Documento | Contenido |
|---|---|---|
| 1 | [Conceptos](01-conceptos.md) | Cada término desde cero: especificidad, PPV, PR-AUC, calibración, incertidumbre, Grad-CAM. Con las matemáticas |
| 2 | [Los datos](02-datos.md) | Qué son exactamente estas imágenes, las 17 clases, el 80.3%, el embudo de filtrado, el aumento de datos |
| 3 | [El entrenamiento](03-entrenamiento.md) | Cada decisión en el orden en que ocurre, explicando tu ejecución real línea a línea |
| 4 | [La evaluación](04-evaluacion.md) | Las cuatro herramientas y qué pregunta responde cada una |
| 5 | [Decisiones y bugs](05-decisiones.md) | Tabla maestra de decisiones, los 7 bugs encontrados, y todas las limitaciones |
| 6 | [El flujo completo](06-flujo.md) | El recorrido de una imagen desde el disco hasta una alerta |

**Si solo vas a leer uno**: el [2](02-datos.md). La naturaleza de los datos determina
el techo de todo lo demás.

**Si vas a leer dos**: el 2 y la sección de limitaciones del [5](05-decisiones.md).

---

## El resumen en diez líneas

1. Tenemos 5.069 fotos de primer plano de hojas de cítrico, 17 clases, de dos datasets
   públicos de Kaggle con licencia permisiva.
2. El 80.3% están afectadas. Un limonar real tiene entre el 2% y el 5%.
3. El 23.3% son enfermedades que **no existen en España**, incluido el HLB.
4. Entrenamos un EfficientNet-B0 preentrenado en ImageNet, 4M de parámetros, para
   decidir sana contra afectada.
5. Tu ejecución dio PR-AUC 1.0000 en 6m 45s sobre una T4.
6. **Ese 1.0000 no es un buen resultado, es una señal de alarma.** Significa tarea
   trivial, fuga de datos, o atajo — y las métricas no pueden distinguirlos.
7. La métrica que importa es el PPV proyectado a prevalencia de campo: a 98.7% de
   especificidad, 1.7 alertas por caso real.
8. Hay cuatro herramientas de evaluación. La cuarta (`attention`, Grad-CAM) audita a
   las otras tres mirando **dónde** mira el modelo.
9. La incertidumbre (MC Dropout) es 4.9× más alta en los errores, lo que hace que
   ordenar la cola de inspección funcione.
10. **La restricción real es imagen aérea de Murcia con ground truth agronómico.** Nada
    en este repositorio mejora hasta que eso exista.

---

## Los números reales de tu ejecución

Todo lo documentado está verificado contra el código y los datos del repositorio, no
estimado.

### Los datos

```
14.224  ficheros en disco (5.5 GB)
 5.951  tras descartar copias pre-aumentadas (8.273 eran aumentadas)
 5.069  tras deduplicar por SHA-256 (882 duplicados exactos)
        ├── train 3.545
        ├── val     762
        └── test    762
```

Prevalencia: 4.071 afectadas (80.3%), 998 sanas (19.7%).

Por relevancia local: 39.8% presentes en Murcia, 23.3% ausentes de España, 19.7% sanas,
17.2% síntomas no específicos.

### El entrenamiento

```
device: cuda (Tesla T4)
model: efficientnet_b0, 4.0M parameters
class weights: [1.606, 0.394]
warmup: backbone frozen for 1 epoch
15 épocas en 6m 45s, ~27s por época
best validation PR-AUC: 1.0000 (alcanzado en la época 10)
```

### La evaluación

```
PR-AUC 1.0000   ROC-AUC 1.0000   F1@0.5 0.9967   prevalencia test 80.3%

umbral  sensibilidad  especificidad    PPV   alertas/caso
 0.009        100.0%          90.0%  16.9%            5.9
 0.013        100.0%          94.7%  27.7%            3.6
 0.046        100.0%          98.7%  60.5%            1.7
```

---

## Estado del código

- **298 tests** pasan (145 antes de esta sesión)
- **4.076 líneas** de código fuente
- **7 bugs** encontrados y arreglados, documentados en el [documento 5](05-decisiones.md)
- Lint (`ruff`) y formato limpios

### Módulos

```
src/citrus_scout/
├── cli.py                    277 líneas  los 6 comandos
├── data/
│   ├── sources.py             88  catálogo con licencias
│   ├── taxonomy.py           102  qué enfermedad existe en Murcia
│   ├── leaf_dataset.py       169  descubrimiento y lectura de imágenes
│   ├── splits.py             170  reparto estratificado y deduplicación
│   ├── build.py              131  ensamblado del conjunto
│   ├── package.py            140  empaquetado para Colab
│   ├── archive_dataset.py    133  lectura del archivo en Colab
│   ├── transforms.py          79  aumento de datos
│   └── download.py           107  descarga de Kaggle
├── models/
│   └── classifier.py         119  construcción del modelo y MC Dropout
├── training/
│   ├── config.py             127  configuración validada
│   ├── loop.py               265  bucle, pesado, parada
│   └── train.py              283  orquestación del entrenamiento
└── evaluation/
    ├── metrics.py            188  PPV, especificidad, puntos de operación
    ├── calibration.py        245  escalado de temperatura, ECE, Brier
    ├── uncertainty.py        224  MC Dropout y triaje
    ├── per_class.py          213  métricas por clase y confusiones
    ├── gradcam.py            343  mapas de atención y detección de atajos
    └── report.py             646  los informes que imprime la CLI
```

---

## Los comandos

```bash
# Datos
citrus-scout data download      # descarga de Kaggle (5.5 GB)
citrus-scout data summary       # qué se ensambló
citrus-scout data relevance     # qué clases importan en Murcia
citrus-scout data package       # empaqueta para Colab (174 MB)

# Entrenamiento
citrus-scout train --config configs/leaf_baseline.yaml

# Evaluación: cuatro preguntas distintas
citrus-scout evaluate    --checkpoint runs/leaf_baseline/best.pt   # ¿qué tan bueno?
citrus-scout uncertainty --checkpoint runs/leaf_baseline/best.pt   # ¿dónde mirar?
citrus-scout per-class   --checkpoint runs/leaf_baseline/best.pt   # ¿qué confunde?
citrus-scout attention   --checkpoint runs/leaf_baseline/best.pt --save-to runs/cam
                                                                   # ¿hace trampas?
```

---

## La idea que organiza todo el proyecto

**Cuando un número sale demasiado bien, es una señal de alarma, no una celebración.**

De eso se derivan casi todas las decisiones:

- No reportar accuracy, porque premia modelos que no detectan nada
- Proyectar el PPV a prevalencia de campo, porque el test miente
- Mostrar la prevalencia del test junto a las métricas, para que no se lean solas
- Imprimir la interpretación al lado del número
- Construir Grad-CAM, porque ninguna métrica puede detectar un atajo
- Parar el entrenamiento diciendo "la métrica se saturó" en vez de "hay una meseta"
- Reportar el ECE antes y después aunque empeore
- Devolver `None` en vez de `True` cuando no hay errores que medir
