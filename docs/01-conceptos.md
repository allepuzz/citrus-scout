# 1. Conceptos: cada palabra, desde cero

Este documento asume que no sabes nada de aprendizaje automático. No asume que seas
tonto. La diferencia importa: voy a definir cada término, pero no voy a simplificar
las matemáticas hasta volverlas falsas.

Orden de lectura: este documento primero. Los demás lo dan por leído.

---

## 1.1 El problema, en una frase

Tenemos fotografías de hojas de cítrico. Queremos un programa que mire una foto
nueva y diga si la hoja está sana o afectada.

Eso es **clasificación de imágenes**. "Clasificación" porque la respuesta es una
categoría (sana / afectada), no un número continuo. Si predijéramos "cuántos gramos
pesa esta hoja" sería **regresión**. Nosotros hacemos clasificación.

Y es **clasificación binaria**, porque hay exactamente dos categorías posibles. Si
distinguiéramos entre las 17 enfermedades concretas, sería **clasificación
multiclase**. El código soporta las dos cosas; por defecto hace binaria, y en
§1.12 explico por qué.

---

## 1.2 Qué es un modelo, y qué significa "entrenar"

Un **modelo** es una función matemática con huecos. Los huecos se llaman
**parámetros** o **pesos**. Nuestro modelo tiene 4.000.000 de ellos
aproximadamente (la salida de tu entrenamiento dijo `4.0M parameters`).

La función toma una imagen y devuelve dos números. Algo así:

```
imagen  ->  [ 4.000.000 de parámetros haciendo multiplicaciones y sumas ]  ->  (número_sana, número_afectada)
```

**Entrenar** es el proceso de buscar valores para esos 4 millones de parámetros
tales que la función dé las respuestas correctas en las fotos que ya sabemos
etiquetar.

Nadie elige esos valores a mano. El proceso es:

1. Empieza con valores casi aleatorios.
2. Pásale una foto. Mira qué responde.
3. Compara su respuesta con la correcta. Calcula cuánto se ha equivocado.
4. Ajusta ligeramente cada parámetro en la dirección que habría reducido el error.
5. Repite unos cuantos miles de veces.

El paso 3 necesita una forma de medir "cuánto se ha equivocado". Eso es la
**función de pérdida** (*loss*). El paso 4 es el **descenso de gradiente**, y el
mecanismo que calcula en qué dirección mover cada parámetro es la
**retropropagación** (*backpropagation*).

---

## 1.3 Logits, softmax y probabilidades

Los dos números que escupe el modelo se llaman **logits**. Son números reales sin
restricción: pueden ser `-3.7` y `8.2`, o `0.001` y `-0.0004`.

Un logit no es una probabilidad. No está entre 0 y 1, y los dos no suman 1. Para
convertirlos en probabilidades se usa la función **softmax**:

```
softmax(z_i) = e^(z_i) / suma_j( e^(z_j) )
```

Con dos clases y logits `(z_sana, z_afectada)`:

```
P(afectada) = e^(z_afectada) / ( e^(z_sana) + e^(z_afectada) )
```

Propiedades que importan:

- El resultado siempre está entre 0 y 1.
- Los dos resultados suman exactamente 1.
- Preserva el orden: si `z_afectada > z_sana`, entonces `P(afectada) > 0.5`.
- Es **invariante a sumar una constante** a todos los logits, pero **no** a
  multiplicarlos. Esto último es la base de la calibración (§1.16).

En el código esto aparece como:

```python
probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
```

`[:, 1]` coge la columna 1, que es la clase "afectada". Por eso el orden de las
clases importa tanto, y por eso se guarda junto a los pesos del modelo
(`save_checkpoint` guarda `classes`). Si cargas un checkpoint con el orden invertido,
el modelo predice lo contrario de lo que crees, **sin dar ningún error**.

---

## 1.4 Umbral de decisión

El modelo da `P(afectada) = 0.73`. ¿Eso es "afectada" o "sana"?

Depende de dónde pongas el **umbral** (*threshold*). Si el umbral es 0.5, entonces
0.73 > 0.5 y decides "afectada". Si el umbral es 0.9, decides "sana".

**El umbral no es parte del modelo.** Es una decisión separada, que se toma después
de entrenar, y que se puede cambiar sin reentrenar nada. Esto es crucial y mucha
gente lo confunde: el modelo produce un *ranking* (qué fotos son más sospechosas que
otras), y el umbral convierte ese ranking en decisiones.

Elegir el umbral bien es, en este proyecto, igual de importante que entrenar bien.
§1.10 explica por qué.

---

## 1.5 Los cuatro resultados posibles

Con un umbral fijado, cada predicción cae en una de cuatro casillas. Esto se llama
**matriz de confusión**:

|  | El modelo dice AFECTADA | El modelo dice SANA |
|---|---|---|
| **Realmente AFECTADA** | Verdadero Positivo (VP) | Falso Negativo (FN) |
| **Realmente SANA** | Falso Positivo (FP) | Verdadero Negativo (VN) |

Traducido al campo:

- **VP**: árbol enfermo, lo detectamos. Bien.
- **VN**: árbol sano, lo descartamos. Bien.
- **FP**: árbol sano, damos la alarma. El técnico camina hasta allí para nada.
- **FN**: árbol enfermo, no lo vemos. La enfermedad sigue propagándose.

"Positivo" siempre significa "la clase que buscamos", que aquí es *afectada*. No
significa "bueno". Un verdadero positivo es un árbol enfermo, lo cual es una mala
noticia agronómica y una buena noticia para el modelo.

---

## 1.6 Sensibilidad (= recall = tasa de verdaderos positivos)

**Definición**: de todos los árboles que están realmente enfermos, qué fracción
detecta el modelo.

```
sensibilidad = VP / (VP + FN)
```

El denominador es "todos los enfermos de verdad". Si hay 100 árboles enfermos y
detectas 90, la sensibilidad es 90%.

Tres nombres para lo mismo: **sensibilidad** (medicina), **recall** (recuperación de
información), **tasa de verdaderos positivos** o TPR (teoría de detección). En este
proyecto uso "sensibilidad" porque es el término del dominio sanitario vegetal.

**Qué la rompe**: un modelo que dice "afectada" a todo tiene sensibilidad del 100%.
No se le escapa ni uno. Y es completamente inútil.

Por eso la sensibilidad nunca se reporta sola.

---

## 1.7 Especificidad

Esta es la palabra por la que preguntaste explícitamente, así que voy despacio.

**Definición**: de todos los árboles que están realmente sanos, qué fracción
descarta correctamente el modelo.

```
especificidad = VN / (VN + FP)
```

El denominador es "todos los sanos de verdad". Si hay 1000 árboles sanos y el modelo
descarta correctamente 990 (dando falsa alarma en 10), la especificidad es 99%.

Es la **imagen espejo** de la sensibilidad:

| | Mira solo... | Responde a... |
|---|---|---|
| Sensibilidad | los enfermos reales | ¿se me escapa alguno? |
| Especificidad | los sanos reales | ¿doy falsas alarmas? |

**Son independientes.** Puedes tener sensibilidad 100% y especificidad 0% (diciendo
"afectada" a todo), o al contrario. Un modelo bueno tiene las dos altas.

**La relación con el umbral**: mover el umbral intercambia una por la otra.

- Umbral bajo (0.01): casi todo se marca como afectado. Sensibilidad alta,
  especificidad baja. Muchas falsas alarmas.
- Umbral alto (0.99): casi nada se marca. Sensibilidad baja, especificidad alta. Se
  escapan enfermos.

Esa curva de intercambio es lo que el código llama **punto de operación**
(*operating point*), y cada fila de la tabla que viste en tu evaluación es uno:

```
threshold  sensitivity  specificity    PPV   alerts per true case
    0.009       100.0%        90.0%  16.9%                    5.9
    0.013       100.0%        94.7%  27.7%                    3.6
    0.046       100.0%        98.7%  60.5%                    1.7
```

Fíjate: la sensibilidad es 100% en las tres filas. Lo único que cambia es la
especificidad, y con ella el PPV. Que es lo siguiente.

---

## 1.8 Por qué la "precisión" (accuracy) es una métrica tóxica aquí

**Accuracy** (exactitud) es la métrica que todo el mundo usa por defecto:

```
accuracy = (VP + VN) / total
```

"Qué fracción de predicciones son correctas". Suena razonable. Es una trampa.

**El problema**: en un limonar bien gestionado, entre el 2% y el 5% de los árboles
están afectados. Supón 2%. Ahora considera este modelo:

```python
def modelo_inutil(imagen):
    return "sana"   # siempre
```

Este modelo no mira la imagen. Su accuracy es del **98%**. Acierta en los 98 árboles
sanos de cada 100 y falla en los 2 enfermos.

Un 98% suena excelente. El modelo detecta exactamente cero enfermedades.

Por eso el código **no reporta accuracy en ningún sitio**. No es una omisión, es una
decisión. El docstring de `metrics.py` lo dice:

> Deliberately does **not** return accuracy: at low prevalence it is misleading and
> tends to justify models that detect nothing.

Cuando una métrica puede recompensar un modelo que no hace nada, esa métrica no
mide lo que te importa.

---

## 1.9 Prevalencia

**Definición**: qué fracción de la población está realmente afectada.

```
prevalencia = (número de afectados) / (total)
```

No es una propiedad del modelo. Es una propiedad **del mundo**, o del conjunto de
datos que estés mirando.

Y aquí está el hecho más importante de todo este proyecto:

| Dónde | Prevalencia |
|---|---|
| Nuestro conjunto de test | **80.3%** |
| Un limonar real en Murcia | **2% a 5%** |

Ese 80.3% no es un error, es verificable en el propio archivo de datos (§2.4).
Los datasets públicos son colecciones de *fotos de enfermedades*: la gente fotografía
hojas enfermas porque son interesantes. Nadie sube 10.000 fotos de hojas sanas.

El resultado es que **toda métrica medida sobre nuestro test está midiendo un mundo
que no existe**. Y algunas métricas se rompen catastróficamente al cambiar de
prevalencia, mientras otras no. Saber cuál es cuál es la diferencia entre un informe
honesto y uno inútil.

---

## 1.10 PPV: la métrica que de verdad decide si esto sirve

**PPV** = *Positive Predictive Value*, valor predictivo positivo. También se llama
**precision** (precisión) en la literatura de aprendizaje automático. Dos nombres,
misma fórmula.

**Definición**: de todas las alarmas que da el sistema, qué fracción son árboles
realmente enfermos.

```
PPV = VP / (VP + FP)
```

Nota la diferencia con la sensibilidad. La sensibilidad divide entre "todos los
enfermos reales". El PPV divide entre "todo lo que el modelo marcó".

**Por qué es la métrica que importa**: el técnico que recibe el informe no ve la
sensibilidad ni la especificidad. Ve una lista de árboles a visitar. El PPV le dice
qué fracción de esa caminata no será en vano.

### La parte que sorprende a todo el mundo

El PPV **depende de la prevalencia**. La sensibilidad y la especificidad no.

Esto viene del **teorema de Bayes**. La derivación, en términos de frecuencias sobre
una población de tamaño N:

```
enfermos reales        = N · prevalencia
sanos reales           = N · (1 - prevalencia)

verdaderos positivos   = N · prevalencia · sensibilidad
falsos positivos       = N · (1 - prevalencia) · (1 - especificidad)

                                  VP
PPV  =  ------------------------------------------------------
                                VP + FP

              prevalencia · sensibilidad
     =  -----------------------------------------------------------------
         prevalencia · sensibilidad  +  (1-prevalencia) · (1-especificidad)
```

La N se cancela. El PPV depende solo de esos tres números.

Esto está implementado literalmente así en `evaluation/metrics.py`:

```python
def ppv_at_prevalence(sensitivity, specificity, prevalence):
    true_positives = sensitivity * prevalence
    false_positives = (1.0 - specificity) * (1.0 - prevalence)
    denominator = true_positives + false_positives
    if denominator == 0.0:
        return 0.0
    return true_positives / denominator
```

### El número que deberías tener grabado

Un modelo con **90% de sensibilidad y 90% de especificidad** suena decente. A una
prevalencia del 2%:

```
VP = 0.02 × 0.90           = 0.018
FP = 0.98 × 0.10           = 0.098
PPV = 0.018 / (0.018+0.098) = 0.155  →  15.5%
```

**El 84.5% de las alarmas son falsas.** Casi 6 caminatas inútiles por cada árbol
enfermo encontrado. El técnico abandona la herramienta en una semana.

Ahora sube solo la especificidad al 99%, dejando la sensibilidad igual:

```
VP = 0.02 × 0.90           = 0.018
FP = 0.98 × 0.01           = 0.0098
PPV = 0.018 / (0.018+0.0098) = 0.647  →  64.7%
```

**Misma sensibilidad. El PPV se cuadruplica.** Eso es toda la razón por la que este
proyecto está diseñado alrededor de la especificidad y no del F1 ni de la accuracy.

### Tabla completa, con sensibilidad 100% y prevalencia 2%

Estas cifras están calculadas con el código del repositorio, no estimadas:

| Especificidad | PPV | Alarmas por caso real |
|---|---|---|
| 50% | 3.9% | 25.5 |
| 80% | 9.3% | 10.8 |
| 90% | 16.9% | 5.9 |
| 95% | 29.0% | 3.5 |
| 98.7% | 61.1% | 1.6 |
| 99% | 67.1% | 1.5 |
| 99.5% | 80.3% | 1.2 |
| 99.9% | 95.3% | 1.0 |

La relación es brutalmente no lineal. Entre el 90% y el 99% de especificidad el PPV
se multiplica por cuatro. Entre el 99% y el 99.9%, por 1.4 más. Cada nueve que
añades vale más que el anterior.

### Y por qué el 80% de prevalencia de nuestro test lo falsea todo

Mismo modelo (90% sens, 99% spec), cambiando solo la prevalencia:

| Prevalencia | PPV |
|---|---|
| 2% (campo real) | 64.7% |
| 5% | 82.6% |
| 10% | 90.9% |
| 30% | 97.5% |
| 80% (nuestro test) | 99.7% |

**El mismo modelo, sin cambiar un solo peso, pasa de 64.7% a 99.7% de PPV solo
porque cambias el conjunto en el que lo mides.**

Por eso la tabla de tu evaluación dice "PPV projected to 2.0% field prevalence". No
mide el PPV sobre el test; mide sensibilidad y especificidad (que no dependen de la
prevalencia) y luego los **proyecta** a la prevalencia real mediante Bayes. Es la
única forma honesta de reportar este número.

---

## 1.11 NPV, el complemento

**NPV** = *Negative Predictive Value*. De todos los árboles que el sistema descarta,
qué fracción están realmente sanos.

```
NPV = VN / (VN + FN)
```

A baja prevalencia el NPV es casi siempre altísimo (más del 99%), porque la mayoría
de los árboles están sanos y acertar "sano" es fácil. Está implementado
(`npv_at_prevalence`) y se calcula, pero no se imprime en la tabla principal porque
un número que siempre dice 99.x% no ayuda a decidir nada.

---

## 1.12 F1

El **F1** combina precisión (PPV) y sensibilidad en un número:

```
F1 = 2 · (PPV · sensibilidad) / (PPV + sensibilidad)
```

Es la **media armónica** de las dos. La media armónica, a diferencia de la
aritmética, castiga los desequilibrios: si PPV = 1.0 y sensibilidad = 0.0, la media
aritmética daría 0.5 mientras que F1 da 0.

**Por qué no es nuestra métrica principal**: el F1 trata los falsos positivos y los
falsos negativos como igual de malos. En nuestro problema no lo son. Saltarse un
árbol sano (FN en el sentido de no visitarlo) cuesta poco; inundar al técnico de
falsas alarmas (FP) mata el producto. El F1 no tiene forma de expresar esa asimetría.

Además, el F1 se calcula a un umbral fijo (el código usa 0.5), y ya hemos visto que
el umbral es una decisión libre. Reportar F1 a 0.5 es reportar el rendimiento en un
punto arbitrario de la curva.

Se reporta (`f1_at_0.5`) por comparabilidad con resultados publicados, y nada más.

---

## 1.13 PR-AUC y ROC-AUC: métricas de ranking

Las métricas anteriores necesitan un umbral. Estas no. Miden la **calidad del
ranking**: si ordenas todas las fotos por `P(afectada)` descendente, ¿quedan las
enfermas arriba?

### ROC-AUC

La curva **ROC** (*Receiver Operating Characteristic*, un nombre heredado del radar
de la Segunda Guerra Mundial) dibuja, para cada umbral posible:

- eje Y: sensibilidad
- eje X: 1 − especificidad (la tasa de falsas alarmas)

El **AUC** es el área bajo esa curva. Interpretación directa y elegante: es la
probabilidad de que, cogiendo un enfermo y un sano al azar, el modelo puntúe más
alto al enfermo.

- 0.5 = azar puro
- 1.0 = ranking perfecto

**Su defecto con datos desbalanceados**: el eje X es `FP / (FP + VN)`. Cuando hay
muchísimos más sanos que enfermos, ese denominador es enorme, y añadir falsos
positivos casi no mueve el eje. La curva se ve bien aunque el sistema sea inútil en
la práctica.

### PR-AUC

La curva **PR** (*Precision-Recall*) dibuja:

- eje Y: precisión (PPV)
- eje X: sensibilidad (recall)

El área bajo ella es el **PR-AUC**, que en sklearn se calcula como
`average_precision_score`.

**Por qué es mejor aquí**: el eje Y es `VP / (VP + FP)`. No hay ningún VN en esa
fórmula. Los verdaderos negativos, que son la inmensa mayoría en nuestro problema, no
pueden inflar el resultado. La métrica es sensible al desbalance de la forma correcta.

Por eso es la métrica principal del proyecto, y por eso la selección del mejor modelo
durante el entrenamiento se hace sobre PR-AUC y no sobre la pérdida de validación
(§3.7).

**Una diferencia sutil que importa**: el PR-AUC mínimo no es 0.5, es la prevalencia.
Un modelo aleatorio sobre datos con 80% de positivos tiene PR-AUC ≈ 0.80. Así que
"PR-AUC = 0.85" sobre nuestro test es *peor que aleatorio*, mientras que sobre un
conjunto al 2% sería extraordinario. Nunca compares PR-AUC entre conjuntos con
prevalencias distintas.

---

## 1.14 Y entonces, ¿qué significa tu PR-AUC de 1.0000?

Tu entrenamiento terminó con PR-AUC = 1.0000 y ROC-AUC = 1.0000 sobre 762 imágenes
de test.

**Eso no es un buen resultado. Es una señal de alarma.**

Un 1.0000 exacto significa separación perfecta: existe un umbral que clasifica bien
las 762 imágenes. Hay tres explicaciones posibles:

1. **La tarea es trivial.** Las fotos de hoja sana y enferma son tan distintas que
   cualquier modelo las separa. Plausible: son primeros planos con fondo controlado.

2. **Hay fuga de datos** (*leakage*). Imágenes casi idénticas aparecen en
   entrenamiento y en test, y el modelo está recordando en vez de generalizando. El
   código lo combate (§2.6) pero no puede eliminarlo del todo.

3. **El modelo usa un atajo.** Ha aprendido algo correlacionado con la etiqueta pero
   irrelevante: el fondo, la iluminación, el encuadre, qué cámara tomó cada clase.

**Ninguna métrica de las anteriores puede distinguir entre las tres.** Las tres
predicen PR-AUC = 1.0. Por eso existe el módulo de Grad-CAM (§6): es la única
herramienta del repositorio que mira *dónde* está mirando el modelo, en vez de
solo qué responde.

Y hay una consecuencia técnica inmediata: con un 1.0000, el conjunto de test **ya
no tiene resolución para medir nada**. No puede distinguir un modelo bueno de uno
excelente. Eso es exactamente lo que rompía el early stopping antes del PR #6
(§3.8).

---

## 1.15 Sobreajuste, y para qué sirven tres conjuntos

**Sobreajuste** (*overfitting*): el modelo memoriza las fotos concretas del
entrenamiento en vez de aprender el patrón general. Acierta en lo que ya vio y falla
en lo nuevo.

Con 4 millones de parámetros y 3.545 imágenes de entrenamiento, hay capacidad de
sobra para memorizar. El sobreajuste es el estado natural; hay que combatirlo
activamente.

La defensa es partir los datos en tres:

| Conjunto | Tamaño real | Para qué |
|---|---|---|
| **Train** | 3.545 | Ajustar los 4M de parámetros |
| **Validación** | 762 | Decidir cuándo parar, qué checkpoint guardar, ajustar la temperatura |
| **Test** | 762 | Medir. Una sola vez, al final |

La regla de oro: **cada decisión que tomes mirando un conjunto, contamina ese
conjunto**. Si eliges el mejor checkpoint mirando validación, las métricas de
validación son optimistas, porque has elegido el que mejor salía ahí.

Por eso el test existe y se toca lo mínimo. Y por eso la calibración (§5) se ajusta
en validación y se mide en test: el código lo fuerza rechazando que ambos sean el
mismo split.

---

## 1.16 Calibración

Un modelo puede ordenar perfectamente y aun así dar probabilidades que no significan
nada.

**Calibrado** significa: de todas las veces que el modelo dice "70% de probabilidad",
aproximadamente el 70% resultan ser afectadas de verdad. El número se puede leer
como una probabilidad real.

**Descalibrado** significa que no. El caso típico es el **exceso de confianza**: el
modelo dice 0.99 cuando la frecuencia real es 0.85. Las redes neuronales modernas
entrenadas con entropía cruzada lo hacen sistemáticamente.

### Cómo lo detectamos en tu ejecución

Mira los umbrales de tu tabla:

```
threshold  sensitivity  specificity
    0.009       100.0%        90.0%
    0.013       100.0%        94.7%
    0.046       100.0%        98.7%
```

Para alcanzar el 90% de especificidad hace falta un umbral de **0.009**. Eso quiere
decir que casi todas las probabilidades están apiladas por encima de 0.009 — el
modelo dice "afectada con 99.x%" a casi todo. Esos no son probabilidades, son
confianzas pegadas a los extremos del intervalo.

### Por qué importa y no es cosmética

El punto de operación se elige **sobre esos números**. Cuando dices "quiero 99% de
especificidad", el código busca el umbral que lo consigue en validación. Si las
probabilidades están mal calibradas y apiladas, ese umbral es numéricamente
inestable: un cambio diminuto de umbral salta muchas muestras de golpe.

Además, cualquier razonamiento posterior que trate la salida como probabilidad (y el
triaje por incertidumbre lo hace) hereda el error.

### ECE: cómo se mide

**ECE** = *Expected Calibration Error*, error de calibración esperado.

Procedimiento:
1. Agrupa las predicciones en cubos por confianza: [0, 0.067), [0.067, 0.133), ...
2. En cada cubo, compara la confianza media con la fracción realmente positiva.
3. Promedia las diferencias absolutas, ponderando por cuántas muestras hay en cada
   cubo.

```
ECE = suma_cubos  (n_cubo / N) · | confianza_media_cubo − fracción_real_cubo |
```

ECE = 0 significa calibración perfecta. ECE = 0.1 significa que de media la
confianza se desvía 10 puntos de la realidad.

**Defecto del ECE**: depende de cómo hagas los cubos, y se puede engañar. Un modelo
que predice siempre la tasa base tiene ECE ≈ 0 y es inútil. Por eso el código
reporta también el **Brier score**, que es una *regla de puntuación propia* (no se
puede engañar así):

```
Brier = media( (probabilidad − resultado)² )
```

### Escalado de temperatura: cómo lo arreglamos

Se divide cada logit por un único número **T** antes del softmax:

```
P(afectada) = softmax( logits / T )[1]
```

- **T = 1**: no cambia nada.
- **T > 1**: los logits se encogen, las probabilidades se acercan a 0.5. Corrige el
  exceso de confianza.
- **T < 1**: las probabilidades se alejan de 0.5. Afila.

**Un solo parámetro**, ajustado en validación minimizando la pérdida.

**La propiedad que lo hace seguro**: dividir por un número positivo es una
transformación **monótona**. No cambia el orden de las puntuaciones. Por tanto
**PR-AUC y ROC-AUC son literalmente idénticos** antes y después. Lo verifiqué:

```
PR-AUC raw        : 0.999157
PR-AUC calibrated : 0.999157
```

Idéntico a nueve decimales. Los umbrales sí se mueven (de 0.013 a 0.034 para la
misma especificidad), que es precisamente el objetivo. Calibrar no cambia lo que el
modelo sabe, cambia cómo lo expresa.

Referencia: Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017.

---

## 1.17 Incertidumbre: aleatórica y epistémica

Hay dos razones distintas por las que un modelo puede no estar seguro, y confundirlas
lleva a decisiones malas.

### Incertidumbre aleatórica

Ruido irreducible en los datos. La foto está borrosa, mal iluminada, o la hoja es
genuinamente ambigua.

**No se arregla con más datos.** Visitar ese árbol otra vez no ayuda si la foto es
mala por naturaleza.

### Incertidumbre epistémica

Ignorancia del modelo. Esta entrada no se parece a nada de lo que vio entrenando.

**Sí se reduce con más datos.** Y, crucialmente, **sí se reduce yendo a mirar**.

### Por qué la distinción decide el diseño

El triaje ordena la cola de inspección por incertidumbre **epistémica**, no por la
total. El razonamiento: si el modelo duda porque la imagen es intrínsecamente
ambigua, mandar a un técnico no resuelve nada. Si duda porque nunca ha visto algo
así, la visita humana es exactamente lo que aporta información.

### MC Dropout: cómo se mide

**Dropout** es una técnica de regularización: durante el entrenamiento se apaga
aleatoriamente una fracción de las neuronas (aquí el 30%) en cada paso. Fuerza al
modelo a no depender de ninguna neurona concreta.

Normalmente el dropout **se desactiva** al hacer predicciones, para que la respuesta
sea determinista.

**MC Dropout** (*Monte Carlo Dropout*) hace lo contrario: lo deja activado y pasa la
misma imagen varias veces (aquí 20). Cada pase apaga neuronas distintas, así que da
una respuesta ligeramente distinta. La **dispersión** de esas 20 respuestas es la
medida de incertidumbre.

Intuición: si la respuesta cambia mucho según qué neuronas se apaguen, el modelo no
tiene una convicción robusta. Si las 20 coinciden, sí.

Gal y Ghahramani (ICML 2016) demostraron que esto aproxima inferencia bayesiana
variacional. Es una aproximación, no una posterior exacta: sirve para **ordenar**,
no para citarle un intervalo de confianza a un agricultor.

### Descomposición matemática

Con `p_1 ... p_T` las T predicciones:

```
p_media = (1/T) · suma_t( p_t )

H(p) = −[ p·ln(p) + (1−p)·ln(1−p) ]        entropía binaria, en nats

Incertidumbre total      = H(p_media)
Incertidumbre aleatórica = (1/T) · suma_t( H(p_t) )
Incertidumbre epistémica = total − aleatórica
```

Esa última resta se llama **información mutua**. Es matemáticamente no negativa; el
código hace `np.maximum(..., 0.0)` porque en coma flotante puede cruzar el cero.

### El número que justifica todo el coste

MC Dropout multiplica por 20 el coste de inferencia. ¿Vale la pena? Solo si la
incertidumbre es realmente más alta donde el modelo se equivoca.

Lo medí sobre un modelo deliberadamente mediocre (54.7% de accuracy, para que
hubiera errores reales que medir):

```
n_errors: 29
mean_uncertainty_correct: 0.0225
mean_uncertainty_wrong:   0.1102
ratio: 4.91
```

**La incertidumbre epistémica es 4.9 veces más alta en las predicciones erróneas.**
Eso significa que ordenar la cola por incertidumbre pone los fallos arriba, que es
exactamente lo que quieres de una cola de inspección.

La función `uncertainty_separates_errors` reporta esta ratio en cada ejecución, en
vez de asumirla. Si algún día sale menor que 1, los 20 pases no están comprando
nada y hay que quitarlos.

---

## 1.18 Grad-CAM y el problema del atajo

Todas las métricas anteriores puntúan **qué** responde el modelo. Ninguna comprueba
**por qué**.

**Aprendizaje por atajo** (*shortcut learning*): el modelo encuentra una
correlación que funciona en el entrenamiento pero no es la causa real. Ejemplos
documentados en la literatura:

- Un detector de neumonía que aprendió a leer el identificador del hospital impreso
  en la esquina de la radiografía, porque un hospital tenía más casos graves.
- Un clasificador de lobos y huskies que aprendió a detectar nieve en el fondo.

**Por qué nuestro dataset es terreno fértil**: son primeros planos de hojas sobre
fondos controlados. Si las fotos de cada clase las tomó una persona distinta, con
otra cámara, otra luz y otro fondo, el modelo puede clasificar perfectamente sin
mirar la hoja ni una vez.

Un modelo así da PR-AUC 1.0 y **cero** valor sobre un limonar real, porque ahí no
hay fondo de estudio.

### Cómo funciona Grad-CAM

*Gradient-weighted Class Activation Mapping*.

1. Pasa la imagen por el modelo.
2. Quédate con los mapas de características de la última capa convolucional. En
   EfficientNet-B0 son 1280 mapas de 7×7 píxeles.
3. Calcula el gradiente de la puntuación de la clase respecto a cada mapa. Eso dice
   cuánto contribuye cada mapa.
4. Pondera los mapas por ese gradiente y súmalos.
5. Aplica ReLU (deja solo lo positivo): interesa la evidencia **a favor** de la
   clase, no en contra.

El resultado es un mapa de calor 7×7 que muestra qué posiciones espaciales apoyaron
la predicción.

### Cómo lo convertimos en una cifra automática

Mirar mapas de calor uno a uno no escala a 762 imágenes. El código los resume:

**`border_fraction`**: qué fracción de la atención cae en el anillo exterior del
encuadre. En un primer plano de hoja centrada, los bordes no contienen información
diagnóstica. Atención ahí es atención en la fotografía, no en la planta.

Pero hay una trampa que encontré al implementarlo, y es instructiva. Los mapas son
de 7×7. Un anillo de una celda de grosor en 7×7 es **24 de 49 celdas, el 49% del
área**. Un mapa perfectamente uniforme pondría el 49% de su atención ahí, y mi
primera versión lo marcaba como sospechoso. De hecho marcaba como sospechoso un
modelo con pesos aleatorios.

| Tamaño del mapa | El borde es... del área |
|---|---|
| 7×7 | 49.0% |
| 14×14 | 26.5% |
| 28×28 | 38.3% |
| 224×224 | 43.8% |

Un umbral absoluto significa cosas distintas a cada resolución. La corrección es
comparar contra la **cuota de área**:

```
border_excess = border_fraction / border_area_fraction
```

- **1.0** = exactamente proporcional. Un mapa uniforme da esto a cualquier
  resolución.
- **> 1.15** = el modelo prefiere los bordes. Sospechoso.

### La validación: un modelo entrenado para hacer trampas

Construí dos datasets donde la clase está codificada solo en una región, y entrené
un modelo en cada uno:

```
señal en el borde:  accuracy 100.0%, border excess 1.51x, 50% marcadas
señal en el centro: accuracy 100.0%, border excess 0.41x,  0% marcadas
```

**Los dos ajustan perfectamente.** Ninguna métrica del proyecto los distingue. El
mapa de calor sí.

El 50% tiene explicación exacta: la señal solo está en la clase positiva, así que la
mitad de las imágenes no tiene nada en el borde que mirar. Marca precisamente las que
hacen trampa.

El control del centro importa tanto como el caso positivo: sin él, un detector que
marcara todo pasaría el test.

---

## 1.19 Glosario de consulta rápida

| Término | Definición en una línea |
|---|---|
| **Accuracy** | Fracción de aciertos. Tóxica a baja prevalencia. No se usa aquí |
| **AMP** | Precisión mixta: cálculos en 16 bits para ir más rápido y gastar menos memoria |
| **Backbone** | La parte del modelo que extrae características, preentrenada |
| **Brier score** | Error cuadrático medio de las probabilidades. Regla de puntuación propia |
| **Checkpoint** | Fichero con los pesos del modelo más lo necesario para interpretarlos |
| **Dropout** | Apagar neuronas al azar durante el entrenamiento, para regularizar |
| **ECE** | Error de calibración esperado: desviación media entre confianza y realidad |
| **Época** | Una pasada completa por todo el conjunto de entrenamiento |
| **Especificidad** | De los sanos reales, cuántos descarto bien. `VN/(VN+FP)` |
| **F1** | Media armónica de PPV y sensibilidad. Asume que FP y FN cuestan igual |
| **Fuga de datos** | Información del test filtrada al entrenamiento. Infla las métricas |
| **Grad-CAM** | Mapa de calor de dónde mira el modelo |
| **Incertidumbre aleatórica** | Ruido irreducible. Más datos no ayudan |
| **Incertidumbre epistémica** | Ignorancia del modelo. Más datos sí ayudan |
| **Logit** | Salida cruda del modelo, antes del softmax. No es probabilidad |
| **Lote (batch)** | Grupo de imágenes procesadas juntas en un paso |
| **MC Dropout** | Dejar el dropout activo al predecir y medir la dispersión |
| **NPV** | De los descartados, cuántos están sanos de verdad |
| **Pérdida (loss)** | Número que mide cuánto se equivoca el modelo. Se minimiza |
| **PPV / Precisión** | De las alarmas dadas, cuántas son reales. `VP/(VP+FP)` |
| **PR-AUC** | Área bajo la curva precisión-recall. Métrica principal aquí |
| **Prevalencia** | Fracción realmente afectada de la población |
| **Punto de operación** | Un umbral concreto, con su sensibilidad y especificidad |
| **ROC-AUC** | Área bajo la curva ROC. Optimista con datos desbalanceados |
| **Sensibilidad / Recall** | De los enfermos reales, cuántos detecto. `VP/(VP+FN)` |
| **Softmax** | Convierte logits en probabilidades que suman 1 |
| **Sobreajuste** | Memorizar el entrenamiento en vez de aprender el patrón |
| **Temperatura** | Divisor de los logits que calibra sin cambiar el ranking |
| **Transferencia** | Reutilizar un modelo entrenado en otra tarea como punto de partida |
| **Umbral** | Valor a partir del cual una probabilidad se decide como positiva |

---

Siguiente: [02-datos.md](02-datos.md), la naturaleza real de los datos.
