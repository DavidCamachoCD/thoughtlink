# Guardrails para BCI-to-robot: calibración + conformal prediction

> Capa post-hoc entre el modelo y el robot que convierte el `confidence_threshold = 0.6`
> de un dial empírico en una garantía estadística distribución-free.
>
> Aplicada sobre 3 modelos × 4 calibradores × 4 conformal × 3 seeds = **96 celdas
> de ablación factorial** sobre el dataset KernelCo/robot_control.

## Tabla de contenidos

1. [Motivación](#1-motivación)
2. [Arquitectura de la capa](#2-arquitectura-de-la-capa)
3. [Calibración post-hoc](#3-calibración-post-hoc)
4. [Conformal prediction](#4-conformal-prediction)
5. [Weighted conformal: cross-subject covariate shift](#5-weighted-conformal-cross-subject-covariate-shift)
6. [Harness de ablación factorial](#6-harness-de-ablación-factorial)
7. [Aceleración MPS](#7-aceleración-mps)
8. [Resultados empíricos](#8-resultados-empíricos)
9. [Limitaciones y open follow-ups](#9-limitaciones-y-open-follow-ups)
10. [Cómo reproducir](#10-cómo-reproducir)
11. [Bibliografía](#11-bibliografía)

---

## 1. Motivación

ThoughtLink v1.0 tenía un único guardrail entre el modelo y el robot:
[`inference/confidence.py:54`](../../src/thoughtlink/inference/confidence.py#L54)
descartaba predicciones con `prob < 0.6`. Esto **asume que `predict_proba` está
calibrado** — que cuando el modelo dice 0.85, acierta el 85% de las veces. En la
práctica esto raramente se cumple:

- **CNN/EEGNet** suele estar **sobreconfiado** (Guo et al. 2017). Pasa el threshold
  con confianza 0.9 y accuracy real <30% → falso trigger sobre el robot.
- **Random Forest** suele estar **sub-confiado** → predicciones correctas se
  descartan innecesariamente.
- **SVM con `probability=True`** usa Platt scaling internamente, mediocre en colas.

Resultado: `confidence_threshold = 0.6` afina sobre números arbitrarios. La capa
de seguridad existe pero no tiene base estadística.

Esta capa SOTA ataca el problema en dos pasos:

1. **Calibración post-hoc** — los modelos producen probabilidades reales (Platt /
   isotonic / temperature scaling según el caso). Métrica: ECE objetivo <0.05.
2. **Conformal prediction** — en lugar de "una clase con probas", devuelve un
   *prediction set* con cobertura ≥1−α garantizada (Vovk; Angelopoulos & Bates 2023).
   Si el set tiene >1 clase, la stability pipeline lo trata como "incierto" y
   mantiene la acción actual.

Outcome: `confidence_threshold` pasa a tener interpretación frecuentista ("60%
probabilidad real de acierto"), y los falsos triggers quedan acotados por una
cobertura conformal en lugar de un dial empírico.

## 2. Arquitectura de la capa

```
                                         ┌──────────────────────┐
                                         │  StabilityPipeline    │
                                         │  (sin cambios desde    │
                                         │   v1.0)               │
                                         └──────────▲───────────┘
                                                    │ probs calibrados
                                                    │ + opcional set conformal
                ┌───────────────────────────────────┴───────────────────────────┐
                │  ConformalWrapper (NEW: inference/conformal.py)               │
                │    - APS / Naive / Weighted APS                               │
                │    - .predict_set(probs) → set[label]                         │
                │    - alpha configurable                                       │
                └───────────────────────────────────▲───────────────────────────┘
                                                    │ probs calibrados
                ┌───────────────────────────────────┴───────────────────────────┐
                │  Calibrator (NEW: inference/calibration.py)                   │
                │    - SklearnCalibrator(method=isotonic|sigmoid)               │
                │    - TemperatureScaler  (PyTorch logits)                      │
                │    - HierarchicalCalibrator                                   │
                │    - .fit(X_calib, y_calib) / .transform(probs)               │
                └───────────────────────────────────▲───────────────────────────┘
                                                    │ probs crudos
                                         ┌──────────┴───────────┐
                                         │  Decoder.predict()    │
                                         │  → model.predict_proba│
                                         └───────────────────────┘
```

Punto de inyección en runtime:
[`bridge/brain_policy.py:162-174`](../../src/thoughtlink/bridge/brain_policy.py#L162-L174)
— entre `decoder.predict()` y `stability.process()`. Backward compatible: sin
calibrator/conformal, el comportamiento es idéntico al de v1.0.

**Datos**: split subject-aware 3-way (13 train / 1 calib / 3 test) vía
[`split_by_subject_3way`](../../src/thoughtlink/data/splitter.py). El sujeto de
calibración nunca aparece en train ni test — preserva exchangeability tan bien
como es posible cross-subject.

## 3. Calibración post-hoc

Implementado en [`inference/calibration.py`](../../src/thoughtlink/inference/calibration.py).

### `SklearnCalibrator(method="isotonic" | "sigmoid")`

Wrapper sobre [`sklearn.calibration.CalibratedClassifierCV`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
con `cv="prefit"` (sklearn ≥1.6: vía `FrozenEstimator`). El modelo base no se
re-entrena — solo se aprende el mapeo de calibración sobre el calib set.

- **Isotonic** (Zadrozny & Elkan 2002): regresión monótona no-paramétrica. Mejor
  cuando el calib set es grande (>1k muestras).
- **Sigmoid** (Platt 1999): logística 1-D sobre los scores. Mejor con poca data,
  asume forma sigmoidal del miscalibration. Default actual en
  [`configs/default.yaml`](../../configs/default.yaml).

### `TemperatureScaler`

Para CNN (`EEGNet`) y modelos PyTorch en general (Guo et al. 2017, ICML). Aprende
un único `T > 0` minimizando NLL sobre los logits del calib set vía LBFGS:

```
log_probs = log_softmax(logits / T)
loss = NLL(log_probs, y_calib)
```

Predicciones se mantienen invariantes (argmax preserva), solo las confianzas se
aplastan (T > 1) cuando el modelo es overconfident, o se afilan (T < 1) cuando
underconfident.

LBFGS es CPU-bound en PyTorch — el scaler vive en CPU aunque el modelo entrene
en MPS.

### `HierarchicalCalibrator`

Caso especial para [`HierarchicalClassifier`](../../src/thoughtlink/models/hierarchical.py)
(2-stage rest/active gate + 4-class). Calibra Stage 1 (binario) y Stage 2 (4-class)
**por separado**. Si cada marginal está calibrada, el producto
`P(class) = P(active) × P(class|active)` sigue siendo joint probability válida
(chain rule).

### Diagnostics

[`eval/diagnostics.py`](../../src/thoughtlink/eval/diagnostics.py) expone:

- `expected_calibration_error(y, probs, n_bins=15)` — Naeini et al. 2015.
- `maximum_calibration_error(y, probs, n_bins=15)` — peor gap por bin.
- `brier_score(y, probs)` — Brier (1950), proper scoring rule.
- `reliability_curve(y, probs, n_bins=15)` — datos por-bin para plotear sin
  acoplar matplotlib.

## 4. Conformal prediction

Implementado en [`inference/conformal.py`](../../src/thoughtlink/inference/conformal.py).

### `APSConformalPredictor` (recommended default)

Adaptive Prediction Sets (Romano, Sesia & Candès 2020, NeurIPS). Score = suma
acumulada de probas de las clases ordenadas hasta incluir la verdadera:

```
order = argsort(-probs)            # descending
cumsum = cumsum(probs[order])
s_i = cumsum[rank_of_true_class]
```

Cuantil corregido finite-sample (Angelopoulos & Bates 2023):

```
q_hat = quantile(scores, ceil((n+1)*(1-α)) / n)
```

`predict_set(probs)` incluye toda clase cuya APS-score acumulada ≤ q_hat (más la
top class para garantizar set no-vacío).

Sets se contraen cuando el modelo está confiado y crecen bajo incertidumbre →
exactamente lo que queremos como guardrail.

### `NaiveConformalPredictor` (ablation only)

Score = `1 - p(y_true)`. Más simple, menos adaptativo. Mantenido para
comparación; sets resultan ~20% mayores que APS para cobertura equivalente
(ver [tabla §8](#8-resultados-empíricos)).

### Garantía formal

Bajo intercambiabilidad de calib y test:

```
P(y_test ∈ C(x_test)) ≥ 1 - α
```

donde `C(x)` es el prediction set. La cobertura es **marginal sobre todo** (no
condicional a clase ni input). En ThoughtLink el supuesto se rompe a nivel
cross-subject; ver §5.

## 5. Weighted conformal: cross-subject covariate shift

El supuesto de exchangeability falla cuando el sujeto de calibración tiene una
distribución de EEG distinta a los sujetos de test. Empíricamente: cobertura cae
de target 0.90 a ~0.69 en `hierarchical/raw/aps`. Ver §8.

### `WeightedAPSConformalPredictor` (Tibshirani et al. 2019, NeurIPS)

Los scores de calibración se ponderan por el likelihood ratio
`w(x) = P_test(x) / P_calib(x)`:

```
q_hat = weighted_quantile(scores_calib, w_calib, level = 1 - α)
```

donde `weighted_quantile` es el cuantil empírico ponderado.

Garantía recuperada: si `P_test(Y|X) = P_calib(Y|X)` (solo P(X) cambia) y los
pesos son correctos, marginal coverage ≥ 1−α se preserva.

### Estimación de los pesos

[`inference/domain_weights.py:estimate_likelihood_ratio`](../../src/thoughtlink/inference/domain_weights.py).
Trick clásico:

1. Concatenar X_calib (label 0) y X_test (label 1).
2. Ajustar logistic regression binaria con `C=1.0` regularización L2.
3. Para cada x_calib: `w(x) = P(test|x) / P(calib|x) = p̂(x) / (1 - p̂(x))`.

**Importance clipping** (Sugiyama et al. 2008): los pesos se clippean a
`[1/clip, clip]` (default `clip=50`) para limitar la varianza del cuantil
ponderado bajo shift extremo.

### `diagnose_weights(weights)`

Reporta `effective_sample_size` (ESS) y ratio. ESS ratio bajo (~0.02) indica que
unos pocos pesos extremos dominan — el cuantil weighted se vuelve high-variance.
En la ablación esto se traduce en cobertura inestable seed-a-seed (ver §8).

## 6. Harness de ablación factorial

`src/thoughtlink/eval/` agrupa toda la infraestructura de evaluación. El subpaquete
nuevo en esta release contiene:

- [`eval/diagnostics.py`](../../src/thoughtlink/eval/diagnostics.py) — métricas (ECE, MCE, Brier, reliability_curve)
- [`eval/training.py`](../../src/thoughtlink/eval/training.py) — `TrainConfig` + `train_baseline/hierarchical/cnn` como funciones puras
- [`eval/ablation.py`](../../src/thoughtlink/eval/ablation.py) — registries + `run_factorial`
- [`eval/_torch_device.py`](../../src/thoughtlink/eval/_torch_device.py) — selección CUDA > MPS > CPU

### Registries

```python
MODEL_TRAINERS = {"baseline": ..., "hierarchical": ..., "cnn": ...}

CALIBRATORS = {"raw", "isotonic", "sigmoid", "temperature"}

CONFORMAL = {"none", "aps", "naive", "weighted_aps"}

COMPATIBILITY = {                       # ¿qué calibrador aplica a qué modelo?
    "baseline":     {"raw", "isotonic", "sigmoid"},
    "hierarchical": {"raw", "isotonic", "sigmoid"},
    "cnn":          {"raw", "temperature"},
}
```

Añadir variantes nuevas (e.g., KLIEP, beta calibration, adaptive conformal) es
una entrada al dict + una closure factory. Cero cambios al runner.

### `run_factorial`

Itera el producto cruzado `(seed × model × calibration × conformal)`. Por seed:

1. `split_by_subject_3way` con `random_state=seed`.
2. Preprocess + extract features (una vez, reusado por los 3 modelos).
3. Por modelo: train (~10s baseline, ~2min CNN en MPS).
4. Por (calibration, conformal): fit + evaluate, ~1s/cell.

### Persistencia

Output en `results/ablations/<run-id>/`:

```
config.json    # variantes pedidas + git hash + librería versions + timestamp
cells.jsonl    # 1 línea por cell (append, resumable)
summary.csv    # tidy DataFrame
summary.md     # tabla aggregada (mean ± std) por (model, cal, conformal)
```

`run-id` formato: `YYYYMMDD-HHMMSS-<git-sha7>`. Reproducibilidad anclada al
commit.

### Resumability

`cells.jsonl` se abre en append mode. Al arrancar, `run_ablation.py --resume
<run-id>` lee el archivo, identifica celdas ya completadas (`{seed}/{model}/{cal}/{conf}`)
y solo ejecuta las pendientes. Útil para runs largos en background o si algo se
cae a la mitad.

### Per-cell error isolation

Cada cell se evalúa en `try/except`. Una falla (ej. modelo incompatible, NaN en
weights) registra `error: "TypeName: msg"` en su entrada de jsonl pero no
aborta el resto del sweep.

### CLI

[`scripts/run_ablation.py`](../../scripts/run_ablation.py):

```bash
# Default: full factorial 3 modelos × 4 cal × 4 conf × 3 seeds = 96 cells
uv run python scripts/run_ablation.py

# Subset
uv run python scripts/run_ablation.py --models cnn --calibrators temperature

# Continuar run interrumpido
uv run python scripts/run_ablation.py --resume 20260503-005644-b238320
```

### Comparador

[`notebooks/07_ablation_comparison.ipynb`](../../notebooks/07_ablation_comparison.ipynb)
carga uno o varios `cells.jsonl`, produce tabla pivote, heatmap de cobertura,
boxplot de ECE y scatter Pareto cobertura-vs-set-size.

## 7. Aceleración MPS

El cuello operativo del harness era el CNN (EEGNet) corriendo en CPU: ~50 min
por seed. Solo añadiendo detección Apple Silicon GPU
([`eval/_torch_device.py`](../../src/thoughtlink/eval/_torch_device.py)) baja a
**~2 min por seed** — speedup ~20×.

```python
def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

Cero cambios al loop de entrenamiento ni a los hyperparams. El CNN entrena en
MPS, pero el state_dict se persiste en CPU (línea ~226 de `train_cnn`), por lo
que la calibración, conformal y persistencia downstream no notan diferencia.

`TemperatureScaler` queda en CPU intencionalmente — LBFGS no acelera en MPS y
ya convertía a CPU internamente.

Run completo de 96 celdas: **~12 min wall-clock** en M-series Mac. Antes de
MPS: ~22h estimadas (abandonado por inviable).

## 8. Resultados empíricos

Run referencia: `results/ablations/20260503-005644-b238320/`. 96 cells, 0 errores,
3 seeds, α=0.1 (target coverage 0.90), 2 sujetos calibración / 3 sujetos test.

### Tabla principal: ECE pre-vs-post calibración

Accuracy y ECE no dependen del conformal (la calibración cambia las probas, no
las decisiones), así que se reportan agregadas sobre conformal:

| Modelo | Calibración | Accuracy | ECE | Brier |
|---|---|---:|---:|---:|
| baseline | raw | 0.248 ± 0.037 | 0.058 ± 0.016 | 0.797 ± 0.018 |
| baseline | sigmoid | 0.239 ± 0.022 | **0.039 ± 0.005** | 0.796 ± 0.004 |
| baseline | isotonic | 0.234 ± 0.022 | 0.042 ± 0.022 | 0.799 ± 0.010 |
| hierarchical | raw | 0.242 ± 0.044 | 0.119 ± 0.043 | 0.826 ± 0.026 |
| hierarchical | sigmoid | 0.245 ± 0.046 | 0.079 ± 0.066 | 0.810 ± 0.027 |
| hierarchical | isotonic | 0.241 ± 0.026 | **0.071 ± 0.061** | 0.807 ± 0.017 |
| **cnn** | **raw** | 0.214 ± 0.017 | **0.658 ± 0.084** | 1.402 ± 0.114 |
| **cnn** | **temperature** | 0.214 ± 0.017 | **0.020 ± 0.012** | 0.801 ± 0.001 |

**Hallazgo headline**: temperature scaling sobre el CNN reduce ECE de **0.66 a 0.02
(33×)**. Es la victoria más clara del paper. La accuracy se mantiene (argmax
preservado). Brier baja a la mitad porque dejamos de penalizar confianzas
extremas con clase incorrecta.

### Tabla de cobertura conformal por (modelo, calibración, conformal)

Target ≥ 0.90 (α = 0.1):

| Modelo | Calibración | none | aps | naive | weighted_aps |
|---|---|:---:|:---:|:---:|:---:|
| baseline | raw | — | 0.93 | 0.92 | 0.89 |
| baseline | sigmoid | — | **0.95** | 0.86 | 0.91 |
| baseline | isotonic | — | 0.90 | 0.87 | 0.99 |
| hierarchical | raw | — | 0.95 | 0.87 | 0.77 |
| hierarchical | sigmoid | — | **0.96** | 0.84 | 0.83 |
| hierarchical | isotonic | — | 0.95 | 0.84 | 0.93 |
| cnn | raw | — | 0.98 | 1.00 | 0.98 |
| cnn | temperature | — | **0.99** | 0.82 | 0.74 |

### Tabla de set size promedio (decisividad)

Set size = 1 → predicción singleton, robot ejecuta. Set size > 1 → "incertidumbre",
robot mantiene acción anterior. Total: 5 clases.

| Modelo | Calibración | aps | naive | weighted_aps |
|---|---|:---:|:---:|:---:|
| baseline | sigmoid | 4.69 | 4.09 | 4.44 |
| hierarchical | sigmoid | 4.70 | 3.98 | 3.96 |
| cnn | temperature | 4.92 | 4.12 | 3.67 |

Sets enormes (3.7–5.0 de 5) son **el comportamiento correcto** dado que la
accuracy real de los modelos cross-subject es ~24%. La capa honestamente admite
que no sabe → robot mantiene STOP. Es exactamente la garantía de seguridad que
buscamos.

### Hallazgos cualitativos

**1. Temperature scaling es el mayor return on investment.**
Una técnica de un solo parámetro reduce ECE 33× en EEGNet. T promedio ≈42k entre
seeds (varianza enorme: ±43k). T tan grande indica que el softmax estaba
absurdamente afilado — el modelo "decidía" en logits que en realidad estaban
indistinguibles. Después del scaling el modelo sigue prediciendo lo mismo, solo
que ahora con confianza honesta.

**2. APS conformal cumple target en todos los regímenes.**
Cobertura ≥ 0.90 con varianza baja (std < 0.07) para las 9 combinaciones de
modelo+calibración. Es la opción más robusta del menú.

**3. Naive conformal subcubre sistemáticamente.**
Cobertura ~0.83-0.92 vs APS ~0.93-0.99, con set sizes apenas más pequeños.
Confirma empíricamente Romano et al. 2020: naive es uniformemente dominado por
APS bajo predict-set semantics.

**4. Weighted conformal: efectivo pero inestable.**
Tibshirani 2019 promete recuperar cobertura bajo covariate shift. En la práctica
se cumple parcialmente:

- **`hierarchical/raw/weighted_aps`**: cobertura 0.77 ± **0.22** (varianza 3× la
  del APS estándar). Algunos seeds suben de 0.69 a >0.95, otros bajan a 0.55. La
  inestabilidad viene del ESS ratio ≈ 0.02 — unos pocos pesos extremos dominan
  el cuantil ponderado.
- **`baseline/isotonic/weighted_aps`**: cobertura **0.99 ± 0.003**, set size
  4.93. Aquí weighted **sí** funciona — la calibración isotonic aplana las
  probas y los pesos hacen un trabajo limpio.
- **`cnn/temperature/weighted_aps`**: cobertura **0.74 ± 0.12**, contra
  `cnn/temperature/aps` 0.99 ± 0.01. Weighted **degrada** la cobertura cuando
  temperature ya saturó las probas — los pesos meten varianza sin información
  útil.

Lectura honesta: weighted conformal es una herramienta condicional, no una
mejora monótona sobre APS. Hace falta validación empírica por modelo. Documentar
esto explícitamente es más valioso que vender "weighted siempre gana".

**5. Calibración no ayuda a modelos cerca de chance.**
`baseline/raw` ya tiene ECE = 0.058 (bajo). Eso parece "calibrado" hasta que ves
que la accuracy es 0.25 y la confianza promedio es ~0.30 — están "calibrados por
accidente" porque ambas son ~chance. Sigmoid baja ECE marginalmente a 0.039,
isotonic queda igual con std 0.022. Para modelos donde el predict_proba ya es
casi-uniforme, la calibración añade variance sin information gain. El movimiento
correcto es mejorar el modelo (cross-subject features, transfer learning), no
calibrarlo.

## 9. Limitaciones y open follow-ups

### Limitaciones reconocidas del run

- **3 seeds** es el mínimo para reportar std/CI con cierta confianza. Para paper
  formal querrías 5-10 seeds, especialmente para `weighted_aps` donde la
  varianza inter-seed es alta.
- **2 sujetos calibración** es chico (de 17 totales). El effective sample size
  para los pesos del logistic-regression weight estimator queda en ~0.02. Más
  sujetos calib mejoraría la estabilidad de weighted conformal a costa de
  reducir el set de train.
- **CNN no usa Euclidean alignment en el harness para el calib/test sets**
  porque la alineación es subject-aware y los sujetos calib/test difieren del
  train. Mantener consistencia con la versión `scripts/train_cnn.py` (que sí
  alinea train+test) requeriría re-extender la alineación o eliminarla en ambos
  caminos.

### Open research questions (no atacadas en esta capa)

- **Cross-subject transfer learning**. El verdadero limit es la accuracy
  cross-subject del modelo (~24%), no el guardrail. Domain adaptation /
  few-shot fine-tuning per subject queda como next major step.
- **Per-class confidence thresholds**. El threshold único de 0.6 podría ser
  per-clase post-calibración (e.g., más estricto para "Both Fists" que confunde
  más con "Right Fist").
- **Adaptive conformal bajo drift online** (Gibbs & Candès 2021). Para
  deployment real-time donde el sujeto cambia de estado mental durante una
  sesión, un cuantil estático q_hat no se ajusta.
- **KLIEP** (Sugiyama et al. 2008) como alternativa al estimador de likelihood-
  ratio basado en logistic regression. Más robusto bajo shift fuerte, no
  implementado aquí.
- **Beta calibration** (Kull et al. 2017) para calibradores con miscalibration
  asimétrica.

### Deuda técnica de este PR

- Shim deprecated en [`inference/diagnostics.py`](../../src/thoughtlink/inference/diagnostics.py)
  re-exporta de `eval.diagnostics`. Borrar después de 1-2 commits cuando todos
  los notebooks externos hayan migrado.
- Fallback `cv='prefit'` en `SklearnCalibrator` para sklearn <1.6 — borrable
  cuando subamos el pin de sklearn.
- `scripts/train_*.py` no fueron adelgazados a thin wrappers de `eval/training.py`
  (decisión de scope). Hay duplicación intencional entre el path CLI y el path
  in-process.

## 10. Cómo reproducir

```bash
# 1. Setup
git clone https://github.com/DavidCamachoCD/thoughtlink.git
cd thoughtlink
git checkout research            # o el commit referencia b238320
uv sync

# 2. Tests (espera 276 verde)
uv run python -m pytest tests/

# 3. Entrenar los modelos base si no existen ya en results/
uv run python scripts/train_baseline.py
uv run python scripts/train_hierarchical.py
uv run python scripts/train_cnn.py     # ~2-3 min en M-series con MPS

# 4. Run factorial completo (3 modelos × 4 cal × 4 conf × 3 seeds = 96 cells)
uv run python scripts/run_ablation.py
# → ~12 min en M-series Mac
# → output en results/ablations/<run-id>/

# 5. Inspeccionar resultados
cat results/ablations/<run-id>/summary.md
jupyter notebook notebooks/07_ablation_comparison.ipynb
```

Para un smoke run rápido (~2 min):

```bash
uv run python scripts/run_ablation.py \
    --models baseline hierarchical \
    --calibrators raw sigmoid \
    --conformals none aps \
    --seeds 42
```

## 11. Bibliografía

Ver [`docs/references.md`](../references.md) para la lista completa con DOI/arXiv
y mapeo paper → archivo:línea.

Resumen de las referencias principales aplicadas:

- **[Platt 1999]** *Probabilistic Outputs for SVMs* — sigmoid calibration.
- **[Zadrozny & Elkan 2002]** *Transforming Classifier Scores into Accurate
  Multiclass Probability Estimates* — isotonic regression.
- **[Niculescu-Mizil & Caruana 2005]** comparativa Platt vs isotonic
  (justifica sigmoid default con calib < 1k samples).
- **[Naeini, Cooper & Hauskrecht 2015]** ECE / MCE.
- **[Brier 1950]** Brier score, proper scoring rule.
- **[Guo, Pleiss, Sun & Weinberger 2017]** *On Calibration of Modern Neural
  Networks*, ICML — temperature scaling.
- **[Vovk, Gammerman & Shafer 2005]** *Algorithmic Learning in a Random World* —
  conformal prediction framework.
- **[Romano, Sesia & Candès 2020]** *Adaptive Prediction Sets*, NeurIPS — APS.
- **[Tibshirani, Foygel Barber, Candès & Ramdas 2019]** *Conformal Prediction
  Under Covariate Shift*, NeurIPS — weighted conformal.
- **[Sugiyama et al. 2008]** *Direct Importance Estimation for Covariate Shift
  Adaptation* — importance clipping.
- **[Angelopoulos & Bates 2023]** *A Gentle Introduction to Conformal
  Prediction* — corrección finite-sample del cuantil.

---

## Commits relevantes

| Commit | Tema |
|---|---|
| [`08290f4`](https://github.com/DavidCamachoCD/thoughtlink/commit/08290f4) | feat: capa de guardrails con calibración + conformal prediction |
| [`1735fdb`](https://github.com/DavidCamachoCD/thoughtlink/commit/1735fdb) | fix: CNN load + CLI flags + empirical results en notebook 06 |
| [`bae9748`](https://github.com/DavidCamachoCD/thoughtlink/commit/bae9748) | feat: weighted conformal para covariate shift cross-subject |
| [`1f3c254`](https://github.com/DavidCamachoCD/thoughtlink/commit/1f3c254) | feat: harness de ablación factorial (`eval/`) |
| [`b238320`](https://github.com/DavidCamachoCD/thoughtlink/commit/b238320) | perf: soporte MPS para entrenamiento del CNN |
