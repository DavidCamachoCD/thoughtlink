# Ablation summary

`4` non-error cells across 1 seeds.

| model | calibration | conformal | n | n_params (mean ± std) | best_train_accuracy (mean ± std) | epochs_run (mean ± std) | accuracy (mean ± std) | ece (mean ± std) | mce (mean ± std) | brier (mean ± std) | q_hat (mean ± std) | coverage (mean ± std) | avg_set_size (mean ± std) | fraction_singleton (mean ± std) | temperature (mean ± std) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cnn | raw | aps | 1 | 5317.000 ± 0.000 | 0.307 ± 0.000 | 43.000 ± 0.000 | 0.232 ± 0.000 | 0.668 ± 0.000 | 0.770 ± 0.000 | 1.417 ± 0.000 | 1.000 ± 0.000 | 0.990 ± 0.000 | 4.924 ± 0.000 | 0.001 ± 0.000 | nan ± 0.000 |
| cnn | raw | none | 1 | 5317.000 ± 0.000 | 0.307 ± 0.000 | 43.000 ± 0.000 | 0.232 ± 0.000 | 0.668 ± 0.000 | 0.770 ± 0.000 | 1.417 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 |
| cnn | temperature | aps | 1 | 5317.000 ± 0.000 | 0.307 ± 0.000 | 43.000 ± 0.000 | 0.232 ± 0.000 | 0.033 ± 0.000 | 0.795 ± 0.000 | 0.801 ± 0.000 | 1.000 ± 0.000 | 0.989 ± 0.000 | 4.924 ± 0.000 | 0.000 ± 0.000 | 30756.593 ± 0.000 |
| cnn | temperature | none | 1 | 5317.000 ± 0.000 | 0.307 ± 0.000 | 43.000 ± 0.000 | 0.232 ± 0.000 | 0.033 ± 0.000 | 0.795 ± 0.000 | 0.801 ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 | nan ± 0.000 | 30756.593 ± 0.000 |
