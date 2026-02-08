"""Comprehensive LOSO-CV training with all accuracy improvements.

Runs ablation study comparing configurations:
1. baseline_42: Band power + Hjorth (42 features)
2. time_domain_66: + time-domain stats (66 features)
3. norm_66: + subject normalization
4. csp_86: + CSP features (86 features)
5. aligned_csp_86: + Euclidean alignment
6. ensemble: + ensemble voting
"""

import sys
import json
import copy
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thoughtlink.data.loader import load_all, CLASS_NAMES, get_class_distribution
from thoughtlink.data.splitter import get_subject_folds
from thoughtlink.preprocessing.eeg import preprocess_all
from thoughtlink.preprocessing.windowing import windows_from_samples
from thoughtlink.preprocessing.alignment import euclidean_align
from thoughtlink.features.eeg_features import extract_features_from_windows
from thoughtlink.features.csp_features import CSPFeatureExtractor
from thoughtlink.features.normalization import normalize_features_by_subject
from thoughtlink.features.fusion import fuse_all_features
from thoughtlink.models.baseline import build_baselines
from thoughtlink.models.hierarchical import HierarchicalClassifier
from thoughtlink.models.ensemble import VotingEnsemble


def run_single_fold(
    train_samples: list[dict],
    val_samples: list[dict],
    use_time_domain: bool = False,
    use_csp: bool = False,
    use_subject_norm: bool = False,
    use_alignment: bool = False,
    use_ensemble: bool = False,
    csp_n_components: int = 4,
) -> dict[str, float]:
    """Run a single LOSO fold with configurable improvements.

    Returns dict of {model_name: accuracy}.
    """
    # Preprocess (deep copy to avoid mutating across configs)
    train_s = copy.deepcopy(train_samples)
    val_s = copy.deepcopy(val_samples)
    preprocess_all(train_s)
    preprocess_all(val_s)

    # Windows
    X_train_win, y_train, train_subj_ids = windows_from_samples(train_s)
    X_val_win, y_val, val_subj_ids = windows_from_samples(val_s)

    # Euclidean alignment (on raw windows, before features)
    if use_alignment:
        X_train_win = euclidean_align(X_train_win, train_subj_ids)
        X_val_win = euclidean_align(X_val_win, val_subj_ids)

    # Standard features
    X_train_feat = extract_features_from_windows(
        X_train_win, include_time_domain=use_time_domain
    )
    X_val_feat = extract_features_from_windows(
        X_val_win, include_time_domain=use_time_domain
    )

    # CSP features
    if use_csp:
        csp = CSPFeatureExtractor(
            n_components=csp_n_components,
            n_classes=len(CLASS_NAMES),
        )
        csp_train = csp.fit_transform(X_train_win, y_train)
        csp_val = csp.transform(X_val_win)
        X_train_feat = fuse_all_features(X_train_feat, csp_features=csp_train)
        X_val_feat = fuse_all_features(X_val_feat, csp_features=csp_val)

    # Subject normalization
    if use_subject_norm:
        X_train_feat, _ = normalize_features_by_subject(X_train_feat, train_subj_ids)
        X_val_feat, _ = normalize_features_by_subject(X_val_feat, val_subj_ids)

    # Train and evaluate models
    results = {}

    # Individual baselines
    models = build_baselines()
    for name, model in models.items():
        model.fit(X_train_feat, y_train)
        y_pred = model.predict(X_val_feat)
        results[name] = accuracy_score(y_val, y_pred)

    # Hierarchical
    hier = HierarchicalClassifier()
    hier.fit(X_train_feat, y_train)
    y_pred = hier.predict(X_val_feat)
    results["hierarchical"] = accuracy_score(y_val, y_pred)

    # Ensemble
    if use_ensemble:
        ensemble_models = [
            ("logreg", build_baselines()["logreg"]),
            ("svm_rbf", build_baselines()["svm_rbf"]),
            ("random_forest", build_baselines()["random_forest"]),
            ("hierarchical", HierarchicalClassifier()),
        ]
        ensemble = VotingEnsemble(models=ensemble_models, voting="soft")
        ensemble.fit(X_train_feat, y_train)
        y_pred = ensemble.predict(X_val_feat)
        results["ensemble"] = accuracy_score(y_val, y_pred)

    return results


def run_loso_cv(
    samples: list[dict],
    config_name: str,
    **kwargs,
) -> dict:
    """Run full LOSO-CV with given configuration."""
    subject_ids = sorted(set(s["subject_id"] for s in samples))
    n_subjects = len(subject_ids)
    folds = get_subject_folds(samples, n_folds=n_subjects)

    all_fold_results = []

    for fold_idx, (train_samples, val_samples) in enumerate(folds):
        val_subjects = sorted(set(s["subject_id"] for s in val_samples))
        print(f"  Fold {fold_idx + 1}/{n_subjects}: test={val_subjects}", end="  ")

        fold_results = run_single_fold(train_samples, val_samples, **kwargs)

        accs = [f"{v:.3f}" for v in fold_results.values()]
        print(f"  accs: {', '.join(accs)}")

        all_fold_results.append(fold_results)

    # Average across folds
    model_names = list(all_fold_results[0].keys())
    avg_results = {}
    for name in model_names:
        accs = [f[name] for f in all_fold_results if name in f]
        avg_results[name] = {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "per_fold": [float(a) for a in accs],
        }

    return avg_results


def main():
    print("=" * 60)
    print("ThoughtLink - LOSO-CV Ablation Study")
    print("=" * 60)

    # Load data once
    print("\nLoading dataset...")
    samples = load_all()
    print(f"Loaded {len(samples)} samples, {len(set(s['subject_id'] for s in samples))} subjects")
    print(f"Class distribution: {get_class_distribution(samples)}")

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    configs = {
        "baseline_42": dict(
            use_time_domain=False, use_csp=False,
            use_subject_norm=False, use_alignment=False, use_ensemble=False,
        ),
        "time_domain_66": dict(
            use_time_domain=True, use_csp=False,
            use_subject_norm=False, use_alignment=False, use_ensemble=False,
        ),
        "norm_66": dict(
            use_time_domain=True, use_csp=False,
            use_subject_norm=True, use_alignment=False, use_ensemble=False,
        ),
        "csp_86": dict(
            use_time_domain=True, use_csp=True,
            use_subject_norm=True, use_alignment=False, use_ensemble=False,
        ),
        "aligned_csp_86": dict(
            use_time_domain=True, use_csp=True,
            use_subject_norm=True, use_alignment=True, use_ensemble=False,
        ),
        "ensemble": dict(
            use_time_domain=True, use_csp=True,
            use_subject_norm=True, use_alignment=True, use_ensemble=True,
        ),
    }

    all_results = {}

    for config_name, config_params in configs.items():
        print(f"\n{'=' * 60}")
        print(f"Configuration: {config_name}")
        print(f"  Params: {config_params}")
        print(f"{'=' * 60}")

        results = run_loso_cv(samples, config_name, **config_params)
        all_results[config_name] = results

        print(f"\n  --- {config_name} Summary ---")
        for model_name, stats in results.items():
            print(f"  {model_name:20s}: {stats['mean']:.3f} +/- {stats['std']:.3f}")

    # Save all results
    with open(output_dir / "improved_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print final comparison table
    print(f"\n{'=' * 80}")
    print("FINAL COMPARISON TABLE (LOSO-CV Mean Accuracy)")
    print(f"{'=' * 80}")

    # Collect all model names
    all_model_names = set()
    for r in all_results.values():
        all_model_names.update(r.keys())
    all_model_names = sorted(all_model_names)

    # Header
    header = f"{'Config':<20}"
    for m in all_model_names:
        header += f" {m:<14}"
    print(header)
    print("-" * len(header))

    # Rows
    for config_name, results in all_results.items():
        row = f"{config_name:<20}"
        for m in all_model_names:
            if m in results:
                row += f" {results[m]['mean']:.3f}{'':>9}"
            else:
                row += f" {'--':>14}"
        print(row)

    print(f"\nChance level: {1/len(CLASS_NAMES):.3f}")
    print(f"\nResults saved to {output_dir / 'improved_results.json'}")


if __name__ == "__main__":
    main()
