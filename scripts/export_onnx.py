#!/usr/bin/env python3
"""Export trained models to ONNX format for optimized inference."""

import pickle
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from thoughtlink.models.cnn import EEGNet


def export_cnn_to_onnx(
    model: EEGNet,
    output_path: Path,
    n_channels: int = 6,
    n_samples: int = 500,
) -> None:
    """Export PyTorch EEGNet to ONNX.

    Args:
        model: Trained EEGNet instance.
        output_path: Path to save .onnx file.
        n_channels: Number of EEG channels.
        n_samples: Samples per window.
    """
    model.eval()
    dummy = torch.randn(1, 1, n_channels, n_samples)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["eeg_input"],
        output_names=["logits"],
        dynamic_axes={
            "eeg_input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )
    print(f"Exported CNN -> {output_path}")


def export_sklearn_to_onnx(
    model_path: Path,
    output_path: Path,
    n_features: int = 66,
) -> None:
    """Export sklearn pipeline to ONNX using skl2onnx.

    Args:
        model_path: Path to pickled sklearn model.
        output_path: Path to save .onnx file.
        n_features: Number of input features.
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("skl2onnx not installed, skipping sklearn export")
        print("Install with: uv add skl2onnx")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=14,
    )

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Exported {model_path.name} -> {output_path}")


def export_sklearn_pipeline_to_onnx(
    pipeline,
    output_path: Path,
    n_features: int = 66,
    name: str = "model",
) -> None:
    """Export an in-memory sklearn pipeline to ONNX.

    Args:
        pipeline: Fitted sklearn Pipeline object.
        output_path: Path to save .onnx file.
        n_features: Number of input features.
        name: Label for log messages.
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("skl2onnx not installed, skipping sklearn export")
        return

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_type,
        target_opset=14,
    )

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Exported {name} -> {output_path}")
    verify_onnx(output_path, (n_features,))


def verify_onnx(onnx_path: Path, input_shape: tuple[int, ...]) -> None:
    """Verify ONNX model inference with dummy data.

    Args:
        onnx_path: Path to .onnx file.
        input_shape: Shape of a single input sample.
    """
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, *input_shape).astype(np.float32)

    results = session.run(None, {input_name: dummy})
    print(f"  Verified {onnx_path.name}: output shape {results[0].shape}")


def main() -> None:
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Export CNN if checkpoint exists
    cnn_path = results_dir / "cnn_model.pt"
    if cnn_path.exists():
        model = EEGNet(n_classes=5, n_channels=6, n_samples=500)
        model.load_state_dict(torch.load(cnn_path, weights_only=True))
        onnx_path = results_dir / "eegnet.onnx"
        export_cnn_to_onnx(model, onnx_path)
        verify_onnx(onnx_path, (1, 6, 500))
    else:
        print(f"No CNN checkpoint at {cnn_path}, skipping CNN export")

    # Export best sklearn baseline
    baseline_path = results_dir / "best_baseline.pkl"
    if baseline_path.exists():
        onnx_path = results_dir / "best_baseline.onnx"
        export_sklearn_to_onnx(baseline_path, onnx_path, n_features=66)
        if onnx_path.exists():
            verify_onnx(onnx_path, (66,))
    else:
        print(f"No baseline model at {baseline_path}, skipping baseline export")

    # Export hierarchical model (sub-stages separately)
    hier_path = results_dir / "hierarchical_model.pkl"
    if hier_path.exists():
        with open(hier_path, "rb") as f:
            hier_model = pickle.load(f)

        # Stage 1: binary (Relax vs Active)
        export_sklearn_pipeline_to_onnx(
            hier_model.stage1_model,
            results_dir / "hierarchical_stage1.onnx",
            n_features=66,
            name="hierarchical_stage1",
        )
        # Stage 2: 4-class active intent
        export_sklearn_pipeline_to_onnx(
            hier_model.stage2_model,
            results_dir / "hierarchical_stage2.onnx",
            n_features=66,
            name="hierarchical_stage2",
        )
    else:
        print(f"No hierarchical model at {hier_path}, skipping")


if __name__ == "__main__":
    main()
