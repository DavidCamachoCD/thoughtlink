"""Streamlit real-time dashboard for ThoughtLink BCI demo.

Usage:
    uv run streamlit run src/thoughtlink/viz/dashboard.py
"""

import sys
import pickle
import time
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from thoughtlink.data.loader import CLASS_NAMES, load_sample
from thoughtlink.preprocessing.eeg import preprocess_eeg
from thoughtlink.features.eeg_features import extract_window_features
from thoughtlink.inference.decoder import RealtimeDecoder
from thoughtlink.inference.confidence import StabilityPipeline
from thoughtlink.bridge.intent_to_action import intent_to_action_name
from thoughtlink.bridge.brain_policy import load_config


ACTION_COLORS = {
    "RIGHT": "#3b82f6",
    "LEFT": "#eab308",
    "FORWARD": "#22c55e",
    "STOP": "#ef4444",
}

ACTION_EMOJIS = {
    "RIGHT": "-->",
    "LEFT": "<--",
    "FORWARD": "^",
    "STOP": "#",
}

CLASS_COLORS = ["#3b82f6", "#eab308", "#22c55e", "#f97316", "#6b7280"]


# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="ThoughtLink BCI", layout="wide")
st.title("ThoughtLink — Brain to Robot")
st.caption("Real-time BCI decoding dashboard")


# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.header("Configuration")

config_path = st.sidebar.text_input("Config", value="configs/default.yaml")
config = load_config(config_path)

model_options = list(Path("results").glob("*.pkl")) if Path("results").exists() else []
if not model_options:
    st.sidebar.warning("No trained models found in results/. Train a model first.")
    st.stop()

model_path = st.sidebar.selectbox(
    "Model",
    options=model_options,
    format_func=lambda p: p.stem,
)

data_dir = Path(config["data"]["cache_dir"])
npz_files = sorted(data_dir.rglob("*.npz")) if data_dir.exists() else []
if not npz_files:
    st.sidebar.warning(f"No .npz files in {data_dir}. Download the dataset first.")
    st.stop()

selected_file = st.sidebar.selectbox(
    "Signal file",
    options=npz_files,
    format_func=lambda p: p.name,
)

speed = st.sidebar.slider("Playback speed", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
run_btn = st.sidebar.button("Run", type="primary", use_container_width=True)


# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model(model_path)


# ── Main layout ──────────────────────────────────────────────
col_action, col_probs = st.columns([1, 2])

with col_action:
    action_placeholder = st.empty()
    info_placeholder = st.empty()

with col_probs:
    probs_chart = st.empty()

st.divider()

col_eeg, col_timeline = st.columns(2)

with col_eeg:
    st.subheader("EEG Signal (6 channels)")
    eeg_plot = st.empty()

with col_timeline:
    st.subheader("Action Timeline")
    timeline_plot = st.empty()

log_expander = st.expander("Step Log", expanded=False)
log_placeholder = log_expander.empty()


# ── Run ──────────────────────────────────────────────────────
if run_btn:
    sample = load_sample(selected_file)
    eeg_clean = preprocess_eeg(sample["eeg"])

    eeg_cfg = config["preprocessing"]["eeg"]
    inf_cfg = config["inference"]
    sfreq = eeg_cfg["sfreq"]
    prediction_hz = inf_cfg["prediction_hz"]
    samples_per_step = int(sfreq / prediction_hz)

    decoder = RealtimeDecoder(
        model=model,
        feature_extractor=lambda w: extract_window_features(w, sfreq=sfreq),
        window_size_s=config["preprocessing"]["window_duration_s"],
        sfreq=sfreq,
    )
    stability = StabilityPipeline(
        confidence_threshold=inf_cfg["confidence_threshold"],
        hysteresis_margin=inf_cfg["hysteresis_margin"],
        debounce_count=inf_cfg["debounce_count"],
        smoother_window=inf_cfg["smoother_window"],
    )

    # State for plots
    action_history: list[str] = []
    time_history: list[float] = []
    probs_history: list[np.ndarray] = []
    log_lines: list[str] = []

    info_placeholder.info(
        f"File: **{selected_file.name}** | Label: **{sample['label']}** | Subject: {sample['subject_id']}"
    )

    progress = st.progress(0)
    total_steps = eeg_clean.shape[0] // samples_per_step

    for step_idx, i in enumerate(range(0, eeg_clean.shape[0], samples_per_step)):
        chunk = eeg_clean[i : i + samples_per_step]
        decoder.feed_samples(chunk)

        probs, latency_ms = decoder.predict()
        if probs is None:
            continue

        raw_idx = int(np.argmax(probs))
        raw_intent = CLASS_NAMES[raw_idx]
        stable_intent = stability.process(probs, CLASS_NAMES)
        action = intent_to_action_name(stable_intent)
        confidence = float(probs[raw_idx])
        timestamp_s = i / sfreq

        action_history.append(action)
        time_history.append(timestamp_s)
        probs_history.append(probs)

        # ── Update action display ──
        color = ACTION_COLORS.get(action, "#888")
        sym = ACTION_EMOJIS.get(action, "?")
        action_placeholder.markdown(
            f"<div style='text-align:center;padding:20px;'>"
            f"<span style='font-size:64px;font-family:monospace;'>{sym}</span><br>"
            f"<span style='font-size:32px;font-weight:bold;color:{color};'>{action}</span><br>"
            f"<span style='font-size:14px;color:#aaa;'>conf: {confidence:.2f} | lat: {latency_ms:.1f}ms</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Update probability bars ──
        fig_probs, ax_probs = plt.subplots(figsize=(6, 2.5))
        bars = ax_probs.barh(CLASS_NAMES, probs, color=CLASS_COLORS)
        ax_probs.set_xlim(0, 1)
        ax_probs.axvline(inf_cfg["confidence_threshold"], color="red", ls="--", lw=1, label="threshold")
        ax_probs.set_xlabel("Probability")
        ax_probs.legend(fontsize=8)
        fig_probs.tight_layout()
        probs_chart.pyplot(fig_probs)
        plt.close(fig_probs)

        # ── Update EEG plot (last 2s) ──
        window_start = max(0, i - int(2 * sfreq))
        eeg_window = eeg_clean[window_start : i + samples_per_step]
        t_axis = np.arange(eeg_window.shape[0]) / sfreq + window_start / sfreq

        fig_eeg, ax_eeg = plt.subplots(figsize=(6, 3))
        channels = ["AFF6", "AFp2", "AFp1", "AFF5", "FCz", "CPz"]
        for ch_idx in range(min(eeg_window.shape[1], 6)):
            offset = ch_idx * 30
            ax_eeg.plot(t_axis, eeg_window[:, ch_idx] + offset, lw=0.5, label=channels[ch_idx])
        ax_eeg.set_xlabel("Time (s)")
        ax_eeg.set_ylabel("Channel")
        ax_eeg.legend(fontsize=6, loc="upper right", ncol=3)
        fig_eeg.tight_layout()
        eeg_plot.pyplot(fig_eeg)
        plt.close(fig_eeg)

        # ── Update timeline ──
        if len(action_history) > 1:
            fig_tl, ax_tl = plt.subplots(figsize=(6, 2))
            action_to_y = {"STOP": 0, "LEFT": 1, "FORWARD": 2, "RIGHT": 3}
            ys = [action_to_y.get(a, 0) for a in action_history]
            colors = [ACTION_COLORS.get(a, "#888") for a in action_history]
            ax_tl.scatter(time_history, ys, c=colors, s=20)
            ax_tl.set_yticks([0, 1, 2, 3])
            ax_tl.set_yticklabels(["STOP", "LEFT", "FORWARD", "RIGHT"])
            ax_tl.set_xlabel("Time (s)")
            ax_tl.set_xlim(0, eeg_clean.shape[0] / sfreq)
            fig_tl.tight_layout()
            timeline_plot.pyplot(fig_tl)
            plt.close(fig_tl)

        # ── Log ──
        log_lines.append(
            f"{timestamp_s:6.1f}s  {action:<8s}  raw={raw_intent:<16s}  conf={confidence:.2f}  lat={latency_ms:.1f}ms"
        )
        log_placeholder.code("\n".join(log_lines[-20:]))

        # ── Progress + pacing ──
        progress.progress(min((step_idx + 1) / max(total_steps, 1), 1.0))
        time.sleep(max(0.01, (1.0 / prediction_hz) / speed))

    progress.progress(1.0)
    st.success(
        f"Done — {len(action_history)} steps | "
        f"Ground truth: **{sample['label']}** | "
        f"Most common action: **{max(set(action_history), key=action_history.count) if action_history else 'N/A'}**"
    )
