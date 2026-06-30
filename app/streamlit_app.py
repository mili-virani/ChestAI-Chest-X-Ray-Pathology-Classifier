import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT))

from dataset import CLASSES  # noqa: E402
from gradcam import generate_gradcam  # noqa: E402
from model import ChestXrayClassifier  # noqa: E402

MODEL_PATH = ROOT / "models" / "best_model.pth"

# Dark-theme palette
BG_COLOR = "#0e1117"
PANEL_COLOR = "#1a1f2e"
TEXT_COLOR = "#fafafa"
ACCENT_COLOR = "#4b9eff"
POSITIVE_COLOR = "#ff4b4b"
NEGATIVE_COLOR = "#4b9eff"
GRID_COLOR = "#2d3344"
SUCCESS_COLOR = "#3dd68c"
WARNING_COLOR = "#f5c542"
DANGER_COLOR = "#ff4b4b"


def _confidence_badge_style(confidence: float) -> tuple[str, str]:
    if confidence > 0.5:
        return DANGER_COLOR, "High"
    if confidence >= 0.3:
        return WARNING_COLOR, "Medium"
    return SUCCESS_COLOR, "Low"


def _apply_dark_theme() -> None:
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            .stApp {{
                background-color: {BG_COLOR};
                color: {TEXT_COLOR};
                font-family: 'Inter', sans-serif;
                margin-top: 0;
            }}
            header[data-testid="stHeader"] {{
                background-color: #0e1117;
            }}
            div[data-testid="stAppViewContainer"] {{
                background-color: #0e1117;
            }}
            div[data-testid="stToolbar"] {{
                background-color: #0e1117;
            }}
            [data-testid="stSidebar"] {{
                background-color: {PANEL_COLOR};
            }}
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
                color: {TEXT_COLOR};
            }}
            [data-testid="stSidebar"] * {{
                color: #e8eaed !important;
            }}
            [data-testid="stSidebar"] .stMarkdown li {{
                color: #e8eaed !important;
                opacity: 1 !important;
            }}
            [data-testid="stSidebar"] .stCaption,
            [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
                color: #9aa4b2 !important;
                opacity: 1 !important;
            }}
            ::selection {{
                background-color: #4b9eff;
                color: #0e1117;
            }}
            .disclaimer {{
                background-color: {PANEL_COLOR};
                border-left: 4px solid {ACCENT_COLOR};
                padding: 1rem 1.25rem;
                border-radius: 0.5rem;
                font-size: 0.9rem;
                color: #c9d1d9;
            }}
            .upload-zone {{
                border: 2px dashed #3d4f72;
                border-radius: 16px;
                padding: 2.5rem 1.5rem;
                min-height: 130px;
                text-align: center;
                background: linear-gradient(145deg, {PANEL_COLOR} 0%, #121826 100%);
                margin-bottom: 0.75rem;
                transition: all 0.25s ease;
                box-shadow: 0 0 0 rgba(75, 158, 255, 0);
            }}
            .upload-zone:hover {{
                border-color: {ACCENT_COLOR};
                box-shadow: 0 0 24px rgba(75, 158, 255, 0.18);
                transform: translateY(-1px);
            }}
            .upload-icon {{
                font-size: 2.75rem;
                margin-bottom: 0.75rem;
                line-height: 1;
            }}
            .upload-title {{
                font-size: 1.15rem;
                font-weight: 600;
                color: {TEXT_COLOR};
                margin-bottom: 0.35rem;
            }}
            .upload-subtitle {{
                font-size: 0.85rem;
                color: #9aa4b2;
            }}
            .section-divider {{
                display: flex;
                align-items: center;
                gap: 1rem;
                margin: 1.75rem 0 1.25rem 0;
            }}
            .section-divider-line {{
                flex: 1;
                height: 1px;
                background: linear-gradient(90deg, transparent, {ACCENT_COLOR}, transparent);
                opacity: 0.55;
            }}
            .section-divider-label {{
                font-size: 0.72rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: #8b949e;
                white-space: nowrap;
            }}
            .sidebar-divider {{
                height: 2px;
                background: linear-gradient(90deg, {ACCENT_COLOR}, transparent);
                border: none;
                margin: 1rem 0;
                opacity: 0.7;
            }}
            .sidebar-brand {{
                font-size: 1.85rem;
                font-weight: 700;
                color: {TEXT_COLOR};
                margin-bottom: 0.15rem;
            }}
            .prediction-banner {{
                position: relative;
                background: linear-gradient(135deg, {PANEL_COLOR} 0%, #141b28 100%);
                border: 1px solid #2d3a52;
                border-radius: 14px;
                padding: 1.35rem 1.5rem;
                margin: 0.5rem 0 1.25rem 0;
                animation: pulse-border 2.4s ease-in-out infinite;
            }}
            @keyframes pulse-border {{
                0%, 100% {{ box-shadow: 0 0 0 0 rgba(75, 158, 255, 0.25); }}
                50% {{ box-shadow: 0 0 0 6px rgba(75, 158, 255, 0.05); }}
            }}
            .prediction-label {{
                font-size: 0.78rem;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                color: #8b949e;
                margin-bottom: 0.5rem;
            }}
            .prediction-disease {{
                font-size: 1.85rem;
                font-weight: 700;
                color: {TEXT_COLOR};
                margin-bottom: 0.75rem;
            }}
            .confidence-badge {{
                display: inline-block;
                padding: 0.35rem 0.85rem;
                border-radius: 999px;
                font-size: 0.9rem;
                font-weight: 600;
                color: #0e1117;
            }}
            .metric-card {{
                background: {PANEL_COLOR};
                border: 1px solid #2d3a52;
                border-radius: 12px;
                padding: 1.1rem 1.25rem;
                height: 100%;
            }}
            .metric-card-accent-red {{ border-top: 3px solid {DANGER_COLOR}; }}
            .metric-card-accent-blue {{ border-top: 3px solid {ACCENT_COLOR}; }}
            .metric-card-accent-green {{ border-top: 3px solid {SUCCESS_COLOR}; }}
            .metric-card-label {{
                font-size: 0.72rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #8b949e;
                margin-bottom: 0.45rem;
            }}
            .metric-card-value {{
                font-size: 1.35rem;
                font-weight: 700;
                color: {TEXT_COLOR};
                line-height: 1.2;
            }}
            .metric-card-sub {{
                font-size: 0.82rem;
                color: #9aa4b2;
                margin-top: 0.35rem;
            }}
            div[data-testid="stFileUploader"] {{
                position: relative;
                margin-top: -130px;
                opacity: 0;
                height: 130px;
                cursor: pointer;
                z-index: 10;
            }}
            div[data-testid="stFileUploader"] button {{
                display: none;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_upload_zone() -> None:
    st.markdown(
        """
        <div class="upload-zone">
            <div class="upload-icon">🫁</div>
            <div class="upload-title">Drop your chest X-ray here</div>
            <div class="upload-subtitle">Supports PNG, JPG up to 200MB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_section_divider(label: str) -> None:
    st.markdown(
        f"""
        <div class="section-divider">
            <div class="section-divider-line"></div>
            <div class="section-divider-label">{label}</div>
            <div class="section-divider-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar_divider() -> None:
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)


def _render_top_prediction_banner(top_class: str, top_confidence: float) -> None:
    badge_color, _ = _confidence_badge_style(top_confidence)
    disease_name = top_class.replace("_", " ")
    st.markdown(
        f"""
        <div class="prediction-banner">
            <div class="prediction-label">Top Prediction</div>
            <div class="prediction-disease">{disease_name}</div>
            <span class="confidence-badge" style="background-color: {badge_color};">
                {top_confidence:.1%} confidence
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metrics_row(top_class: str, top_confidence: float, predictions: dict[str, float]) -> None:
    findings_count = sum(1 for prob in predictions.values() if prob > 0.2)
    _, risk_level = _confidence_badge_style(top_confidence)
    disease_name = top_class.replace("_", " ")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card metric-card-accent-blue">
                <div class="metric-card-label">Top Disease</div>
                <div class="metric-card-value">{disease_name}</div>
                <div class="metric-card-sub">{top_confidence:.1%} confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card metric-card-accent-green">
                <div class="metric-card-label">Findings &gt; 20%</div>
                <div class="metric-card-value">{findings_count}</div>
                <div class="metric-card-sub">pathologies above threshold</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        risk_color = {
            "High": DANGER_COLOR,
            "Medium": WARNING_COLOR,
            "Low": SUCCESS_COLOR,
        }[risk_level]
        st.markdown(
            f"""
            <div class="metric-card metric-card-accent-red">
                <div class="metric-card-label">Overall Risk</div>
                <div class="metric-card-value" style="color: {risk_color};">{risk_level}</div>
                <div class="metric-card-sub">based on top confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


@st.cache_resource
def load_model_and_metadata() -> tuple[ChestXrayClassifier, torch.device, float, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChestXrayClassifier(
        num_classes=len(CLASSES),
        pretrained=False
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    val_auc = 0.7946
    per_class_auc = {}

    if isinstance(checkpoint, dict):
        val_auc = float(checkpoint.get("val_auc", val_auc))
        per_class_auc = checkpoint.get("per_class_auc", per_class_auc)

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    state_dict = {
        key.replace("module.", ""): value
        for key, value in state_dict.items()
    }

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=False
    )

    if missing_keys:
        st.warning(f"Missing keys while loading model: {len(missing_keys)}")

    if unexpected_keys:
        st.warning(f"Unexpected keys while loading model: {len(unexpected_keys)}")

    model.to(device)
    model.eval()

    return model, device, val_auc, per_class_auc


def _plot_per_class_auc(per_class_auc: dict) -> plt.Figure:
    if not per_class_auc:
        per_class_auc = {name: 0.0 for name in CLASSES}

    labels = [name.replace("_", " ") for name in CLASSES]
    values = [float(per_class_auc.get(name, 0.0)) for name in CLASSES]
    order = np.argsort(values)

    fig, ax = plt.subplots(figsize=(6, 7), facecolor=PANEL_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    bars = ax.barh(
        np.array(labels)[order],
        np.array(values)[order],
        color=ACCENT_COLOR,
        height=0.65,
    )
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("AUC-ROC", color=TEXT_COLOR, fontsize=10)
    ax.set_title("Per-Class Validation AUC", color=TEXT_COLOR, fontsize=12, pad=12)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.grid(axis="x", color=GRID_COLOR, linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.spines["left"].set_color(GRID_COLOR)

    for bar, value in zip(bars, np.array(values)[order]):
        ax.text(
            min(value + 0.02, 0.95),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            color=TEXT_COLOR,
            fontsize=8,
        )

    fig.tight_layout()
    return fig


def _plot_prediction_probabilities(predictions: dict[str, float]) -> plt.Figure:
    labels = [name.replace("_", " ") for name in CLASSES]
    probs = [predictions[name] for name in CLASSES]
    order = np.argsort(probs)

    sorted_labels = np.array(labels)[order]
    sorted_probs = np.array(probs)[order]
    colors = [POSITIVE_COLOR if p > 0.5 else NEGATIVE_COLOR for p in sorted_probs]

    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    bars = ax.barh(sorted_labels, sorted_probs, color=colors, height=0.65)
    ax.set_xlim(0, 1.0)
    ax.axvline(0.5, color="#888888", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xlabel("Predicted Probability", color=TEXT_COLOR, fontsize=11)
    ax.set_title("Pathology Predictions", color=TEXT_COLOR, fontsize=14, pad=14)
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    ax.grid(axis="x", color=GRID_COLOR, linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.spines["left"].set_color(GRID_COLOR)

    for bar, prob in zip(bars, sorted_probs):
        ax.text(
            min(prob + 0.02, 0.95),
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.1%}",
            va="center",
            ha="left",
            color=TEXT_COLOR,
            fontsize=9,
        )

    fig.tight_layout()
    return fig


st.set_page_config(
    page_title="ChestAI — X-Ray Pathology Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)
_apply_dark_theme()

try:
    model, device, val_auc, per_class_auc = load_model_and_metadata()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.markdown('<div class="sidebar-brand">🫁 ChestAI</div>', unsafe_allow_html=True)
    st.caption("NIH ChestX-ray14 Pathology Classifier")

    _render_sidebar_divider()
    with st.expander("Per-Class AUC", expanded=False):
        st.pyplot(_plot_per_class_auc(per_class_auc), use_container_width=True)

st.title("Chest X-Ray Pathology Classifier")
st.markdown(
    "Upload a frontal chest radiograph to obtain multi-label pathology predictions "
    "and a Grad-CAM explanation for the top finding."
)

_render_section_divider("Upload")
_render_upload_zone()
uploaded_file = st.file_uploader(
    "Upload chest X-ray (PNG or JPG)",
    type=["png", "jpg", "jpeg"],
    help="Supported formats: PNG, JPG, JPEG",
    label_visibility="collapsed",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Running inference and generating Grad-CAM..."):
        predictions, gradcam_image = generate_gradcam(model, image, device=device)

    top_class = max(predictions, key=predictions.get)
    top_confidence = predictions[top_class]

    _render_section_divider("Results")
    _render_top_prediction_banner(top_class, top_confidence)
    _render_metrics_row(top_class, top_confidence, predictions)

    _render_section_divider("Probabilities")
    st.subheader("Prediction Probabilities")
    st.pyplot(_plot_prediction_probabilities(predictions), use_container_width=True)

    _render_section_divider("Visualization")
    col_original, col_gradcam = st.columns(2)

    with col_original:
        st.subheader("Original X-Ray")
        st.image(image, use_container_width=True)

    with col_gradcam:
        st.subheader(
            f"Grad-CAM Explanation — {top_class.replace('_', ' ')} ({top_confidence:.1%})"
        )
        st.caption("Target layer: `model.backbone.conv_head`")
        st.image(gradcam_image, use_container_width=True)

else:
    st.info("Upload a chest X-ray image to begin analysis.")

st.divider()
st.markdown(
    '<p class="disclaimer"><strong>Disclaimer:</strong> '
    "For research purposes only. Not for clinical use.</p>",
    unsafe_allow_html=True,
)