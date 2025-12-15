import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

from src.preprocess import TitanicPreprocessor
from src.model import MLPBinaryClassifier


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"


def load_model(artifacts_dir: Path):
    ckpt_path = artifacts_dir / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = MLPBinaryClassifier(
        input_dim=int(ckpt["input_dim"]),
        hidden_dims=ckpt.get("hidden_dims", [128, 64]),
        dropout=float(ckpt.get("dropout", 0.2)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_preprocessor(artifacts_dir: Path):
    p = artifacts_dir / "preprocess.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing preprocess artifacts: {p}")
    return TitanicPreprocessor.load(str(p))


@torch.no_grad()
def predict_proba(model, X: np.ndarray) -> np.ndarray:
    x = torch.tensor(X, dtype=torch.float32)
    logits = model(x)
    prob = torch.sigmoid(logits).numpy()
    return prob


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)

    acc = float((y_pred == y_true).mean())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


st.set_page_config(page_title="Titanic Inference", layout="wide")
st.title("Titanic Inference & Evaluation (PyTorch)")

st.sidebar.header("Artifacts")
artifacts_path_str = st.sidebar.text_input("Artifacts folder", str(ARTIFACTS_DIR))

st.sidebar.header("Input CSV")
uploaded = st.sidebar.file_uploader("Upload CSV (train/test style)", type=["csv"])
csv_path_str = st.sidebar.text_input("...or CSV path on disk", "")

threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

# Load artifacts
artifacts_dir = Path(artifacts_path_str)
try:
    model = load_model(artifacts_dir)
    pre = load_preprocessor(artifacts_dir)
    st.sidebar.success("Loaded model + preprocess")
except Exception as e:
    st.sidebar.error(f"Failed to load artifacts: {e}")
    st.stop()

# Load data
df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif csv_path_str.strip():
    df = pd.read_csv(csv_path_str.strip())

if df is None:
    st.info("Upload a CSV or provide a path in the sidebar.")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)

# Preprocess + predict
X = pre.transform(df)
probs = predict_proba(model, X)
preds = (probs >= threshold).astype(int)

out = df.copy()
out["pred_prob"] = probs
out["pred"] = preds

st.subheader("Predictions")
st.dataframe(out.head(50), use_container_width=True)

# If Survived exists, evaluate
if "Survived" in df.columns:
    y_true = df["Survived"].to_numpy(dtype=int)
    metrics = compute_metrics(y_true, probs, threshold=threshold)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    c2.metric("Precision", f"{metrics['precision']:.3f}")
    c3.metric("Recall", f"{metrics['recall']:.3f}")
    c4.metric("F1", f"{metrics['f1']:.3f}")

    st.subheader("Confusion Matrix")
    cm = np.array(metrics["confusion_matrix"])
    fig = plt.figure()
    plt.imshow(cm)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

# Probability histogram
st.subheader("Prediction Probability Histogram")
fig2 = plt.figure()
plt.hist(probs, bins=30)
plt.title("pred_prob histogram")
plt.xlabel("pred_prob")
plt.ylabel("count")
st.pyplot(fig2)

# Download results
st.subheader("Download")
csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
