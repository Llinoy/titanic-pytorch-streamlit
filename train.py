import argparse
import json
import os
import random
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)

from src.preprocess import TitanicPreprocessor
from src.model import MLPBinaryClassifier


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_kaggle_cli_ready(kaggle_json_path: str) -> None:
    """Copy kaggle.json to the location Kaggle CLI expects, then sanity-check."""
    p = Path(kaggle_json_path)
    if not p.exists():
        raise FileNotFoundError(f"kaggle json not found: {kaggle_json_path}")

    cfg_dir = Path("/root/.config/kaggle")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    dst = cfg_dir / "kaggle.json"
    dst.write_bytes(p.read_bytes())
    os.chmod(dst, 0o600)

    subprocess.run(["kaggle", "-v"], check=True)


def download_and_extract_train_csv(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "titanic.zip"
    train_csv = raw_dir / "train.csv"

    if train_csv.exists():
        return train_csv

    if not zip_path.exists():
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", "titanic", "-p", str(raw_dir), "--force"],
            check=True
        )

    if not zip_path.exists():
        raise FileNotFoundError(f"Missing: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract("train.csv", path=str(raw_dir))

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing after extract: {train_csv}")

    return train_csv


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    losses = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys = []
    probs = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.sigmoid(logits).cpu().numpy()
        probs.append(prob)
        ys.append(yb.numpy())

    y_true = np.concatenate(ys).astype(int)
    y_prob = np.concatenate(probs)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["roc_auc"] = None

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--kaggle_json", type=str, default="/content/drive/MyDrive/kaggle (1).json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dims", type=str, default="128,64")
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    set_seed(args.seed)

    project_dir = Path(args.project_dir)
    raw_dir = project_dir / "data" / "raw"
    artifacts_dir = project_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ensure_kaggle_cli_ready(args.kaggle_json)
    train_csv_path = download_and_extract_train_csv(raw_dir)

    df = pd.read_csv(train_csv_path)
    if "Survived" not in df.columns:
        raise ValueError("train.csv must contain Survived")

    df_train, df_val = train_test_split(
        df, test_size=args.val_size, random_state=args.seed, stratify=df["Survived"]
    )

    # Preprocess: fit on train only
    pre = TitanicPreprocessor().fit(df_train)
    X_train = pre.transform(df_train)
    y_train = df_train["Survived"].to_numpy(dtype=np.float32)

    X_val = pre.transform(df_val)
    y_val = df_val["Survived"].to_numpy(dtype=np.float32)

    preprocess_path = artifacts_dir / "preprocess.json"
    pre.save(str(preprocess_path))

    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(",") if x.strip()]
    model = MLPBinaryClassifier(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_score = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, device)
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = train_loss
        history.append(val_metrics)

        key = "roc_auc" if val_metrics.get("roc_auc") is not None else "accuracy"
        score = float(val_metrics[key]) if val_metrics[key] is not None else float(val_metrics["accuracy"])
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d} | loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['f1']:.4f} | "
            f"val_auc={val_metrics.get('roc_auc')}"
        )

    model_path = artifacts_dir / "model.pt"
    torch.save(
        {
            "state_dict": best_state if best_state is not None else model.state_dict(),
            "input_dim": int(X_train.shape[1]),
            "hidden_dims": hidden_dims,
            "dropout": float(args.dropout),
        },
        model_path
    )

    metrics_path = artifacts_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved preprocess to: {preprocess_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
