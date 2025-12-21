# train_torch_autoencoder.py
import os
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler

from utils import setup_pandas_options, load_and_preprocess_data, add_additional_features


# -----------------------------
# Dataset
# -----------------------------
class TabularDataset(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = torch.from_numpy(x).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]


# -----------------------------
# Model
# -----------------------------
class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden=(256//4, 128//4, 64//2)):
        super().__init__()
        h1, h2, h3 = hidden

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(h2, h3),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(h3, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# -----------------------------
# Helpers
# -----------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def reconstruction_errors(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    errs = []
    for batch in loader:
        batch = batch.to(device)
        recon = model(batch)
        # MSE по строке
        e = torch.mean((recon - batch) ** 2, dim=1)
        errs.append(e.detach().cpu().numpy())
    return np.concatenate(errs, axis=0)


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 5,
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for x in train_loader:
            x = x.to(device)
            opt.zero_grad()
            recon = model(x)
            loss = loss_fn(recon, x)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                recon = model(x)
                loss = loss_fn(recon, x)
                val_losses.append(loss.item())

        tr = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va = float(np.mean(val_losses)) if val_losses else float("nan")
        print(f"Epoch {ep:02d} | train={tr:.6f} | val={va:.6f}")

        if va < best_val - 1e-6:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping. Best val={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# -----------------------------
# Main
# -----------------------------
def main(
    file_path: str,
    out_dir: str = "models",
    batch_size: int = 4096//8,
    epochs: int = 50,
    lr: float = 1e-3,
    threshold_quantile: float = 0.999,  # порог аномалий по квантилю ошибки
    seed: int = 42,
):
    setup_pandas_options()
    set_seed(seed)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Load + FE
    df = load_and_preprocess_data(file_path)
    df = add_additional_features(df)

    # 2) Собираем df_numeric: все числовые + timestamp
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    df_numeric = df[numeric_cols].copy()
    df_numeric["timestamp"] = df["timestamp"].values

    # 3) Матрица X: все фичи, кроме timestamp
    feature_cols = [c for c in df_numeric.columns if c != "timestamp"]
    X = df_numeric[feature_cols].astype("float32").values

    # 4) StandardScaler (сохраняем!)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype("float32")

    # 5) Train/Val split (по времени, чтобы не перемешивать)
    n = Xs.shape[0]
    split = int(n * 0.5)
    X_train, X_val = Xs[:split], Xs[split:]

    train_loader = DataLoader(TabularDataset(X_train), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TabularDataset(X_val), batch_size=batch_size, shuffle=False, drop_last=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoEncoder(input_dim=Xs.shape[1])

    model = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        patience=6,
    )

    # 6) Ошибки реконструкции на всём датасете → порог → метки anomaly (1/-1)
    full_loader = DataLoader(TabularDataset(Xs), batch_size=batch_size, shuffle=False, drop_last=False)
    errs = reconstruction_errors(model, full_loader, device=device)

    thr = float(np.quantile(errs, threshold_quantile))
    anomaly = np.where(errs > thr, -1, 1)

    df_out = df_numeric.copy()
    df_out["recon_error"] = errs
    df_out["anomaly"] = anomaly

    # 7) Save artifacts
    torch_path = out / "torch_autoencoder.pt"
    scaler_path = out / "torch_autoencoder_scaler.pkl"
    meta_path = out / "torch_autoencoder_meta.json"
    parquet_out = out / "torch_autoencoder_scored.parquet"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(Xs.shape[1]),
            "feature_cols": feature_cols,
        },
        torch_path,
    )
    joblib.dump(scaler, scaler_path)

    meta = {
        "threshold_quantile": threshold_quantile,
        "threshold_value": thr,
        "n_rows": int(n),
        "seed": seed,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    df_out.to_parquet(parquet_out, index=False)

    print("\nSaved:")
    print(f"- model:   {torch_path}")
    print(f"- scaler:  {scaler_path}")
    print(f"- meta:    {meta_path}")
    print(f"- scored:  {parquet_out}")
    print(f"\nAnomalies: {(df_out['anomaly'] == -1).sum()} / {len(df_out)} (threshold={thr:.66f})")


if __name__ == "__main__":
    # Пример:
    # python train_torch_autoencoder.py
    main(file_path="dataset/ml_datasets/merged_resampled_telemetry_data.parquet")
