# infer_torch_autoencoder.py
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_torch_autoencoder import AutoEncoder, TabularDataset, reconstruction_errors
from utils import (
    setup_pandas_options,
    load_and_preprocess_data,
    add_additional_features,
    visualize_anomalies,
)




def load_artifacts(models_dir: str = "models"):
    models_dir = Path(models_dir)

    torch_path = models_dir / "torch_autoencoder.pt"
    scaler_path = models_dir / "torch_autoencoder_scaler.pkl"
    meta_path = models_dir / "torch_autoencoder_meta.json"

    if not torch_path.exists():
        raise FileNotFoundError(f"Not found: {torch_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Not found: {scaler_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Not found: {meta_path}")

    ckpt = torch.load(torch_path, map_location="cpu")
    scaler = joblib.load(scaler_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return ckpt, scaler, meta


def main(
    file_path: str,
    out_path: str = "models/torch_autoencoder_scored_infer.parquet",
    models_dir: str = "models",
    batch_size: int = 4096//8,
    plot: bool = True,
):
    setup_pandas_options()

    # Названия для графиков (как у вас)
    feature_names = {
        "gps_speed_kmh": "Скорость экскаватора (км/ч)",
        "heading_angle_deg": "Угол направления экскаватора (градусы)",
        "platform_inclination_x_deg": "Угол наклона платформы по оси X (градусы)",
        "platform_inclination_y_deg": "Угол наклона платформы по оси Y (градусы)",
        "boom_inclination_x_deg": "Угол наклона стрелы экскаватора по оси X (градусы)",
        "arm_inclination_deg": "Угол наклона рабочего оборудования (градусы)",
    }

    # 1) Load + FE (как в train)
    df_filtered = load_and_preprocess_data(file_path)
    df_filtered = add_additional_features(df_filtered)

    # 2) df_numeric: все numeric + timestamp
    numeric_cols = df_filtered.select_dtypes(include="number").columns.tolist()
    df_numeric = df_filtered[numeric_cols].copy()
    df_numeric["timestamp"] = df_filtered["timestamp"].values

    # 3) Load artifacts
    ckpt, scaler, meta = load_artifacts(models_dir=models_dir)
    feature_cols = ckpt["feature_cols"]
    input_dim = int(ckpt["input_dim"])
    threshold_value = float(meta["threshold_value"])

    # 4) Проверки согласованности
    missing = [c for c in feature_cols if c not in df_numeric.columns]
    if missing:
        raise ValueError(
            "В данных не хватает колонок, которые ожидает модель:\n"
            + "\n".join(missing[:50])
            + (f"\n...and {len(missing)-50} more" if len(missing) > 50 else "")
        )

    # Важно: порядок фичей должен быть таким же, как при обучении
    X = df_numeric[feature_cols].astype("float32").values
    Xs = scaler.transform(X).astype("float32")

    if Xs.shape[1] != input_dim:
        raise ValueError(f"Input dim mismatch: X has {Xs.shape[1]}, model expects {input_dim}")

    # 5) Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoEncoder(input_dim=input_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # 6) Errors + anomaly
    loader = DataLoader(TabularDataset(Xs), batch_size=batch_size, shuffle=False, drop_last=False)
    errs = reconstruction_errors(model, loader, device=device)
    anomaly = np.where(errs > threshold_value, -1, 1)

    # 7) Добавляем результаты в df_filtered (для графиков) и df_out (для сохранения)
    df_filtered = df_filtered.copy()
    df_filtered["recon_error"] = errs
    df_filtered["anomaly"] = anomaly

    df_out = df_numeric.copy()
    df_out["recon_error"] = errs
    df_out["anomaly"] = anomaly

    # 8) Save
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Anomalies: {(df_out['anomaly'] == -1).sum()} / {len(df_out)} (threshold={threshold_value})")

    # 9) Plot
    if plot:
        # Рисуем только те фичи, которые реально есть в df_filtered
        feature_names_existing = {k: v for k, v in feature_names.items() if k in df_filtered.columns}
        if not feature_names_existing:
            print("plot=True, но ни один из ключей feature_names не найден в df_filtered — графики не построены.")
        else:
            visualize_anomalies(df_filtered, # .iloc[::2],  # каждая 10-я точка
                                feature_names=feature_names_existing)


if __name__ == "__main__":
    # Пример:
    # python infer_torch_autoencoder.py
    main(file_path="dataset/ml_datasets/merged_resampled_telemetry_data.parquet", plot=True)

