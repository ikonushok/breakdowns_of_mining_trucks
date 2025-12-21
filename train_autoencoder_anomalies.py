import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from utils import (
    setup_pandas_options,
    load_and_preprocess_data,
    add_additional_features,
    standardize_data,
    save_parquet,
    visualize_anomalies,
)


class TabularDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = torch.from_numpy(data.astype(np.float32))

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        hidden_dim1 = max(latent_dim * 4, 32)
        hidden_dim2 = max(latent_dim * 2, 16)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec


def prepare_dataframe(file_path: str) -> pd.DataFrame:
    """
    Полный пайплайн подготовки данных:
    1. Загрузка и переименование столбцов.
    2. Добавление временных и инженерных признаков.
    3. Стандартизация базовых физических признаков.
    4. Удаление NaN после сдвигов/rolling.
    """
    df = load_and_preprocess_data(file_path)
    df = add_additional_features(df)
    df = standardize_data(df)
    df = df.dropna().reset_index(drop=True)
    return df


def train_autoencoder(
    df_numeric: pd.DataFrame,
    batch_size: int = 512,
    epochs: int = 20,
    lr: float = 1e-3,
    latent_dim: int = 16,
    device: str | None = None,
) -> tuple[Autoencoder, np.ndarray]:
    """
    Обучение автоэнкодера на числовых признаках.
    Возвращает модель и вектор ошибок реконструкции для всех объектов.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    x = df_numeric.to_numpy(dtype=np.float32)
    dataset = TabularDataset(x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = Autoencoder(input_dim=x.shape[1], latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="mean")

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch:03d} | train MSE = {epoch_loss:.6f}")

    # Оценка ошибок реконструкции на всех объектах
    model.eval()
    with torch.no_grad():
        all_tensor = torch.from_numpy(x).to(device)
        recon_all = model(all_tensor)
        errors = torch.mean((recon_all - all_tensor) ** 2, dim=1).cpu().numpy()

    return model, errors


def main():
    parser = argparse.ArgumentParser(description="Обучение автоэнкодера для поиска аномалий (PyTorch).")
    parser.add_argument(
        "--data-path",
        type=str,
        default="dataset/ml_datasets/merged_resampled_telemetry_data.parquet",
        help="Путь до parquet-файла с телеметрией экскаваторов.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="dataset/ml_datasets/merged_resampled_telemetry_with_ae_anomalies.parquet",
        help="Путь для сохранения таблицы с оценками аномалий.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=16)
    args = parser.parse_args()

    setup_pandas_options()

    # 1. Подготовка данных
    df = prepare_dataframe(args.data_path)
    print("После подготовки данных:", df.shape)

    # 2. Формирование числовой матрицы признаков
    df_numeric = df.select_dtypes(include="number").copy()
    print("Числовых признаков:", df_numeric.shape[1])

    # 3. Обучение автоэнкодера
    model, recon_errors = train_autoencoder(
        df_numeric=df_numeric,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        latent_dim=args.latent_dim,
    )

    # 4. Расчёт порога и формирование флага аномалии
    mean_err = recon_errors.mean()
    std_err = recon_errors.std()
    threshold = mean_err + 2.0 * std_err

    print(f"Средняя ошибка реконструкции: {mean_err:.6f}")
    print(f"Стандартное отклонение ошибки: {std_err:.6f}")
    print(f"Порог аномалии (mu + 3*sigma): {threshold:.6f}")

    anomaly_flag = (recon_errors > threshold).astype(int)
    # Приводим к формату, аналогичному другим детекторам: 1 — норм, -1 — аномалия
    anomaly_iforest_style = np.where(anomaly_flag == 1, -1, 1)

    # 4.1. Подсчёт числа аномалий
    num_anomalies = int(anomaly_flag.sum())
    total_samples = anomaly_flag.shape[0]
    print(
        f"Обнаружено аномалий: {num_anomalies} "
        f"({num_anomalies / total_samples:.4%} от общего числа {total_samples})"
    )

    # 5. Формирование итогового DataFrame (с исходными колонками)
    out_df = df.copy()
    out_df["ae_recon_error"] = recon_errors
    out_df["ae_anomaly_flag"] = anomaly_flag
    out_df["anomaly"] = anomaly_iforest_style  # для совместимости с visualize_anomalies

    # 6. Визуализация аномалий так же, как для других способов
    feature_names = {
        "gps_speed_kmh": "Скорость по GPS, км/ч",
        "heading_angle_deg": "Курс, градусы",
        "platform_inclination_x_deg": "Наклон платформы X, градусы",
        "platform_inclination_y_deg": "Наклон платформы Y, градусы",
        "boom_inclination_x_deg": "Наклон стрелы X, градусы",
        "arm_inclination_deg": "Наклон рукояти, градусы",
    }
    visualize_anomalies(out_df, feature_names=feature_names)

    # 7. Сохранение модели и результатов
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "input_dim": df_numeric.shape[1], "latent_dim": args.latent_dim},
        models_dir / "autoencoder_anomaly.pt",
    )
    print("PyTorch-модель сохранена в models/autoencoder_anomaly.pt")

    save_parquet(out_df, Path(args.output_path))
    print(f"Результирующий DataFrame с аномалиями сохранён в {args.output_path}")


if __name__ == "__main__":
    main()
