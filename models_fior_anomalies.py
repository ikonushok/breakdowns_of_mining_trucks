
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from scipy.stats import zscore
from sklearn.ensemble import IsolationForest


def detect_anomalies_isolation_forest(df_filtered, contamination=0.02, max_samples=0.2,
                                      model_name='IsolationForest.pkl'):
    """
    Используем Isolation Forest для выявления аномалий и сохраняем модель.
    """
    features = df_filtered.select_dtypes(
        include='number').columns  # Используем все числовые признаки, включая gps_speed_kmh
    model = IsolationForest(contamination=contamination, max_samples=max_samples, random_state=42)
    df_filtered['anomaly'] = model.fit_predict(df_filtered[features])  # Применяем модель ко всем числовым признакам

    # Сохраняем модель в файл
    path = Path('models')
    path.mkdir(parents=True, exist_ok=True)
    model_path = path / model_name
    joblib.dump(model, model_path)

    print(f"Модель сохранена в {model_path}")
    return df_filtered



def detect_anomalies_iqr(df_filtered, iqr_multiplier=1.0):
    """
    Используем метод IQR для выявления аномалий.
    """
    Q1 = df_filtered['gps_speed_kmh'].quantile(0.25)
    Q3 = df_filtered['gps_speed_kmh'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR

    df_filtered['anomaly'] = ((df_filtered['gps_speed_kmh'] < lower_bound) |
                              (df_filtered['gps_speed_kmh'] > upper_bound)).astype(int)

    # Переводим 0 в 1 для нормальных значений и 1 в -1 для аномалий, чтобы привести к единому формату
    df_filtered['anomaly'] = df_filtered['anomaly'].replace({0: 1, 1: -1})

    # Выводим количество аномальных точек
    print(f"Количество аномальных точек по IQR: {df_filtered['anomaly'].sum()}")

    return df_filtered




def detect_anomalies_zscore(df_filtered, zscore_threshold=3):
    """
    Используем Z-оценку для выявления аномалий.
    """
    df_filtered['gps_speed_zscore'] = zscore(df_filtered['gps_speed_kmh'])
    df_filtered['anomaly'] = (df_filtered['gps_speed_zscore'].abs() > zscore_threshold).astype(int)

    # Переводим 0 в 1 для нормальных значений и 1 в -1 для аномалий, чтобы привести к единому формату
    df_filtered['anomaly'] = df_filtered['anomaly'].replace({0: 1, 1: -1})

    # Выводим количество аномальных точек
    print(f"Количество аномальных точек по IQR: {df_filtered['anomaly'].sum()}")

    return df_filtered



