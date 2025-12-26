
import shap
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import plotly.io as pio
import plotly.graph_objects as go

from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler

pio.renderers.default = "browser"



def optimize_dtypes(df):
    df_optimized = df.copy()

    # float64 → float32
    float_cols = df_optimized.select_dtypes(include='float64').columns
    df_optimized[float_cols] = df_optimized[float_cols].astype('float32')

    # int64 → int32
    int_cols = df_optimized.select_dtypes(include='int64').columns
    df_optimized[int_cols] = df_optimized[int_cols].astype('int32')

    # object → category (для строк, где мало уникальных значений)
    object_cols = df_optimized.select_dtypes(include='object').columns
    for col in object_cols:
        num_unique_values = df_optimized[col].nunique()
        num_total_values = len(df_optimized[col])
        if num_unique_values / num_total_values < 0.5:
            df_optimized[col] = df_optimized[col].astype('category')

    return df_optimized

def save_parquet(df: pd.DataFrame, path: Path):
    """Сохранение DataFrame в Parquet (требует pyarrow или fastparquet)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = optimize_dtypes(df)  # Оптимизация типов данных перед сохранением
    df.to_parquet(path, index=False)

def setup_pandas_options():
    """Настройка глобальных опций pandas для отображения."""
    pd.set_option("display.precision", 3)
    pd.set_option("expand_frame_repr", False)



# 1) Подготовка телеметрии
def prepare_telemetry(telemetry_df):
    tele = telemetry_df.copy()
    tele["timestamp"] = pd.to_datetime(tele["timestamp"], utc=True)
    tele = tele.sort_values(["asset_id", "mdm_object_name", "timestamp"])  # сортировка перед diff
    tele["rail_error"] = tele["rail_pressure"] - tele["rail_pressure_target"]
    tele["rail_diff"] = tele.groupby(["asset_id", "mdm_object_name"], sort=False)["rail_pressure"].diff()
    return tele

# 2) Агрегация по окнам
def aggregate_window_data(tele, window="15min"):
    base_cols = [
        "rail_pressure", "rail_pressure_target",
        "rail_error", "rail_diff",
        "temp_engine", "pres_coolant_nn",
        "fuel_rate", "engine_load", "vehicle_speed"
    ]

    tele_base = tele[["timestamp"] + ["asset_id", "mdm_object_name"] + base_cols].copy()

    base_agg = (
        tele_base.set_index("timestamp")
        .groupby(["asset_id", "mdm_object_name"], sort=False)[
            ["rail_pressure", "rail_pressure_target", "rail_error", "rail_diff", "temp_engine", "pres_coolant_nn"]
        ]
        .resample(window)
        .agg(
            rail_pressure_std=("rail_pressure", "std"),
            rail_pressure_min=("rail_pressure", "min"),
            rail_pressure_max=("rail_pressure", "max"),
            rail_pressure_target_mean=("rail_pressure_target", "mean"),
            rail_error_mean=("rail_error", "mean"),
            rail_error_std=("rail_error", "std"),
            rail_error_max=("rail_error", "max"),
            rail_diff_std=("rail_diff", "std"),
            rail_diff_max=("rail_diff", "max"),
            temp_engine_mean=("temp_engine", "mean"),
            temp_engine_max=("temp_engine", "max"),
            pres_coolant_nn_mean=("pres_coolant_nn", "mean"),
        )
        .reset_index()
    )
    return base_agg

# 3) Агрегация топливных признаков для стационарных режимов
def aggregate_fuel_features(tele):
    steady = tele[(tele["vehicle_speed"] < 1) & (tele["engine_load"].between(20, 60))]
    fuel_agg = (
        steady.set_index("timestamp")
        .groupby(["asset_id", "mdm_object_name"], sort=False)["fuel_rate"]
        .resample("15min")
        .agg(fuel_rate_mean="mean", fuel_rate_std="std")
        .reset_index()
    )
    return fuel_agg

# 4) Мерджинг оконных признаков
def merge_window_features(base_agg, fuel_agg):
    df_win = base_agg.merge(fuel_agg, on=["asset_id", "mdm_object_name", "timestamp"], how="left", sort=False)
    df_win = df_win.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    return df_win

# 5) Подготовка событий
def prepare_events(idles_df):
    events = (
        idles_df[idles_df["case"] == "fuel_degradation"]
        .rename(columns={"event_time": "event_dt"})
        .copy()
    )
    events["event_dt"] = pd.to_datetime(events["event_dt"], utc=True)
    events = events.sort_values(["asset_id", "event_dt"]).reset_index(drop=True)
    return events

# 6) Слияние событий с окнами
def merge_events_with_windows(df_win, events):
    # Важно: сортируем сначала по времени, потом по asset_id
    df_win["timestamp"] = pd.to_datetime(df_win["timestamp"], utc=True)
    events["event_dt"] = pd.to_datetime(events["event_dt"], utc=True)
    df_win = df_win[df_win["timestamp"].notna()].copy()
    events = events[events["event_dt"].notna()].copy()

    # Важно: сортировка сначала по времени, потом по asset_id
    df_win = df_win.sort_values(["timestamp", "asset_id"], kind="mergesort").reset_index(drop=True)
    events = events.sort_values(["event_dt", "asset_id"], kind="mergesort").reset_index(drop=True)

    df_win = pd.merge_asof(
        df_win,
        events[["asset_id", "event_dt"]],
        left_on="timestamp",
        right_on="event_dt",
        by="asset_id",
        direction="forward",
        allow_exact_matches=True
    )
    return df_win

# 7) Таргет 7..30 дней, фильтрация 0..7 дней
def process_target(df_win, empty_window=7, target_window=30):
    df_win["days_to_event"] = (df_win["event_dt"] - df_win["timestamp"]).dt.total_seconds() / 86400.0
    df_win["target"] = (
            (df_win["days_to_event"] >= empty_window) & (df_win["days_to_event"] <= target_window)
    ).astype("int8")
    df_win = df_win[~((df_win["days_to_event"] >= 0) & (df_win["days_to_event"] < empty_window))]
    return df_win

# 8) Подготовка X и y
def prepare_X_y(df_win):
    df_win = df_win.set_index(["asset_id", "mdm_object_name", "timestamp"]).sort_index()
    y = df_win[f"target"].astype("int8")
    X = df_win.drop(columns=["target", "event_dt", "days_to_event"], errors="ignore")
    return X, y


def handle_nan_in_data(X_model, y):
    """
    Функция для обработки пропусков в данных (NaN).
    Заполняет NaN медианой и добавляет признаки is_missing.

    Parameters:
    X_model (DataFrame): Данные с признаками
    y (Series): Целевая переменная

    Returns:
    X_model (DataFrame): Данные с заполненными NaN и добавленными признаками
    y (Series): Целевая переменная с заполненными NaN
    """

    if X_model.isna().any().any() or y.isna().any():
        # Сохраняем, где были NaN (до заполнения)
        nan_mask = X_model.isna()
        nan_counts = nan_mask.sum()
        nan_cols = nan_counts[nan_counts > 0]

        # Добавляем is_missing признаки для колонок с NaN
        for col in nan_cols.index:
            X_model[f"{col}_is_missing"] = nan_mask[col].astype("int8")

        # Заполняем NaN в X_model медианой
        X_model = X_model.fillna(X_model.median())

        # Заполняем NaN в y медианой
        y = y.fillna(y.mode()[0])

        # Финальная проверка (жёсткая)
        assert not X_model.isna().any().any(), "NaN still present in X_model after filling"
        assert not y.isna().any(), "NaN still present in y after filling"

        print(
            "⚠️ NaN detected and handled:\n"
            f"Columns with NaN:\n{nan_cols}\n"
            f"Added is_missing flags: {len(nan_cols)}"
        )
    else:
        print("✅ No NaN detected in X_model and y")

    return X_model, y

def replace_nan_with_median(X_model, y):
    """
    Функция для обработки NaN в данных:
    - Проверяет наличие NaN в X_model и y
    - Заполняет NaN медианой
    - Добавляет признаки is_missing для NaN
    - Выводит отчёт о NaN до и после заполнения

    Parameters:
    X_model (DataFrame): Признаки модели
    y (Series): Целевая переменная

    Returns:
    X_model (DataFrame): Данные с заполненными NaN и добавленными признаками
    y (Series): Целевая переменная с заполненными NaN
    """
    try:
        # Проверяем на NaN в X_model и y
        assert not X_model.isna().any().any(), "NaN detected in X_model"
        assert not y.isna().any(), "NaN detected in y"
    except AssertionError as e:
        # Считаем количество NaN до заполнения
        X_model_nan_count_before = X_model.isna().sum()
        y_nan_count_before = y.isna().sum()

        # Заполняем NaN в X_model медианой
        X_model = X_model.fillna(X_model.median())  # Заполнение NaN медианой для всех столбцов
        y = y.fillna(y.median())  # Заполнение NaN в y медианой, если это необходимо

        # Проверка на NaN после заполнения
        X_model_nan_count_after = X_model.isna().sum()
        y_nan_count_after = y.isna().sum()

        # Если есть NaN после заполнения, покажем это в сообщении
        if X_model_nan_count_after.any() or y_nan_count_after:
            nan_message = f"NaN count after filling (X_model):\n{X_model_nan_count_after[X_model_nan_count_after > 0]}\n" if X_model_nan_count_after.any() else "No NaN in X_model after filling."
            raise ValueError(
                f"NaN detected and filled with median:\n"
                f"NaN count before filling (X_model):\n{X_model_nan_count_before[X_model_nan_count_before > 0]}\n"
                f"NaN count before filling (y): {y_nan_count_before}\n"
                f"{nan_message}"
                f"NaN count after filling (y): {y_nan_count_after}"
            )
        else:
            print("✅ NaN was detected and successfully filled with median!")

    return X_model, y

def calculate_event_level_precision(pred, threshold=0.6):
    """
    Функция для вычисления Precision на уровне событий.

    Parameters:
    pred (DataFrame): Данные с окнами и предсказаниями
    threshold (float): Порог для классификации (default 0.6)

    Returns:
    float: Precision на уровне событий
    """

    # 1. Фильтруем события, которые находятся в горизонте 7–30 дней
    pred_h = pred[
        (pred["days_to_event"] >= 7) & (pred["days_to_event"] <= 30)].copy()  # .copy() гарантирует работу с копией

    # 2. Создаем флаг, если хотя бы одно окно в пределах события получило proba >= threshold
    pred_h.loc[:, "alert"] = pred_h["proba"] >= threshold  # Используем .loc для модификации

    # 3. Для каждого события, если хотя бы одно окно с proba >= threshold, то событие считается обнаруженным
    event_alerts = (
        pred_h.groupby(["asset_id", "event_dt"])["alert"]
        .max()  # Если хотя бы одно окно события обнаружено, считаем событие обнаруженным
        .reset_index()
    )

    # 4. Добавляем столбец target в event_alerts
    event_alerts = event_alerts.merge(pred[["asset_id", "event_dt", "target"]], on=["asset_id", "event_dt"],
                                      how="left")

    # 5. Вычисляем количество истинных положительных, ложных положительных и ложных отрицательных
    true_positive = event_alerts[(event_alerts["alert"] == 1) & (event_alerts["target"] == 1)]
    false_positive = event_alerts[(event_alerts["alert"] == 1) & (event_alerts["target"] == 0)]
    false_negative = event_alerts[(event_alerts["alert"] == 0) & (event_alerts["target"] == 1)]

    # 6. Вычисляем precision на уровне событий
    precision_event_level = len(true_positive) / (len(true_positive) + len(false_positive)) \
        if len(true_positive) + len(false_positive) > 0 else 0

    print(f"\nPrecision на уровне событий: {precision_event_level:.4f}")
    print(f"True Positives: {len(true_positive)}")
    print(f"False Positives: {len(false_positive)}")
    print(f"False Negatives: {len(false_negative)}\n")

    return precision_event_level

def calculate_early_warning(df_win, proba, threshold=0.6):
    """
    Функция для вычисления ранности первого предупреждения (lead time).

    Parameters:
    df_win (DataFrame): Данные с окнами и предсказаниями.
    proba (array): Массив вероятностей, полученных моделью для каждого окна.
    threshold (float): Порог для классификации (default 0.6).

    Returns:
    DataFrame: Отчёт с первым предупреждением для каждого события.
    """

    # Собираем таблицу с предсказаниями на уровне окон
    pred = (
        df_win.reset_index()[["asset_id", "mdm_object_name", "timestamp", "event_dt", "days_to_event", "target"]]
        .copy()
    )

    pred["proba"] = proba  # proba должен быть в том же порядке, что и df_win/X

    # Берём только окна, которые относятся к горизонту 7..30 (как твой таргет)
    pred_h = pred[(pred["days_to_event"] >= 7) & (pred["days_to_event"] <= 30)].copy()

    # Флаг “сработали”
    pred_h["alert"] = pred_h["proba"] >= threshold

    # Для каждого события: самое раннее срабатывание (максимальный days_to_event)
    first_alert = (
        pred_h[pred_h["alert"]]
        .sort_values(["asset_id", "event_dt", "days_to_event"], ascending=[True, True, False])
        .groupby(["asset_id", "event_dt"], as_index=False)
        .first()[["asset_id", "event_dt", "timestamp", "days_to_event", "proba"]]
        .rename(columns={"timestamp": "first_alert_ts", "days_to_event": "lead_days", "proba": "first_alert_proba"})
    )

    # Список всех “реальных” событий в этом горизонте (чтобы увидеть пропуски)
    all_events = (
        pred_h[pred_h["target"] == 1][["asset_id", "event_dt"]]
        .drop_duplicates()
    )

    # Слияние всех событий с первым предупреждением
    report = all_events.merge(first_alert, on=["asset_id", "event_dt"], how="left")
    report["detected"] = report["lead_days"].notna()

    # Выводим отчёт
    print("=== Early warning report ===")
    print("threshold:", threshold)
    print("events in horizon:", len(report))
    print("detected:", report["detected"].sum(), f"({report['detected'].mean():.1%})")
    print("lead_days (only detected):")
    print(report.loc[report["detected"], "lead_days"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(2))

    return report



