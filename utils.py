
import shap
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import plotly.graph_objects as go

from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler



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

def load_and_preprocess_data(file_path: str):
    """
    Загрузка и предварительная обработка данных:
    1. Переименование столбцов для более понятных названий.
    2. Фильтрация нужных столбцов.
    3. Преобразование timestamp в datetime.
    4. Создание временных признаков.
    """
    df = pd.read_parquet(file_path)

    # Переименовываем столбцы для более понятных названий
    df.rename(columns={
        'mdm_object_uuid': 'excavator_id',
        'create_dt': 'timestamp',
        'speed_gps': 'gps_speed_kmh',
        'direction': 'heading_angle_deg',
        'inclinom_platx': 'platform_inclination_x_deg',
        'inclinom_platy': 'platform_inclination_y_deg',
        'inclinom_boomx': 'boom_inclination_x_deg',
        'inclinom_arm': 'arm_inclination_deg'
    }, inplace=True)

    # Оставляем только нужные столбцы с новыми более понятными именами
    columns_needed = [
        'excavator_id', 'timestamp', 'gps_speed_kmh', 'heading_angle_deg',
        'platform_inclination_x_deg', 'platform_inclination_y_deg',
        'boom_inclination_x_deg', 'arm_inclination_deg'
    ]

    # Создаем новый DataFrame с нужными столбцами
    df_filtered = df[columns_needed].copy()  # Используем .copy() для явного создания копии

    # Преобразуем timestamp в datetime
    df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])

    # Добавляем дополнительные временные признаки
    df_filtered['hour'] = df_filtered['timestamp'].dt.hour
    df_filtered['dayofweek'] = df_filtered['timestamp'].dt.dayofweek
    df_filtered['month'] = df_filtered['timestamp'].dt.month
    df_filtered['year'] = df_filtered['timestamp'].dt.year

    return df_filtered

def standardize_data(df_filtered):
    """
    Стандартизируем числовые столбцы для улучшения работы модели.
    """
    scaler = StandardScaler()
    df_filtered[['gps_speed_kmh', 'heading_angle_deg', 'platform_inclination_x_deg',
                 'platform_inclination_y_deg', 'boom_inclination_x_deg', 'arm_inclination_deg']] = \
        scaler.fit_transform(df_filtered[['gps_speed_kmh', 'heading_angle_deg', 'platform_inclination_x_deg',
                                          'platform_inclination_y_deg', 'boom_inclination_x_deg',
                                          'arm_inclination_deg']])

    return df_filtered

def visualize_anomalies(df_filtered, feature_names):
    """
    Визуализируем указанные признаки с выделением аномалий, используя их русские названия.

    :param df_filtered: DataFrame с данными экскаватора.
    :param feature_names: Словарь признаков с русскими названиями для визуализации.
    """
    # Проходим по каждому признаку и его русскому названию в словаре feature_names
    for feature, feature_title in feature_names.items():
        # Визуализация признака с выделением аномалий
        fig = go.Figure()

        # Добавляем основную линию для признака
        fig.add_trace(go.Scatter(x=df_filtered['timestamp'],
                                 y=df_filtered[feature],
                                 mode='lines',
                                 name=feature,
                                 line=dict(color='blue')))

        # Добавляем аномальные точки (где anomaly == -1)
        anomalies = df_filtered[df_filtered['anomaly'] == -1]
        fig.add_trace(go.Scatter(x=anomalies['timestamp'],
                                 y=anomalies[feature],
                                 mode='markers',
                                 name='Аномалии',
                                 marker=dict(color='red', symbol='x', size=10)))

        # Настройка графика с подписями на русском
        fig.update_layout(
            title=f'{feature_title} с аномалиями (красные точки)',  # Заголовок на русском
            xaxis_title='Время',  # Подпись оси X на русском
            yaxis_title=feature_title,  # Подпись оси Y на русском
            showlegend=True
        )

        # Показываем график в браузере
        fig.show(renderer="browser")

def add_additional_features(df_filtered):
    # Список признаков, для которых будем создавать сдвиги и дифференцированные признаки
    shift_columns = ['gps_speed_kmh', 'heading_angle_deg', 'platform_inclination_x_deg',
                     'platform_inclination_y_deg', 'boom_inclination_x_deg', 'arm_inclination_deg']

    # Добавляем сдвиги на 1, 2, 3 временных шага для каждого признака
    for col in shift_columns:
        for shift_val in range(1, 4):  # сдвиги на 1, 2 и 3
            df_filtered[f'{col}_shift_{shift_val}'] = df_filtered[col].shift(shift_val)

        # Дифференцированные признаки (разница между текущим и предыдущими значениями)
        for diff_val in range(1, 4):  # diff на 1, 2 и 3
            df_filtered[f'{col}_diff_{diff_val}'] = df_filtered[col].diff(diff_val)

    # Временные признаки
    # df_filtered['hour_of_day'] = df_filtered['timestamp'].dt.hour
    # df_filtered['day_of_week'] = df_filtered['timestamp'].dt.dayofweek
    # df_filtered['is_weekend'] = df_filtered['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Скользящие статистики
    window_size = 10  # Размер окна для скользящей статистики
    df_filtered['rolling_mean_speed'] = df_filtered['gps_speed_kmh'].rolling(window=window_size).mean()
    df_filtered['rolling_std_speed'] = df_filtered['gps_speed_kmh'].rolling(window=window_size).std()

    # Агрегированные признаки
    df_filtered['avg_speed_last_5min'] = df_filtered['gps_speed_kmh'].rolling(window=5).mean()
    df_filtered['max_heading_last_hour'] = df_filtered['heading_angle_deg'].rolling(window=60).max()

    # Признаки отношения
    df_filtered['speed_direction_ratio'] = df_filtered['gps_speed_kmh'] / (df_filtered['heading_angle_deg'] + 1e-5)
    df_filtered['platform_inclination_diff'] = df_filtered['platform_inclination_x_deg'] - df_filtered[
        'platform_inclination_y_deg']

    # Дополнительные взаимодействия
    df_filtered['speed_inclination_interaction'] = df_filtered['gps_speed_kmh'] * df_filtered[
        'platform_inclination_x_deg']
    df_filtered['speed_heading_interaction'] = df_filtered['gps_speed_kmh'] * df_filtered['heading_angle_deg']

    df_filtered = df_filtered.dropna()  # Удаляем оставшиеся строки с NaN

    return df_filtered

def calculate_anomalies_iqr(df_filtered):
    """
    Рассчитываем аномалии с использованием IQR (Interquartile Range).
    """
    Q1 = df_filtered['gps_speed_kmh'].quantile(0.25)
    Q3 = df_filtered['gps_speed_kmh'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    anomalies_iqr = df_filtered[(df_filtered['gps_speed_kmh'] < lower_bound) |
                                (df_filtered['gps_speed_kmh'] > upper_bound)]

    return anomalies_iqr

def calculate_anomalies_zscore(df_filtered):
    """
    Рассчитываем аномалии с использованием Z-оценок.
    """
    df_filtered['gps_speed_zscore'] = zscore(df_filtered['gps_speed_kmh'])
    anomalies_zscore = df_filtered[df_filtered['gps_speed_zscore'].abs() > 3]

    return anomalies_zscore

# Функция для обучения и анализа важности признаков
def analyze_feature_importance(df, model_name):
    """
    Функция для анализа важности признаков с использованием SHAP и модели Isolation Forest.
    """
    # Загружаем модель
    model = joblib.load(f'models/{model_name}')
    print(f"\nМодель загружена из {model_name}")
    # Удаляем ненужные признаки
    df = df.drop(columns=[col for col in ['timestamp', 'anomaly', 'gps_speed_zscore']
                          if col in df.columns], errors='ignore')
    print(df)

    # Получаем SHAP значения для модели и используем kmeans для ускорения
    background_data = shap.kmeans(df, 100)  # Снижаем размер данных с помощью kmeans
    explainer = shap.KernelExplainer(model.predict, background_data)  # KernelExplainer для моделей без прямого объяснения
    shap_values = explainer.shap_values(df)  # Вычисляем SHAP значения для всех признаков
    # Визуализация важности признаков
    shap.summary_plot(shap_values, df)
    # shap.dependence_plot('feature_name', shap_values, df)
    shap.bar_plot(shap_values)



