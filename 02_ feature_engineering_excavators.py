from models_fior_anomalies import detect_anomalies_isolation_forest, detect_anomalies_iqr, detect_anomalies_zscore
from utils import load_and_preprocess_data, standardize_data, visualize_anomalies, \
    calculate_anomalies_iqr, calculate_anomalies_zscore, add_additional_features, setup_pandas_options, \
    analyze_feature_importance

setup_pandas_options()



# Путь к файлу
file_path = 'dataset/ml_datasets/merged_resampled_telemetry_data.parquet'

# 1. Загрузка и подготовка данных
df_filtered = load_and_preprocess_data(file_path)
print(f'\ndf_filtered:\n{df_filtered}')

# 2& Feature Engineering
df_filtered = add_additional_features(df_filtered)
print(f'\ndf_filtered:\n{df_filtered}')

# 3. Стандартизация данных
df_filtered = standardize_data(df_filtered)

# 4. Обучение модели
contamination = 0.002
max_samples = 0.2
# Выбор признаков для обучения
# numeric_columns = ['gps_speed_kmh', 'heading_angle_deg', 'platform_inclination_x_deg',
#                    'platform_inclination_y_deg', 'boom_inclination_x_deg', 'arm_inclination_deg']
# Выбираем только числовые признаки
# 1. Выбираем только числовые признаки
# Выбираем только числовые признаки
numeric_columns = df_filtered.select_dtypes(include='number').columns
# Сохраняем столбец timestamp и добавляем его обратно к числовым признакам
df_numeric = df_filtered[numeric_columns].copy()  # Берём копию числовых данных
df_numeric['timestamp'] = df_filtered['timestamp']  # Добавляем столбец timestamp обратно
print(f'\ndf_numeric:\n{df_numeric}')


# Теперь передаем df_features (включая timestamp) в модель
df_filtered = detect_anomalies_isolation_forest(
    df_numeric,  # Передаем весь DataFrame с числовыми признаками и timestamp
    contamination=contamination,
    max_samples=max_samples,
    model_name='IsolationForest.pkl'
)

# Или используйте IQR для обнаружения аномалий:
# df_filtered = detect_anomalies_iqr(df_filtered, iqr_multiplier=3.5)

# Или используйте Z-оценку для обнаружения аномалий:
# df_filtered = detect_anomalies_zscore(df_filtered, zscore_threshold=5)


# 5. Визуализация аномалий
feature_names = {
    'gps_speed_kmh': 'Скорость экскаватора (км/ч)',
    'heading_angle_deg': 'Угол направления экскаватора (градусы)',
    'platform_inclination_x_deg': 'Угол наклона платформы по оси X (градусы)',
    'platform_inclination_y_deg': 'Угол наклона платформы по оси Y (градусы)',
    'boom_inclination_x_deg': 'Угол наклона стрелы экскаватора по оси X (градусы)',
    'arm_inclination_deg': 'Угол наклона рабочего оборудования (градусы)',
    # 'hour_of_day': 'Час суток',
    # 'day_of_week': 'День недели',
    # 'is_weekend': 'Выходной день (1 - да, 0 - нет)',
    # 'month': 'Месяц'
}
visualize_anomalies(df_filtered.iloc[::2],  # каждая 2-я точка
                    feature_names=feature_names)



exit()
# 8. Вызов функции для анализа важности признаков
analyze_feature_importance(
    df_numeric.loc[20_000:25_000],
    model_name='IsolationForest.pkl'
)
