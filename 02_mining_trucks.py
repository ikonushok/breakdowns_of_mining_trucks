import os
import warnings

import pandas as pd

from auxiliary.explanation_mining_trucks import explain_shap_for_event
from auxiliary.models_mining_trucks import train_model_with_anomaly_detection
from auxiliary.utils_minigng_trucks import (setup_pandas_options, optimize_dtypes, calculate_event_level_precision,
                                            calculate_early_warning, prepare_telemetry, aggregate_fuel_features,
                                            aggregate_window_data, merge_window_features, prepare_events,
                                            merge_events_with_windows, process_target, prepare_X_y)

warnings.filterwarnings("ignore",
                        message="LightGBM binary classifier with "
                                "TreeExplainer shap values output has changed to a list of ndarray")
setup_pandas_options()




source_root = 'dataset/ml_datasets/_by_Hack'
files = os.listdir(source_root)
print(files)

# 1. Загрузим данные из файлов
equipment = pd.read_csv('dataset/_by_Hack/reference/equipment.csv')
print(f'\nequipment:\n{equipment}')

# Загрузка данных по частям и оптимизация для idles_df
idles_file = os.path.join(source_root, 'idles.parquet')
idles_df = optimize_dtypes(pd.read_parquet(idles_file))
print(f'\nidles_df:\n{idles_df}')

# Загрузка и оптимизация для oil_lab_df
oil_lab_file = os.path.join(source_root, 'oil_lab_df.parquet')
oil_lab_df = optimize_dtypes(pd.read_parquet(oil_lab_file))
print(f'\noil_lab_df:\n{oil_lab_df}')

# Загрузка и оптимизация для telemetry_df
telemetry_file = os.path.join(source_root, 'telemetry_df.parquet')
telemetry_df = optimize_dtypes(pd.read_parquet(telemetry_file))
print(f'\ntelemetry_df:\n{telemetry_df}')


# 2. Подготовка телеметрии
tele = prepare_telemetry(telemetry_df)
base_agg = aggregate_window_data(tele, window="720min")  # Агрегация по окнам (1d = 1440min)
fuel_agg = aggregate_fuel_features(tele)  # Агрегация топливных признаков
df_win = merge_window_features(base_agg, fuel_agg)  # Объединение оконных признаков
events = prepare_events(idles_df)  # Подготовка событий
df_win = merge_events_with_windows(df_win, events)  # Слияние событий с окнами

# Таргет (prediction_window) 7..30 дней, фильтрация (empty_window) 0..7 дней
empty_window = 7
target_window = 30
df_win = process_target(df_win,
                        empty_window=empty_window,
                        target_window=target_window)
# Подготовка X и y
X, y = prepare_X_y(df_win)

# Вывод
print(f'\ndf_win:\n{df_win}')
print(f'\ndescribe df_win["days_to_event"]:\n{df_win["days_to_event"].describe()}')
print(f'\ny value_counts:\n{y.value_counts(dropna=False)}')
print(f'duplicated columns:\t{X.columns[X.columns.duplicated()].unique()}\n')


# 3. Model
threshold = 0.6
model, proba = train_model_with_anomaly_detection(X, y,
                                                  anomaly_contamination=0.02,
                                                  threshold=threshold)

# 4. Собираем таблицу с предсказаниями на уровне окон
pred = (df_win.reset_index()[
            ["asset_id", "mdm_object_name", "timestamp", "event_dt", "days_to_event", "target"]].copy())
pred["proba"] = proba  # proba должен быть в том же порядке, что и df_win/X


# 5. Ранность предупреждения (first alert lead time)
report = calculate_early_warning(df_win, proba, threshold)
# Пересчет precision на уровне событий
precision_event_level = calculate_event_level_precision(pred, threshold=threshold)


# Если хочешь одну “маркетинговую” цифру:
mean_lead = report.loc[report["detected"], "lead_days"].mean()
median_lead = report.loc[report["detected"], "lead_days"].median()
print(f"mean lead time: {mean_lead:.2f} days")
print(f"median lead time: {median_lead:.2f} days")

exit()
# 6. SHAP и Локальная объяснимость для одного события
explain_shap_for_event(X, model, report, pred, threshold=0.6, sample_size=5000, random_state=42, topn=8)






"""
7. Следующий шаг (могу сделать)
Если хочешь, дальше можем:
Добавить FFT пульсаций rail pressure
Сделать SHAP-интерпретацию
Развести горизонты 7 / 14 / 30 как multi-label
Подготовить структуру ноутбука “как на Kaggle”
Сжать это в инференс-скрипт для продакшена
"""


