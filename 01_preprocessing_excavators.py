
import pandas as pd

from pathlib import Path

from utils import setup_pandas_options, save_parquet

setup_pandas_options()


# Загрузка данных с правильными заголовками
dvs_repair_df = pd.read_excel('dataset/_by_Dmitry/reference/ДВС.xlsx', header=0)
dvs_repair_df.columns = dvs_repair_df.columns.str.strip()  # Убираем лишние пробелы или символы новой строки
dvs_repair_df = dvs_repair_df.dropna(how='all', axis=0)  # Удаляем строки с пропущенными значениями
dvs_repair_df = dvs_repair_df.dropna(how='all', axis=1)  # Удаляем столбцы с пропущенными значениями
dvs_repair_df = dvs_repair_df.loc[:, ~dvs_repair_df.columns.str.contains('^Unnamed')]  # Убираем столбцы с Unnamed
dvs_repair_df = dvs_repair_df[['Марка ДВС', 'Ремонты за 2023-24 год', 'Средняя наработка на ремонт за 2023г']].dropna()  # Выбираем нужные столбцы
print(f'\ndvs_repair_df:\n{dvs_repair_df}')

# Шаг 2: Обработка временных меток в барже и рейсах
barge_trips_df = pd.read_excel('dataset/_by_Dmitry/reference/РейсыБарж.xlsx')
barge_trips_df['BEGIN'] = pd.to_datetime(barge_trips_df['BEGIN'])
barge_trips_df['END'] = pd.to_datetime(barge_trips_df['END'].apply(lambda x: f"2000-01-01 {x}" if isinstance(x, str) else x), format='%Y-%m-%d %H:%M:%S', errors='coerce')
print(f'\nbarge_trips_df:\n{barge_trips_df}')

# Шаг 3: Загрузка телеметрии и слияние с данными о рейсах
telemetry_df = pd.read_csv('dataset/_by_Dmitry/telemetry/telemetry_excavators.csv')
truck_trips_df = pd.read_csv('dataset/_by_Dmitry/reference/truck_trips.csv')
truck_trips_df['start_time'] = pd.to_datetime(truck_trips_df['start_time'])
truck_trips_df['end_time'] = pd.to_datetime(truck_trips_df['end_time'])
print(f'\ntruck_trips_df:\n{truck_trips_df}')

# Шаг 4: Слияние телеметрии с данными о рейсах
merged_df = pd.merge(telemetry_df, truck_trips_df, left_on='mdm_object_uuid', right_on='object_uuid', how='left')
# Ресемплинг на 5 секунд (удалены ссылки на 'temperature')
merged_df['create_dt'] = pd.to_datetime(merged_df['create_dt'])
num_cols = [
    'speed_gps', 'direction', 'inclinom_platx', 'inclinom_platy', 'inclinom_boomx', 'inclinom_arm'
]  # Убираем 'temperature' из списка столбцов

df_resampled = (
    merged_df
    .groupby('mdm_object_uuid', observed=False)
    .resample('5s', on='create_dt')[num_cols]  # Используем '5s' вместо '5S' для корректности
    .mean()
    .reset_index()
)

# Сортировка для корректного diff
df_resampled = df_resampled.sort_values(['mdm_object_uuid', 'create_dt'])

# Обработка пропусков (без 'temperature')
df_resampled[['speed_gps', 'direction', 'inclinom_platx', 'inclinom_platy', 'inclinom_boomx', 'inclinom_arm']] = \
    (df_resampled[['speed_gps', 'direction', 'inclinom_platx', 'inclinom_platy', 'inclinom_boomx', 'inclinom_arm']]
     .interpolate())

# Преобразуем 'create_dt' в df_resampled, убирая временную зону
df_resampled['create_dt'] = df_resampled['create_dt'].dt.tz_localize(None)

# Шаг 5: Обработка данных масляной лаборатории
oil_lab_df = pd.read_excel('dataset/_by_Dmitry/oil/Масляная лаборатория 1.xlsx')
print(f'\noil_lab_df:\n{oil_lab_df}')

# Шаг 6: Слияние данных, теперь с одинаковыми типами данных для времени
df_resampled = pd.merge(df_resampled, oil_lab_df, left_on=['mdm_object_uuid', 'create_dt'], right_on=['CustomerId', 'ReportedDate'], how='left')

# Шаг 7: Слияние с данными ремонтов ДВС
# используем 'mdm_object_uuid' для слияния
df_resampled = pd.merge(df_resampled, dvs_repair_df, left_on='mdm_object_uuid', right_on='Марка ДВС', how='left')
df_resampled['create_dt'] = pd.to_datetime(df_resampled['create_dt'])

# Сохранение итогового датасета в Parquet
print(f'\nmerged_resampled_telemetry_data:\n{df_resampled}')
save_parquet(df_resampled, Path('dataset/ml_datasets/merged_resampled_telemetry_data.parquet'))
print('Данные успешно подготовлены, ресемплированы, объединены и сохранены в Parquet.')






