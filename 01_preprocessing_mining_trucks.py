import os

import pandas as pd
from pathlib import Path

from auxiliary.utils_minigng_trucks import setup_pandas_options, save_parquet, optimize_dtypes

setup_pandas_options()

def data_cleaner(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()  # Убираем лишние пробелы или символы новой строки
    df.replace(-1000000, pd.NA, inplace=True)  # Заменим -1000000 на NaN
    df = df.dropna(how='all', axis=0)  # Удаляем строки с пропущенными значениями
    df = df.dropna(how='all', axis=1)  # Удаляем столбцы с пропущенными значениями
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Убираем столбцы с Unnamed

    # Удалим признаки, где нет разнообразия..
    nunique = df.nunique()
    zero_variance_cols = nunique[nunique <= 1].index.tolist()
    print("Бесполезные признаки:", zero_variance_cols)
    df = df.drop(columns=zero_variance_cols)

    return df


source_root = 'dataset/_by_Hack'

# Загрузка данных с правильными заголовками
print(f'\nidles:')
idles = pd.read_csv(f'{source_root}/reference/idles.csv')
idles = data_cleaner(idles)
print(idles)

idles['GMTBEGINTIME'] = pd.to_datetime(idles['GMTBEGINTIME'], errors='coerce')
idles = idles.dropna(subset=['GMTBEGINTIME'])

idles = idles.rename(columns={
    'OBJECTID': 'asset_id',
    'GMTBEGINTIME': 'event_time',
    'IDLETYPENAME': 'event_name',
    'OBJECTNAME': 'mdm_object_name'
})
print(pd.unique(idles['event_name']))

event_to_case = {
    # 🔹 Кейс 1: Топливная система
    'Топливная система': 'fuel_degradation',
    # 'Замена форсунки': 'fuel_degradation',
    # 'Ремонт ТНВД': 'fuel_degradation',

    # # 🔹 Кейс 2: Наддув / турбокомпрессор
    # 'Гидравлическая система': 'turbo_degradation',
    # 'Пневмосистема': 'turbo_degradation',
    # 'Снижение давления наддува': 'turbo_degradation',

    # # 🔹 Кейс 3: Охлаждение
    # 'Система охлаждения': 'cooling_failure',
    # 'Прокачка ПГП': 'cooling_failure',
    # 'Неисправность насоса': 'cooling_failure',

    # # 🔹 Кейс 4: Масло
    # 'Перегрев КГШ': 'oil_pressure_drop',
    # 'КГШ - замена': 'oil_pressure_drop',
    # 'Низкое давление масла': 'oil_pressure_drop',

    # # 🔹 Кейс 5: Электротяга
    # 'Ремонт АСД': 'electric_failure',
    # 'Неисправность АСД': 'electric_failure',
    # 'Отказ инвертора': 'electric_failure',

    # # 🔹 Кейс 6: Трансмиссия
    # 'Ходовая система': 'gearbox_overheat',
    # 'Ремонт редуктора': 'gearbox_overheat',

    # # 🔹 Кейс 7: Шины
    # 'КГШ - подкачка': 'tire_burst_risk',
    # 'Утечка воздуха в шине': 'tire_burst_risk',

    # # 🔹 Кейс 8: Воздушный тракт
    # 'ТО, КР, ППР': 'air_filter_clogged',
    # 'Замена воздушного фильтра': 'air_filter_clogged',

    # # 🔹 Кейс 9: Режимы вождения
    # 'Ожидание погрузки': 'aggressive_driving',
    # 'Ожидание заправки - очередь': 'aggressive_driving',
    # 'Ожидание разгрузки': 'aggressive_driving'

}

# Применяем
idles['case'] = idles['event_name'].map(event_to_case)
idles_events = idles.dropna(subset=['case'])[['asset_id', 'mdm_object_name', 'event_time', 'case']].drop_duplicates()

idles_df_optimized = optimize_dtypes(idles_events)
save_parquet(idles_df_optimized, Path('dataset/ml_datasets/_by_Hack/idles.parquet'))
print(idles_df_optimized)


# Загрузка данных масляной лаборатории
print(f'\noil_lab_df:')
oil = pd.read_csv(f'{source_root}/oil/oil.csv')
oil = data_cleaner(oil)

oil['TakenDate'] = pd.to_datetime(oil['TakenDate'], errors='coerce')
oil = oil.dropna(subset=['TakenDate'])

oil = oil.rename(columns={
    'UnitNumberField': 'mdm_object_name',
    'TakenDate': 'event_time'
})

if 'ComponentTypeField' in oil.columns:
    oil = oil[oil['ComponentTypeField'].str.contains('двигатель|engine', case=False, na=False)]

oil['is_oil_issue'] = oil['Condition'].isin(['Abnormal', 'Severe'])
oil_issues = oil[oil['is_oil_issue']].copy()
oil_issues['case'] = 'oil_pressure_drop'
oil_issues = oil_issues[['mdm_object_name', 'event_time', 'case']].drop_duplicates().reset_index(drop=True)

manual_mapping = {
    1374: 1383,
    1381: 1581,
    1349: 1381,
    1497: 2186,
    1385: 1384,
    1395: 1661,
}
oil_issues['asset_id'] = oil_issues['mdm_object_name'].map(manual_mapping)

oil_lab_df_optimized = optimize_dtypes(oil_issues)
save_parquet(oil_lab_df_optimized, Path('dataset/ml_datasets/_by_Hack/oil_lab_df.parquet'))
print(oil_lab_df_optimized)



# Загрузка телеметрии
files = os.listdir(f'{source_root}/telemetry')
print(f'\ntelemetry_df:')
print(files)
data_frames = []
for file in files:
    if 'telemetry' in file and file.endswith('.csv'):
        print(f'Loading {file} for combine..')
        df = pd.read_csv(os.path.join(f'{source_root}/telemetry', file))
        df = optimize_dtypes(df)

        # Убираем пробелы в названиях
        df.columns = df.columns.str.strip()

        # Переименовываем
        df = df.rename(columns={
            'mdm_object_id': 'asset_id',
            'create_dt': 'timestamp',
            'load_engine': 'engine_load',
            'inst_fuel': 'fuel_rate',
            'pres_rail_injector_nn': 'rail_pressure',
            'pres_des_rail_injector_nn': 'rail_pressure_target',
            'pres_turbo': 'boost_pressure',
            'engine_coolant_temp': 'coolant_temp',
            'engine_oil_pressure': 'oil_pressure',
            'temp_oil_engine_nn': 'oil_temp',
            'tweather_nn': 'ambient_temp',
            'speed_gps': 'vehicle_speed',
            'spn': 'fault_code'
        })

        # Удаляем ненужные колонки
        df = df.drop(columns=[
            'Unnamed: 0',
            'engine_speed_control',
            'finjection',
            'purgepressure_nn',
            'meta_object_name',
            'meta_model_id',
            'ambient_temp',
            'mdm_object_uuid',
            'mdm_model_id',
            'oil_temp',
            'boost_pressure',
            'meta_model_name',
            'sutep_error',
            'fault_code',
            'distance_nn',
            'fault_code',
            'engine_rpm',
            # 'mdm_object_name',
            'mdm_model_name',
            'coefficient_correction',
            'error_belaz_11',
            'error_belaz_12',
            'fault_code',
            'fmi',
            'spn_weichai',
            'accelerator_pedal_position',
            'transmission_oil_temperature',
            'coefficient_correction',
            'total_vehicle_hours',
            'nominal_torque',
            'fuel_level_can',
            'turbo_pressure',
            'crankcase_purge_pressure',
            'engine_oil_level',
            'coolant_temp',
            'oil_pressure',
            'dfm_in_sum'
        ], errors='ignore')

        # Приводим временные метки
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])

        # Убираем дубли
        df = df.drop_duplicates(subset=['asset_id', 'timestamp'])

        data_frames.append(df)

print('Объединение всех DataFrame в один..')
telemetry_df = pd.concat(data_frames, ignore_index=True)





print('Удалим признаки, где нет разнообразия..')
telemetry_df = data_cleaner(telemetry_df)
telemetry_df_optimized = optimize_dtypes(telemetry_df)
print('save_parquet..')
save_parquet(telemetry_df_optimized, Path('dataset/ml_datasets/_by_Hack/telemetry_df.parquet'))
print(telemetry_df_optimized)
