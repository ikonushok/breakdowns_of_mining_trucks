
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
    df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp']).dt.round('500ms')

    # Добавляем дополнительные временные признаки
    # df_filtered['hour'] = df_filtered['timestamp'].dt.hour
    # df_filtered['dayofweek'] = df_filtered['timestamp'].dt.dayofweek
    # df_filtered['month'] = df_filtered['timestamp'].dt.month
    # df_filtered['year'] = df_filtered['timestamp'].dt.year

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

# =========================
# Episode anomaly visualization (Plotly)
# =========================
def _find_intervals_from_binary(ts: pd.Series, binary: pd.Series):
    """
    Convert a 0/1 series into a list of (start_ts, end_ts) intervals where binary==1.
    """
    ts = pd.to_datetime(ts)
    b = binary.fillna(0).astype(int).values

    intervals = []
    start = None
    for i, v in enumerate(b):
        if v == 1 and start is None:
            start = ts.iloc[i]
        if v == 0 and start is not None:
            end = ts.iloc[i - 1]
            intervals.append((start, end))
            start = None
    if start is not None:
        intervals.append((start, ts.iloc[len(b) - 1]))
    return intervals

def visualize_anomalies(
    df_filtered,
    feature_names,
    *,
    episode_label_col: str = "anomaly_episode",
    point_label_col: str = "anomaly_point",
    show_point_anomalies: bool = False,
    output_dir: str | Path = "outputs/anomalies",
    save_html: bool = True,
    open_in_browser: bool = True,
):

    d = df_filtered.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.sort_values("timestamp")

    excavator_id = "unknown_excavator"
    if "excavator_id" in d.columns and len(d) > 0:
        excavator_id = str(d["excavator_id"].iloc[0])

    # создаём папку
    out_dir = Path(output_dir) / excavator_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # полезно для имени файла (если есть)
    excavator_id = None
    if "excavator_id" in d.columns and len(d) > 0:
        excavator_id = str(d["excavator_id"].iloc[0])

    for feature, feature_title in feature_names.items():
        fig = go.Figure()

        # линия
        fig.add_trace(go.Scatter(
            x=d["timestamp"],
            y=d[feature],
            mode="lines",
            name=feature,
            line=dict(color="blue"),
        ))

        # episode intervals
        if episode_label_col in d.columns:
            intervals = _find_intervals_from_binary(d["timestamp"], d[episode_label_col])
            for start, end in intervals:
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor="red",   # ты просил просто red
                    line_width=0,
                    layer="below"
                )

        # point anomalies (optional)
        if show_point_anomalies and point_label_col in d.columns:
            pts = d[d[point_label_col].fillna(0).astype(int) == 1]
            if len(pts) > 0:
                fig.add_trace(go.Scatter(
                    x=pts["timestamp"],
                    y=pts[feature],
                    mode="markers",
                    name="Аномалии",
                    marker=dict(color="red", symbol="x", size=10),
                ))

        fig.update_layout(
            title=f"{feature_title} с аномалиями",
            xaxis_title="Время",
            yaxis_title=feature_title,
            showlegend=True,
        )

        # save html
        if save_html:
            safe_feat = feature.replace("/", "_")
            safe_id = (excavator_id or "unknown_excavator").replace("/", "_")
            html_path = out_dir / f"{safe_feat}.html"
            fig.write_html(str(html_path), include_plotlyjs="cdn")

        # show (как раньше)
        if open_in_browser:
            fig.show(renderer="browser")

def _robust_center_scale(x: pd.Series):
    """median + MAD (scaled). Returns (median, mad_scaled)."""
    x = pd.to_numeric(x, errors="coerce")
    med = float(np.nanmedian(x.values))
    mad = float(np.nanmedian(np.abs(x.values - med)))
    mad_scaled = 1.4826 * mad  # ~= std for normal
    if not np.isfinite(mad_scaled) or mad_scaled < 1e-9:
        mad_scaled = 1e-9
    return med, mad_scaled

def _episode_id_from_binary(df: pd.DataFrame, label_col: str = "anomaly_episode"):
    """Add episode_id where label==1. 0 means 'not in episode'."""
    b = df[label_col].fillna(0).astype(int).values
    ep = np.zeros(len(b), dtype=int)
    cur = 0
    in_ep = False
    for i, v in enumerate(b):
        if v == 1 and not in_ep:
            cur += 1
            in_ep = True
        if v == 0 and in_ep:
            in_ep = False
        ep[i] = cur if v == 1 else 0
    return ep

def summarize_anomaly_episodes(
    df_scored: pd.DataFrame,
    feature_names: dict,
    *,
    excavator_id: str | None = None,
    episode_label_col: str = "anomaly_episode",
    regime_col: str = "regime",
    ts_col: str = "timestamp",
    baseline_window: str = "7D",
    top_k: int = 3,
    min_episode_points: int = 5,
    examples_per_episode: int = 2,
    output_dir: str | Path = "outputs/anomalies",
    save_txt: bool = True,
    txt_filename: str | None = None,
):

    excavator_safe = str(excavator_id) if excavator_id else "unknown_excavator"
    out_dir = Path(output_dir) / excavator_safe
    out_dir.mkdir(parents=True, exist_ok=True)

    d = df_scored.copy()
    d[ts_col] = pd.to_datetime(d[ts_col])
    d = d.sort_values(ts_col)

    if excavator_id is not None:
        d = d[d["excavator_id"] == excavator_id].copy()

    if episode_label_col not in d.columns:
        raise ValueError(f"Column '{episode_label_col}' not found in dataframe")

    d["episode_id"] = _episode_id_from_binary(d, episode_label_col)

    ep_ids = [eid for eid in sorted(d["episode_id"].unique()) if eid != 0]
    lines: list[str] = []

    if not ep_ids:
        msg = "Нет эпизодов аномалий (anomaly_episode==1)."
        print(msg)
        lines.append(msg)
        if save_txt:
            name = txt_filename or f"{(str(excavator_id) if excavator_id else 'unknown_excavator')}__anomaly_report.txt"
            (out_dir / name).write_text("\n".join(lines), encoding="utf-8")
        return "\n".join(lines)

    for eid in ep_ids:
        ep = d[d["episode_id"] == eid].copy()
        if len(ep) < min_episode_points:
            continue

        start = ep[ts_col].iloc[0]
        end = ep[ts_col].iloc[-1]
        dur = end - start

        regime = None
        if regime_col in ep.columns and not ep[regime_col].mode().empty:
            regime = ep[regime_col].mode().iloc[0]

        # baseline: до начала эпизода, тот же режим, окно baseline_window
        base = d[d[ts_col] < start].copy()
        if regime is not None and regime_col in base.columns:
            base = base[base[regime_col] == regime]
        if baseline_window:
            base = base[base[ts_col] >= (start - pd.Timedelta(baseline_window))]

        # fallback
        if base.empty:
            base = d[d[ts_col] < start].copy()
            if regime is not None and regime_col in base.columns:
                base = base[base[regime_col] == regime]
        if base.empty:
            base = d.copy()
            if regime is not None and regime_col in base.columns:
                base = base[base[regime_col] == regime]

        # contributions
        contrib = []
        for feat in feature_names.keys():
            if feat not in ep.columns or feat not in base.columns:
                continue
            med, mad = _robust_center_scale(base[feat])
            z = (pd.to_numeric(ep[feat], errors="coerce") - med) / mad
            score = float(np.nanmean(np.abs(z.values)))
            contrib.append((feat, score, med, mad))

        contrib.sort(key=lambda x: x[1], reverse=True)
        top = contrib[:top_k]

        lines.append("=" * 80)
        lines.append(f"Эпизод {eid}: {start} — {end}  (длительность: {dur})")
        if regime is not None:
            lines.append(f"Режим: {regime}")
        lines.append(f"Точек в эпизоде: {len(ep)}")
        lines.append("")
        lines.append("Топ-признаки (robust отклонение относительно baseline в этом режиме):")

        for feat, _, med, mad in top:
            ru = feature_names.get(feat, feat)
            ep_med = float(np.nanmedian(pd.to_numeric(ep[feat], errors='coerce').values))
            delta = ep_med - med
            z_med = delta / mad
            direction = "вверх" if delta > 0 else "вниз"
            lines.append(f"- {ru}")
            lines.append(f"  В эпизоде (медиана): {ep_med:.3g}")
            lines.append(f"  В норме (baseline медиана): {med:.3g}")
            lines.append(f"  Отклонение: {delta:+.3g} ({direction}), ~{z_med:+.2f}σ")

        if top:
            zsum = np.zeros(len(ep), dtype=float)
            for feat, _, med, mad in top:
                vals = pd.to_numeric(ep[feat], errors="coerce").values
                z = (vals - med) / mad
                zsum += np.abs(np.nan_to_num(z, nan=0.0))

            idx = np.argsort(-zsum)[:examples_per_episode]
            lines.append("")
            lines.append("Примеры точек внутри эпизода:")

            for j, i in enumerate(idx, 1):
                row = ep.iloc[int(i)]
                t = row[ts_col]
                lines.append("")
                lines.append(f"  Аномалия {j}: время = {t}")
                for feat, _, med, mad in top:
                    ru = feature_names.get(feat, feat)
                    x = float(pd.to_numeric(row[feat], errors="coerce"))
                    delta = x - med
                    z = delta / mad
                    lines.append(f"   - {ru}: {x:.3g}")
                    lines.append(f"     В норме (baseline): {med:.3g}")
                    lines.append(f"     Отклонение: {delta:+.3g} ({z:+.2f}σ)")

    lines.append("=" * 80)

    # print to console
    text = "\n".join(lines)
    print(text)

    # save to txt
    if save_txt:
        safe_id = (str(excavator_id) if excavator_id else "unknown_excavator").replace("/", "_")
        name = txt_filename or "anomaly_report.txt"
        (out_dir / name).write_text(text, encoding="utf-8")

    return text


