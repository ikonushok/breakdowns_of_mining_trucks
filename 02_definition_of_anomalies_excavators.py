
"""
End-to-end script: excavator telemetry -> regime-aware episode anomaly detection -> visualization.

What it does (per your plan):
1) Load + preprocess raw parquet
2) Data quality flags/cleaning (-1e6, invalid heading, etc.)
3) Regime labels
4) Sliding-window features (30/60/120s)
5) Train anomaly detectors on healthy proxy (first N days) per excavator
6) Ensemble score: IsolationForest + robust stats
7) Quantile thresholds (post-hoc), per regime
8) Episode anomalies by aggregating scores over 10 minutes
9) Plotly visualization with shaded episode intervals (operator-friendly)
"""

from auxiliary.anomaly_pipeline import PipelineConfig, run_anomaly_pipeline
from auxiliary.utils_excavators import (
    load_and_preprocess_data,
    setup_pandas_options,
    visualize_anomalies,
    summarize_anomaly_episodes,
)



setup_pandas_options()

FILE_PATH = "dataset/ml_datasets/merged_resampled_telemetry_data.parquet"

# 1) Load
df_filtered = load_and_preprocess_data(FILE_PATH)
print(f'\ndf_filtered:\n{df_filtered}')
df_describe = df_filtered.describe().T
df_describe.loc['timestamp'] = df_describe.loc['timestamp'].map(
    lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if hasattr(x, 'strftime') else x
)
print(f'\ndf_filtered describe:\n{df_describe}')

# 2) Run anomaly pipeline
cfg = PipelineConfig(
    win_seconds=(30, 60, 120),
    agg_minutes=10,
    score_quantile=0.995,
    n_estimators=300,
    max_samples=0.3,
    contamination=0.01,
    healthy_first_days=7,
)

df_scored = run_anomaly_pipeline(df_filtered, cfg)
# print("Scored dataframe head:")
# print(df_scored[[
#     "excavator_id", "timestamp", "regime", "flag_invalid",
#     "anomaly_score", "anomaly_point",
#     "anomaly_score_episode", "anomaly_episode"
# ]].head())

# 3) Visualization (episode anomalies)
feature_names = {
    "gps_speed_kmh": "Скорость экскаватора (км/ч)",
    "heading_angle_deg": "Угол направления экскаватора (градусы)",
    "platform_inclination_x_deg": "Угол наклона платформы по оси X",
    "platform_inclination_y_deg": "Угол наклона платформы по оси Y",
    "boom_inclination_x_deg": "Угол наклона стрелы по оси X",
    "arm_inclination_deg": "Угол наклона рукояти",
}

# Choose one excavator to inspect (first one)
example_excavator_id = df_scored["excavator_id"].iloc[0]

# Печать объяснений по эпизодам (в консоль)
summarize_anomaly_episodes(
    df_scored,
    feature_names,
    excavator_id=example_excavator_id,
    baseline_window="7D",
    top_k=3,
    examples_per_episode=2,
)

# Графики (episode-интервалы)
df_vis = df_scored[df_scored["excavator_id"] == example_excavator_id].iloc[::2].copy()

visualize_anomalies(
    df_vis,
    feature_names,
    show_point_anomalies=False
)
