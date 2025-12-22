
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


SENSOR_COLS = [
    "gps_speed_kmh",
    "heading_angle_deg",
    "platform_inclination_x_deg",
    "platform_inclination_y_deg",
    "boom_inclination_x_deg",
    "arm_inclination_deg",
]


@dataclass
class PipelineConfig:
    # Rolling windows (seconds)
    win_seconds: Tuple[int, ...] = (30, 60, 120)
    # Episode aggregation window (minutes)
    agg_minutes: int = 10
    # Threshold quantile (computed on healthy train proxy)
    score_quantile: float = 0.995

    # IsolationForest
    n_estimators: int = 300
    max_samples: float = 0.3
    contamination: float = 0.01  # technical parameter only
    random_state: int = 42

    # Healthy proxy selection
    healthy_first_days: int = 7

    # Regime rules
    speed_idle_max: float = 0.3
    speed_move_min: float = 1.0

    # Numerical stability
    mad_eps: float = 1e-9


# -----------------------
# Data quality handling
# -----------------------
def flag_and_clean_technical_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags technical/invalid values and replaces known placeholders with NaN.

    IMPORTANT:
    - This is *not* anomaly detection. These are data-quality issues.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["flag_invalid"] = 0

    # -1e6 placeholders (as per your describe)
    for c in ["platform_inclination_x_deg", "platform_inclination_y_deg", "boom_inclination_x_deg", "arm_inclination_deg"]:
        if c in df.columns:
            m = df[c].astype(float).eq(-1_000_000.0)
            df.loc[m, "flag_invalid"] = 1
            df.loc[m, c] = np.nan

    # Heading in [0..360]
    if "heading_angle_deg" in df.columns:
        m = ~df["heading_angle_deg"].between(0, 360)
        df.loc[m, "flag_invalid"] = 1
        df.loc[m, "heading_angle_deg"] = np.nan

    # Speed >= 0
    if "gps_speed_kmh" in df.columns:
        m = df["gps_speed_kmh"] < 0
        df.loc[m, "flag_invalid"] = 1
        df.loc[m, "gps_speed_kmh"] = np.nan

    # Sort + de-dup
    df = df.sort_values(["excavator_id", "timestamp"]).drop_duplicates(["excavator_id", "timestamp"])
    return df

def impute_missing_per_excavator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conservative imputation for short-step telemetry:
    forward-fill then backward-fill within each excavator.
    """
    df = df.copy()
    df[SENSOR_COLS] = (
        df.groupby("excavator_id", group_keys=False, observed=False)[SENSOR_COLS]
        .apply(lambda g: g.ffill().bfill())
    )
    return df

# -----------------------
# Regimes
# -----------------------
def add_regime_labels(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """
    Coarse regime labeling. Refine later if needed.
    """
    df = df.copy()
    speed = df["gps_speed_kmh"].astype(float)

    df["regime_motion"] = np.select(
        [speed <= cfg.speed_idle_max, speed >= cfg.speed_move_min],
        ["idle", "move"],
        default="maneuver",
    )

    boom = df["boom_inclination_x_deg"].astype(float)
    arm = df["arm_inclination_deg"].astype(float)

    df["regime_work"] = np.select(
        [(boom.abs() < 1e-6) & (arm.abs() < 1e-6), (boom.abs() + arm.abs()) > 0],
        ["no_tool_activity", "tool_activity"],
        default="unknown",
    )

    df["regime"] = df["regime_motion"].astype(str) + "__" + df["regime_work"].astype(str)
    return df

# -----------------------
# Window features
# -----------------------
def _infer_step_seconds(ts: pd.Series) -> int:
    dt = pd.to_datetime(ts).diff().dt.total_seconds().dropna()
    if len(dt) == 0:
        return 5
    return int(np.clip(np.nanmedian(dt), 1, 60))

def add_window_features(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    out = []

    for ex_id, g in df.groupby("excavator_id", observed=False, sort=False):  # Добавляем observed=False
        g = g.sort_values("timestamp").copy()
        step = _infer_step_seconds(g["timestamp"])

        # Список новых признаков, которые будем добавлять
        new_columns = {}

        for w_sec in cfg.win_seconds:
            w = max(2, int(round(w_sec / step)))
            minp = max(2, w // 3)

            for c in SENSOR_COLS:
                # Роллинг статистики
                r = g[c].rolling(window=w, min_periods=minp)
                new_columns[f"{c}_mean_{w_sec}s"] = r.mean()
                new_columns[f"{c}_std_{w_sec}s"] = r.std()
                new_columns[f"{c}_min_{w_sec}s"] = r.min()
                new_columns[f"{c}_max_{w_sec}s"] = r.max()

                # Разница между соседними значениями
                d = g[c].diff()
                rd = d.rolling(window=w, min_periods=minp)
                new_columns[f"{c}_diff_mean_{w_sec}s"] = rd.mean()
                new_columns[f"{c}_diff_std_{w_sec}s"] = rd.std()

                # Сдвиг / тренд
                new_columns[f"{c}_slope_{w_sec}s"] = (g[c] - g[c].shift(w)) / max(1, w_sec)

        # Создаём новый DataFrame с вычисленными признаками
        new_features_df = pd.DataFrame(new_columns)

        # Объединяем с оригинальным DataFrame, сбрасывая индексы, чтобы избежать ошибки несовпадения индексов
        g = pd.concat([g.reset_index(drop=True), new_features_df], axis=1)

        out.append(g)

    # Собираем результат
    return pd.concat(out, ignore_index=True)

# -----------------------
# Healthy proxy
# -----------------------
def select_healthy_proxy(df_one: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    g = df_one.sort_values("timestamp").copy()
    t0 = g["timestamp"].iloc[0]
    cutoff = t0 + pd.Timedelta(days=cfg.healthy_first_days)
    return g[(g["timestamp"] < cutoff) & (g["flag_invalid"] == 0)].copy()

# -----------------------
# Models & scoring
# -----------------------
def _feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {
        "excavator_id", "timestamp",
        "regime", "regime_motion", "regime_work",
        "flag_invalid",
        # optional time calendar features; for pure behavior, usually exclude:
        "hour", "dayofweek", "month", "year",
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def fit_iforest_per_regime(train_df: pd.DataFrame, cfg: PipelineConfig) -> Dict[str, tuple[RobustScaler, IsolationForest]]:
    models: Dict[str, tuple[RobustScaler, IsolationForest]] = {}
    feat_cols = _feature_columns(train_df)

    for regime, g in train_df.groupby("regime", sort=False):
        if len(g) < 500:
            continue

        X = g[feat_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        scaler = RobustScaler()
        Xs = scaler.fit_transform(X)

        model = IsolationForest(
            n_estimators=cfg.n_estimators,
            max_samples=cfg.max_samples,
            contamination=cfg.contamination,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
        model.fit(Xs)
        models[regime] = (scaler, model)

    return models

def score_iforest_per_regime(df: pd.DataFrame, models: Dict[str, tuple[RobustScaler, IsolationForest]]) -> pd.Series:
    feat_cols = _feature_columns(df)
    score = pd.Series(index=df.index, dtype=float)

    for regime, g in df.groupby("regime", sort=False):
        if regime not in models:
            score.loc[g.index] = np.nan
            continue

        scaler, model = models[regime]
        X = g[feat_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        Xs = scaler.transform(X)

        normality = model.decision_function(Xs)  # higher => more normal
        score.loc[g.index] = -normality          # higher => more anomalous

    return score.fillna(score.median())

def robust_stat_score(df: pd.DataFrame, cfg: PipelineConfig) -> pd.Series:
    feat_cols = _feature_columns(df)
    X = df[feat_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    med = X.median(axis=0)
    mad = (X - med).abs().median(axis=0) + cfg.mad_eps
    rz = (X - med).abs().div(mad, axis=1)

    k = max(3, int(0.1 * len(feat_cols))) if len(feat_cols) else 1
    return rz.apply(lambda row: float(np.mean(np.sort(row.values)[-k:])), axis=1)

def _robust_norm(series: pd.Series, ref: pd.Series, cfg: PipelineConfig) -> pd.Series:
    m = ref.median()
    mad = (ref - m).abs().median() + cfg.mad_eps
    return (series - m) / mad

def quantile_threshold(ref_scores: pd.Series, q: float) -> float:
    return float(np.quantile(ref_scores.dropna().values, q))

def aggregate_episode_score(df_one: pd.DataFrame, point_score: pd.Series, cfg: PipelineConfig) -> pd.Series:
    g = df_one.sort_values("timestamp").copy()
    step = _infer_step_seconds(g["timestamp"])
    w = max(2, int(round((cfg.agg_minutes * 60) / step)))
    minp = max(2, w // 3)

    s = point_score.reindex(g.index).fillna(point_score.median())
    return s.rolling(window=w, min_periods=minp).mean()

# -----------------------
# End-to-end
# -----------------------
def run_anomaly_pipeline(df: pd.DataFrame, cfg: PipelineConfig | None = None) -> pd.DataFrame:
    """
    Returns df with:
      - flag_invalid
      - regime
      - anomaly_score_if, anomaly_score_stat
      - anomaly_score (ensemble)
      - threshold (regime-specific)
      - anomaly_point (0/1)
      - anomaly_score_episode, anomaly_episode (0/1)
    """
    cfg = cfg or PipelineConfig()

    # 1) Data quality
    df0 = flag_and_clean_technical_values(df)
    df0 = impute_missing_per_excavator(df0)

    # 2) Regime labels
    df0 = add_regime_labels(df0, cfg)

    # 3) Window features
    df0 = add_window_features(df0, cfg)

    results = []

    for ex_id, g in df0.groupby("excavator_id", sort=False, observed=False):
        g = g.sort_values("timestamp").reset_index(drop=True).copy()

        # Healthy proxy (train)
        train = select_healthy_proxy(g, cfg)
        if len(train) < 2000:
            train = g[g["flag_invalid"] == 0].copy()

        # Fit IF per regime
        models = fit_iforest_per_regime(train, cfg)

        # Point scores
        g["anomaly_score_if"] = score_iforest_per_regime(g, models)
        g["anomaly_score_stat"] = robust_stat_score(g, cfg)

        # Normalize on train (global, per-excavator)
        # Compute train scores too for correct normalization
        train_sc_if = score_iforest_per_regime(train, models)
        train_sc_stat = robust_stat_score(train, cfg)

        g["score_if_norm"] = _robust_norm(g["anomaly_score_if"], train_sc_if, cfg)
        g["score_stat_norm"] = _robust_norm(g["anomaly_score_stat"], train_sc_stat, cfg)

        # Ensemble score
        g["anomaly_score"] = 0.5 * g["score_if_norm"] + 0.5 * g["score_stat_norm"]

        # Threshold per regime (computed on healthy proxy)
        g["threshold"] = np.nan
        g["anomaly_point"] = 0

        # Need train ensemble score for thresholds
        train_tmp = train.copy()
        train_tmp["anomaly_score_if"] = train_sc_if.values
        train_tmp["anomaly_score_stat"] = train_sc_stat.values
        train_tmp["score_if_norm"] = _robust_norm(train_tmp["anomaly_score_if"], train_sc_if, cfg)
        train_tmp["score_stat_norm"] = _robust_norm(train_tmp["anomaly_score_stat"], train_sc_stat, cfg)
        train_tmp["anomaly_score"] = 0.5 * train_tmp["score_if_norm"] + 0.5 * train_tmp["score_stat_norm"]

        for regime, gr in g.groupby("regime", sort=False):
            train_r = train_tmp[train_tmp["regime"] == regime]
            ref = train_r["anomaly_score"] if len(train_r) >= 500 else train_tmp["anomaly_score"]
            thr = quantile_threshold(ref, cfg.score_quantile)

            idx = gr.index
            g.loc[idx, "threshold"] = thr
            g.loc[idx, "anomaly_point"] = (g.loc[idx, "anomaly_score"] > thr).astype(int)

        # Episode score + episode label (global threshold on healthy proxy)
        g["anomaly_score_episode"] = aggregate_episode_score(g, g["anomaly_score"], cfg).values
        thr_ep = quantile_threshold(train_tmp["anomaly_score"], cfg.score_quantile)
        g["anomaly_episode"] = (g["anomaly_score_episode"] > thr_ep).astype(int)

        results.append(g)

    out = pd.concat(results, ignore_index=True)
    return out
