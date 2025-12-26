import shap

import numpy as np
import pandas as pd




def engineer_explain(x_row: pd.DataFrame, shap_row: np.ndarray, topn=6):
    # Преобразуем shap_row в pandas Series, чтобы работать с данными
    s = pd.Series(shap_row, index=x_row.columns)
    # Сортируем по абсолютному значению
    s = s.reindex(s.abs().sort_values(ascending=False).index)
    top = s.head(topn)

    parts = []
    for feat, val in top.items():
        direction = "↑ риск" if val > 0 else "↓ риск"
        parts.append(f"{feat} ({direction})")

    # “доменные” подсказки
    domain = []
    if any(f.startswith("rail_diff_") for f in top.index) or any("rail_diff" in f for f in top.index):
        domain.append("рост пульсаций давления в рампе (rail_diff)")
    if any("fuel_rate" in f for f in top.index):
        domain.append("дрейф/рост расхода топлива на стационарных режимах (fuel_rate)")
    if any("rail_error" in f for f in top.index):
        domain.append("увеличение ошибки управления давлением (rail_error)")

    text = "\nКлючевые факторы окна: " + ", ".join(parts)
    if domain:
        text += "\nИнженерная интерпретация: " + "; ".join(domain) + "."

    return text


def prepare_data_for_shap(X_model):
    """
    Функция для подготовки данных, удаляя столбцы с датой и преобразуя их в числовой формат.

    Parameters:
    X_model (DataFrame): Данные для модели

    Returns:
    DataFrame: Подготовленные данные без столбцов с датой и преобразованные в числовой формат
    """

    # Преобразуем datetime столбцы в числовой формат
    for column in X_model.select_dtypes(include=["datetime64"]).columns:
        X_model[column] = (X_model[column] - pd.Timestamp("1970-01-01")) // pd.Timedelta(
            '1s')  # Секунды с начала эпохи UNIX

    # Убедимся, что все данные числовые
    X_model = X_model.apply(pd.to_numeric, errors='coerce')

    return X_model


def explain_shap_for_event(X, model, report, pred, threshold=0.6, sample_size=5000, random_state=42, topn=8):
    """
    Функция для SHAP и локальной объяснимости для одного события.

    Parameters:
    df_win (DataFrame): Данные для модели, без целевой переменной
    model (LGBMClassifier): Обученная модель
    report (DataFrame): Отчёт с событиями, где были сделаны предсказания
    pred (DataFrame): Предсказания для окон
    threshold (float): Порог для предупреждения (по умолчанию 0.6)
    sample_size (int): Размер подвыборки для SHAP (по умолчанию 5000)
    random_state (int): Состояние генератора случайных чисел (по умолчанию 42)
    topn (int): Количество топ фичей для объяснения (по умолчанию 8)

    Returns:
    None: Печатает локальный график SHAP и генерирует текстовое объяснение
    """

    # 1) Подготовка данных для SHAP
    X_model = X.drop(columns=["target_7_30", "event_dt", "days_to_event"], errors="ignore")

    # Преобразуем все datetime столбцы в числовой формат (например, количество секунд с эпохи UNIX)
    for column in X_model.select_dtypes(include=["datetime64"]).columns:
        X_model[column] = (X_model[column] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

    # 3. Создаем объяснителя SHAP
    explainer = shap.TreeExplainer(model)

    # 4. Вычисляем SHAP значения для данных X_model
    shap_values = explainer.shap_values(X_model)

    # 5. Создаем summary plot для SHAP значений
    shap.summary_plot(shap_values, X_model, show=True)

    # 6) Локальное объяснение для одного события
    one = report[report["detected"]].iloc[0]  # Берём первое событие, которое было обнаружено
    aid, edt = one["asset_id"], one["event_dt"]
    alert_ts = one["first_alert_ts"]

    # Найдём строку этого события в X_model
    row_idx = (aid, pred.loc[(pred["asset_id"] == aid) & (pred["event_dt"] == edt) & (
                pred["timestamp"] == alert_ts), "mdm_object_name"].iloc[0], alert_ts)
    x_row = X_model.loc[row_idx:row_idx]  # DataFrame 1xN

    # 7) Получаем SHAP значения для этого события
    sv = explainer.shap_values(x_row)
    sv_pos = sv[1] if isinstance(sv, list) else sv  # Для положительного класса

    # 8) Локальный график (waterfall)
    shap.plots.waterfall(shap.Explanation(
        values=sv_pos[0],  # Значения SHAP для выбранного примера
        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value,
                                                              (list, tuple)) else explainer.expected_value,
        data=x_row.iloc[0],  # Исходные данные для этого примера
        feature_names=x_row.columns  # Имена признаков
    ))

    # 9) Генерируем текст объяснения на основе топ-N SHAP-фичей
    print(engineer_explain(x_row, sv_pos[0], topn=topn))  # Печатаем объяснение для топ-N фич

