from sklearn.ensemble import IsolationForest
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from auxiliary.utils_minigng_trucks import replace_nan_with_median, handle_nan_in_data


def train_model_with_anomaly_detection(X, y, anomaly_contamination=0.02, threshold=0.6, random_state=42):
    """
    Функция для обучения модели с использованием детекции аномалий (Isolation Forest) и классификации риска (LightGBM).

    Parameters:
    X (DataFrame): Признаки для обучения
    y (Series): Целевая переменная
    anomaly_contamination (float): Параметр contamination для IsolationForest (по умолчанию 0.02)
    threshold (float): Порог для классификации (по умолчанию 0.6)
    random_state (int): Состояние генератора случайных чисел (по умолчанию 42)

    Returns:
    model (LGBMClassifier): Обученная модель LightGBM
    proba (array): Вероятности классов для X
    """

    # 4.1 Anomaly score (unsupervised)
    iso = IsolationForest(n_estimators=300, contamination=anomaly_contamination, random_state=random_state)
    X['anomaly_score'] = -iso.fit_predict(X)  # Отрицательные значения будут указывать на аномалии

    # Создаем X_model, добавляя anomaly_score
    X_model = X.copy()
    X_model['anomaly_score'] = X['anomaly_score']

    # 4.2 Классификация риска (supervised)
    model = lgb.LGBMClassifier(n_estimators=500,
                               learning_rate=0.03,
                               num_leaves=64,
                               class_weight='balanced',
                               nan_as_zero=True, random_state=random_state)

    # Проверяем на NaN в X_model и y
    # X_model, y = replace_nan_with_median(X_model, y)
    # X_model, y = handle_nan_in_data(X_model, y)
    # model = lgb.LGBMClassifier(n_estimators=500,
    #                            learning_rate=0.03,
    #                            num_leaves=64,
    #                            class_weight='balanced')

    # Обучаем модель
    model.fit(X_model, y)

    # Получаем вероятности (для AUC метрик)
    proba = model.predict_proba(X_model)[:, 1]

    # Метрики
    roc_auc = roc_auc_score(y, proba)
    pr_auc = average_precision_score(y, proba)
    print(f"\nМетрики:"
          f"\nROC AUC: {roc_auc}"
          f"\nPR AUC: {pr_auc}\n")

    # Precision/Recall на топ-K%
    p, r, t = precision_recall_curve(y, proba)
    for k in [0.01, 0.05, 0.1]:
        idx = int(len(p) * k)
        print(f"Top {int(k * 100)}% precision:", p[idx])

    print()

    return model, proba
