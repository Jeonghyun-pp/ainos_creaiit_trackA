"""
Model C: Additive Component Model (가산형 컴포넌트 모델)
- 각 컴포넌트가 log-odds 점수 출력
- 단순 합산 → sigmoid → 확률
- 설명가능성 최고, 확장성 최고
- EBM(ExplainableBoostingClassifier) + LogisticRegression 사용
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier

from feature_pipeline import (
    build_all_features, temporal_split, get_feature_columns,
    get_domain_arrays, DATA_DIR
)

# ─── 도메인별 컴포넌트 모델 ──────────────────────────────────────────

DOMAIN_MODELS = {
    "demographics_uw": lambda: LogisticRegression(max_iter=1000, C=1.0),
    "medical":         lambda: ExplainableBoostingClassifier(max_rounds=200, interactions=0),
    "claim":           lambda: LogisticRegression(max_iter=1000, C=1.0),
    "policy":          lambda: LogisticRegression(max_iter=1000, C=1.0),
    "behavior":        lambda: ExplainableBoostingClassifier(max_rounds=200, interactions=0),
}


def _get_log_odds(model, X):
    """모델 종류에 따라 log-odds 점수 추출"""
    if hasattr(model, "decision_function"):
        # LogisticRegression
        return model.decision_function(X)
    elif hasattr(model, "predict_proba"):
        # EBM, LightGBM → predict_proba → log-odds 변환
        proba = model.predict_proba(X)[:, 1]
        proba = np.clip(proba, 1e-7, 1 - 1e-7)
        return np.log(proba / (1 - proba))
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {type(model)}")


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def train_additive(features_train, y_train, features_valid, y_valid,
                   features_test, y_test):
    """
    Additive 모델 학습 및 평가.
    반환: metrics dict, 학습된 컴포넌트들, intercept
    """
    domains = list(features_train.keys())

    # ── 컴포넌트별 독립 학습 ──
    print("\n[Additive] 컴포넌트별 독립 학습")
    components = {}

    for domain in domains:
        print(f"  [{domain}]", end=" ")
        df_tr = features_train[domain]
        fcols = get_feature_columns(df_tr)
        X_tr = df_tr[fcols].values.astype(np.float32)

        model = DOMAIN_MODELS[domain]()
        model.fit(X_tr, y_train.values)
        components[domain] = model

        # 개별 성능
        score_tr = _get_log_odds(model, X_tr)
        prob_tr = sigmoid(score_tr)
        auc = roc_auc_score(y_train.values, prob_tr)
        print(f"개별 AUC={auc:.4f}")

    # ── 합산 → 확률 ──
    def predict(features_dict):
        total_score = np.zeros(len(next(iter(features_dict.values()))))
        domain_scores = {}
        for domain in domains:
            df = features_dict[domain]
            fcols = get_feature_columns(df)
            X = df[fcols].values.astype(np.float32)
            score = _get_log_odds(components[domain], X)
            total_score += score
            domain_scores[domain] = score
        return sigmoid(total_score), domain_scores

    y_valid_pred, valid_scores = predict(features_valid)
    y_test_pred, test_scores = predict(features_test)

    # ── 평가 ──
    metrics = {
        "valid": {
            "AUC-ROC": roc_auc_score(y_valid.values, y_valid_pred),
            "PR-AUC": average_precision_score(y_valid.values, y_valid_pred),
            "Brier": brier_score_loss(y_valid.values, y_valid_pred),
        },
        "test": {
            "AUC-ROC": roc_auc_score(y_test.values, y_test_pred),
            "PR-AUC": average_precision_score(y_test.values, y_test_pred),
            "Brier": brier_score_loss(y_test.values, y_test_pred),
        },
    }

    # ── 설명가능성: 도메인별 평균 기여도 ──
    print("\n[Additive] 도메인별 평균 기여도 (test set, log-odds):")
    for domain in domains:
        mean_score = test_scores[domain].mean()
        std_score = test_scores[domain].std()
        print(f"  {domain:20s}: mean={mean_score:+.3f}  std={std_score:.3f}")

    return metrics, components


def explain_individual(components, features_dict, idx=0):
    """개별 예측 설명: 각 도메인이 얼마나 기여했는지"""
    domains = list(components.keys())
    total = 0
    print(f"\n[Additive] 개별 설명 (index={idx}):")
    for domain in domains:
        df = features_dict[domain]
        fcols = get_feature_columns(df)
        X = df[fcols].values[idx:idx+1].astype(np.float32)
        score = _get_log_odds(components[domain], X)[0]
        total += score
        print(f"  {domain:20s}: {score:+.3f}")
    prob = sigmoid(total)
    print(f"  {'합산':20s}: {total:+.3f} → 확률: {prob:.4f}")
    return prob


# ─── 메인 실행 ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Model C: Additive Component Model")
    print("=" * 60)

    outcome = pd.read_csv(DATA_DIR / "modeling_outcome_high_cost_event.csv")
    train_df, valid_df, test_df = temporal_split(outcome)

    print(f"\nTrain: {len(train_df):,}  Valid: {len(valid_df):,}  Test: {len(test_df):,}")

    print("\n--- Feature 추출 (Train) ---")
    feat_train, y_train = build_all_features(train_df)
    print("\n--- Feature 추출 (Valid) ---")
    feat_valid, y_valid = build_all_features(valid_df)
    print("\n--- Feature 추출 (Test) ---")
    feat_test, y_test = build_all_features(test_df)

    metrics, components = train_additive(
        feat_train, y_train, feat_valid, y_valid, feat_test, y_test
    )

    # 개별 설명 예시
    explain_individual(components, feat_test, idx=0)
    explain_individual(components, feat_test, idx=100)

    print("\n" + "=" * 60)
    print("결과:")
    for split, m in metrics.items():
        print(f"\n  [{split}]")
        for k, v in m.items():
            print(f"    {k}: {v:.4f}")
    print("=" * 60)
