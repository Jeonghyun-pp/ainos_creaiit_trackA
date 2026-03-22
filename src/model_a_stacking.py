"""
Model A: Stacked Generalization (스태킹)
- 1단계: 도메인별 모델이 각각 확률 출력
- 2단계: 메타 모델(LogisticRegression)이 최종 판단
- Out-of-Fold 예측으로 Data Leakage 방지
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from lightgbm import LGBMClassifier

from feature_pipeline import (
    build_all_features, temporal_split, get_feature_columns,
    get_domain_arrays, DATA_DIR, DOMAIN_EXTRACTORS
)

# ─── 도메인별 모델 선택 ──────────────────────────────────────────────

DOMAIN_MODELS = {
    "demographics_uw": lambda: LogisticRegression(max_iter=1000, C=1.0),
    "medical":         lambda: LGBMClassifier(n_estimators=200, max_depth=5, verbose=-1),
    "claim":           lambda: LogisticRegression(max_iter=1000, C=1.0),
    "policy":          lambda: LogisticRegression(max_iter=1000, C=1.0),
    "behavior":        lambda: LGBMClassifier(n_estimators=200, max_depth=5, verbose=-1),
}


def train_stacking(features_train, y_train, features_valid, y_valid,
                   features_test, y_test, n_folds=5):
    """
    스태킹 모델 학습 및 평가.
    반환: metrics dict, 학습된 모델들
    """
    domains = list(features_train.keys())

    # ── 1단계: Out-of-Fold 예측 생성 ──
    print("\n[Stacking] 1단계: 도메인별 모델 OOF 학습")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_train = np.zeros((len(y_train), len(domains)))
    pred_valid = np.zeros((len(y_valid), len(domains)))
    pred_test = np.zeros((len(y_test), len(domains)))

    trained_models = {}

    for i, domain in enumerate(domains):
        print(f"  [{domain}]", end=" ")
        X_tr = get_domain_arrays(features_train, domain,
                                 pd.DataFrame({"policyholder_id": features_train[domain]["policyholder_id"],
                                               "anchor_month": features_train[domain]["anchor_month"]}))
        X_val = get_domain_arrays(features_valid, domain,
                                  pd.DataFrame({"policyholder_id": features_valid[domain]["policyholder_id"],
                                                "anchor_month": features_valid[domain]["anchor_month"]}))
        X_te = get_domain_arrays(features_test, domain,
                                 pd.DataFrame({"policyholder_id": features_test[domain]["policyholder_id"],
                                               "anchor_month": features_test[domain]["anchor_month"]}))

        y_tr_arr = y_train.values

        # OOF 예측
        fold_models = []
        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_tr)):
            model = DOMAIN_MODELS[domain]()
            model.fit(X_tr[tr_idx], y_tr_arr[tr_idx])
            oof_train[val_idx, i] = model.predict_proba(X_tr[val_idx])[:, 1]
            pred_valid[:, i] += model.predict_proba(X_val)[:, 1] / n_folds
            pred_test[:, i] += model.predict_proba(X_te)[:, 1] / n_folds
            fold_models.append(model)

        # 전체 데이터로 최종 모델 학습
        final_model = DOMAIN_MODELS[domain]()
        final_model.fit(X_tr, y_tr_arr)
        trained_models[domain] = final_model

        auc = roc_auc_score(y_tr_arr, oof_train[:, i])
        print(f"OOF AUC={auc:.4f}")

    # ── 2단계: 메타 모델 학습 ──
    print("\n[Stacking] 2단계: 메타 모델 학습")
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(oof_train, y_train.values)

    # 검증/테스트 예측
    y_valid_pred = meta_model.predict_proba(pred_valid)[:, 1]
    y_test_pred = meta_model.predict_proba(pred_test)[:, 1]

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

    # 메타 모델 가중치 (각 도메인의 기여도)
    print("\n[Stacking] 메타 모델 가중치:")
    for domain, coef in zip(domains, meta_model.coef_[0]):
        print(f"  {domain}: {coef:.4f}")

    return metrics, trained_models, meta_model


# ─── 메인 실행 ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Model A: Stacked Generalization")
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

    metrics, models, meta = train_stacking(
        feat_train, y_train, feat_valid, y_valid, feat_test, y_test
    )

    print("\n" + "=" * 60)
    print("결과:")
    for split, m in metrics.items():
        print(f"\n  [{split}]")
        for k, v in m.items():
            print(f"    {k}: {v:.4f}")
    print("=" * 60)
