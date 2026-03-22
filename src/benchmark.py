"""
Benchmark: 4가지 모듈형 아키텍처 성능 비교
A. Stacking | B. Multi-Head NN | C. Additive | D. MoE
"""

import pandas as pd
import numpy as np
import time
from feature_pipeline import (
    build_all_features, temporal_split, DATA_DIR
)

from model_a_stacking import train_stacking
from model_b_multihead import train_multihead
from model_c_additive import train_additive
from model_d_moe import train_moe


def run_benchmark():
    print("=" * 70)
    print("  4가지 모듈형 아키텍처 벤치마크")
    print("=" * 70)

    # ── 데이터 로드 & 분할 ──
    outcome = pd.read_csv(DATA_DIR / "modeling_outcome_high_cost_event.csv")
    train_df, valid_df, test_df = temporal_split(outcome)
    print(f"\nTrain: {len(train_df):,}  Valid: {len(valid_df):,}  Test: {len(test_df):,}")

    # ── Feature 추출 (공통) ──
    print("\n" + "─" * 70)
    print("Feature 추출 (전체 데이터)")
    print("─" * 70)
    t0 = time.time()
    print("\n[Train]")
    feat_train, y_train = build_all_features(train_df)
    print("\n[Valid]")
    feat_valid, y_valid = build_all_features(valid_df)
    print("\n[Test]")
    feat_test, y_test = build_all_features(test_df)
    feat_time = time.time() - t0
    print(f"\nFeature 추출 소요: {feat_time:.0f}초")

    # ── 모델별 학습 & 평가 ──
    results = {}

    # A. Stacking
    print("\n" + "─" * 70)
    print("A. Stacking")
    print("─" * 70)
    t0 = time.time()
    metrics_a, _, _ = train_stacking(
        feat_train, y_train, feat_valid, y_valid, feat_test, y_test
    )
    results["A. Stacking"] = {**metrics_a["test"], "time": time.time() - t0}

    # B. Multi-Head NN
    print("\n" + "─" * 70)
    print("B. Multi-Head NN")
    print("─" * 70)
    t0 = time.time()
    metrics_b, _ = train_multihead(
        feat_train, y_train, feat_valid, y_valid, feat_test, y_test
    )
    results["B. Multi-Head NN"] = {**metrics_b["test"], "time": time.time() - t0}

    # C. Additive
    print("\n" + "─" * 70)
    print("C. Additive")
    print("─" * 70)
    t0 = time.time()
    metrics_c, _ = train_additive(
        feat_train, y_train, feat_valid, y_valid, feat_test, y_test
    )
    results["C. Additive"] = {**metrics_c["test"], "time": time.time() - t0}

    # D. MoE
    print("\n" + "─" * 70)
    print("D. MoE")
    print("─" * 70)
    t0 = time.time()
    metrics_d, _ = train_moe(
        feat_train, y_train, feat_valid, y_valid, feat_test, y_test
    )
    results["D. MoE"] = {**metrics_d["test"], "time": time.time() - t0}

    # ── 비교표 출력 ──
    print("\n" + "=" * 70)
    print("  최종 성능 비교 (Test Set)")
    print("=" * 70)

    # 기준선
    baselines = {"AUC-ROC": 0.75, "PR-AUC": 0.40, "Brier": 0.13}
    monolithic = {"AUC-ROC": 0.9415, "PR-AUC": 0.7875, "Brier": 0.0940}

    print(f"\n{'모델':20s} {'AUC-ROC':>10s} {'PR-AUC':>10s} {'Brier':>10s} {'학습시간':>10s}")
    print("─" * 65)

    # 기준선
    print(f"{'기준선 (최소)':20s} {baselines['AUC-ROC']:>10.4f} {baselines['PR-AUC']:>10.4f} {baselines['Brier']:>10.4f} {'─':>10s}")
    print(f"{'Monolithic (참고)':20s} {monolithic['AUC-ROC']:>10.4f} {monolithic['PR-AUC']:>10.4f} {monolithic['Brier']:>10.4f} {'─':>10s}")
    print("─" * 65)

    for name, m in results.items():
        t_str = f"{m['time']:.0f}s"
        print(f"{name:20s} {m['AUC-ROC']:>10.4f} {m['PR-AUC']:>10.4f} {m['Brier']:>10.4f} {t_str:>10s}")

    # 기준선 통과 여부
    print("\n기준선 통과 여부:")
    for name, m in results.items():
        auc_ok = "✓" if m["AUC-ROC"] >= baselines["AUC-ROC"] else "✗"
        pr_ok = "✓" if m["PR-AUC"] >= baselines["PR-AUC"] else "✗"
        brier_ok = "✓" if m["Brier"] <= baselines["Brier"] else "✗"
        print(f"  {name}: AUC {auc_ok}  PR-AUC {pr_ok}  Brier {brier_ok}")


if __name__ == "__main__":
    run_benchmark()
