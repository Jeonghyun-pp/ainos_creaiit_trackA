"""
DS Consultant Agent Tools
- 6개 Tool 함수: Agent가 호출하여 분석/조언에 활용
- 모듈형 리스크 모델(Additive)과 데이터를 기반으로 동작
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA_DIR = Path(__file__).parent.parent / "data"


# ─── Tool 1: Feature Registry 조회 ─────────────────────────────────

def get_feature_registry():
    """
    현재 등록된 feature 목록과 메타데이터를 조회한다.

    반환: feature 목록 (dict 리스트)
      - feature_key, display_name, domain, data_type, cadence
      - baseline_included_flag, student_addable_flag
      - regulatory_sensitivity_class
    """
    df = pd.read_csv(DATA_DIR / "modeling_feature_registry.csv")
    cols = ["feature_key", "display_name", "domain", "data_type", "cadence",
            "baseline_included_flag", "student_addable_flag",
            "regulatory_sensitivity_class", "missingness_semantics"]
    result = df[cols].to_dict(orient="records")

    summary = {
        "total_features": len(df),
        "baseline_count": int(df["baseline_included_flag"].sum()),
        "addable_count": int((~df["baseline_included_flag"]).sum()),
        "domains": df["domain"].value_counts().to_dict(),
        "features": result,
    }
    return summary


# ─── Tool 2: Component Structure 조회 ──────────────────────────────

def get_component_structure():
    """
    현재 모듈형 모델의 컴포넌트 구조를 조회한다.

    반환: 컴포넌트 목록과 각 컴포넌트의 feature 구성
    """
    df = pd.read_csv(DATA_DIR / "modeling_component_registry.csv")
    components = []
    for _, row in df.iterrows():
        feature_keys = row.get("input_feature_keys", "[]")
        # 문자열 파싱
        if isinstance(feature_keys, str):
            import ast
            try:
                feature_list = ast.literal_eval(feature_keys)
            except:
                feature_list = []
        else:
            feature_list = []
        components.append({
            "component_name": row.get("component_name", ""),
            "component_type": row.get("component_type", ""),
            "feature_count": len(feature_list),
            "feature_keys": feature_list,
            "status": row.get("status", ""),
        })

    return {
        "total_components": len(components),
        "components": components,
        "note": "behavior 컴포넌트가 없음 — 5번째 컴포넌트 추가가 확장성 시연의 핵심",
    }


# ─── Tool 3: 상관 분석 ─────────────────────────────────────────────

def analyze_correlation(feature_values, target_values, feature_name="new_feature"):
    """
    새 feature와 타깃(high_cost_event_flag) 간의 상관관계를 분석한다.

    Args:
        feature_values: 새 feature의 값 배열 (list 또는 numpy array)
        target_values: 타깃 변수 값 배열 (0/1)
        feature_name: feature 이름 (설명용)

    반환: 상관계수, p-value, 양성/음성 그룹 통계
    """
    feat = np.array(feature_values, dtype=float)
    target = np.array(target_values, dtype=int)

    # 결측 제거
    valid = ~(np.isnan(feat) | np.isnan(target))
    feat, target = feat[valid], target[valid]

    # Point-biserial correlation (이진 타깃 vs 연속형 feature)
    corr, p_value = stats.pointbiserialr(target, feat)

    # 양성/음성 그룹 비교
    pos_vals = feat[target == 1]
    neg_vals = feat[target == 0]

    # t-test
    t_stat, t_p = stats.ttest_ind(pos_vals, neg_vals, equal_var=False)

    result = {
        "feature_name": feature_name,
        "n_samples": len(feat),
        "n_missing": int((~valid).sum()),
        "correlation": round(float(corr), 4),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "positive_group": {
            "count": len(pos_vals),
            "mean": round(float(pos_vals.mean()), 4),
            "std": round(float(pos_vals.std()), 4),
        },
        "negative_group": {
            "count": len(neg_vals),
            "mean": round(float(neg_vals.mean()), 4),
            "std": round(float(neg_vals.std()), 4),
        },
        "t_test": {
            "t_statistic": round(float(t_stat), 4),
            "p_value": float(t_p),
        },
        "interpretation": _interpret_correlation(corr, p_value),
    }
    return result


def _interpret_correlation(corr, p_value):
    if p_value >= 0.05:
        return "통계적으로 유의하지 않음 (p >= 0.05). 타깃과 관련 없을 가능성 높음."
    strength = abs(corr)
    if strength < 0.1:
        level = "매우 약한"
    elif strength < 0.3:
        level = "약한"
    elif strength < 0.5:
        level = "중간"
    else:
        level = "강한"
    direction = "양의" if corr > 0 else "음의"
    return f"{direction} {level} 상관관계 (r={corr:.4f}, p={p_value:.2e}). 예측력에 기여할 가능성 {'높음' if strength >= 0.2 else '있음'}."


# ─── Tool 4: 공선성 확인 ───────────────────────────────────────────

def check_collinearity(new_feature_values, existing_feature_df, new_feature_name="new_feature"):
    """
    새 feature와 기존 feature 간의 공선성(중복 정보)을 확인한다.

    Args:
        new_feature_values: 새 feature 값 배열
        existing_feature_df: 기존 feature DataFrame (컬럼=feature, 행=샘플)
        new_feature_name: 새 feature 이름

    반환: 기존 각 feature와의 상관계수, VIF 근사치, 권고사항
    """
    new_feat = np.array(new_feature_values, dtype=float)

    correlations = {}
    for col in existing_feature_df.columns:
        existing = existing_feature_df[col].values.astype(float)
        valid = ~(np.isnan(new_feat) | np.isnan(existing))
        if valid.sum() > 10:
            r, p = stats.pearsonr(new_feat[valid], existing[valid])
            correlations[col] = {"correlation": round(float(r), 4), "p_value": float(p)}

    # 가장 높은 상관관계 top 5
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]["correlation"]), reverse=True)
    top_5 = sorted_corrs[:5]

    # 공선성 판단
    max_corr = abs(top_5[0][1]["correlation"]) if top_5 else 0
    if max_corr >= 0.8:
        risk = "높음"
        recommendation = "기존 feature와 매우 높은 상관. 중복 정보일 가능성 큼. 기존 feature 대체 또는 잔차를 사용 권장."
    elif max_corr >= 0.5:
        risk = "중간"
        recommendation = "일부 정보 중복. 추가해도 되지만, 모델 성능 개선 폭이 제한적일 수 있음."
    else:
        risk = "낮음"
        recommendation = "기존 feature와 독립적. 새로운 정보를 제공할 가능성 높음. 추가 권장."

    return {
        "new_feature": new_feature_name,
        "collinearity_risk": risk,
        "max_correlation": round(float(max_corr), 4),
        "top_5_correlations": [
            {"feature": k, **v} for k, v in top_5
        ],
        "recommendation": recommendation,
    }


# ─── Tool 5: 통합 시뮬레이션 ──────────────────────────────────────

def simulate_integration(new_feature_values, target_values,
                         existing_scores, new_feature_name="new_feature"):
    """
    새 feature를 컴포넌트로 추가했을 때 성능 변화를 시뮬레이션한다.

    Args:
        new_feature_values: 새 feature 값 배열 (test set)
        target_values: 타깃 값 (test set)
        existing_scores: 기존 컴포넌트들의 합산 log-odds 점수 (test set)
        new_feature_name: 새 feature 이름

    반환: 추가 전/후 AUC 비교, 기여도
    """
    new_feat = np.array(new_feature_values, dtype=float).reshape(-1, 1)
    target = np.array(target_values, dtype=int)
    existing = np.array(existing_scores, dtype=float)

    # 기존 성능
    sigmoid = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    auc_before = roc_auc_score(target, sigmoid(existing))

    # 새 feature로 간단한 모델 학습 (LogisticRegression)
    # train/test를 반으로 나눠 시뮬레이션
    n = len(target)
    mid = n // 2
    model = LogisticRegression(max_iter=1000)
    model.fit(new_feat[:mid], target[:mid])

    # 새 컴포넌트의 log-odds 점수
    new_score = model.decision_function(new_feat[mid:])

    # 기존 + 새 컴포넌트 합산
    combined = existing[mid:] + new_score
    auc_after = roc_auc_score(target[mid:], sigmoid(combined))
    auc_existing_only = roc_auc_score(target[mid:], sigmoid(existing[mid:]))

    delta = auc_after - auc_existing_only

    return {
        "new_feature": new_feature_name,
        "auc_before": round(float(auc_existing_only), 4),
        "auc_after": round(float(auc_after), 4),
        "auc_delta": round(float(delta), 4),
        "improved": delta > 0,
        "new_component_weight": round(float(model.coef_[0][0]), 4),
        "recommendation": _interpret_integration(delta),
    }


def _interpret_integration(delta):
    if delta > 0.01:
        return f"AUC +{delta:.4f} 유의미한 성능 개선. 새 컴포넌트로 추가 권장."
    elif delta > 0:
        return f"AUC +{delta:.4f} 소폭 개선. 추가 가능하나 기대 효과 제한적."
    else:
        return f"AUC {delta:.4f} 개선 없음. 추가하지 않거나 feature engineering 재검토 필요."


# ─── Tool 6: 규제 민감도 확인 ──────────────────────────────────────

def check_regulatory_sensitivity(feature_name, data_type="numeric",
                                  source_description="", contains_pii=False):
    """
    새 feature의 규제 민감도를 분류하고 필요 조치를 안내한다.

    Args:
        feature_name: feature 이름
        data_type: 데이터 타입 (numeric, binary, categorical, text)
        source_description: 데이터 출처 설명
        contains_pii: 개인식별정보 포함 여부

    반환: 민감도 등급, 규제 체크리스트, 필요 조치
    """
    # 민감 키워드 탐지
    sensitive_keywords = ["gender", "sex", "race", "religion", "disability",
                          "성별", "인종", "종교", "장애", "genetic", "유전"]
    health_keywords = ["diagnosis", "disease", "hospital", "surgery", "medication",
                       "진단", "질병", "입원", "수술", "약물", "치료"]
    financial_keywords = ["income", "salary", "credit", "debt",
                          "소득", "연봉", "신용", "부채"]

    name_lower = (feature_name + " " + source_description).lower()

    # 민감도 판정
    if contains_pii or any(kw in name_lower for kw in sensitive_keywords):
        sensitivity = "restricted"
        risk_level = "높음"
    elif any(kw in name_lower for kw in health_keywords):
        sensitivity = "confidential"
        risk_level = "중간"
    elif any(kw in name_lower for kw in financial_keywords):
        sensitivity = "confidential"
        risk_level = "중간"
    else:
        sensitivity = "public_synthetic"
        risk_level = "낮음"

    # 규제 체크리스트
    checklist = {
        "IFRS_17": {
            "세분화 영향": "이 feature가 코호트/연령/성별별 세분화에 영향을 주는지 확인 필요",
            "문서화": "계리가정 변경으로 간주될 수 있으므로 산출 근거 문서화 필요",
        },
        "금융사지배구조법": {
            "감사추적": "feature 추가 이력을 감사 로그에 기록",
            "승인절차": "위험관리 전담조직의 검토/승인 필요" if risk_level != "낮음" else "일반 절차로 추가 가능",
        },
        "AI_거버넌스": {
            "설명가능성": "해당 feature가 예측에 미치는 영향을 설명 가능해야 함",
            "편향점검": "성별/연령/지역 등 보호 속성과의 상관관계 점검 필요" if risk_level != "낮음" else "기본 편향 점검 수행",
            "비상정지": "해당 컴포넌트를 독립적으로 중단할 수 있어야 함 (모듈형 구조에서 자연스럽게 충족)",
        },
    }

    actions = []
    if sensitivity == "restricted":
        actions.append("개인정보 영향평가(PIA) 실시")
        actions.append("데이터 익명화/가명화 처리 확인")
        actions.append("공정성 테스트: 보호 속성별 예측 결과 편차 점검")
    if sensitivity in ["restricted", "confidential"]:
        actions.append("접근 권한 제한 설정")
        actions.append("AI 윤리위원회 검토 요청")
    actions.append("Model Card에 해당 feature 추가 및 사유 기재")
    actions.append("모듈형 구조에서 해당 컴포넌트 독립 검증 가능 확인")

    return {
        "feature_name": feature_name,
        "sensitivity_class": sensitivity,
        "risk_level": risk_level,
        "data_type": data_type,
        "regulatory_checklist": checklist,
        "required_actions": actions,
    }


# ─── Tool 목록 (Agent에 등록할 때 사용) ────────────────────────────

TOOLS = {
    "get_feature_registry": get_feature_registry,
    "get_component_structure": get_component_structure,
    "analyze_correlation": analyze_correlation,
    "check_collinearity": check_collinearity,
    "simulate_integration": simulate_integration,
    "check_regulatory_sensitivity": check_regulatory_sensitivity,
}


# ─── 테스트 ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Tool 1: Feature Registry ===")
    reg = get_feature_registry()
    print(f"총 {reg['total_features']}개 feature, baseline {reg['baseline_count']}개")
    print(f"도메인: {reg['domains']}")

    print("\n=== Tool 2: Component Structure ===")
    comp = get_component_structure()
    print(f"총 {comp['total_components']}개 컴포넌트")
    for c in comp["components"]:
        print(f"  {c['component_name']}: {c['component_type']} ({c['feature_count']} features)")

    print("\n=== Tool 6: Regulatory Sensitivity ===")
    reg_check = check_regulatory_sensitivity(
        "heart_rate_variability", data_type="numeric",
        source_description="웨어러블 기기에서 수집한 심박변이도"
    )
    print(f"민감도: {reg_check['sensitivity_class']} (위험: {reg_check['risk_level']})")
    print(f"필요 조치: {reg_check['required_actions']}")
