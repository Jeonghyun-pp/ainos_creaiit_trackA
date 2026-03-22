"""
Feature Pipeline: 원시 CSV → 도메인별 feature DataFrame 생성
5개 도메인: demographics_uw, medical, claim, policy, behavior
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).parent.parent / "data"

# ─── 원시 데이터 로드 (캐싱) ───────────────────────────────────────────

_cache = {}

def _load(name):
    if name not in _cache:
        _cache[name] = pd.read_csv(DATA_DIR / f"{name}.csv")
    return _cache[name]


# ─── 유틸리티 ──────────────────────────────────────────────────────────

def _month_offset(anchor, months):
    """앵커 월에서 N개월 이동한 날짜 반환"""
    anchor = pd.Timestamp(anchor)
    return anchor - pd.DateOffset(months=months)


# ─── 도메인 1: 인구통계 + UW (정적) ──────────────────────────────────

def extract_demographics_uw(outcome_df):
    """
    정적 데이터 → 앵커 월과 무관하게 동일
    반환: policyholder_id 기준 feature DataFrame
    """
    ph = _load("core_policyholder")
    policy = _load("core_policy")
    uw = _load("core_underwriting_assessment")

    # policy ↔ uw 조인
    policy_uw = policy.merge(uw, on="policy_id", how="left")

    # 계약자 단위 집계 (여러 계약 보유 가능)
    uw_agg = policy_uw.groupby("policyholder_id").agg(
        uw_class_best=("uw_class", "first"),
        extra_rate_flag=("extra_rate_flag", "max"),
        questionnaire_score_band=("questionnaire_score_band", "first"),
        bp_band=("bp_band", "first"),
        glucose_band=("glucose_band", "first"),
        cholesterol_band=("cholesterol_band", "first"),
    ).reset_index()

    # policyholder + UW 병합
    demo = ph.merge(uw_agg, on="policyholder_id", how="left")

    # age 계산 (2022년 기준)
    demo["age"] = 2022 - demo["birth_year"]
    demo.drop(columns=["birth_year", "synthetic_household_id", "created_at"],
              inplace=True, errors="ignore")

    # Label Encoding
    cat_cols = ["sex_code", "region_tier", "occupation_class", "income_band",
                "bmi_band", "uw_class_best", "questionnaire_score_band",
                "bp_band", "glucose_band", "cholesterol_band"]
    for col in cat_cols:
        if col in demo.columns:
            demo[col] = demo[col].fillna("unknown")
            le = LabelEncoder()
            demo[col] = le.fit_transform(demo[col].astype(str))

    # boolean → int
    bool_cols = ["smoker_flag", "family_history_cancer_flag",
                 "family_history_cv_flag", "extra_rate_flag"]
    for col in bool_cols:
        if col in demo.columns:
            demo[col] = demo[col].astype(int)

    # outcome_df의 (policyholder_id, anchor_month)에 맞춰 반복
    result = outcome_df[["policyholder_id", "anchor_month"]].merge(
        demo, on="policyholder_id", how="left"
    )
    return result


# ─── 도메인 2: 의료이력 (동적, lookback) ─────────────────────────────

def extract_medical(outcome_df):
    """진단 + 입원 이벤트를 앵커 월 기준 lookback으로 집계"""
    dx = _load("medical_diagnosis_event").copy()
    hosp = _load("medical_hospitalization_event").copy()

    dx["diagnosis_date"] = pd.to_datetime(dx["diagnosis_date"])
    hosp["admission_date"] = pd.to_datetime(hosp["admission_date"])

    outcome = outcome_df[["policyholder_id", "anchor_month"]].copy()
    outcome["anchor_dt"] = pd.to_datetime(outcome["anchor_month"])

    # ── 진단 집계 ──
    dx_merged = outcome.merge(dx, on="policyholder_id", how="left")
    # 앵커 월 이전만
    dx_merged = dx_merged[dx_merged["diagnosis_date"] < dx_merged["anchor_dt"]]

    # 전체 이력
    dx_hist = dx_merged.groupby(["policyholder_id", "anchor_month"]).agg(
        hist_dx_count=("diagnosis_event_id", "count"),
        hist_chronic_count=("chronic_flag", "sum"),
        has_cancer_hx=("diagnosis_group", lambda x: int((x == "cancer").any())),
        has_cv_hx=("diagnosis_group", lambda x: int((x == "cardiovascular").any())),
    ).reset_index()

    # 최근 3개월
    dx_3m = dx_merged[
        dx_merged["diagnosis_date"] >= dx_merged["anchor_dt"] - pd.DateOffset(months=3)
    ].groupby(["policyholder_id", "anchor_month"]).agg(
        rdx_count_3m=("diagnosis_event_id", "count"),
        rdx_chronic_3m=("chronic_flag", "sum"),
    ).reset_index()

    # 최근 6개월
    dx_6m = dx_merged[
        dx_merged["diagnosis_date"] >= dx_merged["anchor_dt"] - pd.DateOffset(months=6)
    ].groupby(["policyholder_id", "anchor_month"]).agg(
        rdx_count_6m=("diagnosis_event_id", "count"),
    ).reset_index()

    # ── 입원 집계 ──
    hosp_merged = outcome.merge(hosp, on="policyholder_id", how="left")
    hosp_merged = hosp_merged[hosp_merged["admission_date"] < hosp_merged["anchor_dt"]]

    hosp_hist = hosp_merged.groupby(["policyholder_id", "anchor_month"]).agg(
        hist_hosp_count=("hospitalization_event_id", "count"),
        surgery_count=("surgery_flag", "sum"),
        icu_ever=("icu_flag", "max"),
    ).reset_index()

    hosp_3m = hosp_merged[
        hosp_merged["admission_date"] >= hosp_merged["anchor_dt"] - pd.DateOffset(months=3)
    ].groupby(["policyholder_id", "anchor_month"]).agg(
        rhosp_count_3m=("hospitalization_event_id", "count"),
        rhosp_los_3m=("length_of_stay_days", "sum"),
    ).reset_index()

    hosp_6m = hosp_merged[
        hosp_merged["admission_date"] >= hosp_merged["anchor_dt"] - pd.DateOffset(months=6)
    ].groupby(["policyholder_id", "anchor_month"]).agg(
        rhosp_los_6m=("length_of_stay_days", "sum"),
    ).reset_index()

    # 병합
    result = outcome[["policyholder_id", "anchor_month"]].copy()
    for df in [dx_hist, dx_3m, dx_6m, hosp_hist, hosp_3m, hosp_6m]:
        result = result.merge(df, on=["policyholder_id", "anchor_month"], how="left")

    result.drop(columns=["anchor_dt"], inplace=True, errors="ignore")
    result.fillna(0, inplace=True)
    return result


# ─── 도메인 3: 청구 (동적) ───────────────────────────────────────────

def extract_claim(outcome_df):
    """청구 이벤트를 앵커 월 기준 lookback으로 집계"""
    cl = _load("medical_claim_event").copy()
    cl["service_date"] = pd.to_datetime(cl["service_date"])

    outcome = outcome_df[["policyholder_id", "anchor_month"]].copy()
    outcome["anchor_dt"] = pd.to_datetime(outcome["anchor_month"])

    cl_merged = outcome.merge(cl, on="policyholder_id", how="left")
    cl_merged = cl_merged[cl_merged["service_date"] < cl_merged["anchor_dt"]]

    # 전체 이력
    cl_hist = cl_merged.groupby(["policyholder_id", "anchor_month"]).agg(
        hist_claim_count=("claim_id", "count"),
        hist_claim_amount=("claim_amount_paid", "sum"),
    ).reset_index()

    # 최근 3개월
    cl_3m = cl_merged[
        cl_merged["service_date"] >= cl_merged["anchor_dt"] - pd.DateOffset(months=3)
    ].groupby(["policyholder_id", "anchor_month"]).agg(
        rcl_count_3m=("claim_id", "count"),
        rcl_amount_3m=("claim_amount_paid", "sum"),
        rcl_high_sev_3m=("high_severity_service_flag", "sum"),
    ).reset_index()

    # 최근 6개월
    cl_6m = cl_merged[
        cl_merged["service_date"] >= cl_merged["anchor_dt"] - pd.DateOffset(months=6)
    ].groupby(["policyholder_id", "anchor_month"]).agg(
        rcl_count_6m=("claim_id", "count"),
        rcl_amount_6m=("claim_amount_paid", "sum"),
    ).reset_index()

    result = outcome[["policyholder_id", "anchor_month"]].copy()
    for df in [cl_hist, cl_3m, cl_6m]:
        result = result.merge(df, on=["policyholder_id", "anchor_month"], how="left")

    result.drop(columns=["anchor_dt"], inplace=True, errors="ignore")
    result.fillna(0, inplace=True)
    return result


# ─── 도메인 4: 보험계약 (정적/반정적) ────────────────────────────────

def extract_policy(outcome_df):
    """계약자 단위로 보험계약 정보 집계"""
    pol = _load("core_policy").copy()

    # 계약자 단위 집계
    pol_agg = pol.groupby("policyholder_id").agg(
        n_policies=("policy_id", "count"),
        premium_sum=("premium_amount", "sum"),
        premium_mean=("premium_amount", "mean"),
        face_sum=("face_amount", "sum"),
        face_mean=("face_amount", "mean"),
        policy_month_max=("policy_month", "max"),
        renewable_count=("renewable_flag", "sum"),
        lapse_count=("lapse_flag", "sum"),
    ).reset_index()

    # 주력 상품/채널 (최빈값)
    mode_cols = pol.groupby("policyholder_id").agg(
        product_type_main=("product_type", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown"),
        coverage_type_main=("coverage_type", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown"),
        channel_main=("distribution_channel", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown"),
    ).reset_index()

    pol_feat = pol_agg.merge(mode_cols, on="policyholder_id", how="left")

    # Label Encoding
    for col in ["product_type_main", "coverage_type_main", "channel_main"]:
        le = LabelEncoder()
        pol_feat[col] = le.fit_transform(pol_feat[col].astype(str))

    # boolean → int
    for col in ["renewable_count", "lapse_count"]:
        pol_feat[col] = pol_feat[col].astype(int)

    result = outcome_df[["policyholder_id", "anchor_month"]].merge(
        pol_feat, on="policyholder_id", how="left"
    )
    result.fillna(0, inplace=True)
    return result


# ─── 도메인 5: 행동 (동적, MNAR 처리) ────────────────────────────────

def extract_behavior(outcome_df):
    """행동 관측 데이터를 앵커 월 이전 기준으로 집계 + MNAR 처리"""
    beh = _load("behavior_monthly_observation").copy()
    beh["observation_month"] = pd.to_datetime(beh["observation_month"])

    outcome = outcome_df[["policyholder_id", "anchor_month"]].copy()
    outcome["anchor_dt"] = pd.to_datetime(outcome["anchor_month"])

    merged = outcome.merge(beh, on="policyholder_id", how="left")
    merged = merged[merged["observation_month"] < merged["anchor_dt"]]

    # 수치형 컬럼 평균
    num_cols = ["sleep_irregularity_score", "mobility_change_score",
                "stress_proxy_score", "medication_adherence_score",
                "wellness_program_engagement_score", "app_active_days"]

    beh_agg = merged.groupby(["policyholder_id", "anchor_month"]).agg(
        avg_sleep_irregularity_score=("sleep_irregularity_score", "mean"),
        avg_mobility_change_score=("mobility_change_score", "mean"),
        avg_stress_proxy_score=("stress_proxy_score", "mean"),
        avg_medication_adherence_score=("medication_adherence_score", "mean"),
        avg_wellness_engagement=("wellness_program_engagement_score", "mean"),
        avg_app_active_days=("app_active_days", "mean"),
        wearable_source_flag=("wearable_source_flag", "max"),
        preventive_checkup_count=("preventive_checkup_flag", "sum"),
    ).reset_index()

    # 최근 3개월 스트레스/수면 변화량
    recent_3m = merged[
        merged["observation_month"] >= merged["anchor_dt"] - pd.DateOffset(months=3)
    ].groupby(["policyholder_id", "anchor_month"]).agg(
        stress_recent_3m=("stress_proxy_score", "mean"),
        sleep_recent_3m=("sleep_irregularity_score", "mean"),
        mobility_recent_3m=("mobility_change_score", "mean"),
    ).reset_index()

    result = outcome[["policyholder_id", "anchor_month"]].copy()
    result = result.merge(beh_agg, on=["policyholder_id", "anchor_month"], how="left")
    result = result.merge(recent_3m, on=["policyholder_id", "anchor_month"], how="left")

    result.drop(columns=["anchor_dt"], inplace=True, errors="ignore")

    # MNAR 처리: 결측 자체를 feature로
    result["has_behavior_data"] = result["avg_stress_proxy_score"].notna().astype(int)
    result["wearable_source_flag"] = result["wearable_source_flag"].fillna(False).astype(int)
    result.fillna(0, inplace=True)
    return result


# ─── 전체 파이프라인 ─────────────────────────────────────────────────

DOMAIN_EXTRACTORS = {
    "demographics_uw": extract_demographics_uw,
    "medical": extract_medical,
    "claim": extract_claim,
    "policy": extract_policy,
    "behavior": extract_behavior,
}

# 각 도메인에서 제외할 컬럼 (ID, 키 컬럼)
_KEY_COLS = {"policyholder_id", "anchor_month"}


def get_feature_columns(domain_df):
    """도메인 DataFrame에서 실제 feature 컬럼만 반환"""
    return [c for c in domain_df.columns if c not in _KEY_COLS]


def build_all_features(outcome_df, domains=None):
    """
    전체 feature 추출.
    반환: {도메인명: DataFrame}, target Series
    """
    if domains is None:
        domains = list(DOMAIN_EXTRACTORS.keys())

    features = {}
    for domain in domains:
        print(f"  [{domain}] feature 추출 중...")
        df = DOMAIN_EXTRACTORS[domain](outcome_df)
        features[domain] = df
        fcols = get_feature_columns(df)
        print(f"  [{domain}] {len(fcols)} features: {fcols[:5]}{'...' if len(fcols) > 5 else ''}")

    target = outcome_df.set_index(
        ["policyholder_id", "anchor_month"]
    )["high_cost_event_flag"].astype(int)

    return features, target


def temporal_split(outcome_df):
    """시간 기반 분할: Train 01-08, Valid 09-10, Test 11-12"""
    anchor = pd.to_datetime(outcome_df["anchor_month"])
    month = anchor.dt.month

    train_mask = month <= 8
    valid_mask = month.isin([9, 10])
    test_mask = month.isin([11, 12])

    return (
        outcome_df[train_mask].reset_index(drop=True),
        outcome_df[valid_mask].reset_index(drop=True),
        outcome_df[test_mask].reset_index(drop=True),
    )


def get_domain_arrays(features_dict, domain, outcome_df):
    """도메인 DataFrame → numpy array (feature만) 반환"""
    df = features_dict[domain]
    # outcome_df와 순서 맞추기
    merged = outcome_df[["policyholder_id", "anchor_month"]].merge(
        df, on=["policyholder_id", "anchor_month"], how="left"
    )
    fcols = get_feature_columns(merged)
    return merged[fcols].values.astype(np.float32)


# ─── 테스트 실행 ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Feature Pipeline 테스트 ===\n")

    outcome = pd.read_csv(DATA_DIR / "modeling_outcome_high_cost_event.csv")
    print(f"전체 데이터: {len(outcome):,}행\n")

    train, valid, test = temporal_split(outcome)
    print(f"Train: {len(train):,}  Valid: {len(valid):,}  Test: {len(test):,}\n")

    # 소규모 테스트 (Train의 앵커 월 1개만)
    sample = train[train["anchor_month"] == "2022-01-01"].head(1000)
    print(f"샘플 크기: {len(sample)}\n")

    features, target = build_all_features(sample)

    for domain, df in features.items():
        fcols = get_feature_columns(df)
        print(f"\n{domain}: {df.shape} → {len(fcols)} features")
        print(f"  결측률: {df[fcols].isnull().mean().mean():.4f}")
