"""
RiskLab DB 탐색기
- 모든 핵심 테이블을 조회하여 CSV로 저장
- Jupyter/Colab에서도 사용 가능
"""

import psycopg2
import pandas as pd
import os

# DB 접속
DB_URL = ""
conn = psycopg2.connect(DB_URL)

# 저장 폴더
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 핵심 테이블 목록
tables = {
    # core
    "core.policyholder": "SELECT * FROM core.policyholder",
    "core.policy": "SELECT * FROM core.policy",
    "core.underwriting_assessment": "SELECT * FROM core.underwriting_assessment",
    # medical
    "medical.diagnosis_event": "SELECT * FROM medical.diagnosis_event",
    "medical.hospitalization_event": "SELECT * FROM medical.hospitalization_event",
    "medical.claim_event": "SELECT * FROM medical.claim_event",
    # behavior
    "behavior.monthly_observation": "SELECT * FROM behavior.monthly_observation",
    # modeling
    "modeling.outcome_high_cost_event": "SELECT * FROM modeling.outcome_high_cost_event",
    "modeling.feature_registry": "SELECT * FROM modeling.feature_registry",
    "modeling.component_registry": "SELECT * FROM modeling.component_registry",
    "modeling.target_definition": "SELECT * FROM modeling.target_definition",
}

for name, query in tables.items():
    print(f"다운로드 중: {name}...", end=" ")
    df = pd.read_sql(query, conn)
    filename = name.replace(".", "_") + ".csv"
    df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
    print(f"완료 ({len(df):,}행)")

conn.close()
print(f"\n모든 테이블이 '{OUTPUT_DIR}/' 폴더에 CSV로 저장되었습니다.")
