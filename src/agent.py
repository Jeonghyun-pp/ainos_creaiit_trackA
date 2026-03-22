"""
DS Consultant Agent
- LLM API + Tool Use로 새 데이터 반영 방법을 조언하는 Agent
- 6개 Tool을 등록하고, 사용자 질문에 따라 자동으로 Tool을 호출
- API 선택: anthropic / openai / google (API 키만 바꾸면 됨)
"""

import json
import os

from agent_tools import (
    get_feature_registry,
    get_component_structure,
    analyze_correlation,
    check_collinearity,
    simulate_integration,
    check_regulatory_sensitivity,
)

# ─── 설정 ──────────────────────────────────────────────────────────
# 아래 두 줄만 바꾸면 다른 API로 전환 가능

API_PROVIDER = "openai"  # "anthropic" / "openai" / "google"
API_KEY = ""             # ← 여기에 API 키 입력, 또는 환경변수 사용

# 모델명
MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o-mini",
    "google": "gemini-2.0-flash",
}

# ─── 시스템 프롬프트 ───────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 보험사의 **DS(Data Science) Consultant Agent**입니다.

## 역할
새로운 데이터가 들어왔을 때, 기존 모듈형 리스크 모델에 어떻게 반영해야 할지 실무자에게 조언합니다.

## 배경
- 보험사는 **모듈형 Additive 리스크 모델**을 운영 중
- 5개 도메인 컴포넌트: 인구통계+UW, 의료이력, 청구, 보험계약, 행동
- 각 컴포넌트가 독립적으로 log-odds 점수를 출력 → 합산 → sigmoid → 최종 확률
- 타깃: high_cost_event_flag (12개월 내 고비용 의료 이벤트 발생 여부)

## 분석 절차 (반드시 순서대로)
1. **Feature 구조 파악**: get_feature_registry()와 get_component_structure()로 현재 상태 확인
2. **기존 모델 영향 진단**: analyze_correlation()으로 타깃과의 관계, check_collinearity()로 기존 feature와 중복 확인
3. **통합 경로 비교**: "기존 컴포넌트에 추가" vs "새 컴포넌트 생성" 판단
4. **규제 적합성 체크**: check_regulatory_sensitivity()로 민감도 분류
5. **실행 계획 생성**: 구체적 Step-by-step 계획 출력

## 답변 형식
- 한국어로 답변
- 각 단계에서 Tool을 호출하고, 결과를 근거로 판단
- 최종 답변에는 반드시 **구체적 실행 계획**을 포함
- 규제 검토를 절대 생략하지 않음
"""

# ─── Tool 정의 (공통 형식 → API별로 변환) ──────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "get_feature_registry",
        "description": "현재 등록된 feature 목록과 메타데이터(도메인, 타입, 갱신주기, 민감도 등)를 조회한다.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_component_structure",
        "description": "현재 모듈형 리스크 모델의 컴포넌트 구조를 조회한다. 각 컴포넌트의 이름, 타입, 포함 feature 목록을 반환한다.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "analyze_correlation",
        "description": "새 feature와 타깃 변수(high_cost_event_flag) 간의 상관관계를 분석한다. 상관계수, p-value, 양성/음성 그룹 비교를 반환한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "feature_values": {
                    "type": "array", "items": {"type": "number"},
                    "description": "새 feature의 값 배열",
                },
                "target_values": {
                    "type": "array", "items": {"type": "integer"},
                    "description": "타깃 변수 값 배열 (0 또는 1)",
                },
                "feature_name": {
                    "type": "string",
                    "description": "feature 이름",
                },
            },
            "required": ["feature_values", "target_values", "feature_name"],
        },
    },
    {
        "name": "check_collinearity",
        "description": "새 feature와 기존 feature 간의 공선성(정보 중복)을 확인한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "new_feature_values": {
                    "type": "array", "items": {"type": "number"},
                    "description": "새 feature의 값 배열",
                },
                "existing_feature_names": {
                    "type": "array", "items": {"type": "string"},
                    "description": "비교할 기존 feature 이름 목록",
                },
                "new_feature_name": {
                    "type": "string",
                    "description": "새 feature 이름",
                },
            },
            "required": ["new_feature_values", "existing_feature_names", "new_feature_name"],
        },
    },
    {
        "name": "simulate_integration",
        "description": "새 feature를 컴포넌트로 추가했을 때 모델 성능 변화를 시뮬레이션한다. 추가 전/후 AUC-ROC 비교를 반환한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "new_feature_values": {
                    "type": "array", "items": {"type": "number"},
                    "description": "새 feature의 값 배열 (test set)",
                },
                "target_values": {
                    "type": "array", "items": {"type": "integer"},
                    "description": "타깃 값 배열 (test set)",
                },
                "existing_scores": {
                    "type": "array", "items": {"type": "number"},
                    "description": "기존 컴포넌트들의 합산 log-odds 점수 (test set)",
                },
                "new_feature_name": {
                    "type": "string",
                    "description": "새 feature 이름",
                },
            },
            "required": ["new_feature_values", "target_values", "existing_scores", "new_feature_name"],
        },
    },
    {
        "name": "check_regulatory_sensitivity",
        "description": "새 feature의 규제 민감도를 분류한다. IFRS 17, 금융사지배구조법, AI 거버넌스 관점에서 체크리스트를 반환한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "feature_name": {"type": "string", "description": "feature 이름"},
                "data_type": {"type": "string", "description": "데이터 타입", "enum": ["numeric", "binary", "categorical", "text"]},
                "source_description": {"type": "string", "description": "데이터 출처 설명"},
                "contains_pii": {"type": "boolean", "description": "개인식별정보 포함 여부"},
            },
            "required": ["feature_name"],
        },
    },
]

# ─── Tool 실행 ─────────────────────────────────────────────────────

TOOL_FUNCTIONS = {
    "get_feature_registry": lambda **kw: get_feature_registry(),
    "get_component_structure": lambda **kw: get_component_structure(),
    "analyze_correlation": lambda **kw: analyze_correlation(
        kw["feature_values"], kw["target_values"], kw.get("feature_name", "new_feature")),
    "check_collinearity": lambda **kw: check_collinearity(
        kw["new_feature_values"], None, kw.get("new_feature_name", "new_feature")),
    "simulate_integration": lambda **kw: simulate_integration(
        kw["new_feature_values"], kw["target_values"],
        kw["existing_scores"], kw.get("new_feature_name", "new_feature")),
    "check_regulatory_sensitivity": lambda **kw: check_regulatory_sensitivity(
        kw["feature_name"], kw.get("data_type", "numeric"),
        kw.get("source_description", ""), kw.get("contains_pii", False)),
}


def execute_tool(tool_name, tool_input):
    """Tool 실행 → JSON 결과 반환"""
    try:
        result = TOOL_FUNCTIONS[tool_name](**tool_input)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ─── API별 클라이언트 ──────────────────────────────────────────────

def _get_api_key():
    if API_KEY:
        return API_KEY
    env_keys = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    return os.environ.get(env_keys.get(API_PROVIDER, ""), "")


def run_agent_anthropic(user_question):
    """Anthropic Claude API로 Agent 실행"""
    from anthropic import Anthropic

    client = Anthropic(api_key=_get_api_key())
    model = MODELS["anthropic"]

    # Tool 형식 변환 (Anthropic은 input_schema 사용)
    tools = [
        {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
        for t in TOOL_DEFINITIONS
    ]

    messages = [{"role": "user", "content": user_question}]

    while True:
        response = client.messages.create(
            model=model, max_tokens=4096, system=SYSTEM_PROMPT,
            tools=tools, messages=messages,
        )

        tool_calls = [b for b in response.content if b.type == "tool_use"]
        text_parts = [b.text for b in response.content if b.type == "text"]

        if text_parts:
            print("".join(text_parts))

        if response.stop_reason == "end_turn" or not tool_calls:
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tc in tool_calls:
            print(f"\n  [Tool] {tc.name}")
            result = execute_tool(tc.name, tc.input)
            print(f"  [결과] {result[:150]}...")
            tool_results.append({"type": "tool_result", "tool_use_id": tc.id, "content": result})
        messages.append({"role": "user", "content": tool_results})


def run_agent_openai(user_question):
    """OpenAI API로 Agent 실행"""
    from openai import OpenAI

    client = OpenAI(api_key=_get_api_key())
    model = MODELS["openai"]

    # Tool 형식 변환 (OpenAI는 function 사용)
    tools = [
        {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
        for t in TOOL_DEFINITIONS
    ]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]

    while True:
        response = client.chat.completions.create(
            model=model, messages=messages, tools=tools, tool_choice="auto",
        )
        msg = response.choices[0].message

        if msg.content:
            print(msg.content)

        if not msg.tool_calls:
            break

        messages.append(msg)
        for tc in msg.tool_calls:
            print(f"\n  [Tool] {tc.function.name}")
            args = json.loads(tc.function.arguments)
            result = execute_tool(tc.function.name, args)
            print(f"  [결과] {result[:150]}...")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})


def run_agent_google(user_question):
    """Google Gemini API로 Agent 실행"""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_get_api_key())
    model = MODELS["google"]

    # Tool 형식 변환
    function_declarations = []
    for t in TOOL_DEFINITIONS:
        props = {}
        required = t["parameters"].get("required", [])
        for pname, pdef in t["parameters"].get("properties", {}).items():
            schema_type = pdef.get("type", "STRING").upper()
            if schema_type == "ARRAY":
                props[pname] = types.Schema(type="ARRAY", items=types.Schema(type="NUMBER"))
            elif schema_type == "BOOLEAN":
                props[pname] = types.Schema(type="BOOLEAN")
            else:
                props[pname] = types.Schema(type="STRING")
        function_declarations.append(types.FunctionDeclaration(
            name=t["name"], description=t["description"],
            parameters=types.Schema(type="OBJECT", properties=props, required=required) if props else None,
        ))

    tools_gemini = types.Tool(function_declarations=function_declarations)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT, tools=[tools_gemini],
    )

    chat = client.chats.create(model=model, config=config)

    response = chat.send_message(user_question)

    # Gemini Tool 루프
    while response.candidates[0].content.parts:
        has_function_call = False
        for part in response.candidates[0].content.parts:
            if part.text:
                print(part.text)
            if part.function_call:
                has_function_call = True
                fc = part.function_call
                print(f"\n  [Tool] {fc.name}")
                args = dict(fc.args) if fc.args else {}
                result_str = execute_tool(fc.name, args)
                print(f"  [결과] {result_str[:150]}...")
                result_dict = json.loads(result_str)
                response = chat.send_message(
                    types.Part.from_function_response(name=fc.name, response=result_dict)
                )
        if not has_function_call:
            break


# ─── 통합 실행 함수 ────────────────────────────────────────────────

def run_agent(user_question):
    """설정된 API_PROVIDER에 따라 Agent 실행"""
    print(f"\n{'='*60}")
    print(f"  DS Consultant Agent")
    print(f"  API: {API_PROVIDER} ({MODELS.get(API_PROVIDER, '?')})")
    print(f"  질문: {user_question[:50]}...")
    print(f"{'='*60}\n")

    runners = {
        "anthropic": run_agent_anthropic,
        "openai": run_agent_openai,
        "google": run_agent_google,
    }

    runner = runners.get(API_PROVIDER)
    if not runner:
        print(f"지원하지 않는 API: {API_PROVIDER}")
        return

    key = _get_api_key()
    if not key:
        print(f"API 키가 설정되지 않았습니다.")
        print(f"방법 1: agent.py 상단의 API_KEY에 직접 입력")
        print(f"방법 2: 환경변수 설정")
        print(f"  - Anthropic: export ANTHROPIC_API_KEY=sk-...")
        print(f"  - OpenAI:    export OPENAI_API_KEY=sk-...")
        print(f"  - Google:    export GOOGLE_API_KEY=AI...")
        return

    runner(user_question)

    print(f"\n{'='*60}")
    print("  Agent 답변 완료")
    print(f"{'='*60}")


# ─── 평가 시나리오 ─────────────────────────────────────────────────

SCENARIOS = [
    "심박변이도(HRV) 데이터가 웨어러블 기기에서 월별로 수집됩니다. 이 데이터를 리스크 모델에 어떻게 반영할 수 있을까요?",
    "외부 신용평가사의 신용 점수 데이터를 리스크 모델에 추가하고 싶습니다. 어떻게 해야 하나요?",
    "건강검진 시 식습관 설문 결과(식사 규칙성, 채소 섭취량, 음주 빈도)를 활용하고 싶습니다. 방법을 알려주세요.",
]

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        run_agent(question)
    else:
        print("DS Consultant Agent")
        print(f"API: {API_PROVIDER} ({MODELS.get(API_PROVIDER, '?')})")
        print(f"Tool: {len(TOOL_DEFINITIONS)}개\n")

        print("테스트 시나리오:")
        for i, s in enumerate(SCENARIOS, 1):
            print(f"  {i}. {s}")

        print(f"\n사용법:")
        print(f"  python agent.py '질문 내용'")
        print(f"  또는 코드에서 run_agent('질문') 호출")
