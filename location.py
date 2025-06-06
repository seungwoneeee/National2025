# -*- coding: utf-8 -*-
"""
창업 위치 추천 및 상담 모듈 (유동인구 고려 버전)
---------------------------------------
1) 질문 분류는 LLM_CLASSIFY에 완전 위임
   - location : 위치 추천
   - timing   : 시기 추천
   - none     : 창업 무관한 질문
   - general  : 창업 관련 일반 질문
2) location/timing/none 은 정형화된 응답 고정
3) general 은 LLM_ANSWER_GENERAL로 자유응답
4) LLM (langchain) 환경 아닐 땐 intent_of() fallback
5) 유동인구 데이터를 활용하여 시간대별 업종 추천 강화
"""
from __future__ import annotations
import os, re, json
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from typing import Optional
load_dotenv()

# ────────────────── 데이터 로드 ──────────────────
DATA = Path(os.getenv("STARTUP_DATA_DIR", "/data"))
if not DATA.exists():
    DATA = Path(__file__).parent / "data"

def _load_csv(name, **kwargs):
    return pd.read_csv(DATA/ name, dtype=str, **kwargs)

_dev = _load_csv("dev_area_roads.csv").rename(
    columns={"CTY_NM":"구","RD_NM":"도로명주소","SG_FLG":"발달"}
)
_dev = (_dev[_dev["발달"]=="발달"]["구"].str.strip()
        .to_frame().join(_dev["도로명주소"]))

_store = _load_csv("store_counts_closure.csv").rename(
    columns={"CTY_NM":"구","RD_NM":"도로명주소","통합_업종":"업종"}
)
_store[["SG_CNT","SHUT_SG_CNT"]] = (
    _store[["SG_CNT","SHUT_SG_CNT"]]
    .astype(int)
)

_dense = _load_csv("industry_density.csv").rename(
    columns={"CTY_NM":"구","RD_NM":"도로명주소","GI":"DENSITY","통합_업종":"업종"}
)
_dense["DENSITY"] = _dense["DENSITY"].astype(float)

_reg = _load_csv("seoul_population.csv").rename(
    columns={"자치구":"구","행정동":"동","계":"인구","연령별":"연령"}
)
_reg["인구"] = (
    pd.to_numeric(_reg["인구"].str.replace(",","",regex=False),
                   errors="coerce").fillna(0).astype(int)
)
def intent_of(txt:str)->str:
    # 업종 키워드 확인을 먼저 수행
    category = cat_of(txt)
    
    # 업종 키워드가 있으면 무조건 location으로 처리
    if category != "기타":
        return "location"
    
    if LLM_INT:
        try:
            js = LLM.invoke(LLM_INT.format_prompt(query=txt)).content.strip()
            return next((i for i in ["location","timing","other"] if i in js),"other")
        except Exception:
            pass
    
    # 키워드 기반 intent 분류
    if re.search(r"지금|언제|시기|타이밍|후에|내년|개월",txt):
        return "timing"
    if re.search(r"어디|입지|위치|장소",txt) or re.search(r"차리|창업|가게|점포|매장",txt):
        return "location"
    
    return "other"
# ────────── 유동인구 데이터 로드 ──────────
try:
    _flow_weekday = _load_csv("mobility_weekday.csv")
    _flow_weekend = _load_csv("mobility_weekend.csv")
    
    # 컬럼명 표준화
    _flow_weekday = _flow_weekday.rename(columns={
        "EMD_NM": "동", "SGG_NM": "구",
        **{f"CNT_{i:02d}H": f"{i:02d}" for i in range(24)}
    })
    
    _flow_weekend = _flow_weekend.rename(columns={
        "EMD_NM": "동", "SGG_NM": "구",
        **{f"CNT_{i:02d}H": f"{i:02d}" for i in range(24)}
    })
    
    # 숫자 데이터로 변환
    for h in range(24):
        h_str = f"{h:02d}"
        _flow_weekday[h_str] = pd.to_numeric(_flow_weekday[h_str], errors='coerce').fillna(0).astype(int)
        _flow_weekend[h_str] = pd.to_numeric(_flow_weekend[h_str], errors='coerce').fillna(0).astype(int)
    
    HAVE_FLOW_DATA = True
except Exception as e:
    print(f"유동인구 데이터 로드 실패: {e}")
    HAVE_FLOW_DATA = False

# ─── 시기(타이밍) 키워드 셋 ─────────────────────────
TIME_KWS = [
    "언제", "시기", "몇 월", "몇월", "몇 년", "몇년",
    "올해", "내년", "분기", "상반기", "하반기",
    "초에", "말에", "시즌", "계절",
]

def is_timing_question(q:str) -> bool:
    return any(kw in q for kw in TIME_KWS)

# ─────────── 업종별 핵심 시간대 정의 ─────────────
# 평일 핵심 시간대 (시작 시간)
CORE_HOURS_WEEKDAY = {
    "음식업": [11, 12, 18, 19, 20],              # 점심 11-13시, 저녁 18-21시
    "관광여가오락": [],                          # 주말 중심
    "예술/스포츠": [18, 19, 20, 21],             # 18-22시
    "보건의료": [9, 10, 11, 12, 13, 14, 15, 16], # 09-17시
    "숙박": [20, 21, 22, 23],                    # 20-24시
    "부동산": [9, 10, 11, 12, 13, 14, 15, 16, 17], # 09-18시
    "소매": [17, 18, 19],                        # 17-20시
    "수리 및 개인 서비스": [9, 10, 11, 12, 13, 14, 15, 16, 17], # 09-18시
    "과학 및 기술": [9, 10, 11, 12, 13, 14, 15, 16, 17], # 09-18시 (직장인 시간)
    "교육": [15, 16, 17, 18, 19, 20, 21],        # 15-22시
    "시설 관리 및 임대": [9, 10, 11, 12, 13, 14, 15, 16, 17], # 09-18시
    "생활서비스": [9, 10, 11, 12, 13, 14, 15, 16, 17], # 09-18시
}

# 주말 핵심 시간대 (시작 시간)
CORE_HOURS_WEEKEND = {
    "음식업": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20], # 11-21시
    "관광여가오락": [13, 14, 15, 16, 17, 18, 19, 20, 21], # 13-22시
    "예술/스포츠": [13, 14, 15, 16, 17, 18, 19],         # 13-20시
    "보건의료": [],                                      # 주말 영향 낮음
    "숙박": [20, 21, 22, 23],                           # 20-24시
    "부동산": [],                                       # 주말 영향 낮음
    "소매": [12, 13, 14, 15, 16, 17, 18, 19],           # 12-20시
    "수리 및 개인 서비스": [],                           # 주말 영향 낮음
    "과학 및 기술": [],                                 # 주말 영향 낮음
    "교육": [9, 10, 11, 12, 13, 14, 15, 16, 17],        # 09-18시
    "시설 관리 및 임대": [9, 10, 11, 12, 13, 14, 15, 16, 17], # 09-18시
    "생활서비스": [9, 10, 11, 12, 13, 14, 15, 16, 17],  # 09-18시
}

def cat_of(q: str) -> str:
    """
    질문에서 추정한 업종을 반환.
    - 동의어·fuzzy 매칭 결과가 있으면 그걸,
    - 없으면 디폴트로 '음식업' 반환
    """
    key = contains_business_term(q)
    if key:
        return CAT_SYNONYM.get(key, key)   # 동의어 사전 매핑
    # 대분류가 직접 들어간 경우
    for cat in CATS:
        if cat in q:
            return cat
    return "기타" 

# ─────────── 연령 가중 ───────────
AGE_FOCUS = {
    "교육": {"5-9","10-14","15-19"},
    # 보건의료: 어르신 및 소아 모두 중요하므로 전체 연령대 사용
    "보건의료": set(_reg["연령"].unique())
}
ALL_BUCKET = set(_reg["연령"].unique())
for c in ["음식업","소매","부동산","수리 및 개인 서비스",
          "생활서비스","예술/스포츠","시설 관리 및 임대"]:
    AGE_FOCUS[c] = ALL_BUCKET
# 관광여가오락·숙박·과학기술은 가중 연령 미적용
for c in ["관광여가오락","숙박","과학 및 기술"]:
    AGE_FOCUS[c] = set()

def reg_pop(cat: str):
    foc = AGE_FOCUS.get(cat, set())
    if not foc:
        return _reg.groupby(["구","동"])["인구"].sum().reset_index().assign(가중인구=0)
    sub = _reg[_reg["연령"].isin(foc)]
    return sub.groupby(["구","동"])["인구"].sum().reset_index().rename(columns={"인구":"가중인구"})

# ────────── 유동인구 핵심 시간대 계산 함수 ──────────
def core_float(cat: str):
    """업종별 핵심 시간대의 유동인구를 계산하여 반환"""
    if not HAVE_FLOW_DATA:
        return pd.DataFrame(columns=["동", "CORE_FLOAT"])
    
    # 평일 핵심 시간대 합계
    weekday_hours = CORE_HOURS_WEEKDAY.get(cat, [])
    weekday_df = _flow_weekday.copy()
    if weekday_hours:
        weekday_df["WEEKDAY_FLOW"] = weekday_df[[f"{h:02d}" for h in weekday_hours]].sum(axis=1)
    else:
        weekday_df["WEEKDAY_FLOW"] = 0
    
    # 주말 핵심 시간대 합계
    weekend_hours = CORE_HOURS_WEEKEND.get(cat, [])
    weekend_df = _flow_weekend.copy()
    if weekend_hours:
        weekend_df["WEEKEND_FLOW"] = weekend_df[[f"{h:02d}" for h in weekend_hours]].sum(axis=1)
    else:
        weekend_df["WEEKEND_FLOW"] = 0
    
    # 평일과 주말 데이터 병합
    weekday_flow = weekday_df[["구", "동", "WEEKDAY_FLOW"]]
    weekend_flow = weekend_df[["구", "동", "WEEKEND_FLOW"]]
    
    merged_flow = weekday_flow.merge(weekend_flow, on=["구", "동"], how="outer")
    merged_flow = merged_flow.fillna(0)
    
    # 가중치 적용: 평일 5일 + 주말 2일 기준
    merged_flow["CORE_FLOAT"] = (merged_flow["WEEKDAY_FLOW"] * 5 + merged_flow["WEEKEND_FLOW"] * 2) / 7
    
    return merged_flow[["동", "CORE_FLOAT"]]

# ────────── 위치/동 추출 함수 ──────────
def extract_location(txt: str) -> Optional[str]:
    # 도로명(로, 길, 대로) 또는 역명 패턴 추출
    m = re.search(r'([가-힣0-9]+(?:로|길|대로|역))', txt)
    return m.group(1) if m else None

def extract_dong(txt: str) -> Optional[str]:
    # 행정동 패턴 추출
    m = re.search(r'([가-힣]+동)', txt)
    return m.group(1) if m else None

# ────────── 추천 로직 ──────────
POP_W = {"음식업":1,"관광여가오락":1,"소매":1,"숙박":1}
REG_W = {"보건의료":1,"교육":1,"부동산":1,
         "수리 및 개인 서비스":1,"생활서비스":1}
STORE_AVAILABLE = set(_store["업종"].unique())

def beautify(rd: str) -> str:
    """'월드컵북로15길' → '월드컵북로 15길' 처럼 띄어쓰기"""
    return re.sub(r"로(\d+길$)", r"로 \1", rd)

def recommend(gu: str, q: str,
              dev_override: pd.DataFrame = None,
              cat_override: str = None,
              dong: str | None = None): 
    cat = cat_override or cat_of(q)
    dev = _dev[_dev["구"] == gu]
    dev = (
        dev_override
        if dev_override is not None and not dev_override.empty
        else _dev[_dev["구"] == gu]
    )
    if dev.empty:
        return [], [], cat, {}
    if cat in STORE_AVAILABLE:
        st = _store[( _store["구"]==gu)
                    &( _store["업종"]==cat)
                    &( _store["도로명주소"].isin(dev["도로명주소"]))]
        st = st[st["SG_CNT"]>st["SHUT_SG_CNT"]]
    else:
        st = dev.assign(업종=cat, SG_CNT=0, SHUT_SG_CNT=0)

    if dong:
        st = st[st["동"] == dong]
        if st.empty:
            return [], {"high": [], "low": []}, cat, {}
    
    if st.empty:
        return [], [], cat, {}
    analysis = {
        "업종":cat,
        "총_도로수":len(dev),
        "조건부_도로수":len(st),
        "평균_점포수":st["SG_CNT"].mean(),
        "평균_폐업수":st["SHUT_SG_CNT"].mean()
    }
    st["동"] = st["도로명주소"].str.extract(r"(.+동)")
    
    # 등록 인구 데이터 추가
    rd = reg_pop(cat)
    st = st.merge(rd, on=["구","동"], how="left")
    analysis["평균_가중인구"] = rd["가중인구"].mean()
    
    # 유동 인구 데이터 추가
    flow_data = core_float(cat)
    st = st.merge(flow_data, on="동", how="left")
    
    # 유동인구 평균 계산하여 분석 정보에 추가
    if not flow_data.empty and "CORE_FLOAT" in flow_data.columns:
        analysis["평균_핵심유동인구"] = flow_data["CORE_FLOAT"].mean()
    
    if cat in set(_dense["업종"].unique()):
        dd = _dense[( _dense["구"]==gu)
                   &( _dense["업종"]==cat)][["도로명주소","DENSITY"]]
        st = st.merge(dd, on="도로명주소", how="left")
        analysis["평균_밀집도"] = dd["DENSITY"].mean()
    else:
        st["DENSITY"] = 0
    # 결측값 채우기
    st[["CORE_FLOAT","가중인구","DENSITY"]] = \
        st[["CORE_FLOAT","가중인구","DENSITY"]].fillna(0)
    # 점수 계산 & 중복 제거
    st = (
        st.assign(
            # ① road_type: 길(2) > 로(1) > 기타(0)
            road_type=lambda df: (
                df["도로명주소"].str.contains(r"\d+길$").astype(int) * 2
                + df["도로명주소"].str.endswith("로").astype(int)
            ),
            score=lambda df: (
                (df["SG_CNT"] - df["SHUT_SG_CNT"]) * 0.4
                + df["CORE_FLOAT"] * POP_W.get(cat, 0.3) * 1e-3
                + df["가중인구"] * REG_W.get(cat, 0.3) * 1e-4
            )
        )
        .sort_values(["score", "road_type"], ascending=[False, False])
        )
    st["prefix"] = st["도로명주소"].str.replace(r"\d+길$", "로", regex=True)
    st = (
        st.sort_values(["prefix", "road_type"], ascending=[True, False])
          .drop_duplicates(subset="prefix")
    )

    # ③ beautify → 사람이 읽기 편한 표기
    st["pretty_rd"] = st["도로명주소"].map(beautify)

    top5 = st.head(5).to_dict("records")

    # 밀집도별 추가 추천
    if "DENSITY" in st.columns and len(st) > 5:
        high = (st.sort_values("DENSITY", ascending=False)
                  .head(3).to_dict("records"))
        low  = (st.sort_values("DENSITY")
                  .head(3).to_dict("records"))
        dens_dict = {"high": high, "low": low}
    else:
        dens_dict = {"high": [], "low": []}

    return top5, dens_dict, cat, analysis

# ────────── LLM 프롬프트 ────────── ──────────
CATS = ["음식업","예술/스포츠","관광여가오락","보건의료",
        "숙박","부동산","소매","수리 및 개인 서비스",
        "과학 및 기술","교육","시설 관리 및 임대",
        "생활서비스"]
CAT_SYNONYM = {
    "카페": "음식업", "커피": "음식업", "브런치": "음식업",
    "편의점": "소매",  "마트": "소매",
    "꽃집": "소매",    "플라워": "소매",
    "펫카페": "관광여가오락",
    "학원": "교육",      # 어학·보습 등 포괄
    "어학원": "교육",
    "영어학원": "교육",
    "펜션":"숙박"
}
from rapidfuzz import process, fuzz

ALL_KEYS = list(CATS) + list(CAT_SYNONYM.keys())

def contains_business_term(q: str, thresh: int = 80):
    q = q.lower()
    # (a) 동의어 사전에 substring 이면 바로 반환
    for kw in CAT_SYNONYM:          # "영어학원" → "학원" 매칭
        if kw in q:
            return kw
    # (b) 대분류 이름 그대로 들어 있으면
    for cat in CATS:
        if cat.lower() in q:
            return cat
    # (c) 그 외 fuzzy 매칭 (오타‧복합어 대응)
    term, score, _ = process.extractOne(q, ALL_KEYS, scorer=fuzz.partial_ratio)
    return term if score >= thresh else None

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    LLM = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    LLM_CLASSIFY = ChatPromptTemplate.from_messages([
        ("system","""
당신은 '대감이' 창업 챗봇입니다.
질문을 다음 중 하나로 분류하여 JSON으로 출력하세요:
- location : 창업 위치 추천 요청
- timing   : 창업 시기 추천 요청
- general  : 창업 관련 일반 질문
- none     : 창업과 무관한 질문
{{"type":"..."}}
"""),
        ("user","질문: {query}")
    ])
    LLM_ANSWER_GENERAL = ChatPromptTemplate.from_messages([
        ("system","""
당신은 '대감이' 창업 전문가입니다.
창업 관련 일반 질문에 대해 친절하고 전문적으로 답변하세요.
"""),
        ("user","질문: {query}")
    ])
except:
    LLM = LLM_CLASSIFY = LLM_ANSWER_GENERAL = None

# ────────── 가이드 텍스트 ──────────
HELLO        = "안녕하세요, 저는 대감이입니다!\n\n"
TIMING_GUIDE = HELLO + "창업 시기 추천 서비스를 선택해주세요."
OTHER_GUIDE  = HELLO + "죄송합니다. 대감이는 창업 시기 및 위치에 대해서만 답변 드릴 수 있습니다."

# ────────── 추천 포맷 ──────────
def _fmt(rows):
    return "\n".join(
        f"- {r['도로명주소']} (점포: {r.get('SG_CNT',0)}, 폐업: {r.get('SHUT_SG_CNT',0)}, 밀집도: {r.get('DENSITY',0):.2f})"
        for r in rows
    )

# ────────── 분류 및 응답 생성 ──────────
def answer(space: str, query: str) -> str:
    # ────────── 세부 위치 필터링 ──────────
    sub_loc = extract_location(query)  # 도로명·길·역명 추출
    sub_dong = None
    try:
        sub_dong = extract_dong(query)  # 행정동 추출
    except NameError:
        pass
    dev_override = None
    if sub_dong:
        dev_override = _dev[_dev["도로명주소"].str.contains(sub_dong)]
    elif sub_loc:
        dev_override = _dev[_dev["도로명주소"].str.contains(sub_loc)]

    # 0) DEBUG: 찍어보기
    print(f"DEBUG ▶ query: {query!r}")
    # 예외 처리: '길' 키워드가 들어가면 무조건 위치 추천
    if is_timing_question(query):
        qtype = "timing"
        forced_cat = None          # timing 에선 업종 강제 주입 안 함

    else:
        # 2️⃣ 업종(동의어·fuzzy) → 위치 추천
        biz_key    = contains_business_term(query)
        forced_cat = CAT_SYNONYM.get(biz_key, biz_key) if biz_key else None
        if forced_cat:
            qtype = "location"

        # 3️⃣ '길' 키워드 → 위치 추천
        elif "길" in query:
            qtype = "location"
            forced_cat = None
        else:
            # 1) LLM_CLASSIFY
            qtype = None
            if LLM and LLM_CLASSIFY:
                js = LLM.invoke(LLM_CLASSIFY.format_prompt(query=query)).content.strip()
                try:
                    qtype = json.loads(js).get('type')
                except:
                    qtype = None
        # fallback
        if not qtype:
            fb = intent_of(query)
            qtype = 'none' if fb=='other' else fb
            if qtype not in ('location','timing','none'):
                qtype = 'general'
    # 2) 분기
    if qtype == 'location':
        top5, density_data, cat, analysis = recommend(
            space, query,
            dev_override=dev_override,
            cat_override=forced_cat,
            dong=sub_dong
        )

        # ★ 추가: 기타 업종은 분석 건너뛰고 바로 리턴
        if cat == "기타":
            return HELLO + "입력하신 업종을 찾지 못했습니다. 보다 정확한 추천을 위해 업종을 구체적으로 입력해 주세요.\n"

        # 이후 분석 로직 진행...
        lines = [HELLO, '']

        lines += [
            f"[{cat}] 업종 분석\n\n",
            '',
            f"{space}의 발달상권을 기준으로 분석했습니다.\n\n"
        ]

        # 등록 인구 설명
        if cat in REG_W:
            # 보건의료 세부 업종에 따른 연령대 분기
            if cat == "보건의료":
                if "소아" in query:
                    age_desc = "0-4, 5-9"  # 소아 주요 연령대
                    lines.append('')
                    lines.append(f"소아과의 경우 0-9세 어린이 인구가 중요합니다. {space}의 {age_desc} 인구를 분석했습니다.\n\n")
                elif any(x in query for x in ["이비인후과", "내과"]):
                    age_desc = "0-99"  # 전 연령층 고려
                    lines.append('')
                    lines.append(f"전 연령층 대상 진료인 경우 0-99세 전체 인구를 분석했습니다.\n\n")
                else:
                    age_desc = ", ".join(sorted(AGE_FOCUS.get(cat, [])))
                    lines.append('')
                    lines.append(f"노인 대상 의료업종이라면 주로 고령층 인구가 중요합니다. {space}의 {age_desc} 인구를 분석했습니다.\n\n")
            else:
                buckets = AGE_FOCUS.get(cat, set())
                age_desc = ", ".join(sorted(buckets)) if buckets else ""
                lines.append('')
                lines.append(f"{cat} 업종은 등록 인구가 중요합니다. {space}의 {age_desc} 인구를 분석했습니다.\n\n")

        # 유동인구 설명 추가
        if POP_W.get(cat, 0) > 0 and "평균_핵심유동인구" in analysis:
            lines.append('')
            weekday_hours = CORE_HOURS_WEEKDAY.get(cat, [])
            weekend_hours = CORE_HOURS_WEEKEND.get(cat, [])
            
            weekday_time = ""
            if weekday_hours:
                weekday_time = f"평일 {min(weekday_hours):02d}-{max(weekday_hours)+1:02d}시"
            
            weekend_time = ""
            if weekend_hours:
                weekend_time = f"주말 {min(weekend_hours):02d}-{max(weekend_hours)+1:02d}시"
            
            time_desc = ""
            if weekday_time and weekend_time:
                time_desc = f"{weekday_time}와 {weekend_time}"
            elif weekday_time:
                time_desc = weekday_time
            elif weekend_time:
                time_desc = weekend_time
            
            if time_desc:
                lines.append(f"{cat} 업종은 {time_desc}의 유동인구가 중요합니다. {space}의 해당 시간대 유동인구를 분석했습니다.\n\n")
            else:
                lines.append(f"{cat} 업종은 유동인구가 중요합니다. {space}의 유동인구를 분석했습니다.\n\n")

    # 밀집도 설명
        if "평균_밀집도" in analysis:
            avg_d = analysis.get("평균_밀집도", 0)
            if avg_d < -0.3:
                desc = "매우 낮은"
            elif avg_d < 0:
                desc = "낮은"
            elif avg_d < 0.3:
                desc = "보통 수준의"
            else:
                desc = "높은"
            lines.append('')
            lines.append(f"{space}의 {cat} 평균 밀집도는 {avg_d:.2f}입니다. (이 지역은 {desc} 업종 밀집도를 보입니다.)\n")
            lines.append("- 밀집도가 낮은 곳: 동일 업종 경쟁이 적고 틈새시장 공략에 유리합니다.\n")
            lines.append("- 밀집도가 높은 곳: 이미 상권이 형성되어 있어 고객 유입은 쉬우나 경쟁이 치열합니다.\n\n")
        # 추천 상권 리스트
        lines.append('')
        lines.append(f"분석 결과, {space}에서 {cat} 업종에 적합한 추천 상권은 다음과 같습니다:\n\n")
        lines.append('')
        lines.append("▶ 추천 상권 Top5 (종합 점수 기준)\n")
        for r in top5:
            lines.append(f"- {r['pretty_rd']} (현재 점포 수: {r.get('SG_CNT',0)}, 폐업 점포 수: {r.get('SHUT_SG_CNT',0)}, 밀집도: {r.get('DENSITY',0):.2f})\n")
        # 최종 반환: 구성한 lines를 줄바꿈으로 연결
        lines.append("\n\n대감이의 의견은 참고용입니다. 좋은 창업 결과가 있길 바랍니다.")
        return "".join(lines)
    if qtype == 'timing':
        return TIMING_GUIDE
    if qtype == 'none':
        return OTHER_GUIDE
    # general
    if qtype == 'general' and LLM and LLM_ANSWER_GENERAL:
        return LLM.invoke(LLM_ANSWER_GENERAL.format_prompt(query=query)).content.strip()
    # fallback general 안내
    return HELLO + "창업 관련 어떤 질문이든 물어보세요!"