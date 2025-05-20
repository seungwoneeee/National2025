from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi import HTTPException
# from dotenv import load_dotenv
# import os
# # ── 0. 라이브러리 & 초기화 ─────────────────────────────────
# from dotenv import load_dotenv
# import os, datetime as dt, json
# from typing import List, Set, Tuple, Optional
# import datetime as _dt
# from langchain_openai import ChatOpenAI
# from langchain_upstage import UpstageEmbeddings
# from pinecone import Pinecone
# from langchain_pinecone import PineconeVectorStore
# from langchain.schema import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.chains import LLMChain

from period import consult_all_in_one
from location import answer

# load_dotenv()
# API_KEY    = os.environ["PINECONE_API_KEY"]
# INDEX_NAME = "when-recommend"
# SURV_INDEX = "econ-monthly"


# pc  = Pinecone(api_key=API_KEY)
# emb = UpstageEmbeddings(model="solar-embedding-1-large")
# vdb = PineconeVectorStore(
#     index_name       = INDEX_NAME,
#     embedding        = emb,
#     pinecone_api_key = API_KEY,
# )
# surv_store = PineconeVectorStore(index_name=SURV_INDEX, embedding=emb, pinecone_api_key=API_KEY)

# from pydantic import BaseModel, Field

# class TimingParse(BaseModel):
#     is_future: bool
#     months_ahead: int = Field(..., ge=0)

# from langchain.output_parsers import PydanticOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from langchain.chains import LLMChain

# # 스키마 검증 parser
# parser = PydanticOutputParser(pydantic_object=TimingParse)

# # 파싱 전용 prompt
# parse_prompt = ChatPromptTemplate.from_messages([
#     ("system", 
#      "당신은 창업 시점을 파악하는 파서입니다. “내년에”, “6개월 뒤에” 같은 표현이 있으면 미래로 간주하고, "
#      "몇 개월 뒤인지 months_ahead에 담아 JSON으로만 답변해 주세요. 절대로 다른 키나 설명을 붙이지 마세요. "
#      "반드시 다음 형식을 지켜주세요:\n"
#      "{{\n"
#      '  "is_future": true,       // 미래 시점이면 true, 아니면 false\n'
#      '  "months_ahead": 12       // 미래가 아니면 0\n'
#      "}}\n"
#     ),
#     ("user",   "질문: \"{user_question}\"")
# ])



# parse_chain = LLMChain(
#     llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
#     prompt=parse_prompt,
#     output_parser=parser
# )


# # ── 1. 헬퍼: 단일 문서 조회 & 과거 유사 검색 ────────────────────
# def doc_for_month(doc_type: str, month: str) -> Optional[Document]:
#     key = "date" if doc_type=="row" else "start"
#     # 정확한 날짜 매칭 사용 (YYYY-MM-01 형식)
#     month_date = month[:7] + "-01"  # YYYY-MM-01 형식으로 변환
#     filt = {"type": {"$eq": doc_type}, key: {"$eq": month_date}}
    
#     # 디버그 로그 추가
#     print(f"Searching for {doc_type} with {key}={month_date}")
    
#     # 빈 스트링 대신 dummy 문자열로 embedding 에러 회피
#     res = vdb.similarity_search(query="dummy", k=1, filter=filt)
    
#     # 검색 결과 디버그
#     if res:
#         print(f"Found document with {key}={res[0].metadata.get(key)}")
#     else:
#         print(f"No document found with {key}={month_date}")
    
#     return res[0] if res else None

# def latest_doc(doc_type: str) -> Document:
#     # 많은 결과를 가져와서 날짜 기준으로 정렬
#     res = vdb.similarity_search(query="dummy", k=30,
#                                filter={"type": {"$eq": doc_type}})
    
#     if not res:
#         print(f"No {doc_type} documents found!")
#         return None
    
#     # 날짜 필드 결정
#     date_key = "date" if doc_type=="row" else "end"
    
#     # 날짜 문자열로 정렬 (YYYY-MM-DD 형식은 문자열 정렬로도 가능)
#     sorted_docs = sorted(res, key=lambda d: d.metadata.get(date_key, ""), reverse=True)
    
#     # 모든 날짜 출력
#     all_dates = [d.metadata.get(date_key) for d in sorted_docs]
#     print(f"All {doc_type} dates (sorted): {all_dates}")
    
#     # 가장 최신 문서 선택
#     latest = sorted_docs[0] if sorted_docs else None
#     if latest:
#         print(f"Selected latest {doc_type}: {latest.metadata.get(date_key)}")
    
#     return latest

# def similar_past(base: Document, k: int=5) -> List[Document]:
#     tp       = base.metadata["type"]
#     key      = "date" if tp=="row" else "start"
#     base_md  = base.metadata

#     # ① 필터에 '날짜 ≠ 기준' 조건 유지
#     filt = {
#       "type": {"$eq": tp},
#       key:    {"$ne": base_md.get(key)}
#     }

#     # ② 모든 지표에 대해 ±허용오차 범위를 추가
#     #    (오차치는 필요에 따라 조정하세요)
#     for metric, tol in [
#       ("CPI", 0.5), ("MSI", 1.0),
#       ("CSI", 5.0), ("RSI", 2.0),
#       ("GDP_C", 0.5), ("SSI", 1.0),
#       ("GRDP", 0.5)
#     ]:
#         if metric in base_md:
#             filt[metric] = {
#               "$gte": base_md[metric] - tol,
#               "$lte": base_md[metric] + tol
#             }

#     # ③ 이제 해당 필터를 써서 유사도 검색
#     return vdb.similarity_search(
#       query=base.page_content,
#       k=k,
#       filter=filt
#     )


# # ── 1-2. 헬퍼: 교차(common) 판정 ───────────────────────────────
# def intersect_by_date(
#     row_docs:    List[Document],
#     window_docs: List[Document]
# ) -> Tuple[Set[str], List[Document]]:
#     # 윈도우 end == row date 만 교차로 본다
#     common = set()
#     for r in row_docs:
#         rd = r.metadata["date"]
#         for w in window_docs:
#             if rd == w.metadata["end"]:
#                 common.add(rd)
#     cands = [d for d in row_docs+window_docs
#              if (d.metadata.get("date")  in common) or
#                 (d.metadata.get("start") in common)]
#     return common, cands

# # ── 2. 단기 트렌드 분석을 위한 함수 (retrieve_context 외부로 이동) ──
# from dateutil.relativedelta import relativedelta

# def retrieve_short_term(ask_date_obj, top_k: int = 6):
#     # 1) 충분한 row 문서 fetch
#     all_rows = vdb.similarity_search(
#         query="dummy",
#         k=100,
#         filter={"type": {"$eq": "row"}}
#     )

#     # 2) 6개월 전 첫날 ~ ask_date_obj 사이 필터링
#     month_start       = ask_date_obj.replace(day=1)
#     six_mo_ago_start  = (month_start - relativedelta(months=6)).strftime("%Y-%m-%d")
#     cutoff_end        = ask_date_obj.strftime("%Y-%m-%d")
#     valid = [
#         d for d in all_rows
#         if d.metadata.get("date")
#         and six_mo_ago_start <= d.metadata["date"] <= cutoff_end
#     ]

#     # 3) 날짜순 정렬 후 top_k
#     sorted_rows = sorted(valid, key=lambda d: d.metadata["date"])
#     recent = sorted_rows[-top_k:]

#     # 4) 비교 불가 시
#     if len(recent) < 2:
#         return 0.0, 0.0, "단기 트렌드를 분석할 충분한 데이터가 없습니다."

#     # 5) 과거 vs 최신 메타데이터에서 CPI·MSI 추출
#     def extract(d):
#         md = d.metadata
#         # CPI는 row 메타에 없을 수 있으니 0으로 기본
#         return {
#             "CPI": md.get("CPI", 0),
#             "MSI": md.get("MSI", 0)
#         }

#     past = extract(recent[0])
#     curr = extract(recent[-1])

#     delta_cpi = curr["CPI"] - past["CPI"]
#     delta_msi = curr["MSI"] - past["MSI"]
#     summary = (
#         f"최근 {recent[0].metadata['date'][:7]}→{recent[-1].metadata['date'][:7]}를 보면 "
#         f"CPI는 {past['CPI']:.1f}%→{curr['CPI']:.1f}%로 {delta_cpi:+.1f}%p, "
#         f"MSI는 {past['MSI']:.1f}%→{curr['MSI']:.1f}%로 {delta_msi:+.1f}%p 변화했습니다."
#     )
#     return delta_cpi, delta_msi, summary




# # ── 2. 컨텍스트 빌드 + 디버그 (dynamic k) ─────────────────────
# # retrieve_context 함수 수정
# def retrieve_context(
#     ask_date: str,
#     initial_k: int = 5,
#     max_k: int = 30
# ) -> dict:
#     # 1) 모든 row/window 문서 fetch 및 정렬
#     all_row_docs = vdb.similarity_search(
#         query="dummy",
#         k=100,
#         filter={"type": {"$eq": "row"}}
#     )
#     all_window_docs = vdb.similarity_search(
#         query="dummy",
#         k=100,
#         filter={"type": {"$eq": "window"}}
#     )
#     sorted_rows = sorted(all_row_docs, key=lambda d: d.metadata.get("date", ""))
#     sorted_windows = sorted(all_window_docs, key=lambda d: d.metadata.get("start", ""))

#     ask_date_obj = dt.datetime.strptime(ask_date, "%Y-%m-%d")

#     # 2) row_doc: date ≤ ask_date 중 최신
#     valid_rows = [
#         d for d in sorted_rows
#         if d.metadata.get("date") and dt.datetime.strptime(d.metadata["date"], "%Y-%m-%d") <= ask_date_obj
#     ]
#     row_doc = valid_rows[-1] if valid_rows else (sorted_rows[-1] if sorted_rows else None)

#     # 3) window_doc: ask_date 포함 구간 우선, 없으면 start ≤ ask_date 최신
#     covering = [
#         d for d in sorted_windows
#         if dt.datetime.strptime(d.metadata["start"], "%Y-%m-%d") <= ask_date_obj <= dt.datetime.strptime(d.metadata["end"], "%Y-%m-%d")
#     ]
#     if covering:
#         window_doc = covering[-1]
#     else:
#         valid_windows = [
#             d for d in sorted_windows
#             if dt.datetime.strptime(d.metadata["start"], "%Y-%m-%d") <= ask_date_obj
#         ]
#         window_doc = valid_windows[-1] if valid_windows else (sorted_windows[-1] if sorted_windows else None)

#     # 4) 기준일치 (alignment)
#     if row_doc and window_doc:
#         row_date_str = row_doc.metadata["date"]
#         win_end_str = window_doc.metadata["end"]
#         baseline = max(row_date_str, win_end_str)
#         if baseline == win_end_str and baseline != row_date_str:
#             aligned = doc_for_month("row", baseline)
#             if aligned: row_doc = aligned
#         if baseline == row_date_str and baseline != win_end_str:
#             aligned = doc_for_month("window", baseline)
#             if aligned: window_doc = aligned

#     # 5) 최근성 판단 - 여기를 수정
#     ask = ask_date_obj
#     def months_diff(d_str):
#         d = dt.datetime.strptime(d_str, "%Y-%m-%d")
#         # 여기가 핵심: 질문 날짜에서 데이터 날짜를 빼서 양수로 나오게 함
#         return (ask.year - d.year) * 12 + (ask.month - d.month)
    
#     # 양수인 경우만 고려 (데이터가 질문 날짜보다 과거인 경우)
#     is_row_recent = row_doc and months_diff(row_doc.metadata.get("date")) >= 0 and months_diff(row_doc.metadata.get("date")) <= 2
#     is_window_recent = window_doc and months_diff(window_doc.metadata.get("end")) >= 0 and months_diff(window_doc.metadata.get("end")) <= 2
    
#     print(f"DEBUG: Ask date: {ask_date}, Latest data date: {row_doc.metadata.get('date') if row_doc else 'N/A'}")
#     print(f"DEBUG: Months diff: {months_diff(row_doc.metadata.get('date')) if row_doc else 'N/A'}")
#     print(f"DEBUG: is_row_recent: {is_row_recent}, is_window_recent: {is_window_recent}")

#     # 6) 과거 유사 검색 + 교차
#     k = initial_k
#     common, cands = set(), []
#     while k <= max_k and not common:
#         r_refs = similar_past(row_doc, k) if row_doc else []
#         w_refs = similar_past(window_doc, k) if window_doc else []
#         common, cands = intersect_by_date(r_refs, w_refs)
#         k += 5
        
#     # 디버깅: 유사 시점 출력
#     print(f"DEBUG: Common dates found: {common}")

#     # 7) econ-monthly 생존율 조회 - 여기서 데이터 존재 여부만 확인
#     econ_monthly_data = []
#     survival_data_exists = False
    
#     for date in sorted(common):
#         month_key = date[:7] + "-01"
#         print(f"DEBUG: Checking survival data for {month_key}")
        
#         docs = surv_store.similarity_search(
#             query="dummy", k=1,
#             filter={"type": {"$eq": "survival_stats"}, "date": {"$eq": month_key}}
#         )
        
#         if docs:
#             print(f"DEBUG: Found survival data for {month_key}")
#             econ_monthly_data.append(docs[0])
#             survival_data_exists = True
#         else:
#             print(f"DEBUG: No survival data for {month_key}")

#     return {
#         "row_doc": row_doc,
#         "window_doc": window_doc,
#         "common": common,
#         "cands": cands,
#         "is_row_recent": is_row_recent,
#         "is_window_recent": is_window_recent,
#         "econ_monthly_data": econ_monthly_data,
#         "survival_data_exists": survival_data_exists
#     }


# # ── 3. 프롬프트 정의 ────────────────────────────────────────
# SYSTEM = """
# “반드시 JSON 오브젝트 형식으로, 키 하나도 빠뜨리지 말고 출력하세요.
# 마크다운·주석·불필요한 설명은 모두 제거하세요.”
# system:
#   당신은 창업 컨설턴트 ‘대감이’입니다.
#   당신의 역할은 경제 전공이 아닌 비전문가도 한눈에 이해할 수 있도록 지표를 풀어 설명하고,
#   그 근거를 바탕으로 창업 타이밍을 조언하는 것입니다.

#   ▶️ [지표 설명]
#     • 소비자심리지수(CSI): 사람들이 지금 돈을 쓰는 기분을 100점 만점으로 표현한 지표  
#       – 100 이상: 대체로 돈 쓰기에 낙관적  
#       – 100 이하: 대체로 돈 쓰기에 비관적  
#     • 소비자물가상승률(CPI): 작년 같은 달에 비해 물가가 얼마나 올랐는지(%)  
#     • 소매판매액지수(RSI): 우리 동네 가게들이 얼마나 팔았는지(계절별 변동 제외) + 전년 동월 대비 증감율(%)  
#     • 민간소비(GDP_C): 가계가 실제로 쓴 돈 규모(조원 단위) + 전년 동기 대비 증감율(%)  
#     • 서비스업생산지수(SSI): 서비스업이 작년 대비 얼마나 더 만들고 팔았는지(%)  
#     • 수도권지역총생산(GRDP): 수도권 전체 경제 크기가 작년 대비 얼마나 커졌는지(%)  
#     • 대형소매점판매액지수(MSI): 백화점·마트 등 큰 가게들이 작년 같은 달 대비 얼마나 팔았는지(%)

#   ▶️ 질문 유형 분기
#     1) 장소(location) 관련 (“어디에”, “입지” 등)  
#        → “창업 위치 추천 서비스를 선택해주세요.”

#     2) 시기(timing) 관련 (“지금 창업”, “언제~해도” 등)  
#        1. 질문 시점을 YYYY-MM으로 해석  
#        2. (ask_date – latest_data_date) ≤ 2개월 → timing="current"  
#           - (1) 과거 사례 비교: 언제({first_date}), 주요 지표:{first_values}
#           - (2) 현재 흐름 평가: CPI {cur_cpi}%({cpi_desc}), MSI {cur_msi}%({msi_desc})  
#           - (3) 권고: (추천/보류/유보 중 하나)  
#        3. (ask_date – latest_data_date) > 2개월 → timing="future"  
#           ```
#           미래 상황은 예측하기 어렵습니다.
#           하지만 과거 {first_date}에는 {first_values}였습니다.
#           → 향후 해당 수준에 도달하면 창업을 검토해보세요.
#           ```

#     3) 그 외(other)  
#        → “죄송합니다. 대감이는 창업 시기 및 위치에 대해서만 답변 드릴 수 있습니다.”

#   ▶️ 마무리 멘트  
#     > 대감이의 의견은 참고용입니다. 좋은 창업 결과가 있길 바랍니다.

# """

# USER = """
# 사용자 질문: "{user_question}"
# 질문 시점: {ask_date}   # YYYY-MM-DD

# # (1) 질문 의도 분류 결과
# #    변수: intent  ← "location" / "timing" / "other"

# # (2) timing 분기
# # timing: "{timing}"  
# #    - intent == "timing" 이면,
# #      · 최신 월: {latest_month}   # 예: 2025-04
# #      · current_row:    {current_row}    # 행 단위 텍스트 청크
# #      · current_window: {current_window} # 3개월 추세 텍스트 청크
# #      · similar_refs:   {similar_refs}   # 과거 유사 Top3 (날짜+수치)
# #    - intent == "location" 이면 위치 추천 멘트만
# #    - intent == "other" 이면 서비스 안내 멘트만

# # (3) timing & 현재 시기
# #    아래 변수를 채워 아래 형식에 맞춰 대답하세요.
# #    - first_date:   {first_date}   # 가장 먼저 고려된 과거 교차 시점 (YYYY-MM)
# #    - first_values: {first_values} # 그때 주요 지표 수치(쉽게 풀어쓴 형태)
# #    - cur_cpi:      {cur_cpi}%      # 현재 CPI 수치
# #    - cur_msi:      {cur_msi}%      # 현재 MSI 수치
# #    - cpi_desc:     {cpi_desc}      # "상승/하락/보합" 한 단어
# #    - msi_desc:     {msi_desc}      # "상승/하락/보합" 한 단어

# TIMING이 CURRENT인 경우에는 다음과 같이 대답해주세요:

# 안녕하세요 저는 대감이입니다!

# 현재 시점({latest_month} 기준)의 주요 지표를 보면  
# 소비자물가상승률(CPI)는 {cur_cpi}%로 {cpi_desc} 상태이고,  
# 대형소매점판매액지수(MSI)는 {cur_msi}%로 {msi_desc} 중입니다.  


# {past_explanation}  

# 단기간 시장 분위기는 다음과 같습니다:
# → 소비자물가상승률(CPI) {formatted_delta_cpi}%p {cpi_trend}했고,
# 대형소매점판매액지수(MSI) {formatted_delta_msi}%p {msi_trend}했으므로
# 단기간 시장 분위기는 {market_condition} 되었습니다.

# 소비자물가상승률(CPI)이 높아지면 물가 부담이 커져 사람들이 지갑을 닫기 쉽고,  
# 내수 창업 환경에도 직접적인 부담이 됩니다.  
# 특히 대형소매점판매액지수(MSI)는 백화점·마트 등 큰 가게들의 매출 변화를 보여줘  
# '소비 트렌드가 살아 있는지, 전반적으로 위축되었는지'를 판별하기 좋습니다.  

# 따라서 단기간 시장 분위기 변화({formatted_delta_cpi}%p {cpi_trend}, {formatted_delta_msi}%p {msi_trend})와
# 생존율 통계({survival_summary})를 종합적으로 고려할 때,
# 질문해주신 창업 시기는 {recommendation} 합니다.

# 대감이의 의견은 참고용입니다. 좋은 창업 결과가 있길 바랍니다.

# {future_section}
# """
# import re
# import datetime as dt
# from dateutil.relativedelta import relativedelta

# def parse_future_timing(question: str, base_date: str) -> tuple:
#     # JSON까지 생성→파싱을 한 번에
#     timing_obj: TimingParse = parse_chain.predict_and_parse(user_question=question)
#     is_future    = timing_obj.is_future
#     months_ahead = timing_obj.months_ahead

#     # 날짜 보정
#     base_date_obj = dt.datetime.strptime(base_date, "%Y-%m-%d")
#     adj_date_obj  = base_date_obj + relativedelta(months=months_ahead)
#     adjusted_date = adj_date_obj.strftime("%Y-%m-%d")

#     return is_future, adjusted_date, months_ahead






# prompt = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM),
#     ("user",   USER),
# ])
# llm   = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# chain = LLMChain(llm=llm, prompt=prompt, output_parser=StrOutputParser())

# import re
# import datetime as dt

# import re
# import datetime as dt

# import re
# import datetime as dt

# def consult_all_in_one(user_question: str, ask_date: str) -> str:
#     # 0) 의도 파악 추가
#     intent = "timing"  # 기본값
#     if re.search(r"어디|입지|위치|장소", user_question, re.IGNORECASE):
#         intent = "location"
#     elif re.search(r"지금|언제|시기|타이밍|후에|뒤에|창업할|창업하|창업을", user_question, re.IGNORECASE):
#         intent = "timing"
#     else:
#         intent = "other"

#     is_future_question, adjusted_date, future_months = parse_future_timing(user_question, ask_date)
#     print(f"질문 분석: 미래질문={is_future_question}, 조정된날짜={adjusted_date}, 개월차이={future_months}")
    
#     # 미래 시점 질문이면서 6개월 이상 미래인 경우 바로 미래 시점 응답
#     if is_future_question and future_months >= 6:
#         future_response = f"""안녕하세요, 저는 대감이입니다!

# 말씀하신 {future_months//12}년 {future_months%12}개월 후({adjusted_date[:7]}) 창업 시점은 현재 시점과 차이가 많이 납니다.

# 미래 경제 상황은 예측하기 어렵지만, 창업 성공 가능성을 높이기 위해서는 다음과 같은 경제 지표를 참고하시면 좋겠습니다:
# - 소비자물가상승률(CPI): 2.0% 이하로 안정적일 때
# - 대형소매점판매액지수(MSI): 3.0% 이상 성장하는 소비 환경일 때

# 창업을 계획하시는 {adjusted_date[:7]} 시점이 다가오면 다시 문의해 주시기 바랍니다.
# 그때 최신 경제 데이터를 기반으로 더 정확한 조언을 드릴 수 있습니다.

# 대감이의 의견은 참고용입니다. 좋은 창업 결과가 있길 바랍니다."""
#         return future_response

#     # 각 의도에 따른 바로 응답 처리 추가
#     if intent == "location":
#         return "안녕하세요, 저는 대감이입니다!\n\n창업 위치 추천 서비스를 선택해주세요."
    
#     if intent == "other":
#         return "안녕하세요, 저는 대감이입니다!\n\n죄송합니다. 대감이는 창업 시기 및 위치에 대해서만 답변 드릴 수 있습니다."

#     # 1) 컨텍스트 수집
#     ctx = retrieve_context(ask_date)
#     row = ctx["row_doc"]
#     win = ctx["window_doc"]
    
#     # 2) latest_month 계산
#     if row and win:
#         latest_month = max(row.metadata["date"][:7], win.metadata["end"][:7])
#     elif row:
#         latest_month = row.metadata["date"][:7]
#     else:
#         latest_month = win.metadata["end"][:7] if win else ""
    
#     # 3) 질문 날짜와 최신 데이터 날짜 사이의 차이 계산
#     ask_date_obj = dt.datetime.strptime(ask_date, "%Y-%m-%d")
    
#     # 중요한 변경: timing 분기 - 두 값 모두 true일 때만 current, 그렇지 않으면 future
#     timing = "current" if (ctx["is_row_recent"] or ctx["is_window_recent"]) else "future"
#     print(f"DEBUG: Final timing decision: {timing}")
    
#     # 4) future 타이밍일 경우 더 이상 진행하지 않고 바로 응답
#     if timing == "future":
#         # 미래 시점은 명확한 권고만 제공
#         future_response = f"""안녕하세요, 저는 대감이입니다!

# 죄송합니다. 현재 시점({ask_date[:7]})에 대해 가지고 있는 데이터가 없습니다.
# 저의 최신 데이터는 {latest_month} 기준이며, 현재와 차이가 큽니다.

# 창업 성공 가능성을 높이기 위해서는 다음과 같은 경제 지표를 참고하시면 좋겠습니다:
# - 소비자물가상승률(CPI): 2.0% 이하로 안정적일 때
# - 대형소매점판매액지수(MSI): 3.0% 이상 성장하는 소비 환경일 때

# 이러한 지표들이 개선된 시점에 창업을 검토하시는 것을 권장드립니다.

# 대감이의 의견은 참고용입니다. 좋은 창업 결과가 있길 바랍니다."""
#         return future_response
    
#     # 나머지 코드는 timing이 "current"인 경우에만 실행됨
    
#     # 5) common 중 최신 날짜 선택 (현재는 future 타이밍에는 사용하지 않음)
#     common_all = ctx["common"]
#     latest_common = max(common_all) if common_all else None

#     # 6) 생존율 데이터 텍스트
#     survival_data_text = ""
#     if ctx["survival_data_exists"] and ctx["econ_monthly_data"]:
#         survival_docs = ctx["econ_monthly_data"]
#         survival_data_text = "# — 생존율 분석\n" + "\n".join([
#             f"- {d.metadata.get('date')[:7]}: 평균 영업 기간={d.metadata.get('op_avg')}개월, " +
#             f"폐업까지 평균={d.metadata.get('cl_avg')}개월"
#             for d in survival_docs
#         ])
        
#     # 7) 과거 생존율 데이터 & 리스크 분석
#     past_explanation = ""
#     survival_summary = "관련 데이터 없음"
    
#     # 중요한 변경: 모든 common 날짜에 대해 생존율 데이터를 확인하고, 하나라도 있으면 사용
#     found_survival_data = False
#     survival_doc = None
    
#     # common_all의 모든 날짜를 순회하며 생존율 데이터 검색
#     for date in sorted(common_all, reverse=True):  # 최신순으로 정렬
#         month_date = date[:7] + "-01"
#         docs = surv_store.similarity_search(
#             query="dummy", k=1,
#             filter={"type": {"$eq": "survival_stats"}, "date": {"$eq": month_date}}
#         )
#         if docs:
#             print(f"Using survival data from {month_date}")
#             survival_doc = docs[0]
#             found_survival_data = True
#             break
    
#     # 생존율 데이터를 찾았다면 요약 생성
#     if found_survival_data and survival_doc:
#         m  = survival_doc.metadata["date"][:7]
#         op = survival_doc.metadata["op_avg"]
#         cl = survival_doc.metadata["cl_avg"]
#         # 리스크 비중(%), 절반 운영 시점(년) 계산
#         risk_pct   = (cl / op * 100) if op else 0
#         half_years = (op / 2) / 12
        
#         # 생존 요약 추가
#         survival_summary = f"폐업률 {risk_pct:.0f}%, 위험 시점 약 {half_years:.1f}년"
        
#         past_explanation = (
#             f"\n현재와 유사한 과거 시점인 {m}에는 서울 지역 사업체의 평균 영업 기간이 {op}개월, "
#             f"폐업까지 평균 운영 기간이 {cl}개월이었습니다.\n"
#             f"→ 폐업/운영 비중은 약 {risk_pct:.0f}%로, 전체 운영 기간의 절반(약 {half_years:.1f}년) 전후에 "
#             "폐업 리스크가 집중된 것으로 보입니다.\n"
#             f"따라서 창업을 하신다면, 창업 초기 약 {half_years:.1f}년 동안은 자금·마케팅 전략을 더욱 튼튼히 마련하실 것을 권장드립니다.\n"
#         )
    
#     # 8) 단기 변화 계산
#     ask_dt = dt.datetime.strptime(ask_date, "%Y-%m-%d")
#     delta_cpi, delta_msi, short_summary = retrieve_short_term(ask_dt)
#     cpi_trend = "증가" if delta_cpi > 0 else "감소" if delta_cpi < 0 else "보합"
#     msi_trend = "증가" if delta_msi > 0 else "감소" if delta_msi < 0 else "보합"
    
#     # 미리 포맷팅된 델타 값 생성
#     formatted_delta_cpi = f"{delta_cpi:+.1f}"
#     formatted_delta_msi = f"{delta_msi:+.1f}"
    
#     # 로직 표현식을 미리 계산
#     market_condition = "호전" if delta_msi >= 0 else "악화"
#     recommendation = "추천" if delta_msi > 0 else "보류" if delta_msi < -5 else "유보"

#     # 9) 과거 Top3 유사 사례 텍스트 개선
#     def safe_format(value, default="N/A", precision=1):
#         """숫자일 경우 소수점 형식으로, 아닐 경우 기본값 반환"""
#         if isinstance(value, (int, float)):
#             return f"{value:.{precision}f}"
#         return default
        
#     similar_refs = ""
#     if ctx["cands"]:
#         similar_refs = "\n".join([
#             f"- {d.metadata.get('date', d.metadata.get('start'))[:7]}: "
#             f"CPI={safe_format(d.metadata.get('CPI'))}%, "
#             f"MSI={safe_format(d.metadata.get('MSI'))}%"
#             for d in ctx["cands"][:3]
#         ])
    
#     # 10) 첫 교차 시점 정보
#     first_date = ""
#     first_values = ""
#     if ctx["cands"]:
#         first = ctx["cands"][0]
#         fd    = first.metadata.get("date") or first.metadata.get("start")
#         first_date = fd[:7]
#         fv_cpi = first.metadata.get("CPI", 0)
#         fv_msi = first.metadata.get("MSI", 0)
        
#         # 안전한 포맷팅 적용
#         first_values = f"CPI={safe_format(fv_cpi)}%, MSI={safe_format(fv_msi)}%"

#     # 11) 현재 지표
#     md_row = row.metadata if row else {}
#     row_metrics = {k: v for k, v in md_row.items() if isinstance(v, (int, float))}
    
#     # 더 명확한 포맷팅
#     important_metrics = ["CPI", "MSI", "RSI", "SSI", "GDP_C"] 
#     current_values = ", ".join(
#         f"{k}={safe_format(row_metrics.get(k, 'N/A'))}"
#         for k in important_metrics if k in row_metrics
#     )
    
#     cur_cpi = row_metrics.get("CPI", 0)
#     cur_msi = row_metrics.get("MSI", 0)
#     cpi_desc = "상승" if cur_cpi > 0 else "하락" if cur_cpi < 0 else "보합"
#     msi_desc = "상승" if cur_msi > 0 else "하락" if cur_msi < 0 else "보합"

#     current_row = row.page_content if row else "데이터 없음"
#     current_window = win.page_content if win else "데이터 없음"

#     # 12) 프롬프트 전달 변수 - 미리 계산된 값과 사전 포맷팅된 값으로 대체
#     vars = {
#         "user_question": user_question,
#         "ask_date": ask_date,
#         "intent": intent,
#         "timing": timing,
#         "latest_month": latest_month,
#         "current_row": current_row,
#         "current_window": current_window,
#         "current_values": current_values,
#         "cur_cpi": cur_cpi,
#         "cur_msi": cur_msi,
#         "cpi_desc": cpi_desc,
#         "msi_desc": msi_desc,
#         "past_explanation": past_explanation,
#         "short_summary": short_summary,
#         "delta_cpi": delta_cpi,
#         "delta_msi": delta_msi,
#         "formatted_delta_cpi": formatted_delta_cpi,  # 포맷팅된 값 사용
#         "formatted_delta_msi": formatted_delta_msi,  # 포맷팅된 값 사용
#         "cpi_trend": cpi_trend,
#         "msi_trend": msi_trend,
#         "first_date": first_date,
#         "first_values": first_values,
#         "similar_refs": similar_refs,
#         "survival_data_text": survival_data_text,
#         "survival_summary": survival_summary,
#         "future_section": "",  # 더 이상 사용하지 않음
#         "market_condition": market_condition,
#         "recommendation": recommendation
#     }
        
#     # 13) LLM 호출 - 이제 future 타이밍에는 실행되지 않음
#     json_answer = chain.run(**vars).strip()
    
#     # 14) JSON 파싱 시도 
#     try:
#         # JSON 응답을 파싱
#         parsed = json.loads(json_answer)
#         # 이미 포맷된 답변이 있다면 그대로 반환
#         if isinstance(parsed, str):
#             return parsed
        
#         # 아니면 직접 포맷팅
#         answer = f"""안녕하세요 저는 대감이입니다! 현재 시점({latest_month} 기준)의 주요 지표를 보면  
# 소비자물가상승률(CPI)는 {cur_cpi}%로 {cpi_desc} 상태이고,  
# 대형소매점판매액지수(MSI)는 {cur_msi}%로 {msi_desc} 중입니다.  
# {past_explanation}  
# 최근 6개월 간 단기간 시장 분위기는 다음과 같습니다:
# → 소비자물가상승률(CPI) {formatted_delta_cpi}%p {cpi_trend}했고,
# 대형소매점판매액지수(MSI) {formatted_delta_msi}%p {msi_trend}했으므로
# 단기간 시장 분위기는 {market_condition} 되었습니다.

# 소비자물가상승률(CPI)이 높아지면 물가 부담이 커져 사람들이 지갑을 닫기 쉽고,  
# 내수 창업 환경에도 직접적인 부담이 됩니다.  
# 특히 대형소매점판매액지수(MSI)는 백화점·마트 등 큰 가게들의 매출 변화를 보여줘  
# '소비 트렌드가 살아 있는지, 전반적으로 위축되었는지'를 판별하기 좋습니다.  

# 따라서 단기간 시장 분위기 변화({formatted_delta_cpi}%p {cpi_trend}, {formatted_delta_msi}%p {msi_trend})와
# 생존율 통계({survival_summary})를 종합적으로 고려할 때,
# 질문해주신 창업 시기는 {recommendation} 합니다.

# 대감이의 의견은 참고용입니다. 좋은 창업 결과가 있길 바랍니다."""
#         return answer
#     except json.JSONDecodeError:
#         # JSON 파싱에 실패하면 원본 반환
#         return json_answer

app = FastAPI(title="창업타이밍 컨설턴트 API")

class StartupPeriodRequest(BaseModel):
    timestamp: str  # "2025-05-16" 형식
    user_query: str # 질문 텍스트

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 허용할 도메인을 명시해야 합니다
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StartupLocationRequest(BaseModel):
    space: str       # 구 정보
    userquery: str   # 사용자 질문

class StartupLocationResponse(BaseModel):
    answer: str

@app.post("/locationPredict", response_model=StartupLocationResponse)
async def startup_recommendation(request: StartupLocationRequest):
    """
    창업 위치 및 시기에 대한 추천을 제공합니다.
    
    Parameters:
    - space: 서울시 구 정보 (예: '강남구', '마포구')
    - userquery: 사용자 질문 (예: '강남에서 카페 창업하기 좋은 곳')
    
    Returns:
    - answer: 추천 결과 및 답변
    """
    try:
        if not request.space or not request.userquery:
            raise HTTPException(status_code=400, detail="구 정보와 질문을 모두 입력해야 합니다")
        
        # 기존 answer 함수 호출
        result = answer(request.space, request.userquery)
        return StartupLocationResponse(answer=result)
    except Exception as e:
        # 에러 로깅 (실제 운영 환경에서는 로깅 추가)
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="추천 처리 중 오류가 발생했습니다")

@app.post("/periodPredict", response_class=PlainTextResponse)
async def periodPredict(req: StartupPeriodRequest):
    """
    요청 JSON:
    {
      "timestamp": "YYYY-MM-DD",
      "user_query": "질문"
    }
    반환: 모델의 답변 문자열 (plain text)
    """
    # timestamp를 YYYY-MM-DD 형식으로 전달했다고 가정
    answer: str = consult_all_in_one(req.user_query, req.timestamp)
    return answer

# (선택) 헬스체크
@app.get("/health")
async def health():
    return {"status": "ok"}