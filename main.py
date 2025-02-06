import os
import json
import logging
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable
from langchain.docstore.document import Document
from langchain_chroma import Chroma

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

###########################################
# 1) 임베딩 및 벡터스토어 로드
###########################################
class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
        
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()

# KURE-v1 모델 사용 (예시)
embedding_model = SentenceTransformerEmbeddings(model_name="nlpai-lab/KURE-v1")

def load_vectorstore(persist_dir: str = "./chroma_data") -> Chroma:
    logger.info(f"Loading vector store from {persist_dir}")
    vs = Chroma(
        embedding_function=embedding_model,
        collection_name="job_postings",
        persist_directory=persist_dir
    )
    logger.info("Vector store loaded successfully.")
    return vs

vectorstore = load_vectorstore()

###########################################
# 2) Pydantic 모델
###########################################
class UserProfile(BaseModel):
    age: Optional[str] = ""
    location: Optional[str] = ""
    jobType: Optional[str] = ""

class ChatRequest(BaseModel):
    user_message: str
    user_profile: UserProfile
    session_id: Optional[str] = None

class JobPosting(BaseModel):
    id: str
    location: str
    company: str
    title: str
    salary: str
    workingHours: str
    description: str
    rank: int

class ChatResponse(BaseModel):
    message: str
    jobPostings: List[JobPosting]
    type: str
    user_profile: Optional[UserProfile] = None

###########################################
# 3) 사용자 입력 NER
###########################################
def get_user_ner_runnable() -> Runnable:
    """
    사용자 입력 예: "서울 요양보호사"
    -> LLM이 아래와 같이 JSON으로 추출:
       {"직무": "요양보호사", "지역": "서울", "연령대": ""}
    """
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-4o-mini",  
        temperature=0.0
    )

    prompt = PromptTemplate(
        input_variables=["user_query"],
        template=(
            "사용자 입력: {user_query}\n\n"
            "아래 항목을 JSON으로 추출 (값이 없으면 빈 문자열로):\n"
            "- 직무\n"
            "- 지역\n"
            "- 연령대\n\n"
            "예:\n"
            "json\n"
            "{{\"직무\": \"요양보호사\", \"지역\": \"서울\", \"연령대\": \"\"}}\n"
            "\n"
        )
    )
    return prompt | llm

###########################################
# 4) 내부 필터만 사용 (부분 일치 + 논리 연산)
###########################################
def param_filter_search_with_chroma(
    query: str,
    region: Optional[str] = None,
    job: Optional[str] = None,
    top_k: int = 10,
    use_and: bool = True
) -> List[Document]:
    """
    Chroma의 '$contains' 연산자를 사용한 예:
    1) region, job 각각 부분 일치
    2) use_and=True → $and, False → $or
    3) 필터로 걸러진 결과를 similarity_search_with_score
    4) distance 오름차순 정렬 후 상위 top_k
    """
    # ChromaDB의 올바른 where_document 필터 형식으로 수정
    filter_condition = {}
    if region or job:
        conditions = []
        if region:
            conditions.append({"$contains": region})
        if job:
            conditions.append({"$contains": job})
        
        if len(conditions) > 1:
            filter_condition = {"$and": conditions}
        else:
            filter_condition = conditions[0]

    results_with_score = vectorstore.similarity_search_with_score(
        query=query,
        k=top_k * 3,  
        where_document=filter_condition
    )

    results_with_score.sort(key=lambda x: x[1])  # distance 오름차순
    selected_docs = [doc for doc, score in results_with_score[:top_k]]

    # distance 메타데이터 기록(선택사항)
    for i, (doc, dist) in enumerate(results_with_score[:top_k]):
        doc.metadata["search_distance"] = dist

    return selected_docs

def deduplicate_by_id(docs: List[Document]) -> List[Document]:
    unique = []
    seen = set()
    for d in docs:
        job_id = d.metadata.get("채용공고ID", "no_id")
        if job_id not in seen:
            unique.append(d)
            seen.add(job_id)
    return unique

###########################################
# 5) 직무 동의어 확장
###########################################
def get_job_synonyms_with_llm(job: str) -> List[str]:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-4o-mini",
        temperature=0.0
    )

    prompt = PromptTemplate(
        input_variables=["job"],
        template=(
            "입력된 직무와 유사한 동의어를 추출해주세요. "
            "특히, 요양보호, IT, 건설, 교육 등 특정 산업 분야에서 사용되는 동의어를 포함해주세요.\n\n"
            "입력된 직무: {job}\n\n"
            "동의어를 JSON 배열 형식으로 반환해주세요. 예:\n"
            "json\n"
            "{{{{'synonyms': [\"동의어1\", \"동의어2\", \"동의어3\"]}}}}\n"
            "\n"
        )
    )

    chain = prompt | llm
    response = chain.invoke({"job": job})

    try:
        response_content = response.content.strip().replace("```json", "").replace("```", "").strip()
        synonyms_data = json.loads(response_content)
        synonyms = synonyms_data.get("synonyms", [])
        return synonyms
    except Exception as e:
        logger.warning(f"Failed to parse LLM response for job synonyms: {e}")
        return []

###########################################
# 6) LLM 재랭킹 및 메타데이터 검증
###########################################
def compute_ner_similarity(user_ner: dict, doc_ner: dict) -> float:
    score = 0.0
    keys_to_check = ["직무", "근무 지역", "연령대"]
    for key in keys_to_check:
        user_val = user_ner.get(key, "").strip().lower()
        doc_val = doc_ner.get(key, "").strip().lower()
        if user_val and doc_val:
            if user_val in doc_val or doc_val in user_val:
                score += 1.0
    return score

def verify_document_metadata(doc: Document, idx: int):
    md = doc.metadata
    logger.info(f"[Metadata Verification] Doc {idx+1} metadata: {md}")
    llm_ner_str = md.get("LLM_NER", "{}")
    try:
        llm_ner = json.loads(llm_ner_str)
        required_keys = ["직무", "근무 지역", "연령대"]
        missing_keys = [key for key in required_keys if key not in llm_ner]
        if missing_keys:
            logger.warning(f"[Metadata Verification] Doc {idx+1} missing keys in LLM_NER: {missing_keys}")
        else:
            logger.info(f"[Metadata Verification] Doc {idx+1} LLM_NER structure is valid.")
    except Exception as ex:
        logger.error(f"[Metadata Verification] Doc {idx+1} LLM_NER 파싱 실패: {ex}")

def build_detailed_snippet(doc: Document) -> str:
    md = doc.metadata
    title = md.get("채용제목", "정보없음")
    company = md.get("회사명", "정보없음")
    region = md.get("근무지역", "정보없음")
    salary = md.get("급여조건", "정보없음")
    description = md.get("상세정보", doc.page_content[:100].replace("\n", " "))
    snippet = (
        f"제목: {title}\n"
        f"회사명: {company}\n"
        f"근무지역: {region}\n"
        f"급여조건: {salary}\n"
        f"설명: {description}\n"
    )
    return snippet

def llm_rerank(docs: List[Document], user_ner: dict) -> List[Document]:
    if not docs:
        return []

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-4o-mini",
        temperature=0.3
    )

    cond = []
    if user_ner.get("직무"):
        cond.append(f"직무={user_ner.get('직무')}")
    region_val = user_ner.get("근무 지역", user_ner.get("근무지역", user_ner.get("지역", "")))
    if region_val:
        cond.append(f"근무지역={region_val}")
    if user_ner.get("연령대"):
        cond.append(f"연령대={user_ner.get('연령대')}")
    condition_str = ", ".join(cond) or "조건 없음"

    doc_snippets = []
    for i, doc in enumerate(docs):
        verify_document_metadata(doc, i)
        snippet = build_detailed_snippet(doc)
        doc_snippets.append(f"Doc{i+1}:\n{snippet}\n")

    prompt_text = (
        f"사용자 조건: {condition_str}\n\n"
        "아래 각 문서가 사용자 조건에 얼마나 부합하는지 0에서 5점으로 평가해줘. "
        "점수가 높을수록 조건에 더 부합함.\n\n"
        + "\n".join(doc_snippets) +
        "\n\n답변은 반드시 JSON 형식으로만, 예: {\"scores\": [5, 3, 0, ...]}."
    )
    logger.info(f"[llm_rerank] prompt:\n{prompt_text}")

    resp = llm.invoke(prompt_text)
    content = resp.content.replace("```json", "").replace("```", "").strip()
    logger.info(f"[llm_rerank] raw response: {content}")

    try:
        score_data = json.loads(content)
        llm_scores = score_data.get("scores", [])
    except Exception as ex:
        logger.warning(f"[llm_rerank] Re-rank parse failed: {ex}. Using default scores.")
        llm_scores = [0] * len(docs)

    weight_llm = 0.7
    weight_manual = 0.3

    weighted_scores = []
    for idx, (doc, llm_score) in enumerate(zip(docs, llm_scores)):
        doc_ner_str = doc.metadata.get("LLM_NER", "{}")
        try:
            doc_ner = json.loads(doc_ner_str)
        except Exception as ex:
            logger.warning(f"[llm_rerank] Failed to parse document NER: {ex}")
            doc_ner = {}

        manual_score = compute_ner_similarity(user_ner, doc_ner)
        combined_score = weight_llm * llm_score + weight_manual * manual_score

        logger.info(
            f"[llm_rerank] Doc {idx+1} (ID: {doc.metadata.get('채용공고ID', 'no_id')}) - "
            f"LLM score: {llm_score}, Manual score: {manual_score}, Combined score: {combined_score}"
        )
        weighted_scores.append((doc, combined_score))

    if len(weighted_scores) < len(docs):
        for i in range(len(weighted_scores), len(docs)):
            weighted_scores.append((docs[i], 0))

    ranked_sorted = sorted(weighted_scores, key=lambda x: x[1], reverse=True)
    return [x[0] for x in ranked_sorted]

###########################################
# 7) 다단계 검색 (필터 조건 점진적 완화)
###########################################
def multi_stage_search(user_ner: dict) -> List[Document]:
    """
    1) region, job 모두 포함(AND) → 결과 적으면 OR
    2) 그래도 적으면 region만, job만
    3) 직무 동의어 확장
    4) 결과 계속 적으면 임베딩 검색
    5) 최종 LLM 재랭킹
    """
    region = user_ner.get("지역", "").strip()
    job = user_ner.get("직무", "").strip()

    # Step 1: region+job (AND)
    strict_docs = param_filter_search_with_chroma(
        query=f"{region} {job}", 
        region=region, 
        job=job, 
        top_k=10,
        use_and=True
    )
    logger.info(f"[multi_stage_search] region+job (AND) 결과: {len(strict_docs)} 건")

    # Step 2: OR로 완화
    if len(strict_docs) < 5 and region and job:
        or_docs = param_filter_search_with_chroma(
            query=f"{region} {job}", 
            region=region, 
            job=job, 
            top_k=10,
            use_and=False
        )
        strict_docs = deduplicate_by_id(strict_docs + or_docs)
        logger.info(f"[multi_stage_search] region+job (OR) 결과: {len(strict_docs)} 건")

    # Step 3: region만 / job만
    if len(strict_docs) < 5:
        if region:
            region_only = param_filter_search_with_chroma(
                query=f"{region} {job}",
                region=region,
                job=None,
                top_k=10,
                use_and=True
            )
            strict_docs = deduplicate_by_id(strict_docs + region_only)

        if job:
            job_only = param_filter_search_with_chroma(
                query=f"{region} {job}",
                region=None,
                job=job,
                top_k=10,
                use_and=True
            )
            strict_docs = deduplicate_by_id(strict_docs + job_only)
        logger.info(f"[multi_stage_search] region/job 단독 결과: {len(strict_docs)} 건")

    # Step 4: 직무 동의어 확장
    if job:
        synonyms = get_job_synonyms_with_llm(job)
        for syn in synonyms:
            syn_docs = param_filter_search_with_chroma(
                query=f"{region} {syn}",
                region=region,
                job=syn,
                top_k=10,
                use_and=True
            )
            strict_docs = deduplicate_by_id(strict_docs + syn_docs)
    logger.info(f"[multi_stage_search] 동의어 확장 후: {len(strict_docs)} 건")

    # Step 5: 결과가 너무 적으면 필터 없이 임베딩 검색 (k=15)
    if len(strict_docs) < 15:
        fallback_results = vectorstore.similarity_search_with_score(f"{region} {job}", k=15)
        fallback_docs = []
        for doc, score in fallback_results:
            doc.metadata["search_distance"] = score
            fallback_docs.append(doc)
        strict_docs = deduplicate_by_id(strict_docs + fallback_docs)
        logger.info(f"[multi_stage_search] 필터 없이 추가: {len(strict_docs)} 건")

    # LLM 재랭킹
    final_docs = llm_rerank(strict_docs, user_ner)
    return final_docs

###########################################
# 8) FastAPI 엔드포인트 (채팅)
###########################################
@app.post("/api/v1/chat/", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        user_message = req.user_message.strip()
        if not user_message:
            return ChatResponse(
                message="검색어가 비어있습니다.",
                jobPostings=[],
                type="info",
                user_profile=req.user_profile
            )

        logger.info(f"[chat_endpoint] user_message='{user_message}'")

        # 1) 사용자 입력 NER 추출
        ner_chain = get_user_ner_runnable()
        ner_res = ner_chain.invoke({"user_query": user_message})
        ner_str = ner_res.content if hasattr(ner_res, "content") else str(ner_res)
        cleaned = ner_str.replace("```json", "").replace("```", "").strip()
        try:
            user_ner = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"[chat_endpoint] NER parse fail: {cleaned}")
            user_ner = {}

        logger.info(f"[chat_endpoint] user_ner={user_ner}")

        # 2) 다단계 검색
        doc_list = multi_stage_search(user_ner)
        top_docs = doc_list[:5]

        job_postings = []
        for i, doc in enumerate(top_docs, start=1):
            md = doc.metadata
            job_postings.append(JobPosting(
                id=md.get("채용공고ID", "no_id"),
                location=md.get("근무지역", ""),
                company=md.get("회사명", ""),
                title=md.get("채용제목", ""),
                salary=md.get("급여조건", ""),
                workingHours=md.get("근무시간", "정보없음"),
                description=md.get("상세정보", "상세정보 없음"),
                rank=i
            ))
        logger.info(f"[chat_endpoint] 검색 결과: {len(job_postings)} 건")
        
        if job_postings:
            msg = f"'{user_message}' 검색 결과, 상위 {len(job_postings)}건을 반환합니다."
            res_type = "jobPosting"
        else:
            msg = "조건에 맞는 채용공고를 찾지 못했습니다."
            res_type = "info"

        return ChatResponse(
            message=msg,
            jobPostings=job_postings,
            type=res_type,
            user_profile=req.user_profile
        )

    except Exception as e:
        logger.error(f"Error in chat_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

###########################################
# 9) 디버그용: 전체 DB 유사도 검색
###########################################
@app.post("/api/v1/debug/similarity")
def debug_similarity_search(query: str = Query(..., description="유사도 검색에 사용할 쿼리")):
    try:
        results_with_score = vectorstore.similarity_search_with_score(query, k=1000)
        debug_results = []
        for idx, (doc, score) in enumerate(results_with_score):
            md = doc.metadata
            logger.info(
                f"[Debug Similarity] Doc {idx+1} (ID: {md.get('채용공고ID', 'no_id')}) - "
                f"Title: {md.get('채용제목', '정보없음')}, Distance: {score}"
            )
            debug_results.append({
                "rank": idx+1,
                "id": md.get("채용공고ID", "no_id"),
                "title": md.get("채용제목", "정보없음"),
                "distance": score,
                "metadata": md
            })
        return debug_results
    except Exception as e:
        logger.error(f"Error in debug_similarity_search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

###########################################
# 실행부
###########################################
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
