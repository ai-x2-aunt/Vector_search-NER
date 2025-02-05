import os  # 운영체제 기능 사용을 위한 모듈. 예: 파일 경로 및 환경 변수 접근.
import json  # JSON 데이터 인코딩 및 디코딩을 위한 모듈. 예: JSON 문자열을 파이썬 객체로 변환.
import logging  # 애플리케이션 로그 기록을 위한 모듈. 예: 실행 정보나 오류 메시지 출력.
from typing import Optional, List  # Optional, List 등의 타입 힌트를 제공하는 모듈. 예: 함수 인자 타입 명시.
from fastapi import FastAPI, HTTPException, Query  # FastAPI 애플리케이션, 예외 처리, 쿼리 파라미터를 위한 클래스들.
from pydantic import BaseModel  # 데이터 모델 검증 및 직렬화를 위한 Pydantic의 기본 클래스.
from fastapi.middleware.cors import CORSMiddleware  # CORS 미들웨어를 사용해 크로스 도메인 요청을 허용하기 위한 모듈.

from langchain_community.chat_models import ChatOpenAI  # LangChain 커뮤니티 제공 LLM 인터페이스. 예: OpenAI 모델 호출.
from langchain.prompts import PromptTemplate  # 사용자 정의 프롬프트 템플릿을 생성하기 위한 클래스.
from langchain.schema.runnable import Runnable  # 파이프라인의 실행 가능한 구성요소 인터페이스.
from langchain.docstore.document import Document  # 문서 객체를 표현하는 클래스. 예: 텍스트와 메타데이터를 포함.
from langchain_huggingface import HuggingFaceEmbeddings  # HuggingFace 기반 임베딩 모델 래퍼 클래스.
from langchain_chroma import Chroma  # Chroma 벡터스토어와 상호작용하기 위한 클래스.

from dotenv import load_dotenv  # .env 파일로부터 환경변수를 로드하기 위한 모듈.

load_dotenv()  # .env 파일에 저장된 환경변수를 현재 프로세스에 로드. 예: OPENAI_API_KEY 설정.

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# 로그 기본 설정: INFO 레벨 이상의 메시지를 날짜/시간, 레벨, 메시지 형식으로 출력.
logger = logging.getLogger(__name__)
# 현재 모듈 이름(__name__)을 기반으로 로거 객체 생성.

app = FastAPI()
# FastAPI 애플리케이션 인스턴스 생성. 엔드포인트 등록과 요청 처리를 담당.

app.add_middleware(
    CORSMiddleware,  # CORS 미들웨어 추가하여 다른 도메인에서의 요청 허용.
    allow_origins=["*"],  # 모든 출처(origin)에서의 요청을 허용.
    allow_credentials=True,  # 인증 정보(쿠키 등) 포함 요청 허용.
    allow_methods=["*"],  # 모든 HTTP 메서드(GET, POST 등) 허용.
    allow_headers=["*"],  # 모든 HTTP 헤더 허용.
)

###########################################
# 1) 임베딩 및 벡터스토어 로드
###########################################
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# 텍스트 임베딩에 사용할 HuggingFace 모델 이름 설정.
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
# 지정한 모델로 임베딩 객체 생성. 예: 텍스트를 벡터로 변환.

def load_vectorstore(persist_dir: str = "./chroma_data") -> Chroma:
    # 지정된 persist_dir 경로에서 벡터스토어(Chroma)를 로드하는 함수.
    logger.info(f"Loading vector store from {persist_dir}")
    # 벡터스토어 로드 경로를 로그에 기록.
    vs = Chroma(
        embedding_function=embedding_model,  # 텍스트 임베딩에 사용할 함수 전달.
        collection_name="job_postings",  # 채용공고 문서를 저장할 컬렉션 이름 지정.
        persist_directory=persist_dir  # 벡터 데이터가 저장된 디렉토리 경로.
    )
    logger.info("Vector store loaded successfully.")
    # 벡터스토어 로드 성공 메시지를 로그에 기록.
    return vs
    # 로드된 벡터스토어 객체 반환.

vectorstore = load_vectorstore()
# 전역 변수 vectorstore에 load_vectorstore 함수를 통해 벡터스토어 객체 저장.

###########################################
# 2) Pydantic 모델
###########################################
class UserProfile(BaseModel):
    # 사용자 프로필 정보를 담기 위한 Pydantic 모델.
    age: Optional[str] = ""          # 사용자의 나이 정보 (문자열), 기본값은 빈 문자열.
    location: Optional[str] = ""     # 사용자의 위치 정보, 기본값은 빈 문자열.
    jobType: Optional[str] = ""      # 사용자의 직업 유형 정보, 기본값은 빈 문자열.

class ChatRequest(BaseModel):
    # 클라이언트로부터 전달받는 채팅 요청 데이터 모델.
    user_message: str                # 사용자가 입력한 메시지.
    user_profile: UserProfile        # 사용자 프로필 정보를 포함.
    session_id: Optional[str] = None # 선택적 세션 식별자, 기본값은 None.

class JobPosting(BaseModel):
    # 채용 공고 정보를 나타내는 모델.
    id: str                          # 채용 공고의 고유 식별자.
    location: str                    # 근무 지역.
    company: str                     # 회사 이름.
    title: str                       # 채용 제목.
    salary: str                      # 급여 조건.
    workingHours: str                # 근무 시간.
    description: str                 # 상세 채용 정보.
    rank: int                        # 검색 결과 내 순위.

class ChatResponse(BaseModel):
    # 채팅 응답 데이터 모델.
    message: str                     # 사용자에게 보여질 응답 메시지.
    jobPostings: List[JobPosting]    # 검색된 채용공고 목록.
    type: str                        # 응답 타입 (예: "jobPosting", "info").
    user_profile: Optional[UserProfile] = None  # 선택적으로 포함되는 사용자 프로필 정보.

###########################################
# 3) 사용자 입력 NER (연령대는 선택적)
###########################################
def get_user_ner_runnable() -> Runnable:
    """
    사용자 입력 예: "서울 요양보호사"
    -> LLM이 아래와 같이 JSON으로 추출:
       {"직무": "요양보호사", "지역": "서울", "연령대": ""}
    quadruple braces를 사용해 JSON 리터럴을 출력하도록 함.
    """
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    # 환경변수에서 OPENAI_API_KEY 값을 읽어옴.
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
        # API 키가 없으면 예외 발생.
    
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,  # API 키 전달.
        model_name="gpt-4o-mini",         # 사용할 LLM 모델 이름. 필요 시 gpt-4로 변경 가능.
        temperature=0.0                 # 텍스트 생성의 무작위성 없이 결정적인 결과 생성.
    )
    
    prompt = PromptTemplate(
        input_variables=["user_query"],  # 프롬프트에서 사용할 입력 변수 이름.
        template=(
            "사용자 입력: {user_query}\n\n"  # 사용자 입력을 포함하는 템플릿.
            "아래 항목을 JSON으로 추출 (값이 없으면 빈 문자열로):\n"
            "- 직무\n"
            "- 지역\n"
            "- 연령대\n\n"
            "예:\n"
            "json\n"
            "{{{{\"직무\": \"요양보호사\", \"지역\": \"서울\", \"연령대\": \"\"}}}}\n"
            "\n"
            # quadruple braces를 사용해 JSON 리터럴 예시를 그대로 출력.
        )
    )
    return prompt | llm
    # 프롬프트 템플릿과 LLM을 파이프라인으로 연결하여 Runnable 객체 반환.

###########################################
# 4) 파라메트릭 필터 검색 (수정: 소문자 비교)
###########################################
def param_filter_search(region: Optional[str], job: Optional[str], top_k: int = 10) -> List[Document]:
    # region(지역)과 job(직무) 필터를 사용해 벡터스토어에서 검색 후 상위 top_k 문서 반환 함수.
    q = job if job else ""
    # 검색 쿼리로 job 값을 사용. job 값이 없으면 빈 문자열 사용.
    results_with_score = vectorstore.similarity_search_with_score(q, k=10)
    # 벡터스토어의 유사도 검색을 수행하여 상위 10개 결과(문서와 유사도 점수) 반환.
    filtered = []
    # 필터링된 문서를 저장할 빈 리스트 초기화.
    for (doc, score) in results_with_score:
        md = doc.metadata  # 각 문서의 메타데이터 추출.
        doc_region = md.get("근무지역", "").lower()
        # 문서의 근무지역 값을 소문자로 변환하여 추출.
        doc_title = md.get("채용제목", "").lower()
        # 문서의 채용제목 값을 소문자로 변환하여 추출.
        if region and region.lower() not in doc_region:
            continue  # region 필터가 주어졌으나 문서의 근무지역에 포함되지 않으면 건너뜀.
        if job and job.lower() not in doc_title:
            continue  # job 필터가 주어졌으나 문서의 채용제목에 포함되지 않으면 건너뜀.
        filtered.append(doc)  # 조건을 만족하면 문서를 filtered 리스트에 추가.
    return filtered[:top_k]
    # 필터링된 문서 중 상위 top_k 개만 반환.

###########################################
# 5) 중복 제거 (동일 채용공고ID 제거)
###########################################
def deduplicate_by_id(docs: List[Document]) -> List[Document]:
    # 동일한 채용공고ID를 가진 문서를 제거하는 함수.
    unique = []  # 고유 문서를 저장할 리스트 초기화.
    seen = set()  # 이미 처리한 채용공고ID를 저장할 집합.
    for d in docs:
        job_id = d.metadata.get("채용공고ID", "no_id")
        # 각 문서의 메타데이터에서 채용공고ID를 추출. 없으면 "no_id" 사용.
        if job_id not in seen:
            unique.append(d)  # 처음 등장한 채용공고ID면 리스트에 추가.
            seen.add(job_id)  # 해당 채용공고ID를 집합에 추가.
    return unique
    # 중복 제거된 문서 리스트 반환.

###########################################
# 6) 유사 직무 동의어
###########################################
def get_job_synonyms_with_llm(job: str) -> List[str]:
    """
    LLM을 사용하여 입력된 직무에 대한 동의어를 추출합니다.
    """
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    # 환경변수에서 API 키를 읽어옴.
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
        # API 키가 없으면 예외 발생.
    
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-4o-mini",  # 사용할 LLM 모델 설정.
        temperature=0.0           # 결정적인 결과를 위해 무작위성 제거.
    )
    
    prompt = PromptTemplate(
        input_variables=["job"],  # 프롬프트에서 사용할 입력 변수 "job" 지정.
        template=(
            "입력된 직무와 유사한 동의어를 추출해주세요. "
            "특히, 요양보호, IT, 건설, 교육 등 특정 산업 분야에서 사용되는 동의어를 포함해주세요.\n\n"
            "입력된 직무: {job}\n\n"
            "동의어를 JSON 배열 형식으로 반환해주세요. 예:\n"
            "json\n"
            "{{\"synonyms\": [\"동의어1\", \"동의어2\", \"동의어3\"]}}\n"
            "\n"
            # 입력된 직무에 대한 동의어 예시를 JSON 배열 형식으로 출력하도록 템플릿 작성.
        )
    )
    
    chain = prompt | llm
    # 프롬프트와 LLM을 파이프라인으로 연결.
    response = chain.invoke({"job": job})
    # 입력된 job 값을 사용하여 체인 실행 후 응답 받음.
    
    try:
        response_content = response.content.strip().replace("```json", "").replace("```", "").strip()
        # 응답 문자열에서 마크다운 구문(예: ```json) 제거 후 정리.
        synonyms_data = json.loads(response_content)
        # 정리된 문자열을 JSON 객체로 파싱.
        synonyms = synonyms_data.get("synonyms", [])
        # "synonyms" 키에 해당하는 값을 추출, 없으면 빈 리스트 사용.
        return synonyms
        # 동의어 리스트 반환.
    except Exception as e:
        logger.warning(f"Failed to parse LLM response for job synonyms: {e}")
        # 파싱 실패 시 경고 로그 기록.
        return []
        # 빈 리스트 반환.

###########################################
# 7) LLM 재랭킹 및 메타데이터 검증 (VectorDB의 LLM_NER 값을 그대로 사용)
###########################################
def compute_ner_similarity(user_ner: dict, doc_ner: dict) -> float:
    """
    사용자 NER 정보와 문서의 NER 정보를 비교하여 매칭 점수를 산출합니다.
    - 키워드: "직무", "근무 지역"(또는 "근무지역"), "연령대" 등
    """
    score = 0.0  # 초기 점수를 0으로 설정.
    keys_to_check = ["직무", "근무 지역", "연령대"]
    # 비교할 주요 키워드 목록 설정.
    for key in keys_to_check:
        user_val = user_ner.get(key, "").strip().lower()
        # 사용자 NER에서 해당 키의 값을 추출, 소문자 및 좌우 공백 제거.
        doc_val = doc_ner.get(key, "").strip().lower()
        # 문서 NER에서 해당 키의 값을 추출, 소문자 및 좌우 공백 제거.
        if user_val and doc_val:
            if user_val in doc_val or doc_val in user_val:
                score += 1.0  # 두 값이 서로 포함되면 1점 추가.
    return score
    # 최종 매칭 점수 반환.

def verify_document_metadata(doc: Document, idx: int):
    """
    각 문서의 메타데이터 및 LLM_NER 정보를 검증하고 로그로 출력합니다.
    VectorDB에 저장된 LLM_NER 값 그대로를 사용하므로,
    required_keys는 실제 값에 맞춰 "근무 지역" (공백 포함)으로 설정합니다.
    """
    md = doc.metadata  # 문서의 메타데이터 추출.
    logger.info(f"[Metadata Verification] Doc {idx+1} metadata: {md}")
    # 현재 문서의 메타데이터를 로그에 기록.
    llm_ner_str = md.get("LLM_NER", "{}")
    # 메타데이터에서 LLM_NER 정보를 문자열로 추출, 없으면 빈 JSON 객체 문자열 사용.
    try:
        llm_ner = json.loads(llm_ner_str)
        # LLM_NER 문자열을 JSON 객체로 파싱.
        required_keys = ["직무", "근무 지역", "연령대"]
        # 필수 키 목록 설정.
        missing_keys = [key for key in required_keys if key not in llm_ner]
        # 필수 키 중 누락된 키 리스트 작성.
        if missing_keys:
            logger.warning(f"[Metadata Verification] Doc {idx+1} missing keys in LLM_NER: {missing_keys}")
            # 누락된 키가 있으면 경고 로그 출력.
        else:
            logger.info(f"[Metadata Verification] Doc {idx+1} LLM_NER structure is valid.")
            # 모든 필수 키가 존재하면 유효하다는 로그 출력.
    except Exception as ex:
        logger.error(f"[Metadata Verification] Doc {idx+1} LLM_NER 파싱 실패: {ex}")
        # 파싱 오류 발생 시 에러 로그 출력.

def build_detailed_snippet(doc: Document) -> str:
    """
    문서의 주요 정보를 포함한 상세 스니펫을 생성합니다.
    """
    md = doc.metadata  # 문서 메타데이터 추출.
    title = md.get("채용제목", "정보없음")  # 채용제목 추출, 없으면 "정보없음" 사용.
    company = md.get("회사명", "정보없음")   # 회사명 추출.
    region = md.get("근무지역", "정보없음")    # 근무지역 추출.
    salary = md.get("급여조건", "정보없음")    # 급여조건 추출.
    description = md.get("상세정보", doc.page_content[:100].replace("\n", " "))
    # 상세정보 추출, 없으면 문서의 첫 100자를 사용하며 줄바꿈은 공백으로 치환.
    snippet = (
        f"제목: {title}\n"
        f"회사명: {company}\n"
        f"근무지역: {region}\n"
        f"급여조건: {salary}\n"
        f"설명: {description}\n"
    )
    # 각 정보를 결합하여 상세 스니펫 문자열 생성.
    return snippet  # 생성된 스니펫 반환.

def llm_rerank(docs: List[Document], user_ner: dict) -> List[Document]:
    """
    각 문서에 대해 LLM을 이용해 사용자 조건 부합도를 평가하고,
    LLM 평가 점수와 NER 직접 비교 점수를 가중치 합산하여 최종 순위를 산출합니다.
    또한, 각 문서의 메타데이터 검증을 수행합니다.
    """
    if not docs:
        return []  # 입력 문서 리스트가 비어있으면 빈 리스트 반환.
    
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    # 환경변수에서 API 키를 읽어옴.
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-4o-mini",
        temperature=0.3  # 약간의 무작위성을 부여하기 위해 temperature 값을 0.3으로 설정.
    )
    
    cond = []  # 사용자 조건을 구성할 리스트 초기화.
    if user_ner.get("직무"):
        cond.append(f"직무={user_ner.get('직무')}")
        # 사용자 NER의 직무 정보가 있으면 조건 문자열에 추가.
    region_val = user_ner.get("근무 지역", user_ner.get("근무지역", user_ner.get("지역", "")))
    # "근무 지역", "근무지역", "지역" 중 존재하는 값을 region_val에 저장.
    if region_val:
        cond.append(f"근무지역={region_val}")
        # region_val이 있으면 조건 문자열에 추가.
    if user_ner.get("연령대"):
        cond.append(f"연령대={user_ner.get('연령대')}")
        # 사용자 NER의 연령대가 있으면 조건 문자열에 추가.
    condition_str = ", ".join(cond) or "조건 없음"
    # 구성된 조건들을 콤마로 연결하여 최종 조건 문자열 생성.
    
    # 개선된 프롬프트: 각 문서의 상세 스니펫 포함
    doc_snippets = []  # 문서 스니펫들을 저장할 리스트 초기화.
    for i, doc in enumerate(docs):
        verify_document_metadata(doc, i)
        # 각 문서의 메타데이터를 검증하고 로그 출력.
        snippet = build_detailed_snippet(doc)
        # 문서의 상세 스니펫 생성.
        doc_snippets.append(f"Doc{i+1}:\n{snippet}\n")
        # "Doc1:", "Doc2:" 등의 형식으로 문서 번호와 스니펫 결합 후 리스트에 추가.
    
    prompt_text = (
        f"사용자 조건: {condition_str}\n\n"
        "아래 각 문서가 사용자 조건에 얼마나 부합하는지 0에서 5점으로 평가해줘. "
        "점수가 높을수록 조건에 더 부합함.\n\n"
        + "\n".join(doc_snippets) +
        "\n\n답변은 반드시 JSON 형식으로만, 예: {\"scores\": [5, 3, 0, ...]}."
    )
    # 사용자 조건 및 각 문서 스니펫을 포함하여 LLM에게 평가 요청하는 최종 프롬프트 텍스트 구성.
    logger.info(f"[llm_rerank] prompt:\n{prompt_text}")
    # 구성된 프롬프트 텍스트를 로그에 기록.
    
    resp = llm.invoke(prompt_text)
    # 프롬프트 텍스트를 LLM에 전달하여 평가 점수 응답 생성.
    content = resp.content.replace("```json", "").replace("```", "").strip()
    # 응답 문자열에서 마크다운 구문 제거 후 정리.
    logger.info(f"[llm_rerank] raw response: {content}")
    # 원시 응답 내용을 로그에 기록.
    
    try:
        score_data = json.loads(content)
        # 응답 문자열을 JSON 객체로 파싱.
        llm_scores = score_data.get("scores", [])
        # JSON 객체에서 "scores" 리스트를 추출.
    except Exception as ex:
        logger.warning(f"[llm_rerank] Re-rank parse failed: {ex}. Using default scores.")
        llm_scores = [0] * len(docs)
        # 파싱 실패 시 모든 문서에 대해 기본 점수 0 사용.
    
    weight_llm = 0.7     # LLM 평가 점수에 부여할 가중치.
    weight_manual = 0.3  # NER 직접 비교 점수에 부여할 가중치.
    
    weighted_scores = []  # 가중치가 적용된 최종 점수를 저장할 리스트 초기화.
    for idx, (doc, llm_score) in enumerate(zip(docs, llm_scores)):
        doc_ner_str = doc.metadata.get("LLM_NER", "{}")
        # 각 문서의 LLM_NER 정보를 문자열로 추출.
        try:
            doc_ner = json.loads(doc_ner_str)
            # 문자열을 JSON 객체로 파싱.
        except Exception as ex:
            logger.warning(f"[llm_rerank] Failed to parse document NER: {ex}")
            doc_ner = {}
            # 파싱 실패 시 빈 딕셔너리 사용.
        manual_score = compute_ner_similarity(user_ner, doc_ner)
        # 사용자 NER과 문서 NER의 매칭 점수를 계산.
        combined_score = weight_llm * llm_score + weight_manual * manual_score
        # LLM 점수와 수동 NER 비교 점수를 가중치 합산하여 최종 점수 산출.
        
        logger.info(
            f"[llm_rerank] Doc {idx+1} (ID: {doc.metadata.get('채용공고ID', 'no_id')}) - "
            f"LLM score: {llm_score}, Manual score: {manual_score}, Combined score: {combined_score}"
        )
        # 각 문서의 LLM 점수, 수동 점수, 최종 점수를 로그에 기록.
        weighted_scores.append((doc, combined_score))
        # 문서와 최종 점수 튜플을 weighted_scores 리스트에 추가.
    
    if len(weighted_scores) < len(docs):
        for i in range(len(weighted_scores), len(docs)):
            weighted_scores.append((docs[i], 0))
            # 만약 계산된 점수 리스트의 길이가 부족하면 남은 문서에 대해 기본 점수 0 할당.
    
    ranked_sorted = sorted(weighted_scores, key=lambda x: x[1], reverse=True)
    # 가중치가 적용된 점수를 기준으로 내림차순 정렬하여 높은 점수가 먼저 오도록 함.
    return [x[0] for x in ranked_sorted]
    # 정렬된 튜플에서 문서만 추출하여 최종 재랭킹된 문서 리스트 반환.

###########################################
# 추가: LLM_NER 기반 필터링 함수
###########################################
def search_by_llm_ner(user_ner: dict, docs: List[Document]) -> List[Document]:
    """
    각 문서의 metadata에 저장된 LLM_NER 정보를 활용해 사용자 조건과 일치하는 문서를 필터링합니다.
    """
    matching_docs = []  # 사용자 조건과 일치하는 문서를 저장할 리스트 초기화.
    for doc in docs:
        llm_ner_str = doc.metadata.get("LLM_NER", "{}")
        # 문서 메타데이터에서 LLM_NER 정보를 문자열로 추출.
        try:
            doc_llm_ner = json.loads(llm_ner_str)
            # 문자열을 JSON 객체로 파싱.
        except Exception as ex:
            logger.warning(f"LLM_NER 파싱 실패: {ex}")
            continue  # 파싱 실패 시 해당 문서는 건너뜀.
        
        job_match = True    # 직무 매칭 여부 초기값 True.
        region_match = True # 지역 매칭 여부 초기값 True.
        
        if user_ner.get("직무"):
            user_job = user_ner["직무"].strip().lower()
            # 사용자 NER의 직무 정보를 소문자로 정리.
            doc_job = doc_llm_ner.get("직무", "").strip().lower()
            # 문서 NER의 직무 정보를 소문자로 정리.
            if user_job and user_job not in doc_job:
                job_match = False  # 사용자 직무가 문서에 포함되지 않으면 False.
        
        if user_ner.get("지역") or user_ner.get("근무지역"):
            user_region = user_ner.get("지역", user_ner.get("근무지역", "")).strip().lower()
            # 사용자 NER의 지역 정보를 소문자로 정리.
            doc_region = doc_llm_ner.get("근무 지 역", "").strip().lower() or doc_llm_ner.get("근무지역", "").strip().lower()
            # 문서 NER에서 "근무 지 역" 또는 "근무지역" 정보를 소문자로 정리하여 추출.
            if user_region and user_region not in doc_region:
                region_match = False  # 사용자 지역이 문서에 포함되지 않으면 False.
        
        if job_match and region_match:
            matching_docs.append(doc)
            # 두 조건 모두 만족하면 해당 문서를 matching_docs 리스트에 추가.
    return matching_docs
    # 사용자 조건과 매칭된 문서 리스트 반환.

###########################################
# 8) 다단계 검색 (첫 검색은 LLM_NER 기반, 이후 param_filter_search 등 병합)
###########################################
def multi_stage_search(user_ner: dict) -> List[Document]:
    # 사용자 NER 정보를 바탕으로 여러 단계의 검색을 수행하는 함수.
    region = user_ner.get("지역", "").strip()
    # 사용자 NER에서 "지역" 값을 추출 및 정리.
    job = user_ner.get("직무", "").strip()
    # 사용자 NER에서 "직무" 값을 추출 및 정리.
    
    # 1. 첫 검색: 전체 DB를 대상으로 VectorDB에 저장된 LLM_NER 값으로 필터링
    initial_query = f"{job} {region}".strip()
    # 직무와 지역을 결합하여 초기 검색 쿼리 생성.
    initial_results_with_score = vectorstore.similarity_search_with_score(initial_query, k=1000)
    # 초기 쿼리로 벡터스토어에서 상위 1000개 문서를 유사도 점수와 함께 검색.
    all_docs = [doc for doc, score in initial_results_with_score]
    # 검색 결과에서 문서 부분만 추출.
    initial_candidates = search_by_llm_ner(user_ner, all_docs)
    # LLM_NER 기반 필터링을 통해 초기 후보 문서 리스트 생성.
    logger.info(f"[multi_stage_search] LLM_NER 초기 검색 결과: {len(initial_candidates)} 건")
    # 초기 후보 건수를 로그에 기록.
    
    # 충분한 후보가 없으면 전체 문서로 fallback
    if not initial_candidates or len(initial_candidates) < 5:
        initial_candidates = all_docs
        # 초기 후보 문서가 5건 미만이면 전체 문서를 초기 후보로 사용.
    
    # 2. 기존 다단계 검색 (param_filter_search 기반)으로 후보 보강
    docs_stage1 = param_filter_search(region, job, top_k=10) if region and job else []
    # region과 job 모두 존재할 경우, param_filter_search로 10개 문서 추출.
    docs_stage2 = param_filter_search(region=None, job=job, top_k=10) if job else []
    # 직무만 있는 경우, param_filter_search로 10개 문서 추출.
    docs_stage3 = []
    if job:
        synonyms = get_job_synonyms_with_llm(job)
        # 입력된 직무에 대한 동의어 추출.
        for syn in synonyms:
            docs_stage3 += param_filter_search(region=None, job=syn, top_k=10)
            # 각 동의어로 param_filter_search 실행하여 후보 문서를 추가.
    combined_multi = deduplicate_by_id(docs_stage1 + docs_stage2 + docs_stage3)
    # 다단계 검색 결과를 결합한 후 중복 제거.
    logger.info(f"[multi_stage_search] 다단계 검색 결과 (param_filter_search 기반): {len(combined_multi)} 건")
    # 다단계 검색 결과 건수를 로그에 기록.
    
    # 3. 초기 후보와 다단계 검색 결과 병합
    merged_candidates = deduplicate_by_id(initial_candidates + combined_multi)
    # 초기 후보와 다단계 검색 결과를 결합 후 중복 제거.
    logger.info(f"[multi_stage_search] 병합 후보 수: {len(merged_candidates)} 건")
    # 병합 후보 문서 수를 로그에 기록.
    
    # 후보 문서 수가 부족하면 fallback: 전체 임베딩 검색 추가
    if len(merged_candidates) < 15:
        hybrid_results = vectorstore.similarity_search_with_score(initial_query, k=15)
        # 병합 후보가 15건 미만일 경우, 초기 쿼리로 추가 15개 문서 검색.
        hybrid_docs = [doc for doc, _ in hybrid_results]
        # 추가 검색 결과에서 문서 부분만 추출.
        merged_candidates = deduplicate_by_id(merged_candidates + hybrid_docs)
        # 기존 후보와 추가 검색 결과를 결합 후 중복 제거.
    
    # 4. 최종 재랭킹 적용
    final_docs = llm_rerank(merged_candidates, user_ner)
    # LLM 재랭킹을 통해 최종 문서 순서를 결정.
    return final_docs
    # 최종 재랭킹된 문서 리스트 반환.

###########################################
# 9) FastAPI 엔드포인트 (채팅)
###########################################
@app.post("/api/v1/chat/", response_model=ChatResponse)
# "/api/v1/chat/" 경로로 POST 요청 시 ChatResponse 모델로 응답.
def chat_endpoint(req: ChatRequest):
    # ChatRequest 모델을 입력으로 받아 채팅 요청을 처리하는 엔드포인트 함수.
    try:
        user_message = req.user_message.strip()
        # 요청에서 사용자 메시지를 추출하고 양쪽 공백 제거.
        if not user_message:
            return ChatResponse(
                message="검색어가 비어있습니다.",
                jobPostings=[],
                type="info",
                user_profile=req.user_profile
            )
            # 사용자 메시지가 없으면 정보 메시지와 빈 결과 반환.
        
        logger.info(f"[chat_endpoint] user_message='{user_message}'")
        # 사용자 메시지를 로그에 기록.
        
        # 1) 사용자 입력 NER 추출
        ner_chain = get_user_ner_runnable()
        # 사용자 입력에 대한 NER 처리 체인 생성.
        ner_res = ner_chain.invoke({"user_query": user_message})
        # 사용자 메시지를 입력으로 NER 체인 실행.
        ner_str = ner_res.content if hasattr(ner_res, "content") else str(ner_res)
        # 응답에서 content 속성이 있으면 사용, 없으면 문자열 변환.
        cleaned = ner_str.replace("```json", "").replace("```", "").strip()
        # 응답 문자열에서 마크다운 구문 제거 후 정리.
        try:
            user_ner = json.loads(cleaned)
            # 정리된 문자열을 JSON 객체로 파싱하여 사용자 NER 정보 획득.
        except json.JSONDecodeError:
            logger.warning(f"[chat_endpoint] NER parse fail: {cleaned}")
            # 파싱 실패 시 경고 로그 출력.
            user_ner = {}  # 파싱 실패 시 빈 딕셔너리 사용.
        
        logger.info(f"[chat_endpoint] user_ner={user_ner}")
        # 추출된 사용자 NER 정보를 로그에 기록.
        
        # 2) 다단계 검색 실행 (첫 검색은 LLM_NER 기반)
        doc_list = multi_stage_search(user_ner)
        # 사용자 NER 정보를 기반으로 다단계 검색 수행하여 후보 문서 리스트 획득.
        top_docs = doc_list[:5]
        # 상위 5개 문서만 선택.
        
        job_postings = []
        # 반환할 JobPosting 객체들을 저장할 리스트 초기화.
        for i, doc in enumerate(top_docs, start=1):
            md = doc.metadata
            # 각 문서의 메타데이터 추출.
            job_postings.append(JobPosting(
                id=md.get("채용공고ID", "no_id"),  # 채용공고ID 추출, 없으면 "no_id" 사용.
                location=md.get("근무지역", ""),     # 근무지역 추출.
                company=md.get("회사명", ""),        # 회사명 추출.
                title=md.get("채용제목", ""),         # 채용제목 추출.
                salary=md.get("급여조건", ""),         # 급여조건 추출.
                workingHours=md.get("근무시간", "정보없음"),  # 근무시간 추출, 없으면 "정보없음" 사용.
                description=md.get("상세정보", "상세정보 없음"),  # 상세정보 추출, 없으면 "상세정보 없음" 사용.
                rank=i  # 문서 순위를 할당.
            ))
        logger.info(f"[chat_endpoint] 검색 결과: {len(job_postings)} 건")
        # 검색된 채용공고 건수를 로그에 기록.
        if job_postings:
            msg = f"'{user_message}' 검색 결과, 상위 {len(job_postings)}건을 반환합니다."
            res_type = "jobPosting"
            # 검색 결과가 있으면 적절한 메시지와 응답 타입 설정.
        else:
            msg = "조건에 맞는 채용공고를 찾지 못했습니다."
            res_type = "info"
            # 검색 결과가 없으면 정보 메시지와 응답 타입 설정.
        
        return ChatResponse(
            message=msg,
            jobPostings=job_postings,
            type=res_type,
            user_profile=req.user_profile
        )
        # 구성된 ChatResponse 객체를 반환하여 API 응답 전송.
    
    except Exception as e:
        logger.error(f"Error in chat_endpoint: {e}", exc_info=True)
        # 예외 발생 시 상세 에러 로그 기록.
        raise HTTPException(status_code=500, detail=str(e))
        # HTTP 500 에러를 발생시켜 클라이언트에 오류 응답 반환.

###########################################
# 10) 디버그용: 전체 DB 유사도 검색 과정 확인 엔드포인트
###########################################
@app.post("/api/v1/debug/similarity")
# "/api/v1/debug/similarity" 경로로 POST 요청 시 디버그 정보를 반환.
def debug_similarity_search(query: str = Query(..., description="유사도 검색에 사용할 쿼리 문자열")):
    """
    전체 DB(또는 충분히 많은 문서)를 대상으로 입력된 쿼리에 대한 유사도 검색 과정을 확인합니다.
    - vectorstore.similarity_search_with_score를 사용해 k 값을 크게 설정하여 전체 문서(또는 많은 문서)를 대상으로 검색합니다.
    - 각 문서의 메타데이터와 유사도 점수를 로그에 기록하며 결과를 반환합니다.
    """
    try:
        results_with_score = vectorstore.similarity_search_with_score(query, k=1000)
        # 입력된 쿼리를 사용해 벡터스토어에서 상위 1000개 문서를 유사도 점수와 함께 검색.
        debug_results = []  # 디버그 결과를 저장할 리스트 초기화.
        for idx, (doc, score) in enumerate(results_with_score):
            md = doc.metadata  # 각 문서의 메타데이터 추출.
            logger.info(
                f"[Debug Similarity] Doc {idx+1} (ID: {md.get('채용공고ID', 'no_id')}) - "
                f"Title: {md.get('채용제목', '정보없음')}, Score: {score}"
            )
            # 각 문서의 ID, 제목, 유사도 점수를 로그에 기록.
            debug_results.append({
                "id": md.get("채용공고ID", "no_id"),
                "title": md.get("채용제목", "정보없음"),
                "score": score,
                "metadata": md
            })
            # 각 문서의 상세 정보를 포함한 딕셔너리를 debug_results 리스트에 추가.
        return debug_results
        # 디버그 결과 리스트를 반환.
    except Exception as e:
        logger.error(f"Error in debug_similarity_search: {e}", exc_info=True)
        # 예외 발생 시 에러 로그 기록.
        raise HTTPException(status_code=500, detail=str(e))
        # HTTP 500 에러를 발생시켜 클라이언트에 오류 응답 반환.

###########################################
# 실행부
###########################################
if __name__ == "__main__":
    # 스크립트가 직접 실행될 경우 아래 코드를 실행.
    import uvicorn  # uvicorn 웹 서버 모듈 임포트.
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # uvicorn을 사용하여 FastAPI 앱을 호스트 0.0.0.0, 포트 8000에서 실행.
