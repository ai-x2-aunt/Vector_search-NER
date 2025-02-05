import os  # 운영체제와 상호작용하기 위한 모듈을 불러옵니다. 예: 파일 경로 존재 여부 확인
import json  # JSON 형식의 데이터를 다루기 위한 모듈을 불러옵니다. 예: JSON 문자열을 딕셔너리로 변환
import re  # 정규표현식을 사용하여 문자열 검색 및 치환을 수행하는 모듈을 불러옵니다. 예: HTML 태그 제거
import logging  # 프로그램 실행 중 발생하는 이벤트를 기록하기 위한 모듈을 불러옵니다. 예: 에러 메시지 기록
from typing import Tuple, List, Dict, Any  # 코드 가독성을 높이기 위한 타입 힌트(Tuple, List, Dict, Any)를 불러옵니다
from dotenv import load_dotenv  # .env 파일에 저장된 환경변수를 불러오기 위한 모듈을 임포트합니다

# ========== LangChain / Chroma / LLM 관련 ==========
from langchain_community.chat_models import ChatOpenAI  # ChatOpenAI 모델을 불러와 대화형 LLM 기능을 사용합니다. 예: 챗봇 응답 생성
from langchain.prompts import PromptTemplate  # 사용자 정의 프롬프트 템플릿을 만들기 위한 클래스를 임포트합니다
from langchain.schema.runnable import Runnable  # 실행 가능한 파이프라인 구성 요소를 나타내는 인터페이스를 불러옵니다
from langchain_huggingface import HuggingFaceEmbeddings  # HuggingFace 기반 임베딩 모델 래퍼를 불러옵니다. 예: 텍스트 임베딩 생성
from langchain_chroma import Chroma  # Chroma 벡터스토어와 상호작용하기 위한 클래스를 임포트합니다
from langchain.docstore.document import Document  # 문서 객체를 생성하기 위한 클래스를 불러옵니다. 예: 텍스트와 메타데이터 저장
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 긴 텍스트를 일정 크기로 분할하는 도구를 임포트합니다
from chromadb.config import Settings  # ChromaDB의 클라이언트 설정을 구성하기 위한 설정 객체를 불러옵니다

import shutil  # 파일이나 디렉토리 삭제 등 고수준 파일 작업을 위한 모듈을 불러옵니다

persist_dir = "./chroma_data"  # 벡터스토어 데이터를 저장할 디렉토리 경로를 지정합니다. 예: 현재 폴더 내 "chroma_data" 폴더
if os.path.exists(persist_dir):  # 지정한 디렉토리가 이미 존재하는지 확인합니다.
    shutil.rmtree(persist_dir)  # 존재하면 디렉토리와 그 안의 모든 파일을 삭제합니다. 예: 이전 실행 결과 삭제

load_dotenv()  # .env 파일에 저장된 환경변수들을 메모리에 로드합니다. 예: OPENAI_API_KEY를 설정

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# 로그의 기본 설정을 합니다. 레벨은 INFO 이상, 출력 형식은 "시간 [레벨] 메시지" 형식입니다.
logger = logging.getLogger(__name__)  # 현재 모듈 이름을 기반으로 로거 객체를 생성합니다.

# 1) 임베딩 모델
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 사용할 임베딩 모델의 이름을 지정합니다.
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
# 지정된 모델을 바탕으로 HuggingFaceEmbeddings 객체를 생성합니다. 예: 텍스트를 벡터로 변환

# 2) Chunking 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,    # 한 청크(chunk)의 최대 문자 수를 300자로 지정합니다. 예: 긴 텍스트를 300자씩 나눔
    chunk_overlap=50   # 청크 간에 50자씩 겹치도록 설정합니다. 예: 정보 손실 방지를 위해 중복 영역 포함
)
# 이 객체는 텍스트를 설정한 크기로 나누어 후속 처리(임베딩 등)에 사용됩니다.

# 3) LLM + NER 프롬프트
def get_ner_runnable() -> Runnable:
    # 채용공고 텍스트에서 NER(개체명 인식) 정보를 추출하는 Runnable 객체를 생성하는 함수입니다.
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    # 환경변수에서 OPENAI_API_KEY 값을 읽어옵니다.
    if not openai_api_key:
        # API 키가 존재하지 않으면
        raise ValueError("OPENAI_API_KEY is not set in environment.")
        # API 키가 없다는 에러 메시지와 함께 예외를 발생시킵니다.

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,  # API 키를 사용하여 LLM 클라이언트를 초기화합니다.
        model_name="gpt-4o-mini",         # 사용할 모델 이름을 지정합니다. 예: gpt-4o-mini, gpt-3.5-turbo 등
        temperature=0.0                   # 생성되는 텍스트의 창의성(랜덤성)을 0으로 설정하여 결정적인 결과를 유도합니다.
    )

    # NER 추출 프롬프트 생성: 사용자가 추출해야 하는 정보를 명시합니다.
    ner_prompt = PromptTemplate(
        input_variables=["text"],  # 프롬프트에 필요한 입력 변수 이름을 지정합니다.
        template=(
            "다음 채용공고 텍스트에서 주요 정보를 JSON 형식으로 추출해줘.\n\n"  # 지시문
            "추출해야 할 정보:\n"  # 추출 대상 항목 나열
            "- **직무**\n"
            "- **회사명**\n"
            "- **근무 지역**\n"
            "- **연령대**\n"
            "- **경력 요구 사항**\n"
            "- **학력 요건**\n"
            "- **급여 정보**\n"
            "- **고용 형태**\n"
            "- **복리후생**\n\n"
            "JSON 예:\n"
            "{{\n"
            "  \"직무\": \"백엔드 개발자\",\n"
            "  \"회사명\": \"ABC 테크\",\n"
            "  \"근무 지역\": \"서울\",\n"
            "  \"연령대\": \"20~30대 선호\",\n"
            "  \"경력 요구 사항\": \"경력 3년 이상\",\n"
            "  \"학력 요건\": \"대졸 이상\",\n"
            "  \"급여 정보\": \"연봉 4000만원 이상\",\n"
            "  \"고용 형태\": \"정규직\",\n"
            "  \"복리후생\": [\"4대보험\", \"식대 지원\"]\n"
            "}}\n\n"
            "채용 공고:\n"
            "{text}\n\n"
            "채용 공고 내에 없는 정보는 비워두거나 적절한 방식으로 처리하고 위 정보를 JSON으로만 응답해줘."
            # {text} 부분에 실제 채용공고 텍스트가 삽입됩니다.
        )
    )

    runnable = ner_prompt | llm
    # 프롬프트 템플릿과 LLM을 파이프라인 방식으로 연결하여 입력 텍스트에 대해 NER 처리를 수행하도록 구성합니다.
    return runnable
    # 구성된 Runnable 객체를 반환합니다.

# 4) JSON 로드
def load_data(json_file: str = "jobs.json") -> dict:
    # 주어진 JSON 파일에서 데이터를 읽어와 파이썬 딕셔너리로 반환하는 함수입니다.
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            # 파일을 읽기 모드("r")와 UTF-8 인코딩으로 엽니다.
            data = json.load(f)
            # 파일 내용을 JSON 형식으로 파싱하여 data에 저장합니다.
        logger.info(f"Loaded {len(data.get('채용공고목록', []))} postings.")
        # 로드된 데이터에서 "채용공고목록" 리스트의 길이를 로그에 기록합니다.
        return data
        # 파싱된 데이터를 반환합니다.
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        # 에러 발생 시 에러 메시지를 로그에 기록합니다.
        return {"채용공고목록": []}
        # 에러가 발생한 경우 빈 채용공고목록을 반환합니다.

# 5) 텍스트 전처리
def clean_text(text: str) -> str:
    # 주어진 텍스트에서 HTML 태그를 제거하고 불필요한 공백을 정리하는 함수입니다.
    if not isinstance(text, str):
        # 입력이 문자열이 아닌 경우
        return ""  # 빈 문자열을 반환합니다.
    return re.sub(r"<[^>]+>", "", text).replace("\n", " ").strip()
    # 정규표현식을 사용해 HTML 태그를 제거하고, 줄바꿈(\n)을 공백으로 치환한 후 양쪽 공백을 제거합니다.

# ================================
# 6) Document 준비: NER 결과를 page_content에 병합
# ================================
# 수정된 prepare_documents 함수 (ID 생성 로직 추가)
def prepare_documents(data: dict) -> Tuple[List[Document], List[str]]:
    # JSON 데이터에서 채용 공고 목록을 읽어 Document 객체와 각 문서의 고유 ID 리스트를 생성하여 반환합니다.
    ner_runnable = get_ner_runnable()
    # NER 처리를 위한 Runnable 객체를 생성합니다.
    documents = []  # Document 객체들을 저장할 빈 리스트를 초기화합니다.
    doc_ids = []  # 각 Document의 고유 ID를 저장할 리스트를 초기화합니다.

    for idx_item, item in enumerate(data.get("채용공고목록", [])):
        # 채용공고목록 내 각 아이템에 대해 인덱스(idx_item)와 함께 순회합니다.
        posting_id = item.get("공고번호", "no_id")
        # 채용 공고의 고유 번호를 추출하며, 없으면 "no_id"를 사용합니다.
        job_title = clean_text(item.get("채용제목", ""))
        # 채용제목을 전처리(clean_text)하여 job_title에 저장합니다.
        company_name = clean_text(item.get("회사명", ""))
        # 회사명을 전처리하여 company_name에 저장합니다.
        work_location = clean_text(item.get("근무지역", ""))
        # 근무지역 정보를 전처리하여 work_location에 저장합니다.
        salary_condition = clean_text(item.get("급여조건", ""))
        # 급여조건 정보를 전처리하여 salary_condition에 저장합니다.
        job_posting_id = item.get("채용공고ID", "정보없음")
        # 채용공고ID를 추출하며, 없으면 "정보없음"으로 지정합니다.
        job_posting_url = item.get("채용공고URL", "정보없음")
        # 채용공고 URL을 추출하며, 없으면 "정보없음"으로 지정합니다.

        details = item.get("상세정보", {})
        # 상세정보 부분을 추출합니다. 없으면 빈 딕셔너리를 사용합니다.
        job_description = ""  # 직무 내용 초기화를 합니다.
        requirements_text = ""  # 세부요건 내용을 저장할 문자열을 초기화합니다.

        if isinstance(details, dict):
            # 상세정보가 딕셔너리 형태라면
            job_description = clean_text(details.get("직무내용", ""))
            # 상세정보에서 "직무내용"을 전처리하여 job_description에 저장합니다.
            requirements_list = details.get("세부요건", [])
            # 세부요건 리스트를 추출합니다. 없으면 빈 리스트를 사용합니다.
            for requirement in requirements_list:
                # 각 세부요건(딕셔너리)을 순회합니다.
                for k, v in requirement.items():
                    # 각 키(k)와 값(v)에 대해
                    if isinstance(v, list):
                        # 값이 리스트 형태인 경우
                        requirements_text += f"{k}: {' '.join(v)}\n"
                        # 리스트의 항목들을 공백으로 연결하여 문자열로 추가합니다. 예: "스킬: Python Java"
                    else:
                        requirements_text += f"{k}: {v}\n"
                        # 값이 문자열이면 그대로 문자열에 추가합니다.
        else:
            job_description = clean_text(str(details))
            # 상세정보가 딕셔너리가 아니면 문자열로 변환 후 전처리하여 사용합니다.

        # 1) 원본 텍스트 생성: 채용 제목, 회사명, 근무지역, 급여조건, 직무내용, 세부요건 등을 포함
        combined_text = (
            f"{job_title}\n"
            f"회사명: {company_name}\n"
            f"근무지역: {work_location}\n"
            f"급여조건: {salary_condition}\n"
            f"{job_description}\n{requirements_text}"
        )

        # 2) LLM NER 호출: 채용공고 텍스트에서 주요 정보를 추출합니다.
        try:
            res = ner_runnable.invoke({"text": combined_text})
            # NER Runnable을 호출하여 combined_text에서 개체명 정보를 추출합니다.
            if hasattr(res, "content"):
                ner_result_str = res.content
                # 응답 객체에 content 속성이 있으면 그 값을 사용합니다.
            else:
                ner_result_str = str(res)
                # 그렇지 않으면 응답 객체를 문자열로 변환합니다.

            # 백틱(```)을 제거하는 등 마크다운 문법 정리
            ner_str_cleaned = ner_result_str.replace("```json", "").replace("```", "").strip()

            try:
                ner_data = json.loads(ner_str_cleaned)
                # 정리된 문자열을 JSON 형식으로 파싱하여 딕셔너리 형태로 변환합니다.
                logging.info(f"NER JSON parse : {ner_data}")
                # 파싱 성공 시 결과를 로그에 기록합니다.
            except json.JSONDecodeError:
                logger.warning(f"[{posting_id}] NER JSON parse fail: {ner_result_str}")
                # 파싱 실패 시 경고 메시지를 로그에 기록합니다.
                ner_data = {}  # 실패 시 빈 딕셔너리를 사용합니다.
        except Exception as e:
            logger.warning(f"[{posting_id}] NER invoke fail: {e}")
            # NER 호출 중 예외가 발생하면 경고 메시지를 로그에 기록합니다.
            ner_data = {}  # 예외 발생 시 빈 딕셔너리를 사용합니다.

        # 3) NER 정보를 문자열로 변환하여 최종 텍스트에 병합합니다.
        if ner_data:
            ner_as_text = f"\n[NER 추출 정보]\n{json.dumps(ner_data, ensure_ascii=False, indent=2)}\n"
            # NER 데이터가 있으면 보기 좋게 포매팅하여 문자열로 변환합니다.
        else:
            ner_as_text = "\n[NER 추출 정보]\n{}"
            # NER 데이터가 없으면 빈 JSON 객체 문자열을 사용합니다.

        # 최종 텍스트: 원본 텍스트와 NER 정보를 합칩니다.
        final_text = combined_text + ner_as_text

        # 4) chunking: 최종 텍스트를 지정된 청크 크기로 분할합니다.
        splits = text_splitter.split_text(final_text)

        # 5) Document 생성: 각 청크별로 Document 객체와 고유 ID를 생성합니다.
        for idx_chunk, chunk_text in enumerate(splits):
            # 각 청크에 대해 인덱스와 텍스트를 순회합니다.
            doc_id = f"{posting_id}_chunk{idx_chunk}_{hash(chunk_text[:50])}"
            # 공고번호, 청크 인덱스, 청크의 앞 50자에 대한 해시값을 결합하여 고유 ID를 생성합니다.
            doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id)
            # 고유 ID에서 알파벳, 숫자, _ , - 외의 문자는 '_'로 치환하여 안전한 문자열로 만듭니다.

            doc = Document(
                page_content=chunk_text,  # 청크 텍스트를 문서 내용으로 지정합니다.
                metadata={
                    "공고번호": posting_id,  # 원본 채용 공고의 고유 번호
                    "채용제목": job_title,   # 채용 제목
                    "회사명": company_name,  # 회사명
                    "근무지역": work_location,  # 근무지역
                    "급여조건": salary_condition,  # 급여 조건
                    "채용공고ID": job_posting_id,    # 채용공고 ID
                    "채용공고URL": job_posting_url,  # 채용공고 URL
                    "직무내용": job_description,      # 직무 내용
                    "세부요건": requirements_text,   # 세부 요건
                    "LLM_NER": json.dumps(ner_data, ensure_ascii=False),  # NER 결과를 JSON 문자열로 저장
                    "chunk_index": idx_chunk,  # 해당 청크의 인덱스
                    "unique_id": doc_id      # 생성한 고유 ID
                }
            )
            documents.append(doc)  # 생성된 Document 객체를 documents 리스트에 추가합니다.
            doc_ids.append(doc_id)  # 생성된 고유 ID를 doc_ids 리스트에 추가합니다.

        logger.info(f"[{idx_item+1}/{len(data['채용공고목록'])}] 공고번호: {posting_id}, "
                    f"문서 {len(splits)}개 생성, NER keys={list(ner_data.keys())}")
        # 현재 채용 공고 처리 결과(공고번호, 생성된 청크 수, NER에서 추출된 키 목록)를 로그에 기록합니다.

    logger.info(f"총 {len(documents)}개의 Document가 생성되었습니다.")
    # 전체 Document 객체 생성 완료 후 총 개수를 로그에 기록합니다.
    return documents, doc_ids  # 생성된 Document 리스트와 고유 ID 리스트를 함께 반환합니다.

# 7) 벡터스토어 생성 & 저장
def build_vectorstore(documents: List[Document], ids: List[str], persist_dir: str = "./chroma_data") -> Chroma:
    # Document 리스트와 해당 문서들의 고유 ID 리스트를 받아 Chroma 벡터스토어를 생성 및 저장하는 함수입니다.
    logger.info("Building vector store with Chroma...")
    try:
        client_settings = Settings(anonymized_telemetry=False)
        # Chroma 클라이언트 설정을 구성합니다. 익명화 텔레메트리를 비활성화합니다.
        vectorstore = Chroma.from_documents(
            documents=documents,      # 생성된 Document 리스트를 전달합니다.
            ids=ids,                  # 각 Document의 고유 ID 리스트를 전달합니다.
            embedding=embedding_model,  # 임베딩 모델을 전달하여 텍스트를 벡터로 변환합니다.
            collection_name="job_postings",  # 컬렉션 이름을 "job_postings"로 지정합니다.
            persist_directory=persist_dir,   # 벡터 데이터가 저장될 디렉토리를 지정합니다.
            client_settings=client_settings  # 클라이언트 설정 객체를 전달합니다.
        )
        total_docs = vectorstore._collection.count()
        # 벡터스토어 내 저장된 전체 문서 수를 확인합니다.
        logger.info(f"Stored {total_docs} documents in Chroma.")
        # 저장된 문서 수를 로그에 기록합니다.
        return vectorstore  # 생성된 벡터스토어 객체를 반환합니다.
    except Exception as e:
        logger.error(f"Error building vector store: {e}")
        # 벡터스토어 생성 중 예외 발생 시 에러 메시지를 로그에 기록합니다.
        raise  # 예외를 호출자에게 전달합니다.

# 8) main() 함수
def main():
    # 프로그램 실행의 시작점인 main 함수입니다.
    data = load_data("jobs.json")
    # "jobs.json" 파일에서 데이터를 로드합니다.
    if not data.get("채용공고목록"):
        # 채용공고목록이 비어있으면
        logger.error("No job postings found in the JSON.")
        # 에러 로그를 기록한 후
        return  # 함수 실행을 중단합니다.

    # 중복 공고번호 체크: 같은 공고번호가 여러 번 있는지 확인합니다.
    seen = set()  # 이미 처리한 공고번호를 저장할 집합을 초기화합니다.
    duplicates = []  # 중복된 공고번호를 저장할 리스트를 초기화합니다.
    for item in data["채용공고목록"]:
        # 채용공고목록의 각 항목에 대해 순회합니다.
        공고번호 = item.get("공고번호")
        # 각 항목에서 공고번호를 추출합니다.
        if 공고번호 in seen:
            duplicates.append(공고번호)
            # 이미 처리된 공고번호라면 duplicates 리스트에 추가합니다.
        seen.add(공고번호)
        # 현재 공고번호를 seen 집합에 추가합니다.
    
    if duplicates:
        logger.warning(f"중복 공고번호 발견: {list(set(duplicates))}")
        # 중복된 공고번호가 있다면 경고 메시지를 로그에 기록합니다.

    docs, ids = prepare_documents(data)
    # JSON 데이터를 기반으로 Document 객체 리스트와 고유 ID 리스트를 생성합니다.
    if not docs:
        logger.error("No documents were prepared.")
        # 문서 생성에 실패하면 에러 로그를 기록한 후
        return  # 함수 실행을 중단합니다.

    vectorstore = build_vectorstore(docs, ids)
    # 생성된 Document 리스트와 ID 리스트를 이용하여 벡터스토어를 생성합니다.
    final_count = vectorstore._collection.count()
    # 최종적으로 벡터스토어에 저장된 문서의 총 개수를 확인합니다.
    logger.info(f"최종 저장된 문서 수: {final_count}")
    # 저장된 문서 수를 로그에 기록합니다.

if __name__ == "__main__":
    # 현재 스크립트가 직접 실행될 때 main() 함수를 호출합니다.
    main()  # main 함수 실행하여 전체 프로세스를 시작합니다.
