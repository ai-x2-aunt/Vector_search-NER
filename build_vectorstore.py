import os
import json
import re
import logging
from typing import Tuple, List, Dict, Any
from dotenv import load_dotenv

# ========== LangChain / Chroma / LLM 관련 ==========
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings

from sentence_transformers import SentenceTransformer

import shutil
# main() 시작 부분에 추가
persist_dir = "./chroma_data"  # 추가
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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

# KURE-v1 모델 사용
embedding_model = SentenceTransformerEmbeddings(model_name="nlpai-lab/KURE-v1")

# 2) Chunking 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

# 3) LLM + NER 프롬프트
def get_ner_runnable() -> Runnable:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment.")

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-4o-mini",  # 또는 gpt-3.5-turbo / gpt-4
        temperature=0.0
    )

    # NER 추출 프롬프트
    ner_prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "다음 채용공고 텍스트에서 주요 정보를 JSON 형식으로 추출해줘.\n\n"
            "추출해야 할 정보:\n"
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
        )
    )

    runnable = ner_prompt | llm
    return runnable

# 4) JSON 로드
def load_data(json_file: str = "jobs.json") -> dict:
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data.get('채용공고목록', []))} postings.")
        return data
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {"채용공고목록": []}

# 5) 텍스트 전처리
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"<[^>]+>", "", text).replace("\n", " ").strip()

# ================================
# 6) Document 준비: NER 결과를 page_content에 병합
# ================================
# 수정된 prepare_documents 함수 (ID 생성 로직 추가)
def prepare_documents(data: dict) -> Tuple[List[Document], List[str]]:
    ner_runnable = get_ner_runnable()
    documents = []
    doc_ids = []  # ID 리스트 추가

    for idx_item, item in enumerate(data.get("채용공고목록", [])):
        posting_id = item.get("공고번호", "no_id")
        job_title = clean_text(item.get("채용제목", ""))
        company_name = clean_text(item.get("회사명", ""))
        work_location = clean_text(item.get("근무지역", ""))
        salary_condition = clean_text(item.get("급여조건", ""))
        job_posting_id = item.get("채용공고ID", "정보없음")
        job_posting_url = item.get("채용공고URL", "정보없음")

        details = item.get("상세정보", {})
        job_description = ""
        requirements_text = ""

        if isinstance(details, dict):
            job_description = clean_text(details.get("직무내용", ""))
            requirements_list = details.get("세부요건", [])
            for requirement in requirements_list:
                for k, v in requirement.items():
                    if isinstance(v, list):
                        requirements_text += f"{k}: {' '.join(v)}\n"
                    else:
                        requirements_text += f"{k}: {v}\n"
        else:
            job_description = clean_text(str(details))

        # 1) 원본 텍스트
        combined_text = (
            f"{job_title}\n"
            f"회사명: {company_name}\n"
            f"근무지역: {work_location}\n"
            f"급여조건: {salary_condition}\n"
            f"{job_description}\n{requirements_text}"
        )

        # 2) LLM NER 호출
        try:
            res = ner_runnable.invoke({"text": combined_text})
            if hasattr(res, "content"):
                ner_result_str = res.content
            else:
                ner_result_str = str(res)

            # 백틱 제거 등
            ner_str_cleaned = ner_result_str.replace("```json", "").replace("```", "").strip()

            try:
                ner_data = json.loads(ner_str_cleaned)
                logging.info(f"NER JSON parse : {ner_data}")
            except json.JSONDecodeError:
                logger.warning(f"[{posting_id}] NER JSON parse fail: {ner_result_str}")
                ner_data = {}
        except Exception as e:
            logger.warning(f"[{posting_id}] NER invoke fail: {e}")
            ner_data = {}

        # 3) NER 정보를 문자열로 변환하여 page_content에 추가
        #    예) "NER 추출 정보: 직무=..., 회사명=..., ..."
        ner_as_text = ""
        if ner_data:
            # 간단히 JSON 전체를 문자열로
            ner_as_text = f"\n[NER 추출 정보]\n{json.dumps(ner_data, ensure_ascii=False, indent=2)}\n"
        else:
            ner_as_text = "\n[NER 추출 정보]\n{}"

        # 최종적으로 임베딩에 반영될 텍스트
        final_text = combined_text + ner_as_text

        # 4) chunking
        splits = text_splitter.split_text(final_text)

        # 5) Document 생성 부분 수정
        for idx_chunk, chunk_text in enumerate(splits):
            # 고유 ID 생성: 공고번호 + 청크인덱스 + 추가유니크조합
            doc_id = f"{posting_id}_chunk{idx_chunk}_{hash(chunk_text[:50])}"  # 첫 50자 해시값 추가
            doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id)  # 특수문자 처리

            doc = Document(
                page_content=chunk_text,
                metadata={
                    "공고번호": posting_id,
                    "채용제목": job_title,
                    "회사명": company_name,
                    "근무지역": work_location,
                    "급여조건": salary_condition,
                    "채용공고ID": job_posting_id,   
                    "채용공고URL": job_posting_url,
                    "직무내용": job_description,
                    "세부요건": requirements_text,
                    "LLM_NER": json.dumps(ner_data, ensure_ascii=False),
                    "chunk_index": idx_chunk,
                    "unique_id": doc_id  # 메타데이터에 고유 ID 추가
                }
            )
            documents.append(doc)
            doc_ids.append(doc_id)  # ID 리스트에 추가

        logger.info(f"[{idx_item+1}/{len(data['채용공고목록'])}] 공고번호: {posting_id}, "
                    f"문서 {len(splits)}개 생성, NER keys={list(ner_data.keys())}")

    logger.info(f"총 {len(documents)}개의 Document가 생성되었습니다.")
    return documents, doc_ids  # 두 값을 함께 반환

# 7) 벡터스토어 생성 & 저장
# 수정된 build_vectorstore 함수
def build_vectorstore(documents: List[Document], ids: List[str], persist_dir: str = "./chroma_data") -> Chroma:
    logger.info("Building vector store with Chroma...")
    try:
        client_settings = Settings(anonymized_telemetry=False)
        vectorstore = Chroma.from_documents(
            documents=documents,
            ids=ids,  # 명시적 ID 지정 추가
            embedding=embedding_model,
            collection_name="job_postings",
            persist_directory=persist_dir,
            client_settings=client_settings
        )
        total_docs = vectorstore._collection.count()
        logger.info(f"Stored {total_docs} documents in Chroma.")
        return vectorstore
    except Exception as e:
        logger.error(f"Error building vector store: {e}")
        raise

# 8) main() 함수
# 수정된 main 함수
def main():
    data = load_data("jobs.json")
    if not data.get("채용공고목록"):
        logger.error("No job postings found in the JSON.")
        return

    # 중복 공고번호 체크
    seen = set()
    duplicates = []
    for item in data["채용공고목록"]:
        공고번호 = item.get("공고번호")
        if 공고번호 in seen:
            duplicates.append(공고번호)
        seen.add(공고번호)
    
    if duplicates:
        logger.warning(f"중복 공고번호 발견: {list(set(duplicates))}")

    docs, ids = prepare_documents(data)  # ID 리스트 반환 받기
    if not docs:
        logger.error("No documents were prepared.")
        return

    vectorstore = build_vectorstore(docs, ids)  # ID 리스트 전달
    final_count = vectorstore._collection.count()
    logger.info(f"최종 저장된 문서 수: {final_count}")

if __name__ == "__main__":
    main()
