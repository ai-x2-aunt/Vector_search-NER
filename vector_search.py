import logging
import os
from typing import List, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from dotenv import load_dotenv

# ============ LLM & NER ============
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def load_vectorstore(
    persist_dir: str = "./chroma_data",
    collection_name: str = "job_postings"
) -> Chroma:
    logger.info("Loading existing vector store")
    vectorstore = Chroma(
        embedding_function=embedding_model,
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    logger.info("Vector store loaded successfully")
    return vectorstore

# 1) 사용자 쿼리 NER (LLMChain)
def setup_query_ner_chain() -> LLMChain:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set in .env")

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-4o-mini",
        temperature=0.0
    )
    prompt = PromptTemplate(
        input_variables=["query"],
        template=(
            "사용자 입력: {query}\n\n"
            "위 문장에서 찾을 수 있는 직무, 지역, 연령대, 기타 중요한 요소를 JSON으로 추출해줘.\n"
            "예: {\"직무\":\"간호\", \"지역\":\"서울\", \"연령대\":\"50대\"} 등\n"
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

# 2) 검색 결과 요약 (LLMChain)
def setup_summary_chain() -> LLMChain:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-4o-mini",
        temperature=0.5
    )
    # 간단한 요약 프롬프트
    summary_prompt = PromptTemplate(
        input_variables=["query", "raw_docs"],
        template=(
            "사용자 검색어: {query}\n\n"
            "아래 채용공고 목록에 대해, 핵심 정보와 사용자 의도에 맞는지 분석하여 간략히 요약해줘.\n\n"
            "{raw_docs}\n\n"
            "응답 예시:\n"
            "- 총 {N}개 공고가 검색되었습니다.\n"
            "- 첫 번째 공고: ~ (요약)\n"
        )
    )
    chain = LLMChain(llm=llm, prompt=summary_prompt)
    return chain

def main():
    vectorstore = load_vectorstore()
    query_ner_chain = setup_query_ner_chain()
    summary_chain = setup_summary_chain()

    # 사용자 검색 입력
    query = input("\n🔍 검색어를 입력하세요: ").strip()
    if not query:
        print("검색어가 비어있습니다.")
        return

    # 1) 사용자 쿼리 NER
    ner_result_str = query_ner_chain.run({"query": query})
    import json
    try:
        ner_dict = json.loads(ner_result_str)
    except json.JSONDecodeError:
        ner_dict = {}

    # 2) NER 기반 쿼리 확장
    query_terms = []
    for k, v in ner_dict.items():
        if isinstance(v, str) and v:
            query_terms.append(v)
    # 원본 쿼리도 추가
    query_terms.append(query)
    final_query = " ".join(query_terms)

    # 3) 벡터 검색 (with score)
    #    유사도 점수도 함께 받아서 threshold 필터 가능
    results_with_score: List[Tuple[Document, float]] = vectorstore.similarity_search_with_score(final_query, k=5)
    if not results_with_score:
        print("검색된 결과가 없습니다.")
        return

    # 4) 결과 문서 표시
    print("\n==== 📌 검색된 채용공고 ====")
    doc_texts = []
    for i, (doc, score) in enumerate(results_with_score):
        md = doc.metadata
        snippet = (
            f"[{i+1}] 제목: {md.get('채용제목')} / "
            f"회사: {md.get('회사명')} / "
            f"근무지: {md.get('근무지역')} / "
            f"NER(공고): {md.get('NER', {})}"
            f"\n유사도 점수: {score}\n"
        )
        print(snippet)
        doc_texts.append(snippet)

    # 5) LLM을 통한 요약
    joined_docs = "\n".join(doc_texts)
    summary = summary_chain.run({"query": query, "raw_docs": joined_docs})
    print("\n=== LLM 요약 ===")
    print(summary)


if __name__ == "__main__":
    main()
