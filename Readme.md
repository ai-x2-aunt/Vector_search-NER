### 프로젝트 소개
build_vectorstore.py:
데이터 소스를 기반으로 벡터 스토어를 생성하는 스크립트입니다.
(예: 텍스트 임베딩, 문서 검색 등을 위한 벡터 데이터 생성)

main.py:
생성된 벡터 스토어를 활용하여 주요 애플리케이션 기능을 실행합니다.
(예: 질의 응답, 검색 기능 실행 등)

jobs.json : 전체 데이터
Jobs_copy.json : 로그 확인을 테스트를 위한 데이터

---

### 설치 방법
Git 클론
```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

패키지 설치
```bash
pip install -r requirements.txt
```
- 부족한 패키지는 직접 설치
