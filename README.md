# SCRABLE / Scarble

HyperCLOVA 1.5B 기반 고객 리뷰 답변 생성·평가·프롬프트 최적화 구현입니다.
사용자가 제시한 Algorithm 1/2와 RAG 흐름을 모듈로 구성했습니다.
**가중치 fine-tuning/LoRA가 아닌, 평가 피드백을 이용한 프롬프트 학습**입니다.

## 빠른 시작

프로젝트 **전체 폴더**를 GPU 서버에 복사하고 [train.ipynb](train.ipynb)를 열어 **Run All**을 실행하세요.
Python 3.10 이상(권장 3.11)이 필요합니다. 노트북 파일만 복사하면 `scrable/` 패키지를 찾을 수 없습니다.
노트북은 설치 → 모델 접근 확인 → AARSynth 자동 다운로드 → RAG 구축 → 반복 최적화 →
테스트 비교 → 그래프/JSON/프롬프트/ZIP 저장 → 새 리뷰 추론 순으로 실행됩니다.

기본 모델은 [`naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B`](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B)입니다.
공식 chat template, `tool_list` 역할과 `<|endofturn|>` / `<|stop|>` 종료 문자열을 적용합니다.
생성기·평가자·프롬프트 생성기가 **모델 객체 하나를 공유**합니다.
CUDA는 지원 시 bfloat16, 그 외 CUDA/MPS는 float16, CPU는 float32를 사용합니다.

### 최초 1회 모델 접근 설정

HyperCLOVA 공식 저장소는 접근 동의가 필요한 gated 모델입니다.
모델 페이지에서 조건에 동의한 계정으로 서버에서 `hf auth login`을 실행하거나 `HF_TOKEN`을 설정하세요.
노트북도 필요한 경우 토큰을 숨김 입력으로 받습니다. 모델 접근 동의 자체는 자동화할 수 없습니다.
기본 실행에는 유료 LLM API 키가 필요하지 않습니다.
모델 가중치 약 3.2 GB 외에 패키지/캐시 공간을 확보하세요. CPU 실행도 가능하지만 반복 추론은 오래 걸립니다.

### SSH 서버에서 Jupyter 실행

```bash
# 서버의 프로젝트 폴더에서
bash scripts/setup_gpu.sh
.venv/bin/hf auth login
.venv/bin/jupyter lab --ip=127.0.0.1 --port=8888 --no-browser --allow-root
```

```bash
# 개인 PC의 별도 터미널
ssh -L 8888:127.0.0.1:8888 user@your-gpu-server
```

서버 Jupyter가 출력한 토큰 URL을 PC 브라우저에서 열고 `SCRABLE (HyperCLOVA)` 커널을 선택합니다.
Python 경로는 `PYTHON_BIN=python3.11 bash scripts/setup_gpu.sh`로 지정할 수 있습니다.
CUDA용 PyTorch가 이미 설치된 환경을 사용해도 됩니다.

### 터미널 실행

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[local,notebook]'
hf auth login
python -m scrable.experiment --config configs/hyperclova.json

# LLM 없이 데이터 다운로드·분할만 검증
python -m scrable.experiment --prepare-only
```

## 파일 구성

```text
train.ipynb              전체 실행 노트북
configs/hyperclova.json  CLI 설정
scrable/
  config.py             모델·데이터·최적화 설정
  data.py               다운로드, JSONL 로드, 분할, TXT/MD/PDF 로드
  llm.py                Transformers/OpenAI 호환 API, 캐시, JSON 재시도
  prompts.py            기본 프롬프트와 고정 평가 기준
  pipeline.py           ResponseGen, Judge, ScoredResponseGen, IdentifyIR, PromptGen
  optimizer.py          Algorithm 1 반복 최적화
  rag.py                LangChain 분할, 임베딩, BM25+벡터 검색
  azure_rag.py           선택적 Azure AI Search
  experiment.py         실험 실행·독립 테스트 평가·결과 저장
tests/                  계산, 데이터 격리, 종료 조건, 통합 검증
scripts/setup_gpu.sh     서버 환경 설치
scripts/make_notebook.py 노트북 생성기
scripts/smoke_notebook.py 테스트용 LLM으로 노트북 실행 흐름 검증
```

## 알고리즘

1. 기본 프롬프트로 모든 train 리뷰의 답변과 평가 피드백을 생성합니다.
2. **하위 30% + 나머지에서 무작위 10%**를 선택합니다. 작은 데이터에서는 각 비율을 올림하고 중복을 제외합니다.
3. 선정된 답변과 개선 제안으로 프롬프트를 수정합니다.
4. 수정 후 **전체 train 리뷰**를 다시 생성·평가합니다.
5. train 평균이 threshold 이상이거나 최대 수정 횟수에 도달하면 종료합니다.

초기 점수가 이미 목표에 도달하면 추가 수정 없이 종료합니다. `max_iterations=0`은 기본 평가만 합니다.
Judge는 관련성·앱 특화성·정확도·문법을 **각각 별도 호출**로 1~5점 채점합니다.
`(raw - 1) / 4`로 정규화하고 `1:1:2:1` 가중 평균을 계산합니다. threshold는 0~1 값입니다.
해당 리뷰의 참고 답변은 **정확도 Judge에만** 제공하고 ResponseGen에는 제공하지 않습니다.
잘못된 JSON/점수는 수정 요청으로 재시도하고, 실패하면 오류로 중단합니다. 임의 점수를 만들지 않습니다.

validation은 프롬프트 생성에 사용하지 않고 최종 프롬프트 선택에만 사용합니다. 동점이면 앞선 프롬프트를 선택합니다.
test는 모든 최적화·선택이 끝난 뒤 기본/선택 프롬프트를 비교할 때만 사용합니다.
`best_prompt.txt`는 validation 최고 프롬프트, `final_prompt.txt`는 마지막 프롬프트입니다.
최종 선택이 기본 프롬프트일 수도 있으며, 테스트 점수 개선은 보장되지 않습니다.
동일 LLM이 생성과 평가를 수행하므로 결과 점수는 사람의 품질 평가와 구분해야 합니다.

기본값은 train 10 / validation 4 / test 4 / RAG 과거 사례 64개, 프롬프트 수정 최대 2회입니다.
이는 흐름 확인용 소규모 실험입니다. 규모를 늘릴 때 설정과 `output_dir`를 함께 바꾸세요.
캐시·JSON 재시도 제외 최대 호출 수는
`5 × ((train + validation) × (max_iterations + 1) + 2 × test) + max_iterations`이며 기본값은 252회입니다.
답변 생성 1회와 기준별 심사 4회가 포함되며 노트북의 새 리뷰 추론은 1회 추가됩니다.

## 데이터와 RAG

원 논문의 익명 앱 리뷰 49개는 공개 다운로드 링크를 확인하지 못했습니다.
대신 [AARSynth](https://huggingface.co/datasets/recmeapp/AARSynth)의 **실제 영어 앱 리뷰/개발자 답변**을 사용합니다.
고정 revision `49068c54a4551304208aea80749f1a4d5ab893d2`의 `replies/AllRepliesWithRRGen.txt`를 스트리밍합니다.
기본 앱은 `com.ubercab.eats`이며 적격 리뷰 3,000개 또는 원본 100,000줄까지 읽고 표본만 캐시에 저장합니다.
전체 236 MB를 반드시 다운로드하지는 않습니다. **파일 앞부분의 앱 표본**이므로 전체 모집단의 균일 표본이 아닙니다.
다른 앱을 선택하면 `max_scan_lines`를 늘려야 할 수 있습니다. 다운로드 실패 시 가짜 데이터로 대체하지 않습니다.

대소문자·공백을 정규화해 중복 리뷰를 제거하고 RAG/train/validation/test를 분리합니다.
개발자 답변은 과거 참고 답변으로, 독립적인 전문가 검증 정답은 아닙니다.
기본 RAG 자료는 **분리한 과거 리뷰·답변**이며 원 논문의 제품 매뉴얼과 다릅니다.
실제 TXT/MD/PDF 제품 문서는 `documents_dir`로 추가할 수 있습니다.
외부 문서에 테스트 정답을 넣지 않는 것은 데이터 작성자의 책임입니다.

문서를 LangChain으로 약 500 `cl100k_base` 토큰씩 분할하고 50 토큰 중첩을 줍니다.
이 토큰 수는 HyperCLOVA 토큰 수와 다릅니다. 기본 검색은 MiniLM 임베딩과 BM25의 순위를
Reciprocal Rank Fusion으로 결합하여 **같은 앱**의 상위 4개 문서를 반환합니다.
MiniLM 입력 길이보다 긴 청크는 여러 window를 평균하여 뒤쪽 내용을 버리지 않습니다.
기본 임베딩은 영어용이므로 한국어 데이터에는 `embedding_model`을 다국어 모델로 바꾸세요.
과거 정책과 `<url>`/`<email>` 자리표시자를 현재 정책/실제 주소로 날조하지 않도록 공통 지침을 적용합니다.

### 사용자 데이터

UTF-8 JSONL 한 줄마다 아래 구조를 사용하고 `custom_data`를 지정하세요.

```json
{"id":"r001","app_id":"my.app","review":"앱 실행 시 종료됩니다.","expert_response":"앱 버전과 기기 정보를 알려주세요."}
```

`expert_response`는 생략하거나 `null`로 둘 수 있습니다. 참고 답변이 없다면 `documents_dir`로 제품 문서를 공급하세요.
RAG용 리뷰도 따로 예약하므로 분할 크기의 합 이상으로 중복 없는 리뷰를 준비해야 합니다.

### Azure Search / Ada 선택

```bash
python -m pip install -e '.[azure]'
```

`AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_API_KEY`, `OPENAI_API_KEY`를 환경변수로 공급하고
`rag_backend="azure"`, `azure_embedding_model="text-embedding-ada-002"`를 지정합니다.
미리 준비한 Azure Search 서비스에 내용 해시별 인덱스를 생성하고 문서/벡터를 업로드합니다.
같은 데이터는 동일 인덱스를 보완·재사용하고, 다른 분할은 별도 인덱스를 사용합니다.
이 경로는 외부 서비스 사용과 비용이 발생하며 기본 로컬 실행에서는 사용되지 않습니다.
사용하지 않는 실험 인덱스는 Azure 포털에서 삭제하세요.

### OpenAI 호환 LLM API 선택

`python -m pip install -e '.[api]'` 후 `backend="openai"`, `model="사용 가능한 모델 ID"`를 지정하세요.
`OPENAI_API_KEY`, 필요 시 `OPENAI_BASE_URL` 환경변수를 사용합니다. `.env.example`은 변수 안내용이며 자동 로드하지 않습니다.
`judge_model`, `prompt_model`로 역할별 모델을 다르게 설정할 수도 있습니다.

## 결과와 재실행

`outputs/hyperclova/`에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `manifest.json` | 설정, 출처/revision, 분할 원문, 문서, 패키지 버전 |
| `iteration_000.json`, `history.json` | 프롬프트, 답변, 점수/이유/제안, 선정 리뷰 ID |
| `best_prompt.txt`, `final_prompt.txt` | 검증 최고 프롬프트 / 마지막 프롬프트 |
| `metrics.json`, `test_predictions.json` | 독립 테스트 비교와 생성 답변 |
| `result.json` | 전체 최적화 결과 |
| `llm_cache/`, `rag/` | LLM 응답, 청크/임베딩 캐시 |
| `score_history.png`, `results.zip` | 노트북에서 만드는 그래프·결과 묶음 |

중단 후 **같은 설정으로 재실행**하면 완료된 요청의 캐시를 재사용하여 반복을 재구성합니다.
진행 중이던 단일 호출은 다시 실행됩니다. 가중치 체크포인트 복원이 아닙니다.
다른 실험은 별도 output_dir를 사용해야 하며 설정이 다른 결과를 같은 폴더에 섞으면 오류가 납니다.
데이터·결과·캐시·토큰은 Git에 포함되지 않습니다.

## 검증

```bash
python -m pip install -e '.[dev,local,notebook]'
python -m pytest -q
ruff check scrable tests scripts
python scripts/smoke_notebook.py
```

단위/통합 테스트와 노트북 smoke test는 명시적인 테스트용 LLM fixture를 사용합니다.
이는 실행 흐름 검증이며 HyperCLOVA 실제 답변 품질 결과가 아닙니다.
실제 데이터 다운로드와 CPU 임베딩 검색은 별도로 확인합니다.
Azure 실서비스 검증에는 사용자 자격증명과 서비스가 필요합니다.

구현 시 확인한 결과:
- 자동 테스트 **22개 통과**, 정적 검사와 의존성 검사 통과.
- 실제 AARSynth 적격 리뷰 **3,000개 다운로드** 및 분할 성공.
- CPU MiniLM 임베딩으로 **64개 청크 인덱싱 / 상위 4개 검색** 성공.
- 테스트용 LLM을 주입한 노트북에서 최적화·평가·그래프·새 리뷰 추론·ZIP 저장까지 성공.
- 공개 초소형 테스트 가중치로 Transformers CPU 실제 생성 호출 성공.
- 현재 검증 환경에 gated 모델 접근 권한이 없어 **HyperCLOVA 실제 추론/성능 검증은 미실행**.
  위 테스트 결과는 HyperCLOVA의 점수 향상을 의미하지 않습니다.

## 참고 자료와 재현 범위

- Azov et al., [Self-Improving Customer Review Response Generation Based on LLMs](https://aclanthology.org/2024.ecnlp-1.5/), ECNLP 2024.
- Farooq et al., [App-Aware Response Synthesis for User Reviews](https://arxiv.org/abs/2007.15793), [데이터 원저자 저장소](https://github.com/AARSynth/Dataset).
- [AARSynth 데이터 카드](https://huggingface.co/datasets/recmeapp/AARSynth): 메타데이터상 CC-BY-4.0. 원저자와 출처를 유지하세요.
- [HyperCLOVA 사용 예제·라이선스](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B).
- [Azure AI Search 공식 문서](https://learn.microsoft.com/en-us/azure/search/search-get-started-vector).
- [OpenAI Chat Completions 공식 문서](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create) — 선택적 API backend.

원 논문의 GPT-4/Ada/Azure/제품 문서와 기본 실행의 HyperCLOVA/MiniLM/로컬 검색/과거 답변은 다릅니다.
이 프로젝트는 **아키텍처 구현**이며 원 논문의 성능 수치를 재현한 결과가 아닙니다.
