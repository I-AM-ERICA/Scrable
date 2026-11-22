# GPU 없이 노트북에서 사이트까지 돌려 보기

가게비서 웹 화면의 **고객응대**에서 리뷰와 **가게 정보**를 넣으면, 노트북 CPU에서 도는 SCRABLE(최종 모델)이 답글을 씁니다. 학습은 하지 않고 돌리기만 하므로 GPU가 필요 없습니다.

```
브라우저 localhost:7001 (프론트, WSL)
   → 백엔드 localhost:8000 (WSL, H2 메모리 DB, 데모 계정)
      → SCRABLE 127.0.0.1:50053 (WSL, CPU, bfloat16)
```

- 셋 다 WSL 안에서 localhost로만 돕니다. Windows 방화벽은 건드리지 않습니다
- 메모리: SCRABLE 약 3.5GB + 백엔드 약 1GB. WSL에 6GB 이상 비어 있어야 합니다(`free -m`의 available)
- 답글 한 개에 CPU로 20~40초 걸립니다(GPU는 2~4초)
- 내부 키는 스크립트가 실행할 때마다 새로 만들어 SCRABLE과 백엔드에만 넘깁니다. 화면에 나오지 않습니다

## 가게 정보 칸

고객응대 화면의 별점·말투 아래에 **가게 정보**(선택) 칸이 있습니다. 메뉴, 영업시간, 배달·환불·연락 안내, 진행 중인 이벤트를 적으면 SCRABLE이 그 글을 가게 문서로 써서 RAG로 리뷰와 관련된 부분을 찾아 답글에 넣습니다(학습할 때와 같은 `retrieved_documents` 칸). 이 브라우저에 가게별로 저장돼서 다음에도 채워져 있습니다. 서버 DB에는 저장하지 않습니다.

바뀐 곳(모두 로컬 `feat/store-info` 브랜치, 아직 GitHub에 올리지 않음):

| 레포 | 변경 |
|---|---|
| am-erica-frontend | 고객응대 화면에 가게 정보 칸, 요청에 `storeInfo` 추가 |
| am-erica-backend | `ReplyDraftRequest.storeInfo`(2,000자) → gRPC `store_info`. 별점 `@Min(1)` → `@Min(0)`(프론트의 "모름"이 0을 보내서 400이 나던 문제) |
| Scrable | `customer.proto`에 `store_info = 5`, 서버가 그 글을 가게 문서로 사용, CPU bfloat16 옵션(`SCRABLE_CPU_BF16=1`) |

## 1. [WSL] 처음 한 번만

이 venv는 uv로 만들어서 `.venv/bin/pip`가 없습니다. uv로 설치합니다.

```bash
cd ~/naver/Scrable && uv pip install --python .venv/bin/python "sentence-transformers>=3"
```

임베딩 모델(공개, 약 470MB)을 미리 받아 둡니다. 실행할 때는 인터넷 없이 돌도록 해 두었습니다.

```bash
.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
```

최종 어댑터 풀기 (드라이브에서 받은 zip이 Windows `Downloads`에 있을 때):

WSL에 `unzip`이 없어도 되게 Python으로 풉니다.

```bash
cd ~/naver/Scrable && mkdir -p outputs/stage4_photo && .venv/bin/python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall('/tmp/adapter_zip')" "/mnt/c/Users/$(cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r')/Downloads/scrable-stage4-photo-adapter.zip" && cp -r /tmp/adapter_zip/scrable-stage4-photo-adapter/adapter outputs/stage4_photo/
```

HyperCLOVA 1.5B 모델: Windows에 이미 받아 두었다면 복사하고, 아니면 Hugging Face에 로그인해서 받습니다.

```bash
mkdir -p ~/.cache/huggingface/hub && cp -r "/mnt/c/Users/$(cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r')/.cache/huggingface/hub/models--naver-hyperclovax--HyperCLOVAX-SEED-Text-Instruct-1.5B" ~/.cache/huggingface/hub/
```

백엔드·프론트는 RUN_SITE.md 0~1번(Java 21, Node 22, `npm install`)이 되어 있으면 됩니다.

## 2. [WSL] 실행

```bash
cd ~/naver/Scrable && bash scripts/run_local_site.sh
```

`1/3 SCRABLE`, `2/3 Backend`, `3/3 Frontend`가 차례로 `ready`가 되면 준비된 겁니다(처음엔 5분 안팎). 끝낼 때는 `Ctrl+C` 한 번으로 셋 다 꺼집니다.

## 3. [브라우저] 데모 계정 로그인과 리뷰 넣기

Windows 브라우저에서 **http://localhost:7001** → `F12` → Console에 붙여넣기:

```js
const r = await fetch('/api/auth/login', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({email: 'demo@america.local', password: 'demo1234!'})});
const d = await r.json();
sessionStorage.setItem('gage-auth-session', JSON.stringify({accessToken: d.accessToken, expiresAt: Date.now() + 3600 * 1000}));
location.hash = '#customer'; location.reload();
```

고객응대 화면에서:

1. **가게 정보**에 예를 들어 `메뉴: 떡볶이, 김밥, 모둠튀김 / 단무지는 주문 1건당 1개 기본 제공 / 누락은 배달앱 고객센터나 매장 전화로 알려 주시면 확인 후 안내`
2. **고객 리뷰**에 `김밥이 빠져서 왔어요`
3. **리뷰 답글 만들기** → 20~40초 뒤 답글에 가게 정보(누락 안내 경로 등)가 들어가면 RAG가 동작하는 겁니다

가게 정보를 비우고 같은 리뷰를 넣으면 일반적인 사과만 나옵니다. 두 결과를 나란히 보여 주면 차이가 바로 보입니다.

## 문제가 생기면

| 증상 | 원인 | 할 일 |
|---|---|---|
| `SCRABLE stopped` + `Killed` | WSL 메모리 부족 | 다른 WSL 작업을 끄고 다시. `free -m`으로 확인 |
| `SCRABLE stopped` + `401`/`gated` | 모델 파일이 없음 | 1번의 모델 복사 또는 `hf auth login` |
| 답글 대신 "시간이 초과됐어요" | CPU가 느림 | 리뷰를 짧게, 다른 프로그램 종료 |
| `[MOCK]`이 붙은 답글 | 백엔드가 mock으로 답함 | 스크립트로 띄웠는지 확인(`AI_CUSTOMER_MOCK_ENABLED=false`) |
| `Port 8000 already in use` | 이전 백엔드가 떠 있음 | `pkill -f bootRun` 후 다시 |
