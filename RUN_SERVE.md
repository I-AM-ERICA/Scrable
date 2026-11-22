# SCRABLE gRPC 서버 실행 가이드

팀 백엔드(am-erica-backend)는 리뷰 답글을 gRPC `CustomerAssistant.DraftReply`(포트 50053)로 부릅니다. 백엔드 `local` 설정에서는 이 자리에 가짜 답(Mock)이 들어 있습니다. 이 서버는 그 자리에 **지금 서버 모델(1.5B + 사진 어댑터 LoRA, `prompts/stage4_prompt.txt`)로 진짜 답글을 쓰는 서버**입니다. 백엔드 설정은 백엔드 main `ec5c233`(2026-11-19) 기준입니다.

| 항목 | 내용 |
|---|---|
| 코드 | `scrable/serve.py`, 설정 `configs/serve.json` |
| 약속서(proto) | `scrable/proto/america/ai/v1/*.proto` (백엔드 레포에서 그대로 복사) |
| 받는 것 | 가게 정보(`store`), 리뷰(`review_text`), 별점(`rating`, 모르면 0), 말투(`tone`) |
| 별점이 없을 때 | 사이트는 별점을 안 보냅니다. 학습 입력에는 늘 별점이 있었으므로, 리뷰 글의 칭찬·불만 단어로 별점을 추정해 넣습니다(불만 2, 칭찬 5, 둘 다 3, 둘 다 없으면 별점 없음) |
| 말투 | 학습에 말투 지시가 없었고, 사이트가 늘 보내는 "정중하고 따뜻하게"를 넣으면 답글이 더 어색해져서 기본은 쓰지 않습니다. `serve.json`에 `"use_tone": true`를 넣으면 다시 씁니다 |
| 돌려주는 것 | `reply`(답글), `sentiment`(positive/neutral/negative), `keywords` |
| 인증 | 요청 헤더 `x-internal-key` = 이 서버의 환경 변수 `AI_INTERNAL_KEY`. 백엔드가 고객응대에 쓰는 키(`AI_CUSTOMER_INTERNAL_KEY`, 안 정하면 `AI_INTERNAL_KEY`)와 같은 값 |
| 상태 확인 | 표준 gRPC 헬스체크. 백엔드가 2초마다 확인하는 그 방식이라 키 없이도 응답합니다 |
| 사진 | 아직 안 받습니다. proto에 사진 칸이 없습니다. 서버 코드는 준비되어 있습니다(RESEARCH_LOG 할 일 참고) |

- **감정(`sentiment`)** 은 별점으로 정합니다: 4~5점은 positive, 3점과 0점(모름)은 neutral, 1~2점은 negative. 백엔드 가짜 답과 같은 규칙입니다. 별점이 없으면 위의 추정 별점으로 정합니다.
- **키워드(`keywords`)** 는 정해 둔 주제 목록(배달 지연, 음식 온도, 누락, 맛 등)에서 리뷰에 나온 것을 고릅니다.
- **가게 정보**: `data/store_docs/<store_id>/` 폴더가 있으면 그 문서를 씁니다. 없으면 백엔드가 보낸 가게 이름·업종·지역·주소로 짧은 문서를 만들어 씁니다. 백엔드 `store_id`(숫자)를 실험용 가게 폴더에 연결하려면 `configs/serve.json`의 `store_aliases`에 `{"1": "bunsik-01"}`처럼 적습니다.
- 답글은 GPU에서 한 번에 하나씩 만듭니다. 기다리는 요청이 32개를 넘으면 "혼잡" 오류(`RESOURCE_EXHAUSTED`)를 돌려주고, 백엔드는 "AI가 혼잡합니다"를 보여 줍니다.
- 서버는 리뷰 내용을 파일이나 로그에 남기지 않습니다. 로그에는 가게 ID, 별점, 걸린 시간만 남습니다.

`[PC]`는 Windows PowerShell, `[서버]`는 SSH로 접속한 서버 창입니다.

---

## 1. [PC] 코드를 서버로 보내기

```powershell
tar -czf "$env:USERPROFILE\Downloads\scrable_serve.tgz" --exclude=.venv --exclude=outputs --exclude=data --exclude=*.egg-info --exclude=_grpc_gen -C "$env:USERPROFILE\naver_AI_service" scrable
```

```powershell
scp "$env:USERPROFILE\Downloads\scrable_serve.tgz" root@<서버IP>:/opt/scrable/
```

---

## 2. [서버] 풀고 gRPC 패키지 설치 (2분)

학습한 어댑터(`outputs/stage4_photo/adapter`, 드라이브의 `scrable-stage4-photo-adapter.zip`)와 가게 문서(`data/store_docs`)가 있는 기존 폴더에 풉니다.

```bash
cd /opt/scrable/scrable_stage1 && tar -xzf ../scrable_serve.tgz --strip-components=1
```

```bash
.venv/bin/pip install "grpcio>=1.66,<2" "grpcio-tools>=1.66,<2" "grpcio-health-checking>=1.66,<2"
```

마지막에 `Successfully installed ...`가 나오면 됩니다. 필요한 파일이 다 있는지 확인합니다:

```bash
ls configs/serve.json outputs/stage4_photo/adapter data/store_docs
```

---

## 3. [서버] 내부 키 정하기

서버와 백엔드는 같은 키를 써야 합니다. 백엔드는 고객응대(우리 서버)에만 따로 키를 줄 수 있어서(`AI_CUSTOMER_INTERNAL_KEY`), 다른 비서들의 키와 나눠 쓰는 것을 권합니다.

**오늘 밤 혼자 시험만 할 때**는 아무 키나 임시로 만들어 씁니다. 같은 서버 창 안에서 서버와 시험 프로그램이 같은 값을 씁니다:

```bash
export AI_INTERNAL_KEY=$(openssl rand -hex 16)
```

**나중에 백엔드와 연결할 때**는 백엔드 담당과 정한 값을 입력합니다 (화면에 안 보이는 게 정상):

```bash
read -rs AI_INTERNAL_KEY && export AI_INTERNAL_KEY
```

키는 카톡 같은 채팅이나 코드, GitHub에 그대로 붙이지 마세요.

---

## 4. [서버] 서버 켜기 (1분 안팎)

```bash
nohup .venv/bin/python -m scrable.serve --config configs/serve.json > serve.log 2>&1 &
```

```bash
tail -f serve.log
```

이 줄이 나오면 준비된 것입니다 (`Ctrl+C`는 보기만 멈춥니다):

```
SCRABLE gRPC server listening on 0.0.0.0:50053 (adapter outputs/stage4_photo/adapter)
```

---

## 5. [서버] 백엔드처럼 불러 보기

같은 서버 창에서 실행합니다 (키가 같아야 하므로):

```bash
.venv/bin/python scripts/grpc_client.py
```

이렇게 나오면 성공입니다 (답글과 숫자는 예시):

```
health: SERVING
seconds: 3.10
sentiment: negative
keywords: ['배달 지연', '음식 온도']
reply:
고객님, 떡볶이는 맛있게 드셨다니 다행입니다. ...
```

실험용 가게 문서가 없는 가게(백엔드 기본 정보만 있는 경우)도 시험해 봅니다:

```bash
.venv/bin/python scripts/grpc_client.py --store-id 999 --store-name 테스트카페 --category 카페 --rating 5 --review "커피가 진하고 맛있어요. 또 올게요"
```

**두 출력을 붙여 넣어 주세요.** 처음 한 번은 조금 더 느릴 수 있습니다.

---

## 6. [서버] 끄기

```bash
pkill -f scrable.serve
```

---

## 7. 백엔드와 연결할 때 (백엔드 담당과 함께)

| 백엔드 환경 변수 | 값 |
|---|---|
| `AI_CUSTOMER_HOST` | 이 GPU 서버 주소 |
| `AI_CUSTOMER_PORT` | `50053` |
| `AI_CUSTOMER_INTERNAL_KEY` | 3절에서 정한 값과 **같게**. 안 정하면 백엔드는 `AI_INTERNAL_KEY`(다른 비서와 같은 키)를 씁니다. 키를 파일로 줄 때는 `AI_CUSTOMER_KEY_FILE`에 파일 경로 |
| `AI_CUSTOMER_MOCK_ENABLED` | `false` (고객응대 가짜 서버만 끄기) |
| `AI_CUSTOMER_DEADLINE` | 안 정해도 됩니다. 기본 90초이고 답글은 2~4초입니다 |

- 백엔드 `local` 설정은 네 비서의 가짜 서버를 백엔드 PC의 `127.0.0.1`에 띄웁니다. `AI_CUSTOMER_MOCK_ENABLED=false`로 고객응대 가짜 서버만 끄고 나머지 비서는 그대로 둡니다. SSH 터널로 `AI_CUSTOMER_HOST=127.0.0.1`을 쓸 때([RUN_SITE.md](RUN_SITE.md))는 꼭 꺼야 포트가 겹치지 않습니다. 백엔드 기본 설정(`application.yml`)은 가짜 서버가 꺼져 있습니다.
- **포트 50053은 백엔드 서버 IP에만 열어 주세요.** NCP 서버라면 ACG(방화벽)에서 허용 IP를 백엔드 주소 하나로 제한합니다. 키가 있어도 아무에게나 열 필요는 없습니다. 오늘 밤 5절 시험은 서버 안에서만 하므로 포트를 열지 않아도 됩니다.
- 백엔드는 먼저 헬스체크로 서버 상태를 보고, `/actuator/health`의 `aiAssistants.customer`가 `SERVING`이면 연결된 것입니다.

---

## 자주 나는 오류

| 오류 메시지 | 원인 | 해결 |
|---|---|---|
| `Set AI_INTERNAL_KEY ...` | 키를 안 정함 | 3절 실행 후 다시 켜기 |
| `grpcio-tools is required` | 2절 설치를 안 함 | 2절 설치 명령 실행 |
| `Cannot listen on 0.0.0.0:50053` | 이미 서버가 켜져 있음 | `pkill -f scrable.serve` 후 다시 켜기 |
| 클라이언트 `UNAUTHENTICATED` | 서버와 클라이언트의 키가 다름 | 같은 창에서 실행했는지 확인, 서버를 끄고 3절부터 다시 |
| 클라이언트 `UNAVAILABLE` | 서버가 안 켜졌거나 아직 모델을 불러오는 중 | `tail serve.log`로 `listening` 줄 확인 |
| 사이트 답글에 `[MOCK]`이 붙음 | 백엔드의 고객응대 가짜 서버가 켜져 있음 | 백엔드를 `AI_CUSTOMER_MOCK_ENABLED=false`로 다시 켜기 |
| 백엔드가 인증 오류(`UNAUTHENTICATED`) | 백엔드 고객응대 키와 3절 키가 다름 | 백엔드의 `AI_CUSTOMER_INTERNAL_KEY`(안 정했으면 `AI_INTERNAL_KEY`)와 3절 값 확인 |
| `CUDA out of memory` | 다른 작업이 GPU를 쓰는 중 | `nvidia-smi` 확인 후 다시 |
| 그 밖의 `Traceback` | | `tail -30 serve.log` 출력을 보내 주세요 |
