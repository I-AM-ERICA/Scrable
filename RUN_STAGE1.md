# 1단계 실험 실행 가이드 (처음부터 차근차근)

이 문서는 **내 Windows 노트북에 있는 코드를 GPU 서버로 옮겨서 1단계 실험을 돌리고, 결과를 다시 받아오는 과정**을 순서대로 설명합니다.
명령어는 그대로 복사해서 붙여 넣으면 됩니다. `< >`로 감싼 부분만 내 값으로 바꾸세요.

---

## 0. 먼저 알아둘 것

### 폴더가 두 군데 있습니다

| 이름 | 어디에 있나 | 무엇인가 |
|---|---|---|
| **내 PC 프로젝트 폴더** | `C:\Users\wounc\naver_AI_service\scrable` | 지금 코드가 있는 곳. 여기서 수정합니다. |
| **서버 프로젝트 폴더** | 서버의 `/opt/scrable/scrable_stage1` | 실험을 실제로 돌리는 곳. 이 가이드에서 새로 만듭니다. |

> 서버에 이미 있는 `/opt/scrable/scrable_v2`는 이전 버전입니다. 건드리지 않고 그대로 둡니다.

### 창(터미널)도 두 종류를 씁니다

| 창 | 여는 방법 | 맨 앞에 보이는 글자 | 하는 일 |
|---|---|---|---|
| **Windows PowerShell** | 시작 메뉴에서 `PowerShell` 검색 → 실행 | `PS C:\Users\wounc>` | 내 PC에서 파일 묶기, 서버로 보내기, 결과 받기 |
| **서버 창 (SSH)** | PowerShell에서 `ssh` 명령으로 접속 | `root@...:~#` | 서버에서 설치·실행 |

명령어를 **어느 창에 붙여 넣는지**가 중요합니다. 각 단계 제목에 `[PC]` 또는 `[서버]`로 표시했습니다.

### 필요한 것 3가지

1. **서버 주소와 비밀번호**: 예전에 SSH로 접속할 때 쓰던 것 그대로입니다. 아래에서 `<서버IP>`라고 쓴 부분에 넣으세요.
2. **Hugging Face 토큰**: HyperCLOVA 1.5B 모델을 내려받는 데 필요합니다. ([1단계](#1-hugging-face-토큰-준비-처음-한-번))
3. **Anthropic API 키**: Claude Haiku(채점기)를 쓰는 데 필요합니다. ([2단계](#2-anthropic-api-키-준비-처음-한-번))

---

## 1. Hugging Face 토큰 준비 (처음 한 번)

**Hugging Face**는 AI 모델 파일을 올려 두는 사이트입니다. HyperCLOVA 1.5B는 "이용 조건에 동의한 사람만" 받을 수 있어서, 내 계정임을 증명하는 **토큰(긴 비밀번호 같은 문자열)**이 필요합니다.

> 예전에 서버에서 v2 실험을 돌렸다면 이미 해 둔 것이라 **건너뛰어도 됩니다.** [7단계](#7-서버-hugging-face-로그인-확인)에서 확인하는 방법이 있습니다.

1. https://huggingface.co 에 가입하고 로그인합니다.
2. 모델 페이지 https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B 에서 이용 조건에 동의합니다.
3. 오른쪽 위 프로필 → **Settings → Access Tokens → Create new token**
   - Token type: **Read**
   - 이름: 아무거나 (예: `scrable-server`)
4. 만들어진 `hf_`로 시작하는 문자열을 복사해 메모장에 잠깐 붙여 둡니다. **다른 사람에게 보여주거나 코드/깃허브에 올리면 안 됩니다.**

---

## 2. Anthropic API 키 준비 (처음 한 번)

**API 키**는 내 계정으로 Claude를 호출할 수 있게 해 주는 비밀 문자열입니다. 사용한 만큼 요금이 나가므로 비밀번호처럼 관리하세요.

1. https://console.anthropic.com 에 가입하고 로그인합니다.
2. **Billing(결제)** 메뉴에서 크레딧 상태를 확인합니다. 무료 크레딧이 없으면 소액(예: 5달러)을 충전합니다.
   - 1단계는 모든 Claude 역할이 Haiku 5.5라서 연결 확인 1회는 1센트 안팎, 본 실험 1회는 약 0.5달러 이하로 예상합니다.
3. **API Keys → Create Key** 를 누르고 `sk-ant-`로 시작하는 키를 복사해 메모장에 잠깐 붙여 둡니다. (창을 닫으면 다시 볼 수 없습니다.)

---

## 3. [PC] 코드를 압축 파일 하나로 묶기

PowerShell을 열고 아래를 그대로 붙여 넣습니다.

```powershell
tar -czf "$env:USERPROFILE\Downloads\scrable_stage1.tgz" --exclude=.venv --exclude=outputs --exclude=data --exclude=*.egg-info -C "$env:USERPROFILE\naver_AI_service" scrable
```

- 결과: `C:\Users\wounc\Downloads\scrable_stage1.tgz` 파일이 생깁니다.
- `.venv`(내 PC용 설치 파일), `outputs`(이전 결과), `data`(개인 데이터)는 빼고 코드만 묶습니다.
- 아무 메시지 없이 다음 줄로 넘어가면 성공입니다.

---

## 4. [PC] 압축 파일을 서버로 보내기

같은 PowerShell 창에서:

```powershell
scp "$env:USERPROFILE\Downloads\scrable_stage1.tgz" root@<서버IP>:/opt/scrable/
```

- 처음이면 `Are you sure you want to continue connecting (yes/no)?` → `yes` 입력
- `password:`가 나오면 서버 비밀번호를 입력합니다. **입력해도 화면에 아무것도 안 보이는 게 정상**입니다. 다 치고 Enter.
- `100%`가 표시되면 끝입니다.

---

## 5. [서버] 접속해서 압축 풀기

PowerShell에서 서버에 접속합니다.

```powershell
ssh root@<서버IP>
```

비밀번호를 입력하면 맨 앞 글자가 `root@...:~#`로 바뀝니다. 이제부터는 **서버 창**입니다.

```bash
cd /opt/scrable
mkdir -p scrable_stage1
tar -xzf scrable_stage1.tgz -C scrable_stage1 --strip-components=1
cd scrable_stage1
ls
```

`ls` 결과에 `README.md`, `RUN_STAGE1.md`, `configs`, `scrable` 등이 보이면 성공입니다.

---

## 6. [서버] 설치 (처음 한 번, 5~15분)

```bash
python3 --version
```

`Python 3.10.x` 또는 그 이상이면 아래를 실행합니다.

```bash
PYTHON_BIN=python3 bash scripts/setup_gpu.sh
```

- 필요한 프로그램(PyTorch 등)을 `.venv` 폴더에 설치합니다. 시간이 걸립니다.
- 마지막에 `Setup complete.`가 나오면 성공입니다.
- `ensurepip is not available` 같은 오류가 나면 `apt install -y python3-venv`를 실행한 뒤 다시 실행하세요.

---

## 7. [서버] Hugging Face 로그인 확인

```bash
.venv/bin/hf auth whoami
```

- **내 Hugging Face 아이디가 나오면** 이미 로그인된 상태입니다. 다음 단계로 넘어가세요.
- `Not logged in`이 나오면 로그인합니다.

```bash
.venv/bin/hf auth login
```

- `Enter your token`이 나오면 [1단계](#1-hugging-face-토큰-준비-처음-한-번)에서 복사한 `hf_...` 토큰을 붙여 넣습니다. (화면에 안 보여도 정상)
- `Add token as git credential?`는 `n`을 입력해도 됩니다.
- 토큰은 서버에 저장되므로 **한 번만** 하면 됩니다.

---

## 8. [서버] Anthropic API 키 입력 (서버에 새로 접속할 때마다)

```bash
read -rs ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY
```

- 이 명령을 치면 커서만 깜빡입니다. 그때 [2단계](#2-anthropic-api-키-준비-처음-한-번)의 `sk-ant-...` 키를 붙여 넣고 Enter를 누르세요. (화면에 안 보여도 정상)
- 키를 화면과 명령 기록에 남기지 않는 방법입니다.
- SSH 창을 닫으면 사라지므로, **서버에 다시 접속할 때마다** 이 단계를 반복합니다.
- 제대로 들어갔는지 확인: `echo ${#ANTHROPIC_API_KEY}` → 0이 아닌 숫자(키 길이)가 나오면 됩니다.

---

## 9. [서버] 연결 확인 실행 (예시 6건, 1센트 안팎)

실제 데이터 없이, 예시 가게 리뷰 6건으로 **모든 부품이 연결되는지만** 확인합니다.

```bash
.venv/bin/python -m scrable.experiment --config configs/stage1_smoke_example.json
```

진행 순서:
1. HyperCLOVA 1.5B 모델 다운로드 (처음 한 번, 약 3GB)
2. 한국어 검색용 임베딩 모델 다운로드 (처음 한 번)
3. `Iteration 0 ... generate`, `... judge` 진행 막대가 나타납니다
4. 마지막에 점수표가 나오면 성공

```text
Held-out test (n=2), LLM-as-a-Judge (anthropic:claude-haiku-5-5), not human evaluation
generator                    prompt     overall   relevancy specificity    accuracy     grammar
local:...Instruct-1.5B       base         0.xxx ...
local:...Instruct-1.5B       optimized    0.xxx ...
anthropic:claude-haiku-5-5   base         0.xxx ...
```

> 예시 데이터는 6건뿐이라 이 점수는 의미가 없습니다. **오류 없이 표가 나오는지만** 보면 됩니다.

---

## 10. [서버 → PC] 결과 받아오기

서버 창에서 결과를 묶습니다.

```bash
cd /opt/scrable/scrable_stage1
tar -czf stage1_smoke_result.tgz outputs/stage1_smoke/metrics.json outputs/stage1_smoke/test_predictions.json outputs/stage1_smoke/history.json outputs/stage1_smoke/best_prompt.txt
```

**새 PowerShell 창**을 하나 더 열고(서버 창은 그대로 둡니다) 내 PC로 받습니다.

```powershell
scp root@<서버IP>:/opt/scrable/scrable_stage1/stage1_smoke_result.tgz "$env:USERPROFILE\Downloads\"
```

`C:\Users\wounc\Downloads\stage1_smoke_result.tgz`를 채팅에 첨부하면 같이 확인합니다.

---

## 11. 본 실험 (실제 데이터 80건 이상)

### 11-1. [PC] 데이터 만들기

내 PC 프로젝트 폴더에 `data` 폴더를 만들고 아래처럼 파일을 둡니다.

```text
C:\Users\wounc\naver_AI_service\scrable\data\
├─ store_reviews.jsonl              ← 리뷰 (한 줄에 하나)
└─ store_docs\
   ├─ store-001\info.md             ← 가게 1의 정보 (메뉴, 영업시간, 배달·환불 안내)
   └─ store-002\info.md             ← 가게 2의 정보
```

`store_reviews.jsonl` 한 줄 예시 (`examples\store_reviews.sample.jsonl`을 복사해서 고쳐 쓰면 편합니다):

```json
{"id": "r001", "app_id": "store-001", "rating": 2, "review": "배달이 1시간 넘게 걸렸어요.", "expert_response": null}
```

- `id`: 리뷰마다 다른 이름 (r001, r002, ...)
- `app_id`: 가게 이름표. `store_docs` 안의 **폴더 이름과 똑같이** 써야 그 가게 정보를 찾아 씁니다.
- `rating`: 별점 1~5 (모르면 `null`)
- `expert_response`: 사장님이 실제로 단 답글 (없으면 `null`)
- 파일은 **UTF-8**로 저장하세요. (VS Code는 기본이 UTF-8입니다.)
- 서로 다른 리뷰가 **최소 80건** 필요합니다. (학습 30 / 검증 10 / 테스트 40)

### 11-2. [PC] 데이터를 서버로 보내기

```powershell
scp -r "$env:USERPROFILE\naver_AI_service\scrable\data" root@<서버IP>:/opt/scrable/scrable_stage1/
```

### 11-3. [서버] 실행 (창을 닫아도 계속 돌게)

본 실험은 시간이 걸리므로, SSH 창이 끊겨도 계속 돌도록 `nohup`으로 실행합니다.

```bash
cd /opt/scrable/scrable_stage1
read -rs ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY     # 새로 접속했다면
nohup .venv/bin/python -m scrable.experiment --config configs/stage1_claude_judge.json > stage1_run.log 2>&1 &
```

- 진행 상황 보기: `tail -f stage1_run.log` (빠져나오기: `Ctrl + C`, 실험은 계속 돕니다)
- 끝났는지 확인: 로그 마지막에 점수표가 있으면 끝입니다.
- 결과 받기: [10단계](#10-서버--pc-결과-받아오기)에서 `stage1_smoke`를 `stage1`로 바꿔서 똑같이 하면 됩니다.
- 중간에 끊겨도 **같은 명령으로 다시 실행**하면, 끝난 호출은 저장된 결과를 다시 써서 이어서 진행합니다(추가 비용 거의 없음).

---

## 12. 코드를 고친 뒤 다시 올릴 때

PC에서 코드를 고쳤다면 [3단계](#3-pc-코드를-압축-파일-하나로-묶기) → [4단계](#4-pc-압축-파일을-서버로-보내기) → 서버에서 아래만 하면 됩니다. 설치(6단계)는 다시 안 해도 됩니다.

```bash
cd /opt/scrable
tar -xzf scrable_stage1.tgz -C scrable_stage1 --strip-components=1
```

`data`, `outputs`, `.venv`는 압축에 안 들어가므로 서버에 있던 것이 그대로 남습니다.

---

## 자주 나는 오류

| 화면에 나온 메시지 | 원인 | 해결 |
|---|---|---|
| `Permission denied (publickey,password)` | 서버 비밀번호가 틀림 | 다시 입력 (화면에 안 보이는 게 정상) |
| `401 ... gated repo` / `Access to model ... is restricted` | Hugging Face 모델 이용 동의 안 함 또는 로그인 안 됨 | [1단계](#1-hugging-face-토큰-준비-처음-한-번) 2번 동의 → [7단계](#7-서버-hugging-face-로그인-확인) 로그인 |
| `authentication_error` / `Could not resolve authentication method` | Anthropic 키를 입력하지 않았거나 틀림 | [8단계](#8-서버-anthropic-api-키-입력-서버에-새로-접속할-때마다) 다시 |
| `credit balance is too low` | Anthropic 크레딧 부족 | 콘솔 Billing에서 충전 |
| `output_dir contains a different experiment` | 설정을 바꿨는데 같은 결과 폴더를 씀 | 설정 파일의 `output_dir`를 새 이름으로 (예: `outputs/stage1_v2`) |
| `Need 80 unique reviews, got ...` | 리뷰 수 부족 또는 중복 | 리뷰를 더 넣거나 설정의 `train_size`/`test_size`를 줄이기 |
| `CUDA out of memory` | GPU 메모리 부족 | 다른 실험(예: 노트북 커널)이 GPU를 쓰는지 `nvidia-smi`로 확인 후 종료 |
| `bash: $'\r': command not found` | 스크립트 줄바꿈 문제 | `sed -i 's/\r$//' scripts/*.sh` 후 다시 실행 |
