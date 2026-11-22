# 사이트 화면에서 SCRABLE 확인하기 (노트북에서)

가게비서 웹 화면의 **고객응대**에서 리뷰를 넣으면, 진짜 SCRABLE 모델(GPU 서버)이 답글을 쓰는지 확인합니다. 다른 세 비서(정책·홍보·경영)는 백엔드의 mock 응답 그대로 둡니다.

```
브라우저 localhost:7001 (프론트, WSL)
   → 백엔드 localhost:8000 (WSL, H2 메모리 DB, 데모 계정)
      → SSH 터널 localhost:55053
         → GPU 서버 127.0.0.1:50053 (SCRABLE gRPC 서버)
```

- SCRABLE 포트는 서버 밖으로 열지 않습니다. 서버에서 `127.0.0.1`로만 띄우고, 노트북에서는 SSH 터널로 접속합니다
- 내부 키(`AI_INTERNAL_KEY`)는 이번 확인용으로 새로 만든 임시 값을 씁니다. 채팅·GitHub·카톡에 붙여넣지 않습니다
- `[서버]`는 GPU 서버(bash), `[WSL]`은 노트북 WSL(zsh)입니다. **창 네 개**를 씁니다: 서버 1개, WSL 3개(터널, 백엔드, 프론트)

학습이나 평가가 GPU에서 돌고 있으면 끝난 뒤에 하세요. 같이 돌리면 둘 다 느려집니다.

---

## 0. [WSL] 처음 한 번: Java와 Node 설치

백엔드는 Java 17 이상, 프론트는 Node 22가 필요합니다.

```bash
sudo apt update && sudo apt install -y openjdk-21-jdk-headless
```
```bash
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash && source ~/.zshrc && nvm install 22
```
```bash
java -version && node -v
```

`21...`과 `v22...`가 나오면 됩니다.

## 1. [WSL] 처음 한 번: 백엔드·프론트 최신 코드 받기

```bash
cd ~/naver && gh repo clone I-AM-ERICA/am-erica-backend && gh repo clone I-AM-ERICA/am-erica-frontend
```

이미 받아 두었다면 각 폴더에서 `git pull`만 하면 됩니다. 레포 이름이 다르면 `gh repo list I-AM-ERICA`로 확인하세요.

```bash
cd ~/naver/am-erica-frontend && npm install
```

## 2. [서버] SCRABLE 서버 띄우기 (127.0.0.1에서만)

```bash
cd /opt/scrable/scrable_stage1 && sed 's/"host": "0.0.0.0"/"host": "127.0.0.1"/' configs/serve.json > configs/serve_local.json && grep host configs/serve_local.json
```

`"host": "127.0.0.1"`이 나오면, 임시 내부 키를 만들어서 화면에 한 번 보여 줍니다(4번에서 백엔드에 같은 값을 넣어야 함).

```bash
export AI_INTERNAL_KEY=$(openssl rand -hex 16) && echo "$AI_INTERNAL_KEY"
```

```bash
nohup .venv/bin/python -m scrable.serve --config configs/serve_local.json > serve.log 2>&1 &
```
```bash
tail -f serve.log
```

`SCRABLE gRPC server listening on 127.0.0.1:50053`이 나오면 됩니다(모델 불러오기 1~2분). `Ctrl+C`로 보기만 멈춥니다.

## 3. [WSL 창 1] SSH 터널 열기

```bash
ssh -N -L 55053:127.0.0.1:50053 root@<서버IP>
```

아무것도 안 나오고 멈춰 있으면 정상입니다. 이 창은 그대로 둡니다.

## 4. [WSL 창 2] 백엔드 실행 (고객응대만 진짜 서버로)

`<키>`에는 2번에서 서버에 나온 값을 붙여넣습니다.

```bash
cd ~/naver/am-erica-backend && AI_CUSTOMER_MOCK_ENABLED=false AI_CUSTOMER_HOST=127.0.0.1 AI_CUSTOMER_PORT=55053 AI_CUSTOMER_INTERNAL_KEY='<키>' ./gradlew bootRun
```

처음에는 Gradle 다운로드로 몇 분 걸립니다. `Started ... in ... seconds`가 나오면 됩니다. 연결 확인:

```bash
curl -s localhost:8000/actuator/health
```

다른 창에서 실행하세요. 결과에 `customer`가 `UP`이면 SCRABLE까지 연결된 겁니다.

## 5. [WSL 창 3] 프론트 실행

```bash
cd ~/naver/am-erica-frontend && npm run dev
```

Windows 브라우저에서 **http://localhost:7001** 을 엽니다.

## 6. [브라우저] 데모 계정으로 로그인

프론트는 네이버 로그인만 있어서, 백엔드가 자동으로 만든 **데모 계정**(`demo@america.local`, 로컬 H2 전용) 토큰을 브라우저에 직접 넣습니다.

1. http://localhost:7001 에서 `F12` → **Console** 탭
2. 아래를 붙여넣고 Enter (처음이면 "allow pasting"을 입력하라고 나올 수 있음)

```js
const r = await fetch('/api/auth/login', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({email: 'demo@america.local', password: 'demo1234!'})});
const d = await r.json();
sessionStorage.setItem('gage-auth-session', JSON.stringify({accessToken: d.accessToken, expiresAt: Date.now() + 3600 * 1000}));
location.hash = '#customer'; location.reload();
```

고객응대 화면이 열리면 로그인된 겁니다. 업종 선택 화면이 나오면 아무 업종이나 고르고 고객응대로 갑니다.

## 7. [브라우저] 리뷰 넣어 보기

리뷰를 입력하고 **리뷰 답글 만들기**를 누릅니다. 2~4초 뒤 답글이 나오면 성공입니다. `[MOCK]`이 붙어 있으면 아직 mock이 답한 겁니다(4번 환경변수 확인).

시험해 볼 리뷰 예:

- `배달이 한 시간 넘게 걸려서 다 식었어요`
- `떡볶이 맛있어요! 다음에 또 시킬게요`
- `주문한 튀김이 빠져서 왔어요. 확인 부탁드려요`

서버 쪽에서 요청이 들어왔는지:

```bash
tail -n 5 /opt/scrable/scrable_stage1/serve.log
```

`DraftReply store=1 rating=0 2.1s`처럼 나옵니다(리뷰 내용은 로그에 남기지 않음).

**참고**: 데모 매장(ID 1)에는 SCRABLE 가게 문서가 없어서, 백엔드가 보내는 가게 이름·업종만 보고 답합니다. 메뉴나 정책을 짚는 답은 가게 문서가 있을 때 나옵니다.

## 8. 끝낼 때

- WSL 창 1~3: `Ctrl+C`
- 서버: `pkill -f scrable.serve`

## 문제가 생기면

| 증상 | 원인 | 할 일 |
|---|---|---|
| 답글 대신 "고객응대 비서에 연결하지 못했어요" (503) | 터널이 끊겼거나 서버가 안 떠 있음 | 3번 창 확인, 서버에서 `tail serve.log` |
| 답글이 안 나오고 서버 로그에 `invalid internal key` | 4번 키와 2번 키가 다름 | 서버에서 `echo $AI_INTERNAL_KEY`로 다시 확인 |
| "가게 정보를 먼저 저장해 주세요" | 매장 ID가 없음 | 6번을 다시 하고 업종 선택 |
| `Port 8000 already in use` | 이전 백엔드가 떠 있음 | 그 창에서 `Ctrl+C` |
| Console에서 `accessToken` 오류 | 백엔드가 안 떠 있음 | 4번 확인 |
