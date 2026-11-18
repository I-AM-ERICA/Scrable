# Vision-3B 시험 실행 가이드

네이버 `HyperCLOVAX-SEED-Vision-Instruct-3B`(글 + 이미지, 약 3.6B)를 우리 서버에서 시험합니다. 지금 모델(1.5B)보다 약 2배 크고 이미지를 볼 수 있습니다.

| 순서 | 내용 | 비용 | 시간 |
|---|---|---|---|
| 1절~3절 | 준비 (코드 보내기, 추가 설치) | 0원 | 10분 |
| 4절 | **동작 확인**: 모델이 뜨는지, 한국어 답을 쓰는지, 몇 초 걸리는지 | 0원 | 처음엔 다운로드 포함 15~30분 |
| 5절 (선택) | **리뷰 테스트 40건**: 학습 없이 1.5B·2단계 LoRA·Haiku와 비교 | Haiku 채점 160번, 몇십 센트 | 10분 안팎 |
| 6절 | **사진 리뷰 (A안)**: Vision-3B가 사진 설명 → 지금 최종 모델이 답글 | 사진이 들어가서 글보다 비쌈 | 사진 준비 + 20분 안팎 |
| 7절 | **사진 리뷰 두 번째 시험**: 같은 사진으로 짧은 리뷰·불만 리뷰 | 6절의 약 2배 | 25분 안팎 |
| 8절 | **사진 설명 고친 뒤 세 번째 시험**: 설명 먼저 0원 확인 → 평가 한 번 | 7절의 절반 | 15분 안팎 |

Vision-3B LoRA 학습은 결과를 보고 다음에 합니다. `[PC]`는 Windows PowerShell, `[서버]`는 SSH로 접속한 서버 창입니다.

### 답변 속도도 같이 잽니다

- 리뷰 답글은 사장님이 확인하고 올리면 되니 몇 초 늦어도 괜찮습니다.
- 메일과 상담은 손님이 기다리고 있어서, 늦으면 떠날 수 있습니다.

그래서 이번부터 평가 결과 끝에 `seconds per reply`(답 하나에 걸린 초)가 나옵니다. 4절에서는 같은 리뷰로 Vision-3B와 1.5B의 속도를 나란히 잽니다.

---

## 0. 시작 전에: 메일 학습이 끝났는지 확인

메일 학습(`RUN_EMAIL.md` 8절)이 아직 돌고 있으면 **기다리세요.** 그 작업이 끝나기 전에 코드를 바꾸면, 뒤에 이어지는 평가가 바뀐 코드로 돌 수 있습니다. GPU도 하나라 동시에 돌리면 느려지거나 메모리가 부족할 수 있습니다.

```bash
pgrep -af email_dpo
```

**아무것도 안 나오면** 끝난 것이니 시작하면 됩니다.

---

## 1. [PC] 코드를 서버로 보내기

```powershell
tar -czf "$env:USERPROFILE\Downloads\scrable_vision.tgz" --exclude=.venv --exclude=outputs --exclude=data --exclude=*.egg-info -C "$env:USERPROFILE\naver_AI_service" scrable
```

```powershell
scp "$env:USERPROFILE\Downloads\scrable_vision.tgz" root@<서버IP>:/opt/scrable/
```

---

## 2. [서버] 풀고 확인

```bash
cd /opt/scrable/scrable_stage1
tar -xzf ../scrable_vision.tgz --strip-components=1
ls configs/vision_eval.json scripts/vision_smoke.py
```

파일 2개가 보이면 성공입니다.

---

## 3. [서버] 추가 설치와 확인 (5분)

Vision 모델이 쓰는 패키지 4개를 설치합니다:

```bash
.venv/bin/pip install einops timm av decord
```

마지막에 `Successfully installed ...`가 나오면 성공입니다. `decord`에서 오류가 나면 오류 메시지를 붙여 넣어 주세요.

transformers 버전과 남은 디스크 공간을 확인합니다:

```bash
.venv/bin/pip show transformers | head -2 && df -h /opt
```

- `Version: 4.`로 시작하면 됩니다. 모델 페이지는 4.52.4에서 확인했다고 적혀 있습니다.
- `/opt` 줄의 `Avail`(남은 공간)이 **20G 이상**이어야 합니다. 모델 파일이 약 16GB입니다.

---

## 4. [서버] 동작 확인 (처음엔 15~30분)

처음 실행할 때 Hugging Face에서 모델(약 16GB)을 내려받습니다. 화면을 닫아도 되게 `nohup`으로 돌립니다:

```bash
nohup .venv/bin/python scripts/vision_smoke.py > vision_smoke.log 2>&1 &
```

```bash
tail -f vision_smoke.log
```

`Ctrl+C`는 보기만 멈춥니다. 끝나면 이런 식으로 나옵니다 (내용과 숫자는 예시):

```
peak GPU memory so far: 7.5 GB

[Vision-3B review reply] 3.10 s
안녕하세요, 고객님. 단무지가 빠져 불편을 드려 죄송합니다. ...

[1.5B review reply (no adapter)] 1.40 s
안녕하세요, 고객님. ...
```

**성공**은 두 모델 모두 한국어 답글을 쓰고, 마지막 줄까지 오류 없이 끝나는 것입니다. **`vision_smoke.log` 전체를 붙여 넣어 주세요.** 답의 형식이 이상하면(프롬프트가 같이 나오거나 `<|im_end|>` 같은 글자가 섞이는 등) 그걸 보고 코드를 고칩니다.

### (선택) 이미지 확인

**직접 찍은 음식 사진**을 하나 준비하세요. 실제 손님 사진은 허락 없이 쓰면 안 됩니다.

[PC]에서 사진을 서버로 보냅니다 (파일 이름은 바꿔 쓰세요):

```powershell
scp "$env:USERPROFILE\Downloads\food.jpg" root@<서버IP>:/opt/scrable/scrable_stage1/
```

[서버]:

```bash
.venv/bin/python scripts/vision_smoke.py --image food.jpg --no-compare
```

`[Vision-3B image]` 아래에 사진 설명이 나옵니다. 사진 속 음식을 맞게 설명하는지 확인해 주세요.

---

## 5. [서버] 리뷰 테스트 40건 (10분 안팎)

4절이 성공한 뒤에 합니다. 학습 없이 Vision-3B로 2단계와 같은 리뷰 40건에 답하고, 1단계(1.5B)·2단계 LoRA·Haiku와 비교합니다.

새로 접속했다면 API 키부터 입력합니다:

```bash
read -rs ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY
```

```bash
nohup .venv/bin/python -m scrable.evaluate --config configs/vision_eval.json --label vision-3b --reference outputs/stage1/test_predictions.json outputs/stage2_eval/test_predictions.json > vision_eval.log 2>&1 &
```

`grep -c "Saved results" vision_eval.log`가 **1**이 되면 끝입니다. 결과를 봅니다:

```bash
grep -A6 "^vision-3b" vision_eval.log; grep "seconds per reply" vision_eval.log
```

**출력을 그대로 붙여 넣어 주세요.** 비교 상대는 이렇습니다:

| 이름 | 뜻 | 종합 점수 |
|---|---|---|
| `stage1:baseline` | 1.5B, 기본 프롬프트 | 0.621 |
| `stage1:optimized` | 1.5B, 1단계 최적화 프롬프트 (Vision-3B와 같은 프롬프트) | 0.653 |
| `dpo` 등 2단계 이름 | 1.5B + 2단계 LoRA (지금 최종 모델) | 0.752 |
| `anthropic:...` | Haiku | 0.791 |

### 결과를 어떻게 보나

| 결과 | 뜻 | 다음 할 일 |
|---|---|---|
| Vision-3B가 `stage1:optimized`(0.653)보다 확실히 높음 | 모델이 크면 같은 프롬프트에서 더 잘함 | Vision-3B에 LoRA 학습을 하면 2단계(0.752)를 넘을 가능성이 있음 |
| 특히 **구체성**이 확실히 높음 | 구체성 문제는 모델 크기 탓이었을 가능성이 큼 | 위와 같음 |
| `stage1:optimized`와 비슷함 | 2배 크기로는 차이가 없음 | 1.5B를 유지하고, Vision-3B는 이미지 설명용으로만 씀 |

`seconds per reply`는 Vision-3B가 답 하나에 걸린 시간입니다. 1.5B 속도는 4절 로그에 있습니다.

---

## 6. 사진 리뷰 시험 (A안)

**4절이 성공한 뒤에** 합니다. 5절(글 실력 비교)은 건너뛰어도 됩니다.

A안의 순서는 이렇습니다.

1. 손님 사진 → **Vision-3B가 사진 설명을 씀** (예: "떡볶이 위에 머리카락으로 보이는 이물질이 있습니다")
2. 리뷰 + 사진 설명 + 가게 정보 → **지금 최종 모델(1.5B + 2단계 LoRA)이 답글**
3. **채점기(Haiku)는 사진을 직접 보고** 채점 → 사진 설명이 틀려서 답글이 틀리면 점수가 깎임

같은 리뷰를 **사진 설명 없이**도 한 번 더 돌려서, 사진 설명이 도움이 되는지 비교합니다.

- 사진이 없는 리뷰는 전과 똑같이 동작합니다. 사진 설명이 있을 때만 입력에 `photo_description` 칸과 사진 규칙("사진 설명에 있는 것만 말하고, 원인은 단정하지 않기")이 붙습니다.
- 사진이 있으면 Vision-3B가 사진 설명을 쓰는 시간도 답 하나당 시간(`seconds per reply`)에 들어갑니다.

### 6-1. [PC] 사진 준비

**직접 찍었거나 써도 된다는 허락을 받은 사진만** 쓰세요. 실제 손님 사진은 허락 없이 쓰면 안 됩니다.

PC의 `다운로드` 폴더에 아래처럼 폴더를 만들고 사진을 넣습니다. **폴더 이름은 가게 이름과 똑같아야 합니다.**

```
다운로드\review_photos\
    bunsik-01\     ← 분식 사진 (떡볶이, 김밥, 튀김 등)
    cafe-01\       ← 카페 사진 (커피, 디저트)
    chicken-01\    ← 치킨 사진
    gukbap-01\     ← 국밥 사진
```

- 가게마다 3~5장, **모두 12~20장**이면 충분합니다.
- **문제 있는 사진과 괜찮은 사진을 섞어 주세요.** 예: 포장이 흐트러짐, 국물이 샘, 양이 적어 보임, 깔끔하게 잘 온 음식. 문제 사진은 연출해도 됩니다.
- 파일 형식은 `.jpg`, `.jpeg`, `.png`, `.webp` 중 하나입니다. 휴대폰 사진 그대로 괜찮습니다. 채점기로 보낼 때는 자동으로 줄입니다.
- 한 폴더 안에서 파일 이름(확장자 빼고)이 겹치면 안 됩니다.
- 사진에 사람 얼굴, 영수증의 이름·주소·전화번호가 보이지 않게 해 주세요.

### 6-2. [PC] 코드와 사진을 서버로 보내기

4절 이후 코드가 바뀌어서 코드도 다시 보냅니다.

```powershell
tar -czf "$env:USERPROFILE\Downloads\scrable_photo.tgz" --exclude=.venv --exclude=outputs --exclude=data --exclude=*.egg-info -C "$env:USERPROFILE\naver_AI_service" scrable
```

```powershell
scp "$env:USERPROFILE\Downloads\scrable_photo.tgz" root@<서버IP>:/opt/scrable/
```

```powershell
scp -r "$env:USERPROFILE\Downloads\review_photos" root@<서버IP>:/opt/scrable/scrable_stage1/data/
```

### 6-3. [서버] 풀고 확인

```bash
cd /opt/scrable/scrable_stage1 && tar -xzf ../scrable_photo.tgz --strip-components=1
ls configs/photo_eval.json && ls data/review_photos/*
```

설정 파일과, 가게 폴더 4개 안의 사진 이름이 보이면 성공입니다.

API 키를 입력합니다 (새로 접속했다면):

```bash
read -rs ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY
```

### 6-4. [서버] 사진마다 리뷰 만들기 (1~2분)

Haiku가 사진과 가게 정보를 보고, 그 사진을 찍은 손님이 쓸 법한 리뷰와 별점을 씁니다.

```bash
.venv/bin/python -m scrable.synth_photo_reviews --photos data/review_photos --docs data/store_docs --out data/photo_reviews.jsonl
```

마지막 줄이 `Wrote N photo reviews ... (ratings {1: .., 2: .., ...})`이면 성공입니다. N은 사진 수와 같아야 합니다.

만들어진 리뷰를 눈으로 확인합니다:

```bash
cut -c1-200 data/photo_reviews.jsonl
```

리뷰가 사진과 안 맞으면(예: 멀쩡한 사진인데 불만 리뷰) 붙여 넣어 주세요.

### 6-5. [서버] 평가 두 번 (사진 설명 있음 → 없음, 10~15분)

```bash
nohup bash -c '.venv/bin/python -m scrable.evaluate --config configs/photo_eval.json --split all --label with-photos && .venv/bin/python -m scrable.evaluate --config configs/photo_eval.json --split all --label no-photos --no-photos --output-dir outputs/photo_eval/no_photos --reference outputs/photo_eval/with_photos/all_predictions.json' > photo_eval.log 2>&1 &
```

`grep -c "Saved results" photo_eval.log`가 **2**가 되면 끝입니다.

### 6-6. [서버] 결과 보기

점수 비교와 속도:

```bash
grep -A6 "^no-photos - " photo_eval.log; grep "seconds per reply" photo_eval.log
```

`no-photos - with-photos`의 숫자는 **"사진 설명 없음 − 있음"** 입니다. **음수이면 사진 설명이 도움이 된 것**입니다. `seconds per reply`는 첫 줄이 사진 설명 있음, 둘째 줄이 없음입니다.

리뷰마다 사진 설명과 두 답글을 나란히 보기:

```bash
.venv/bin/python scripts/photo_report.py outputs/photo_eval/with_photos/all_predictions.json outputs/photo_eval/no_photos/all_predictions.json
```

**두 출력을 모두 붙여 넣어 주세요.** 특히 이것들을 확인합니다.
- `[사진 설명]`이 사진과 맞는지 (틀린 설명이 있으면 어떤 사진인지 알려 주세요)
- 답글이 사진 속 문제를 언급하는지
- 원인을 단정하지 않는지 (예: "직원 실수로 들어갔습니다" 같은 말)

### 비용

사진이 들어가서 글만 있을 때보다 Haiku 비용이 더 듭니다. 사진 20장 기준으로 리뷰 만들기 20번, 채점 160번(사진 20장 × 4기준 × 2번)입니다. 정확한 금액은 Anthropic 콘솔의 Usage에서 확인하세요.

---

## 7. 사진 리뷰 두 번째 시험 (짧은 리뷰 + 불만 리뷰)

6절에서는 리뷰 글이 이미 사진 내용을 다 말하고 있어서, 사진 설명의 효과를 볼 수 없었습니다. 그래서 **같은 사진 16장**으로 리뷰를 두 종류 새로 만듭니다.

| 종류 | 예시 | 별점 | 확인할 것 |
|---|---|---|---|
| `short` (짧은 리뷰) | "잘 먹었어요 또 시킬게요" | 4~5 | 글에 없는 정보를 사진 설명이 더해 주는지 (사진 설명 있음이 **구체성**에서 이기는지) |
| `complaint` (불만 리뷰) | "떡볶이가 다 식어서 왔어요" | 1~3 | 사진으로 알 수 없는 문제를 **사진에서 봤다고 지어내지 않는지** (`acc<=2`가 적은지) |

사진 설명 프롬프트도 바꿨습니다. 6절에서는 설명에 "포장 상태는 확인할 수 없으나…" 같은 군말과 추측이 많아서, 이제는 두 줄만 쓰게 합니다.

```
음식: 떡볶이와 튀김
문제: 없음
```

### 7-1. [PC] 코드 보내기 (사진은 이미 서버에 있음)

```powershell
tar -czf "$env:USERPROFILE\Downloads\scrable_photo2.tgz" --exclude=.venv --exclude=outputs --exclude=data --exclude=*.egg-info -C "$env:USERPROFILE\naver_AI_service" scrable
```

```powershell
scp "$env:USERPROFILE\Downloads\scrable_photo2.tgz" root@<서버IP>:/opt/scrable/
```

### 7-2. [서버] 풀기, API 키

```bash
cd /opt/scrable/scrable_stage1 && tar -xzf ../scrable_photo2.tgz --strip-components=1 && ls configs/photo_eval_v2.json
```

```bash
read -rs ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY
```

### 7-3. [서버] 리뷰 32개 만들기 (2~3분)

```bash
.venv/bin/python -m scrable.synth_photo_reviews --photos data/review_photos --docs data/store_docs --out data/photo_reviews_v2.jsonl --style short complaint
```

마지막 줄이 `Wrote 32 photo reviews ...`이고, `ratings`에 1~3점과 4~5점이 섞여 있으면 성공입니다. 리뷰를 눈으로 봅니다:

```bash
cut -c1-160 data/photo_reviews_v2.jsonl
```

### 7-4. [서버] 평가 두 번 (20분 안팎)

```bash
nohup bash -c '.venv/bin/python -m scrable.evaluate --config configs/photo_eval_v2.json --split all --label with-photos && .venv/bin/python -m scrable.evaluate --config configs/photo_eval_v2.json --split all --label no-photos --no-photos --output-dir outputs/photo_eval_v2/no_photos --reference outputs/photo_eval_v2/with_photos/all_predictions.json' > photo_eval_v2.log 2>&1 &
```

`grep -c "Saved results" photo_eval_v2.log`가 **2**가 되면 끝입니다.

### 7-5. [서버] 결과 보기

```bash
grep -A6 "^no-photos - " photo_eval_v2.log; grep "seconds per reply" photo_eval_v2.log
```

```bash
.venv/bin/python scripts/photo_report.py outputs/photo_eval_v2/with_photos/all_predictions.json outputs/photo_eval_v2/no_photos/all_predictions.json
```

맨 위에 종류별 표가 나오고, 그 아래로 리뷰 32개가 하나씩 나옵니다. **두 출력을 모두 붙여 넣어 주세요.** 너무 길면 표만 먼저 보내도 됩니다:

```bash
.venv/bin/python scripts/photo_report.py outputs/photo_eval_v2/with_photos/all_predictions.json outputs/photo_eval_v2/no_photos/all_predictions.json --summary
```

`no-photos - with-photos`는 **"사진 설명 없음 − 있음"** 이라, **음수이면 사진 설명이 도움이 된 것**입니다.

---

## 8. 사진 설명 고친 뒤 다시 시험 (세 번째)

7절에서는 사진 설명 16개 중 15개가 "알 수 없음, 없음"이었습니다. 형식 예시의 기본값을 Vision-3B가 그대로 베꼈기 때문입니다. 그래서 7절의 "사진 설명 있음" 점수(+0.044)는 사진 설명이 아니라, 함께 붙는 **사진 규칙 세 줄** 덕분이었을 가능성이 큽니다.

이번에는 빈칸 형식을 빼고 질문 하나로 묻습니다. **먼저 0원으로 설명만 확인하고, 괜찮을 때만** 평가를 돌립니다.

이번 결과를 7절의 두 결과와 비교하면 효과를 나눠서 볼 수 있습니다.

| 비교 | 알 수 있는 것 |
|---|---|
| 이번(설명 있음) − 7절 `with-photos`(규칙만, 설명 비어 있음) | **사진 설명 자체**의 효과 |
| 이번 − 7절 `no-photos`(규칙도 설명도 없음) | 규칙 + 설명을 합친 효과 |

### 8-1. [PC] 코드 보내기

```powershell
tar -czf "$env:USERPROFILE\Downloads\scrable_photo3.tgz" --exclude=.venv --exclude=outputs --exclude=data --exclude=*.egg-info -C "$env:USERPROFILE\naver_AI_service" scrable
```

```powershell
scp "$env:USERPROFILE\Downloads\scrable_photo3.tgz" root@<서버IP>:/opt/scrable/
```

### 8-2. [서버] 풀고, 사진 설명만 확인 (0원, 2~3분)

```bash
cd /opt/scrable/scrable_stage1 && tar -xzf ../scrable_photo3.tgz --strip-components=1
```

```bash
.venv/bin/python scripts/describe_photos.py 2>&1 | grep -v "Warning\|warn"
```

사진마다 `[가게/파일 이름] 걸린 초`와 설명이 나오고, 마지막 줄에 `N descriptions shorter than 15 characters`가 나옵니다. **출력 전체를 붙여 넣어 주세요.**

- **N이 0~2이고 설명이 사진과 대체로 맞으면** → 8-3으로
- **N이 많거나 설명이 엉뚱하면** → 여기서 멈추고 붙여 넣어 주세요. 프롬프트를 다시 고쳐 0원으로 다시 확인합니다

### 8-3. [서버] 평가 한 번 (10분 안팎)

API 키를 입력합니다 (새로 접속했다면):

```bash
read -rs ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY
```

사진 설명은 8-2에서 이미 만들어 둬서 다시 만들지 않습니다.

```bash
nohup .venv/bin/python -m scrable.evaluate --config configs/photo_eval_v3.json --split all --label with-desc --reference outputs/photo_eval_v2/with_photos/all_predictions.json outputs/photo_eval_v2/no_photos/all_predictions.json > photo_eval_v3.log 2>&1 &
```

`grep -c "Saved results" photo_eval_v3.log`가 **1**이 되면 끝입니다.

### 8-4. [서버] 결과 보기

```bash
grep -A6 "^with-desc - " photo_eval_v3.log; grep "seconds per reply" photo_eval_v3.log
```

```bash
.venv/bin/python scripts/photo_report.py outputs/photo_eval_v3/with_desc/all_predictions.json outputs/photo_eval_v2/with_photos/all_predictions.json outputs/photo_eval_v2/no_photos/all_predictions.json
```

**두 출력을 붙여 넣어 주세요.** 이번에는 `with-desc - X` 형식이라, **양수이면 이번(설명 있음)이 나은 것**입니다(7절과 방향이 반대).

---

## 자주 나는 오류

| 오류 메시지 | 원인 | 해결 |
|---|---|---|
| `No module named 'einops'` (또는 timm, av, decord) | 3절 설치를 안 함 | 3절 다시 실행 |
| `401` / `gated` / `Access to model ... is restricted` | Hugging Face에서 모델 사용 동의가 필요함 | 모델 페이지에서 동의 버튼을 누르고 다시 실행. 서버에 저장된 HF 토큰으로 접근합니다 |
| `No space left on device` | 디스크 부족 | `df -h /opt` 결과를 보내 주세요 |
| `CUDA out of memory` | 다른 작업이 GPU를 쓰는 중 | `nvidia-smi`로 확인하고, 끝나면 다시 실행 |
| `Reference run ... covers different test reviews` | 비교 파일 경로가 다름 | `ls outputs/stage1 outputs/stage2_eval` 결과를 보내 주세요 |
| `No photos in data/review_photos/<store>/` | 사진 폴더 구조가 다름 | `ls -R data/review_photos` 결과를 보내 주세요 |
| `No store documents for photo folder ...` | 사진 폴더 이름이 가게 이름과 다름 | 폴더 이름을 `bunsik-01` 등으로 바꾸세요 |
| `Image not found` | 사진 경로가 바뀜 | 6-4를 다시 실행해 `data/photo_reviews.jsonl`을 새로 만드세요 |
| 그 밖의 `Traceback` | 원격 코드나 버전 문제 | `tail -30 vision_smoke.log` (또는 `vision_eval.log`) 출력을 보내 주세요 |
