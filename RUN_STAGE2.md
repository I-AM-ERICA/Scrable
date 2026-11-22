# 2단계 실험 실행 가이드 (LoRA DPO)

1단계에서 확인한 것: 프롬프트를 고쳐도 1.5B의 **정확성은 거의 그대로**였습니다(0.662 → 0.669). 금지 규칙을 프롬프트에 적어도 지키지 못했습니다. 2단계에서는 **모델 가중치 일부(LoRA)를 학습**해서 이 습관을 고칩니다.

이 문서는 [RUN_STAGE1.md](RUN_STAGE1.md)를 끝낸 상태를 가정합니다. 서버 접속, Hugging Face 로그인, API 키 입력 방법은 1단계 가이드를 참고하세요. `[PC]`는 Windows PowerShell, `[서버]`는 SSH로 접속한 서버 창입니다.

---

## 0. 2단계가 하는 일

| 순서 | 명령 | 하는 일 | 결과 파일 |
|---|---|---|---|
| ① 리뷰 만들기 | `scrable.synth_reviews` | 학습용 가상 매장 4곳의 리뷰 320건을 Haiku로 만듭니다 | `data/dpo_reviews.jsonl` |
| ② 선호 쌍 만들기 | `scrable.dpo pairs` | 리뷰마다 1.5B 답변 3개 + Haiku 답변 1개를 만들고, 1단계와 같은 채점기로 점수를 매겨 **가장 좋은 답(chosen)** 과 **1.5B의 가장 나쁜 답(rejected)** 을 짝짓습니다 | `outputs/stage2/pairs.jsonl` |
| ③ 학습 | `scrable.dpo train` | chosen 쪽 확률은 올리고 rejected 쪽 확률은 내리도록 LoRA를 학습합니다 (DPO) | `outputs/stage2/adapter/` |
| ④ 평가 | `scrable.evaluate` | 1단계와 **똑같은 테스트 리뷰 40건**에 답하게 하고, 1단계 결과와 리뷰별로 비교합니다 | `outputs/stage2_eval/` |

### 왜 학습 매장과 테스트 매장을 나눴나

- 학습: 피자·중식·족발·베이커리 (`data/train_store_docs/`, 새로 만든 가상 매장)
- 테스트: 1단계와 같은 분식·치킨·카페·국밥 (`data/store_docs/`)

같은 매장으로 학습하면 "그 가게의 정책을 외워서" 점수가 오를 수 있습니다. 매장을 나누면 점수가 올랐을 때 **답글 쓰는 습관이 나아진 것**(근거 없는 약속 안 하기, 고객 탓 안 하기, 연락 경로 안내하기, '당신' 안 쓰기)이라고 말할 수 있습니다.

### 비용과 시간 (추정)

| 단계 | Haiku 호출 | 비용 | 시간 (L4 GPU) |
|---|---|---|---|
| ① 리뷰 만들기 | 44번 | 몇 센트 | 3~5분 |
| ② 선호 쌍 만들기 | 약 5,000번 (답변 320 + 채점 4,800) | **약 1~2달러** | 약 1시간 |
| ③ 학습 | 0번 (GPU만 사용) | 0원 | 10~20분 |
| ④ 평가 | 약 160번 | 몇 센트 | 5분 |

정확한 금액은 Anthropic 콘솔의 Usage에서 확인하세요. 모든 호출 결과는 `llm_cache`에 저장되므로, 중간에 끊겨도 **같은 명령을 다시 실행하면 이미 끝난 호출은 다시 돈을 내지 않고** 이어서 진행합니다.

---

## 1. [PC] 코드와 학습 매장 정보를 서버로 보내기

코드 묶기 (1단계와 같은 명령, 파일 이름만 다름):

```powershell
tar -czf "$env:USERPROFILE\Downloads\scrable_stage2.tgz" --exclude=.venv --exclude=outputs --exclude=data --exclude=*.egg-info -C "$env:USERPROFILE\naver_AI_service" scrable
```

학습 매장 정보 묶기 (`data` 폴더는 코드 묶음에서 빠지므로 따로 보냅니다):

```powershell
tar -czf "$env:USERPROFILE\Downloads\scrable_stage2_data.tgz" -C "$env:USERPROFILE\naver_AI_service\scrable" data/train_store_docs
```

서버로 보내기:

```powershell
scp "$env:USERPROFILE\Downloads\scrable_stage2.tgz" "$env:USERPROFILE\Downloads\scrable_stage2_data.tgz" root@<서버IP>:/opt/scrable/
```

---

## 2. [서버] 풀고 peft 설치하기

```bash
cd /opt/scrable/scrable_stage1
tar -xzf ../scrable_stage2.tgz --strip-components=1
tar -xzf ../scrable_stage2_data.tgz
ls data
```

- 같은 폴더(`scrable_stage1`)에 덮어씁니다. 코드만 바뀌고 `data/store_reviews.jsonl`, `outputs/stage1`, `.venv`는 그대로 남습니다.
- `ls data` 결과에 `store_docs`, `store_reviews.jsonl`, `train_store_docs`가 보이면 성공입니다.

LoRA 학습 라이브러리(peft)를 설치합니다. 처음 한 번만 하면 됩니다.

```bash
.venv/bin/python -m pip install -e '.[dpo]'
```

```bash
.venv/bin/python -c "import peft, transformers; print(peft.__version__, transformers.__version__)"
```

버전 두 개가 출력되면 성공입니다. (transformers는 4.x여야 합니다.)

API 키를 입력합니다 (서버에 새로 접속할 때마다):

```bash
read -rs ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY
```

붙여 넣어도 화면에 안 보이는 게 정상입니다. Enter를 누르세요.

---

## 3. [서버] ① 학습용 리뷰 만들기

```bash
.venv/bin/python -m scrable.synth_reviews --docs data/train_store_docs --out data/dpo_reviews.jsonl --avoid data/store_reviews.jsonl
```

- `--avoid`: 테스트 리뷰와 너무 비슷한 문장은 버립니다(테스트 데이터가 학습에 새지 않게).
- 마지막에 `Wrote 3xx reviews ...`가 나오면 성공입니다. 비슷한 문장을 버려서 320건보다 조금 적을 수 있습니다.

몇 개만 눈으로 확인해 보세요:

```bash
head -5 data/dpo_reviews.jsonl
```

---

## 4. [서버] ② 선호 쌍 만들기 (약 1시간)

오래 걸리므로 `nohup`으로 백그라운드에서 돌립니다. SSH 창을 닫아도 계속 돕니다.

```bash
nohup .venv/bin/python -m scrable.dpo pairs --config configs/stage2_dpo.json > stage2_pairs.log 2>&1 &
```

진행 상황 보기 (`Ctrl+C`로 보기만 멈춥니다. 작업은 계속 돕니다):

```bash
tail -f stage2_pairs.log
```

진행 막대가 세 번 나옵니다: `sample small-model answers`(1.5B 답변) → `teacher answers`(Haiku 답변) → `judge`(채점).

끝나면 요약이 나옵니다. 확인할 것:

```bash
cat outputs/stage2/pairs_summary.json | head -30
```

| 항목 | 뜻 | 기대하는 값 |
|---|---|---|
| `pairs` | 만들어진 학습 쌍 수 | 150 이상이면 충분합니다 |
| `skipped` | 버린 리뷰와 이유 (`margin_too_small`: 좋은 답과 나쁜 답의 차이가 작음) | |
| `chosen_from` | chosen이 1.5B 답(`policy`)인지 Haiku 답(`teacher`)인지 | |
| `mean_score_by_source` | 1.5B 답 평균 vs Haiku 답 평균 | 1단계처럼 Haiku가 높게 나와야 정상 |

`pairs`가 100보다 적으면 학습 전에 결과를 보내 주세요.

---

## 5. [서버] ③ LoRA DPO 학습 (10~20분)

```bash
nohup .venv/bin/python -m scrable.dpo train --config configs/stage2_dpo.json > stage2_train.log 2>&1 &
```

```bash
tail -f stage2_train.log
```

에포크(전체 쌍을 한 번 다 보는 것)마다 이런 줄이 나옵니다:

```
{"epoch": 1, "train": {... "reward_accuracy": 0.8 ...}, "eval": {"loss": 0.55, "reward_accuracy": 0.75, ...}}
```

- `reward_accuracy`: 학습된 모델이 chosen을 rejected보다 더 선호하는 비율. **0.5보다 커지면** 배우고 있는 것입니다.
- `eval`: 학습에 쓰지 않은 쌍(10%)으로 잰 값. train만 오르고 eval이 안 오르면 외우기만 한 것입니다.
- 마지막에 `Saved LoRA adapter to .../outputs/stage2/adapter`가 나오면 성공입니다.

> **LoRA란?** 1.5B 가중치 전체를 바꾸지 않고, 옆에 작은 가중치(약 1,600만 개, 전체의 약 1%)를 붙여 그것만 학습하는 방법입니다. 결과물(`adapter` 폴더)도 수십 MB로 작습니다.

---

## 6. [서버] ④ 평가: 1단계와 비교 (약 5분)

```bash
nohup .venv/bin/python -m scrable.evaluate --config configs/stage2_eval.json --label dpo --reference outputs/stage1/test_predictions.json > stage2_eval.log 2>&1 &
```

```bash
tail -40 stage2_eval.log
```

- 1단계와 같은 테스트 리뷰 40건, 같은 프롬프트(1단계에서 고른 프롬프트, `prompts/stage1_best_prompt.txt`), 같은 채점기를 씁니다. **바뀐 것은 LoRA 어댑터 하나**입니다.
- 테스트 리뷰가 1단계와 다르면 `covers different test reviews` 오류로 멈춥니다(공정한 비교를 위한 확인).

결과 읽는 법:

```
dpo - stage1:optimized  (95% CI; win/lose/tie)
  overall      +0.080 [+0.030, +0.130] 25/10/5  clear
  accuracy     +0.120 [+0.050, +0.190] 20/6/14  clear
```

(숫자는 예시입니다.)

| 비교 대상 | 뜻 |
|---|---|
| `stage1:optimized` | **DPO의 효과.** 같은 프롬프트에서 학습 전 1.5B와 비교 |
| `stage1:baseline` | 1단계 + 2단계를 합친 효과 (기본 프롬프트의 1.5B와 비교) |
| `claude-haiku-5-5:base` | Haiku와의 남은 차이 (음수면 아직 Haiku보다 낮음) |

- `clear`: 95% 신뢰구간에 0이 없음 → 40건으로도 차이가 확실합니다.
- `chance?`: 신뢰구간에 0이 있음 → 우연일 수 있습니다. 발표에서 "올랐다"고 단정하지 마세요.

---

## 7. [서버→PC] 결과 받기

```bash
tar -czf stage2_result.tgz outputs/stage2/pairs_summary.json outputs/stage2/pairs.jsonl outputs/stage2/train_log.json outputs/stage2_eval/metrics.json outputs/stage2_eval/test_predictions.json data/dpo_reviews.jsonl stage2_pairs.log stage2_train.log stage2_eval.log
```

[PC] PowerShell에서:

```powershell
scp root@<서버IP>:/opt/scrable/scrable_stage1/stage2_result.tgz "$env:USERPROFILE\Downloads\"
```

---

## 7-1. [PC] 사람 블라인드 평가 (API 비용 없음)

심판과 교사가 모두 Haiku라서 생기는 편향을 확인하는 단계입니다. 팀원이 출처를 모르는 답변 세 개(1단계 최고 / DPO / Haiku)에 1~5점을 줍니다.

1. 평가 페이지 만들기. 결과 파일을 `outputs/stage1/`, `outputs/stage2_eval/`에 둔 뒤 실행합니다:

   ```powershell
   .venv\Scripts\python scripts\human_eval.py build --run stage1:optimized=outputs/stage1/test_predictions.json --run dpo=outputs/stage2_eval/test_predictions.json --run claude-haiku-5-5:base=outputs/stage1/test_predictions.json
   ```

   - `outputs/human_eval/human_eval.html`: 팀원에게 보여 줄 페이지
   - `outputs/human_eval/blind_key.json`: 어느 칸이 어느 모델인지 적힌 정답지. 팀원에게 보내지 마세요.
2. 팀원은 페이지에서 평가합니다. 자동 저장이 안 되는 사람은 끝난 뒤 "내 결과 복사"를 눌러 받은 글을 보내 줍니다. 받은 글은 사람마다 `outputs/human_eval/ratings/이름.json`으로 저장합니다.
3. 채점:

   ```powershell
   .venv\Scripts\python scripts\human_eval.py score outputs/human_eval/ratings/*.json --judge dpo=outputs/stage2_eval/test_predictions.json --judge stage1:optimized=outputs/stage1/test_predictions.json
   ```

   **짧은 버전: 구글 설문 (팀원·지인용, 5분 이내)**. 리뷰 6개마다 답글 2개 중 더 나은 쪽 하나만 고릅니다. 세 모델의 짝 6가지(순서 포함)가 리뷰마다 하나씩 들어갑니다. 응답은 구글 시트에 자동으로 모입니다.

   ```powershell
   .venv\Scripts\python scripts\human_eval.py build --mode pair --subset 6 --out-dir outputs/human_eval_pair --seed human-eval-pair-1 --run stage1:optimized=outputs/stage1/test_predictions.json --run dpo=outputs/stage2_eval/test_predictions.json --run claude-haiku-5-5:base=outputs/stage1/test_predictions.json
   ```

   ```powershell
   .venv\Scripts\python scripts\human_eval.py form --dir outputs/human_eval_pair
   ```

   1. https://script.google.com 에서 "새 프로젝트"를 만들고, 기본 코드를 지운 뒤 `outputs\human_eval_pair\make_google_form.gs` 내용을 전부 붙여 넣습니다.
   2. 위쪽 함수 선택 칸이 `makeForm`인지 확인하고 **실행**을 누릅니다. 처음에는 권한 요청이 뜹니다. 본인이 만든 스크립트라서 "확인되지 않은 앱" 경고가 나오는데, **고급 → (프로젝트 이름)(으)로 이동 → 허용**을 누르면 됩니다.
   3. 아래 실행 로그에 "보낼 링크", "설문 편집", "응답 시트" 주소가 나옵니다. **보낼 링크**를 팀원·지인에게 보냅니다.
   4. 응답이 모이면 응답 시트에서 **파일 → 다운로드 → CSV**로 받아 `outputs\human_eval_pair\responses.csv`로 저장하고 채점합니다:

   ```powershell
   .venv\Scripts\python scripts\human_eval.py score --csv outputs/human_eval_pair/responses.csv --key outputs/human_eval_pair/blind_key.json --judge dpo=outputs/stage2_eval/test_predictions.json --judge stage1:optimized=outputs/stage1/test_predictions.json --judge claude-haiku-5-5:base=outputs/stage1/test_predictions.json
   ```

   출력 마지막 줄은 LLM 심판이 사람과 같은 답을 더 낫다고 고른 비율입니다. 설문 문항이나 선택지 글자를 구글에서 손으로 고치면 채점이 응답을 못 알아봅니다. 고치려면 스크립트를 고쳐 새로 만드세요. 결과를 이미 본 사람(개발자 본인)은 응답하지 않습니다.

   사람 점수에서도 순서가 LLM 심판과 같으면(Haiku > DPO > 1단계 최고) 편향 걱정을 덜 수 있습니다. 순서가 바뀌면 그 사실을 발표에 그대로 밝힙니다.

---

## 8. 자주 나는 오류

| 오류 메시지 | 원인 | 해결 |
|---|---|---|
| `No module named 'peft'` | 2단계 설치를 안 함 | [2단계](#2-서버-풀고-peft-설치하기)의 `pip install -e '.[dpo]'` |
| `credit balance is too low` | Anthropic 크레딧 부족 | 콘솔에서 충전 후 **같은 명령을 다시 실행** (끝난 호출은 저장돼 있음) |
| `ANTHROPIC_API_KEY` 관련 인증 오류 | 새로 접속해서 키가 사라짐 | `read -rs ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY` 후 다시 실행 |
| `No pairs in outputs/stage2/pairs.jsonl` | ② 선호 쌍 만들기를 안 했거나 실패 | `stage2_pairs.log` 마지막 30줄 확인 |
| `CUDA out of memory` (학습 중) | GPU 메모리 부족 | `configs/stage2_dpo.json`에 `"gradient_checkpointing": true`를 추가하거나 `"max_length": 1536`으로 줄이기 |
| `No LoRA adapter files` (평가 중) | ③ 학습이 끝나지 않음 | `stage2_train.log`에서 `Saved LoRA adapter`가 나왔는지 확인 |
| `covers different test reviews` | 평가 설정의 데이터·크기·seed가 1단계와 다름 | `configs/stage2_eval.json`을 바꾸지 말고 1단계 데이터 그대로 사용 |

---

## 9. 발표 때 밝힐 한계

- 학습 리뷰도 Claude(Haiku)가 만든 합성 데이터이고, chosen 답의 상당수도 Haiku 답변이며, 채점도 Haiku가 합니다. 점수가 "Haiku 취향"으로 치우쳤을 수 있습니다. 사람 평가 30~50건으로 확인하는 것이 좋습니다.
- 테스트는 40건이라 작은 차이는 `chance?`로 나옵니다.
- 학습 설정(beta 0.1, 학습률 5e-5, 2 에포크, LoRA r=16)은 흔히 쓰는 시작값이고, 튜닝하지 않았습니다.
