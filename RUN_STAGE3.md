# 3단계 실험 실행 가이드 (구체성 개선)

2단계 DPO 모델은 정확성은 Haiku를 거의 따라잡았지만, **구체성**(가게 정보를 꺼내 쓰는 정도)은 그대로였습니다. 확인해 보니 가게 정보는 모델 입력에 이미 들어가 있었고 답이 잘린 것도 아니었습니다. **모델이 정보를 보고도 쓰지 않는 것**이 원인입니다.

3단계는 2단계에서 이미 만든 데이터(`outputs/stage2/candidates.jsonl`)를 다시 써서, 학습 방법만 바꾼 모델 3개를 만들어 비교합니다. **새로 답변을 만들거나 다시 채점하지 않으므로** 학습 데이터 비용은 0원입니다.

이 문서는 [RUN_STAGE2.md](RUN_STAGE2.md)를 끝낸 상태를 가정합니다. `[PC]`는 Windows PowerShell, `[서버]`는 SSH로 접속한 서버 창입니다.

---

## 0. 무엇을 비교하나

| 이름 | 바꾼 것 | 노리는 효과 |
|---|---|---|
| `dpo-v1` | (2단계 모델 그대로) | 비교 기준 |
| `sft1` | 같은 학습 쌍, **좋은 답 따라 쓰기 비중** 0.2 → 1.0 | Haiku의 구체적인 답을 더 직접 따라 쓰게 |
| `spec` | 구체성 차이가 확실한 쌍만 사용(좋은 답 구체성 4점 이상, 2점 이상 차이), 비중 0.5 | 구체성 신호만 남기기 |
| `generic` | 나쁜 답을 **"틀리진 않았지만 뻔한 답"** 으로 교체(없으면 원래대로), `spec`과 같은 필터, 비중 0.5 | "틀리지 마라"가 아니라 "가게 정보를 써라"를 배우게 |

### 왜 테스트 40건 말고 dev 세트로 고르나

모델 3개를 테스트 40건으로 비교해서 제일 좋은 걸 고르면, **시험 문제를 보고 답을 고르는 셈**이 됩니다. 그래서 고르는 건 dev 세트(1단계의 train 30 + validation 10 = 40건, 테스트와 겹치지 않음)로 하고, 테스트 40건은 **최종으로 고른 1개만** 한 번 돌립니다.

### 비용과 시간 (추정)

| 단계 | Haiku 호출 | 비용 | 시간 |
|---|---|---|---|
| 쌍 다시 고르기 ×3 | 0번 | 0원 | 몇 초 |
| dev 평가 (기준 + 3개) | 약 640번 | 1달러 미만 | 각 3~5분 |
| 학습 ×3 | 0번 | 0원 | 각 3~5분 |
| 최종 테스트 평가 1개 | 약 160번 | 몇 센트 | 5분 |

---

## 1. [PC] 코드를 서버로 보내기

```powershell
tar -czf "$env:USERPROFILE\Downloads\scrable_stage3.tgz" --exclude=.venv --exclude=outputs --exclude=data --exclude=*.egg-info -C "$env:USERPROFILE\naver_AI_service" scrable
```

```powershell
scp "$env:USERPROFILE\Downloads\scrable_stage3.tgz" root@<서버IP>:/opt/scrable/
```

---

## 2. [서버] 풀고 준비 확인

```bash
cd /opt/scrable/scrable_stage1
tar -xzf ../scrable_stage3.tgz --strip-components=1
ls configs/stage3_*.json outputs/stage2/candidates.jsonl
```

설정 파일 3개와 `candidates.jsonl`이 보이면 성공입니다. `candidates.jsonl`이 없다고 나오면 멈추고 알려 주세요.

API 키를 입력합니다 (서버에 새로 접속할 때마다):

```bash
read -rs ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY
```

---

## 3. [서버] 학습 쌍 다시 고르기 (몇 초, 비용 0)

```bash
for v in sft1 spec generic; do .venv/bin/python -m scrable.dpo select --config configs/stage3_$v.json; done
```

각각 `Wrote N pairs ...` 줄과 요약이 나옵니다. 확인할 것:

| 이름 | `pairs` 기대값 | 참고 |
|---|---|---|
| `sft1` | 316 (2단계와 같아야 함) | 다르면 멈추고 알려 주세요 |
| `spec` | 170 안팎 | |
| `generic` | 170 안팎 | `rejected_accurate_but_generic`: 나쁜 답이 "뻔한 답"으로 바뀐 쌍 수. 30보다 적으면 `spec`과 거의 같은 실험이 되니 알려 주세요 |

---

## 4. [서버] 기준 모델(dpo-v1)을 dev 세트로 평가 (3~5분)

```bash
nohup .venv/bin/python -m scrable.evaluate --config configs/stage2_eval.json --split dev --label dpo-v1 --output-dir outputs/dev/dpo_v1 > stage3_dev_v1.log 2>&1 &
```

```bash
tail -5 stage3_dev_v1.log
```

`Saved results to .../outputs/dev/dpo_v1`이 나오면 끝입니다. **이게 끝난 뒤에** 다음 단계를 시작하세요.

---

## 5. [서버] 3개 학습 + dev 평가를 한 번에 (약 25분)

한 줄로 셋을 차례로 학습하고, 끝날 때마다 dev 세트로 평가해 dpo-v1과 비교합니다. GPU는 한 번에 하나만 씁니다.

```bash
nohup bash -c 'for v in sft1 spec generic; do .venv/bin/python -m scrable.dpo train --config configs/stage3_$v.json && .venv/bin/python -m scrable.evaluate --config configs/stage2_eval.json --split dev --label $v --adapter outputs/stage3_$v/adapter --output-dir outputs/dev/$v --reference outputs/dev/dpo_v1/dev_predictions.json; done' > stage3_run.log 2>&1 &
```

```bash
tail -f stage3_run.log
```

`Ctrl+C`는 보기만 멈춥니다. 다 끝나면 비교 결과만 모아 봅니다:

```bash
grep -A6 " - dpo-v1" stage3_run.log
```

이렇게 셋이 나오면 끝입니다 (숫자는 예시):

```
spec - dpo-v1  (95% CI; win/lose/tie)
  overall      +0.040 [+0.005, +0.075] 15/6/19  clear
  relevancy    ...
  specificity  +0.120 [+0.060, +0.180] 18/3/19  clear
  accuracy     -0.010 [-0.060, +0.040] 6/7/27  chance?
```

**좋은 결과**: `specificity`는 `clear`로 오르고, `accuracy`는 떨어지지 않는 것(음수 + `clear`가 아님)입니다. 구체성을 밀면 가게 정보에 없는 말을 지어낼 수 있어서, 정확성이 같이 떨어지면 실패로 봅니다.

---

## 6. [서버→PC] 결과 보내기

```bash
tar -czf stage3_result.tgz outputs/stage3_*/pairs_summary.json outputs/stage3_*/train_log.json outputs/dev stage3_*.log
```

[PC]:

```powershell
scp root@<서버IP>:/opt/scrable/scrable_stage1/stage3_result.tgz "$env:USERPROFILE\Downloads\"
```

결과를 보고 하나를 고른 뒤, **그 모델만** 테스트 40건으로 평가합니다. 예를 들어 `spec`을 고르면:

```bash
nohup .venv/bin/python -m scrable.evaluate --config configs/stage2_eval.json --label spec --adapter outputs/stage3_spec/adapter --output-dir outputs/stage3_eval --reference outputs/stage2_eval/test_predictions.json > stage3_eval.log 2>&1 &
```

그다음 새 모델 vs DPO vs Haiku로 구글 설문을 다시 돌립니다 ([RUN_STAGE2.md](RUN_STAGE2.md) 7-1절).

---

## 7. 프롬프트 시험 (5절 결과가 모두 `chance?`였을 때)

학습만으로는 구체성이 확실히 오르지 않아서, 프롬프트에 "가게 정보에서 리뷰와 관련된 사실을 하나 이상 넣으라"는 규칙을 더한 `prompts/stage3_specific_prompt.txt`로 dpo-v1과 `generic`을 dev 세트에서 다시 평가합니다. 기준은 똑같이 dpo-v1(원래 프롬프트)입니다.

```bash
nohup bash -c '.venv/bin/python -m scrable.evaluate --config configs/stage2_eval.json --split dev --label v1-prompt --prompt prompts/stage3_specific_prompt.txt --output-dir outputs/dev/v1_prompt --reference outputs/dev/dpo_v1/dev_predictions.json && .venv/bin/python -m scrable.evaluate --config configs/stage2_eval.json --split dev --label generic-prompt --adapter outputs/stage3_generic/adapter --prompt prompts/stage3_specific_prompt.txt --output-dir outputs/dev/generic_prompt --reference outputs/dev/dpo_v1/dev_predictions.json' > stage3_prompt.log 2>&1 &
```

`grep -c "Saved results" stage3_prompt.log`가 2가 되면 `grep -B2 -A6 " - dpo-v1" stage3_prompt.log` 결과를 확인합니다.

---

## 자주 나는 오류

| 오류 메시지 | 원인 | 해결 |
|---|---|---|
| `No candidates.jsonl in outputs/stage2` | 2단계 결과 폴더가 없거나 다른 곳에 있음 | `ls outputs/stage2` 결과를 보내 주세요 |
| `covers different test reviews` | `--reference`가 다른 split의 결과 | dev 평가에는 `outputs/dev/dpo_v1/dev_predictions.json`을 씁니다 |
| `No pairs in ...` | 다시 고르기에서 쌍이 0개 | 3절 출력의 `skipped`를 보내 주세요 |
| `CUDA out of memory` | GPU 메모리 부족 | 설정 파일에 `"gradient_checkpointing": true`를 추가하고 다시 실행 |
