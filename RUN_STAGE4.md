# 4단계 실험 실행 가이드 (공개 리뷰 데이터로 다시 학습)

3단계에서 같은 데이터로 학습 방법만 바꿨더니 점수가 오르지 않았습니다. 4단계는 **새 데이터**를 넣습니다.

- **실제 사람이 쓴 리뷰**: Mendeley 배달앱 리뷰 (CC BY 4.0, 음식점 136곳, 128,668건). 식당 20곳 × 15건 = 300건
- **합성 불만 리뷰**: 같은 식당 20곳 × 5건(2점 3건, 1점 2건) = 100건. 공개 데이터는 5점이 많아서 채웁니다
- **가게 문서**: 공개 데이터에는 가게 정보가 없어서, 식당마다 리뷰에 맞는 **가상** 가게 문서를 Haiku가 씁니다
- **프롬프트**: 지금 프롬프트에 불만 리뷰 규칙 두 줄을 더한 `prompts/stage4_prompt.txt`
- **쌍 고르기**: 좋은 답의 구체성이 3점(1~5점) 이상인 쌍만 씁니다. 2단계 쌍 316개도 같이 학습합니다
- **평가**: ① 기존 테스트 40건 ② **학습에 안 쓴 식당 10곳의 실제 리뷰 40건**(새 테스트)

`[WSL]`은 노트북 WSL 터미널(zsh), `[서버]`는 SSH로 접속한 서버 창(bash)입니다. 서버 폴더는 `/opt/scrable/scrable_stage1`입니다.

## 비용과 시간 (추정)

| 단계 | 비용 | 시간 |
|---|---|---|
| 가게 문서 30개 | 약 $0.1 | 3~5분 |
| 합성 불만 리뷰 100건 | 약 $0.1 | 2~3분 |
| 파일럿 10건 (비용 확인) | 몇 센트 | 5분 |
| 쌍 만들기 400건 | 약 $1.5 | 1~1.5시간 |
| 학습 (쌍 약 700개) | 0원 | 10~15분 |
| 평가 (테스트 40건 + 실제 리뷰 40건 × 3) | 약 $0.3 | 30분 |

---

## 1. [PC] Mendeley 데이터 받기

브라우저에서 https://data.mendeley.com/datasets/rnyrpzyw3h/2 를 열고 **`Reviews Written in Korean.csv`** 를 받습니다(영어 번역본은 필요 없음). Windows `다운로드` 폴더에 저장되면 됩니다.

## 2. [WSL] 코드와 데이터를 서버로 보내기

```bash
cd ~/naver/Scrable && git pull
```
```bash
tar -czf ~/scrable_stage4.tgz scrable configs prompts scripts tests pyproject.toml RUN_STAGE4.md
```
```bash
scp ~/scrable_stage4.tgz root@<서버IP>:/opt/scrable/
```
```bash
scp "/mnt/c/Users/wounc/Downloads/Reviews Written in Korean.csv" root@<서버IP>:/opt/scrable/
```

## 3. [서버] 코드 풀기

```bash
cd /opt/scrable/scrable_stage1
```
```bash
tar -xzf /opt/scrable/scrable_stage4.tgz
```
```bash
mkdir -p data/raw && mv "/opt/scrable/Reviews Written in Korean.csv" data/raw/
```

`data/`와 `outputs/`는 덮어쓰지 않습니다. 2단계 어댑터와 쌍은 그대로 남습니다.

## 4. CSV 열 (확인 끝남)

노트북에서 미리 확인했습니다. 12만 8,668행이고, 리뷰 글은 `Review`, 식당은 `RestaurantID`(135곳), 별점은 `Total`(1~5)입니다. 5점이 91%라 5절에서 식당마다 5점 미만 리뷰를 먼저(최대 절반) 고릅니다. 다시 보고 싶으면:

```bash
.venv/bin/python -m scrable.public_reviews inspect "data/raw/Reviews Written in Korean.csv"
```

## 5. [서버] 리뷰 고르기 (0원)

노트북에서 같은 명령으로 시험했을 때 학습 299건(5점 189, 4점 67, 3점 24, 2점 8, 1점 11)과 테스트 40건이 나왔습니다.

```bash
.venv/bin/python -m scrable.public_reviews build "data/raw/Reviews Written in Korean.csv" --text-col Review --store-col RestaurantID --rating-col Total
```

`Wrote 299 train reviews ... 40 test reviews` 가 나오면 됩니다. 실제 식당 이름은 리뷰 파일에 들어가지 않고 `data/public_store_map.json`에만 남습니다(git에 안 올라감).

## 6. [서버] 가게 문서와 불만 리뷰 만들기 (약 $0.2)

```bash
read -rs ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY
```
```bash
.venv/bin/python -m scrable.synth_store_docs --reviews data/public_train.jsonl --out data/public_train_docs
```
```bash
.venv/bin/python -m scrable.synth_store_docs --reviews data/public_test.jsonl --out data/public_test_docs
```
```bash
.venv/bin/python -m scrable.synth_reviews --docs data/public_train_docs --out data/public_complaints.jsonl --counts "2:3,1:2" --avoid data/store_reviews.jsonl data/public_test.jsonl
```

가게 문서 하나를 열어서 이상하지 않은지 확인하세요(실제 상호나 전화번호가 없어야 함).

```bash
cat data/public_train_docs/pub-001/info.md
```

학습 파일과 파일럿 파일을 만듭니다.

```bash
cat data/public_train.jsonl data/public_complaints.jsonl > data/stage4_reviews.jsonl && head -n 10 data/stage4_reviews.jsonl > data/stage4_pilot.jsonl && wc -l data/stage4_reviews.jsonl
```

`400` 안팎이면 됩니다(비슷한 합성 리뷰는 버려서 조금 적을 수 있음).

## 7. [서버] 파일럿 10건으로 비용 확인

먼저 GPU를 비워 둡니다. gRPC 서버가 떠 있으면 끕니다.

```bash
pkill -f scrable.serve; nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

Anthropic Console의 Usage 화면에서 **지금 사용 금액을 적어 두고** 파일럿을 돌립니다.

```bash
.venv/bin/python -m scrable.dpo pairs --config configs/stage4_pilot.json 2>&1 | tail -n 30
```

끝나면 Console 금액을 다시 봅니다. **늘어난 금액 × 40**이 본 실행 비용입니다. $3을 넘으면 멈추고 알려 주세요. 파일럿 결과는 캐시에 남아서 본 실행에서 다시 돈이 들지 않습니다.

## 8. [서버] 쌍 만들기 (약 1~1.5시간)

```bash
nohup .venv/bin/python -m scrable.dpo pairs --config configs/stage4_dpo.json > stage4_pairs.log 2>&1 &
```
```bash
tail -f stage4_pairs.log
```

`Ctrl+C`는 보기만 멈춥니다. 인터넷이 끊겨도 계속 돕니다. 끝나면 요약을 봅니다.

```bash
grep -A12 '"reviews"' stage4_pairs.log | head -n 20
```

`pairs`가 250개 이상이면 다음 단계로 갑니다. 훨씬 적으면 `skipped` 줄을 보내 주세요.

## 9. [서버] 학습 (10~15분)

```bash
nohup .venv/bin/python -m scrable.dpo train --config configs/stage4_dpo.json > stage4_train.log 2>&1 &
```
```bash
tail -f stage4_train.log
```

`Saved LoRA adapter to .../outputs/stage4/adapter`가 나오면 끝입니다. 처음에 `Training pairs by file:`에 새 쌍과 2단계 쌍 316개가 같이 나와야 합니다.

## 10. [서버] 평가 (약 30분)

한 번에 이어서 돌립니다.

```bash
nohup sh -c '
.venv/bin/python -m scrable.evaluate --config configs/stage4_eval.json --label stage4 --reference outputs/stage1/test_predictions.json outputs/stage2_eval/test_predictions.json &&
.venv/bin/python -m scrable.evaluate --config configs/public_test_stage2.json --split all --label stage2 &&
.venv/bin/python -m scrable.evaluate --config configs/public_test_haiku.json --split all --label haiku &&
.venv/bin/python -m scrable.evaluate --config configs/public_test_stage4.json --split all --label stage4 --reference outputs/public_test/stage2/all_predictions.json outputs/public_test/haiku/all_predictions.json
' > stage4_eval.log 2>&1 &
```

끝났는지는 이걸로 봅니다. `4`가 나오면 다 끝난 겁니다.

```bash
grep -c "Saved results" stage4_eval.log
```

결과 보기:

```bash
grep -A6 "^stage4 - " stage4_eval.log
```

**출력을 그대로 보내 주세요.**

## 11. 결과 판단과 1000건으로 늘리기

**유의미하게 좋아졌다**는 기준: `stage4 - dpo`(기존 테스트) 또는 `stage4 - stage2`(실제 리뷰 테스트)에서 **overall이 + 이고 `clear`**, 그리고 accuracy가 `clear`로 떨어지지 않은 것.

그러면 1000건으로 늘립니다. 식당을 20곳 → 50곳으로 늘리는 것만 바뀝니다. **테스트 40건과 앞의 식당 20곳은 그대로라서**, 이미 만든 답과 채점은 캐시에서 다시 쓰고 새 600건 비용만 듭니다(약 $2.5).

```bash
.venv/bin/python -m scrable.public_reviews build "data/raw/Reviews Written in Korean.csv" --text-col Review --store-col RestaurantID --rating-col Total --train-stores 50
```

그다음 6절(테스트 문서 명령은 빼도 됨) → 8절 → 9절 → 10절을 다시 합니다. 9절 전에 이전 어댑터를 보관해 두세요.

```bash
mv outputs/stage4/adapter outputs/stage4/adapter_400
```

## 12. 서버 모델 바꾸기 (좋아졌을 때만)

`configs/serve.json`의 `base`에서 두 줄을 바꿉니다.

- `"response_adapter": "outputs/stage4/adapter"`
- `"base_prompt_file": "prompts/stage4_prompt.txt"`

프롬프트도 같이 바꿔야 합니다. 학습한 프롬프트와 서버 프롬프트가 다르면 성능이 떨어집니다.

---

## 데이터 출처 표기 (CC BY 4.0)

발표 자료와 `data/DATA_SOURCE.md`에 출처를 적습니다.

> "Bias-Free Dataset of Food Delivery App Reviews with Data Poisoning Attacks", Mendeley Data, V2 (2024), https://data.mendeley.com/datasets/rnyrpzyw3h/2 , CC BY 4.0. 리뷰를 골라 쓰고(정리·중복 제거·식당 단위 분할), 가게 정보와 일부 불만 리뷰는 Claude Haiku로 만든 가상 데이터입니다.

원본 CSV와 변환한 리뷰 파일은 git에 올리지 않습니다.

## 문제가 생기면

| 메시지 | 원인 | 할 일 |
|---|---|---|
| `Column '...' not in CSV` | 열 이름이 다름 | 명령의 열 이름 오타 확인 |
| `Only N restaurants have 15+ usable reviews` | 거를 리뷰가 많음 | `--train-per-store 10 --train-stores 30`처럼 바꿈 |
| `Invalid structured LLM output` | Haiku가 형식을 어김 | 같은 명령을 다시 실행(성공한 건 캐시에 있음) |
| `CUDA out of memory` | GPU에 다른 프로그램이 있음 | `nvidia-smi`로 확인하고 끄기 |
| `authentication_error` | API 키 문제 | `read -rs ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY` 다시 |
