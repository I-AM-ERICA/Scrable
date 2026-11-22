# 사진 사실 실험: 정확한 사진 정보를 주면 답글이 그걸 살리는가

사진 설명을 바꾸거나(Qwen, 메뉴 힌트) 답글 모델·어댑터를 키워도(7B, DoRA) 사진 답글 점수가 오르지 않았습니다. 같은 Vision-3B 설명을 Haiku에게 줘도 +0.023(우연)이었습니다. 그래서 지금 모델은 그대로 두고, **사람이 사진을 보고 확인한 사실**을 넣었을 때 답글이 달라지는지 봅니다. 이걸로 다음에 고칠 곳이 비전 출력인지, 답글 학습인지, 채점인지 정합니다.

## 비교 조건

답글 모델(1.5B + 사진 어댑터), 프롬프트(`prompts/stage4_prompt.txt`), 가게 문서는 모두 같습니다. 사진 리뷰 32건(사진 16장)입니다.

| 조건 | 답글 모델이 받는 사진 정보 | 결과 위치 |
|---|---|---|
| ① 없음 | 사진 정보 칸 자체가 없음 (`--no-photos`) | `outputs/photo_eval_nophoto` (새로) |
| ② 지금 | Vision-3B 설명 | `outputs/photo_eval_stage4` (이미 있음) |
| ③ 사람 확인 사실 | `configs/photo_facts.json`의 사실, ②와 같은 칸과 비슷한 길이·말투 | `outputs/photo_eval_facts` (새로) |
| (선택) Haiku + ③ | ③과 같은 사실을 Haiku가 읽음 | `outputs/photo_eval_haiku_facts` |

점수와 별도로, 리뷰마다 사람이 표시합니다: 사진이 리뷰 글에 없는 쓸 만한 정보를 줬는지 → Vision-3B 설명에 그 정보가 있었는지 → 각 답글이 그 정보를 맞게 썼는지, 그리고 근거 없는 주장이 있었는지.

## 읽는 법 (결과 보기 전에 정함)

| 관찰 | 먼저 고칠 곳 |
|---|---|
| ③이 ②보다 확실히 높고(clear), ③ 답글은 사진 정보를 잘 쓰는데 ② 설명에는 그 정보가 없음 | **비전 출력**: 긴 설명 대신 답글에 쓸 수 있는 짧은 관찰로. 불확실한 메뉴 이름은 빼기 |
| ③도 ②와 비슷하고, 정확한 정보가 입력에 있어도 답글이 계속 흘림 | **답글 학습 데이터**: 운영처럼 음식 이름 힌트 없이 만든 입력으로, 사진 정보를 잘 쓴 답과 놓치거나 지어낸 답을 짝짓기 |
| 사람이 보기엔 ③ 답글이 사진을 더 잘 쓰는데 점수는 비슷함 | **채점**: 사진 근거 활용률과 근거 없는 주장 비율을 따로 재기 |

- 사진 16장을 32건이 나눠 쓰고, 이 사진들로 이미 여러 번 비교했습니다. 이 결과로 **방향만** 정하고, 효과는 새 사진으로 다시 확인합니다.
- 지금 사진은 모두 멀쩡한 음식 사진이라, 이물질·흘림 같은 불만 사진에서의 효과는 이 실험으로 알 수 없습니다.
- ① 결과는 처음 만듭니다. 예전 `no-photos − with-photos`(−0.044)는 2단계 모델과 옛 설명으로 낸 값이라 지금 모델과 다릅니다.

## 0. [PC] 사진 사실 확인 (사람, 30분)

`configs/photo_facts.json`에 16장의 사실 초안이 있습니다(Claude가 사진을 보고 씀). `다운로드\review_photos`의 사진을 한 장씩 보면서 고칩니다.

- 사진에 보이는 것만 씁니다. 맛, 온도, 조리법은 쓰지 않습니다
- 음식 이름이 확실하지 않으면 "~로 보이는"으로 쓰거나 뺍니다
- 한두 문장, 길어도 세 문장. Vision-3B 설명과 길이가 비슷해야 내용만 비교됩니다
- 다 봤으면 그 사진의 `"checked"`를 `true`로 바꿉니다. 하나라도 `false`면 3단계에서 멈춥니다

## 1. [WSL] 코드 보내기 / [서버] 풀기

```bash
cd ~/naver/Scrable && git pull && tar -czf ~/scrable_stage4.tgz scrable configs prompts scripts tests pyproject.toml && scp ~/scrable_stage4.tgz root@<서버IP>:/opt/scrable/
```
```bash
cd /opt/scrable/scrable_stage1 && tar -xzf /opt/scrable/scrable_stage4.tgz && grep -c '"checked": true' configs/photo_facts.json
```

`16`이 나오면 됩니다.

## 2. [서버] Vision-3B 설명이 잘렸는지 확인 (0원)

사진 설명은 128토큰까지만 만듭니다. 끝이 문장 중간에서 끊겼으면 ②가 정보를 덜 받은 것입니다.

```bash
.venv/bin/python -c "import json; p=json.load(open('outputs/photo_eval_stage4/all_predictions.json',encoding='utf-8'))['predictions']; d={x['photo_description'] for x in p if x.get('photo_description')}; [print(len(t), repr(t[-40:])) for t in d]"
```

## 3. [서버] 사실 파일 만들기 (0원)

```bash
.venv/bin/python scripts/photo_facts.py build
```

`Wrote facts for 32 reviews to outputs/photo_facts/descriptions.json`이면 성공입니다.

## 4. [서버] 평가 ①, ③ (20분 안팎, 몇십 센트)

키가 없는 창이면 `read -rs ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY`를 따로 먼저.

```bash
nohup sh -c '
.venv/bin/python -m scrable.evaluate --config configs/photo_eval_stage4.json --split all --no-photos --label facts-none --output-dir outputs/photo_eval_nophoto --reference outputs/photo_eval_stage4/all_predictions.json &&
.venv/bin/python -m scrable.evaluate --config configs/photo_eval_facts.json --split all --label facts-checked --reference outputs/photo_eval_stage4/all_predictions.json outputs/photo_eval_nophoto/all_predictions.json
' > photo_facts.log 2>&1 &
```

(선택) Haiku + 사람 확인 사실, 10분 안쪽:

```bash
nohup .venv/bin/python -m scrable.evaluate --config configs/photo_eval_haiku_facts.json --split all --label haiku-facts --reference outputs/photo_eval_haiku/all_predictions.json outputs/photo_eval_facts/all_predictions.json > haiku_facts.log 2>&1 &
```

`grep -a -c "Saved results" photo_facts.log`가 `2`면 끝입니다.

```bash
grep -a -E "^facts-[a-z]+: overall|^facts-[a-z]+ - " -A6 photo_facts.log
```

- `facts-none - photo-stage4`: ① − ②, Vision-3B 설명이 지금 얼마나 도움이 되는지
- `facts-checked - photo-stage4`: ③ − ②, 설명이 정확해지면 얼마나 오르는지
- `facts-checked - facts-none`: ③ − ①

## 5. 리뷰별 확인 시트 (사람, 40분)

```bash
.venv/bin/python scripts/photo_facts.py sheet --vision outputs/photo_eval_stage4/all_predictions.json --facts-run outputs/photo_eval_facts/all_predictions.json --none outputs/photo_eval_nophoto/all_predictions.json > facts_sheet.csv
```
```bash
scp root@<서버IP>:/opt/scrable/scrable_stage1/facts_sheet.csv /mnt/c/Users/wounc/Downloads/
```

엑셀에서 열고, 뒤쪽 빈 칸에 `예`/`아니오`를 적습니다.

| 칸 | 적는 법 |
|---|---|
| 사진이 리뷰에 없는 쓸 만한 정보를 줌 | 리뷰 글에는 없고 사진에만 있는, 답글에 쓸 만한 정보(메뉴 구성, 양, 담음새, 눈에 보이는 문제)가 있으면 `예` |
| ② 설명에 그 정보가 있음 | 위가 `예`일 때만. Vision-3B 설명에 그 정보가 맞게 들어 있으면 `예` |
| ② / ③ 답글이 그 정보를 맞게 씀 | 위가 `예`일 때만. 그 답글이 사진 정보를 맞게 썼으면 `예` |
| ② / ③ 답글에 근거 없는 주장 | 모든 리뷰. 사진, 리뷰, 가게 문서 어디에도 없는 메뉴나 사실을 단정했으면 `예` |

저장할 때 형식은 CSV 그대로 둡니다. WSL에서 셉니다:

```bash
cd ~/naver/Scrable && python3 scripts/photo_facts.py tally /mnt/c/Users/wounc/Downloads/facts_sheet.csv
```

점수 결과와 이 숫자를 같이 보고 위 "읽는 법" 표로 다음 작업을 정합니다.
