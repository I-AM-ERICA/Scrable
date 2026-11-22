# 7B 후속 실험: 사진과 리뷰를 직접 보는 모델로 답글 쓰기

사진 설명을 바꿔도(Vision-3B 힌트, Qwen 설명) 답글 점수가 오르지 않았습니다. 병목이 1.5B 답글 모델이라는 뜻이라, 답글 모델 자리에 사진을 직접 볼 수 있는 **Qwen2.5-VL-7B-Instruct**(Apache-2.0)를 넣어 비교합니다. 학습은 하지 않고, 프롬프트(`prompts/stage4_prompt.txt`)와 가게 문서 검색은 지금과 같습니다.

## 비교 조건

| 평가셋 | A · 지금 서버 모델 | B · 7B + 사진 설명 | C · 7B + 원본 사진 |
|---|---|---|---|
| 사진 리뷰 32건 | 1.5B + 사진 어댑터, Vision-3B 설명 (이미 있음) | A와 **같은 설명**을 7B가 읽음 | 7B가 **사진을 직접** 봄 |
| 글 리뷰 40건 | 이미 있음 | — | 7B, 같은 글 입력 |
| 실제 리뷰 40건 | 이미 있음 | — | 7B, 같은 글 입력 |

- C − A: 지금 구조를 7B로 바꾼 효과
- B − A: 같은 설명일 때 답글 모델만 바꾼 효과
- C − B: 같은 7B에서 설명 대신 원본 사진을 준 효과

A 답글은 다시 만들지 않습니다: `outputs/photo_eval_stage4`, `outputs/stage4_photo_eval`, `outputs/public_test/stage4_photo`.

## 판정 기준 (결과 보기 전에 정함, 0~1점)

| 판정 | 기준 |
|---|---|
| 사진 답글을 7B로 | C − A 평균 +0.03 이상, 신뢰구간 하한 0 초과, C에서 새 중대 오류 0건 |
| 원본 사진의 기여 | C − B가 + 쪽인지 (불확실하면 보류) |
| 글 답글까지 7B로 | 글 40건, 실제 40건 각각 C − A 평균이 −0.02 이상이고 clear하게 나빠지지 않음 |
| 운영 가능 | GPU 최대 메모리와 답글 시간(p90)이 정책 비서와 같이 돌려도 감당되는지 |

- 신뢰구간은 리뷰 단위 짝 비교입니다(가게가 4곳뿐이라 가게 단위로 묶지 않음).
- 사진 32건은 사진 설명 비교에 이미 여러 번 썼습니다. C가 통과하면 **새 사진 리뷰로 한 번 더** 확인한 뒤 바꿉니다.

## 0. [WSL] 코드 보내기 / [서버] 풀기

```bash
cd ~/naver/Scrable && git pull && tar -czf ~/scrable_stage4.tgz scrable configs prompts scripts tests pyproject.toml RUN_7B.md && scp ~/scrable_stage4.tgz root@<서버IP>:/opt/scrable/
```
```bash
cd /opt/scrable/scrable_stage1 && tar -xzf /opt/scrable/scrable_stage4.tgz && ls configs/exp7b_*.json
```

GPU를 쓰는 다른 프로세스(gRPC 서버 등)는 끄고 시작합니다: `pkill -f scrable.serve; nvidia-smi`

## 1. [서버] 파일럿 3건 (사진 전달, 답글 잘림 확인)

키가 없는 창이면 `read -rs ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY`를 따로 먼저.

```bash
.venv/bin/python -m scrable.evaluate --config configs/exp7b_photo_image.json --split all --label pilot --limit 3 --output-dir outputs/exp7b/pilot 2>&1 | tail -n 12
```

답글이 한국어로 끝까지 나오는지, `seconds per reply`와 `GPU peak memory`를 봅니다. 답글 3개 보기:

```bash
.venv/bin/python -c "import json; [print('-', p['prediction'][:150]) for p in json.load(open('outputs/exp7b/pilot/all_predictions.json', encoding='utf-8'))['predictions']]"
```

## 2. [서버] 본 실행 (40분 안팎, 1달러 안쪽)

```bash
nohup sh -c '
.venv/bin/python -m scrable.evaluate --config configs/exp7b_photo_desc.json --split all --label 7b-desc --reference outputs/photo_eval_stage4/all_predictions.json &&
.venv/bin/python -m scrable.evaluate --config configs/exp7b_photo_image.json --split all --label 7b-image --reference outputs/photo_eval_stage4/all_predictions.json outputs/exp7b/photo_desc/all_predictions.json &&
.venv/bin/python -m scrable.evaluate --config configs/exp7b_text.json --label 7b-text --reference outputs/stage4_photo_eval/test_predictions.json &&
.venv/bin/python -m scrable.evaluate --config configs/exp7b_public.json --split all --label 7b-public --reference outputs/public_test/stage4_photo/all_predictions.json
' > exp7b.log 2>&1 &
```

`grep -a -c "Saved results" exp7b.log`가 `4`면 끝입니다. 결과:

```bash
grep -a -E "^7b-[a-z]+: overall|^7b-[a-z]+ - |seconds per reply|GPU peak" -A2 exp7b.log
```

## 3. 중대 오류 확인 (C만, 사람이)

```bash
.venv/bin/python scripts/review_sheet.py outputs/exp7b/photo_image/all_predictions.json > review_c.csv
```

노트북으로 가져와 엑셀에서 열고, 답글마다 `중대 오류`에 예/아니오를 적습니다. 중대 오류는 사장님이 그대로 올리면 문제가 되는 것입니다: 사진이나 리뷰로 확인할 수 없는 메뉴·사실을 단정, 가게 문서 근거 없는 환불·보상 약속, 손님 탓. 답글을 쓰지 못한 경우도 오류로 셉니다.

```bash
scp root@<서버IP>:/opt/scrable/scrable_stage1/review_c.csv /mnt/c/Users/wounc/Downloads/
```

## 4. 운영 확인 (C가 통과했을 때)

- 결과 로그의 `seconds per reply`(평균, p90)와 `GPU peak memory`를 기록합니다.
- 정책 비서와 같은 GPU 서버에 올라가므로, 정책 담당자와 같이 돌려 메모리 부족이나 응답 지연이 없는지 확인합니다.
- 지금 gRPC 서버(`scrable.serve`)는 1.5B 전용이고 사진도 받지 않습니다. 7B로 바꾸기로 하면 서버 코드와 백엔드 규약(사진 필드)을 따로 고쳐야 합니다.
