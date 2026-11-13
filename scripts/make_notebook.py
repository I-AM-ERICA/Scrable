"""Regenerate train.ipynb (no execution outputs or secrets)."""

import json
import textwrap
from pathlib import Path


def cell(kind, text, tag=None):
    value = {
        "cell_type": kind,
        "metadata": {"tags": [tag]} if tag else {},
        "source": textwrap.dedent(text).strip() + "\n",
    }
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


cells = [
    cell(
        "markdown",
        """
    # SCRABLE — HyperCLOVA 1.5B 고객 리뷰 답변 프롬프트 최적화

    **전체 셀 실행**: 설치 → 모델 접근 확인 → AARSynth 자동 다운로드 → RAG 구축 →
    프롬프트 반복 최적화 → 테스트 비교 → 결과 저장 → 새 리뷰 응답.

    이 노트북의 train은 **모델 가중치 학습이 아닌 프롬프트 최적화**입니다.
    기본 모델은 `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B`이며 세 역할이 모델 하나를 공유합니다.
    Python 3.10+ (권장 3.11), NVIDIA GPU 서버에서 프로젝트 전체를 복사한 후 실행하세요.
    CPU에서도 실행되지만 LLM 반복 추론은 오래 걸립니다. 디스크에는 모델 약 3.2 GB와 의존성 공간이 필요합니다.

    **최초 1회 필수**: [모델 페이지](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B)에서
    접근 조건에 동의하고, 서버에서 `hf auth login`을 실행하거나 `HF_TOKEN` 환경변수를 설정하세요.
    로그인하지 않은 경우 아래 셀에서 토큰을 숨김 입력으로 요청합니다. 접근 동의는 자동화할 수 없습니다.

    [참고 논문](https://aclanthology.org/2024.ecnlp-1.5/)의 공개되지 않은 원본 49건 대신
    [AARSynth](https://huggingface.co/datasets/recmeapp/AARSynth)의 실제 앱 리뷰/개발자 답변을 사용합니다.
    기본 데이터는 영어입니다. 개발자 답변은 검증된 전문가 정답이 아닌 참고 답변입니다.
    """,
    ),
    cell(
        "code",
        """
    import os
    import sys
    import subprocess
    from pathlib import Path

    assert sys.version_info >= (3, 10), "Python 3.10 이상 커널을 선택하세요."
    ROOT = Path.cwd()
    if not (ROOT / "pyproject.toml").exists():
        raise RuntimeError("프로젝트 루트(Scrable 폴더)에서 train.ipynb를 실행하세요.")
    # 현재 Jupyter 커널 환경에 설치합니다. GPU용 PyTorch가 이미 있으면 호환되는 설치를 유지합니다.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", ".[local,api,notebook]"])
    print("설치 완료:", sys.executable)
    """,
        "install",
    ),
    cell(
        "markdown",
        """
    ## 1. 실험 설정

    기본값은 흐름 확인용 소규모 실험입니다. 본 실험은 `train_size`, `validation_size`, `test_size`,
    `max_iterations`를 늘리고 `output_dir`도 새 경로로 바꾸세요.
    기본 설정의 최대 호출 수는 JSON 재시도·예시 추론을 제외하고
    `5 × ((10+4) × (2+1) + 2×4) + 2 = 252`회입니다.
    답변당 생성 1회 + 기준별 심사 4회가 필요하며 같은 요청은 캐시를 재사용합니다.
    """,
    ),
    cell(
        "code",
        """
    from scrable import Config

    config = Config(
        backend="local",
        model="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
        device="auto",                 # cuda 자동 선택. CPU 검증은 "cpu"
        train_size=10,
        validation_size=4,
        test_size=4,
        knowledge_size=64,
        max_iterations=2,
        threshold=0.95,
        lowest_fraction=0.30,
        random_fraction=0.10,
        seed=42,
        app_id="com.ubercab.eats",
        output_dir="outputs/hyperclova",
        # custom_data="my_reviews.jsonl",   # id, review, app_id, expert_response(선택)
        # documents_dir="my_product_docs",  # TXT/MD/PDF 제품 문서 추가
    )
    # API 선택 시 backend="openai", model="실제 사용 가능한 모델 ID"로 변경하세요.
    # 로컬 OpenAI 호환 서버는 OPENAI_BASE_URL, OPENAI_API_KEY 환경변수를 설정하세요.
    print(config.to_dict())
    """,
        "config",
    ),
    cell(
        "code",
        """
    import getpass
    import torch
    from huggingface_hub import get_token, hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
    elif config.backend == "local":
        print("CPU/MPS로 실행합니다. GPU보다 오래 걸릴 수 있습니다.")
    if config.backend == "local":
        for model_name in dict.fromkeys([config.model, config.judge_model or config.model,
                                        config.prompt_model or config.model]):
            if Path(model_name).is_dir():
                continue
            try:
                hf_hub_download(model_name, "config.json")
            except GatedRepoError:
                if not get_token():
                    os.environ["HF_TOKEN"] = getpass.getpass("접근 승인된 Hugging Face Read 토큰: ")
                    hf_hub_download(model_name, "config.json")
                else:
                    raise RuntimeError(f"{model_name}의 모델 페이지에서 접근 동의를 완료한 계정 토큰이 필요합니다.")
        print("모델 다운로드 접근 확인 완료")
    elif not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI 또는 호환 서버 API key: ")
    """,
        "preflight",
    ),
    cell(
        "markdown",
        """
    ## 2. 데이터 자동 다운로드·분할

    고정 revision의 AARSynth 파일을 스트리밍하여 최대 3,000개 적격 리뷰를 캐시에 저장합니다.
    파일 앞부분의 해당 앱 표본이므로 전체 데이터셋의 균일 표본은 아닙니다.
    중복 리뷰를 제거한 뒤 RAG 자료 / train / validation / test를 seed로 나눕니다.
    RAG 기본 자료는 별도 분할한 과거 개발자 답변이며 제품 설명서와는 다릅니다.
    실제 제품 문서가 있다면 `documents_dir`에 넣으세요. 테스트 정답은 문서에 넣지 마세요.
    """,
    ),
    cell(
        "code",
        """
    import pandas as pd
    from IPython.display import display
    from scrable.experiment import prepare_data, run_experiment

    prepared = prepare_data(config)
    splits, documents, provenance = prepared
    display(pd.DataFrame([{"split": name, "reviews": len(rows)} for name, rows in splits.items()]))
    print("RAG 문서:", len(documents))
    print("데이터 출처:", provenance)
    display(pd.DataFrame([vars(r) for r in splits["train"][:3]]))
    """,
        "data",
    ),
    cell(
        "markdown",
        """
    ## 3. RAG 구축 → 반복 최적화 → 최종 평가

    LangChain으로 약 500 토큰씩 분할하고 로컬 임베딩 + BM25 검색 순위를 결합합니다.
    낮은 점수 30% + 나머지에서 무작위 10%를 선택합니다(소수점 올림, 중복 제외).
    정확도 심사에만 참고 정답을 제공하고, 답변 생성에는 해당 리뷰의 정답을 제공하지 않습니다.
    점수는 `(raw−1)/4`, 가중치는 관련성 1 / 앱 특화성 1 / 정확도 2 / 문법 1입니다.
    학습 평균이 threshold 이상이거나 최대 수정 횟수에 도달하면 종료합니다.

    validation 최고 프롬프트를 저장한 뒤, 사용하지 않았던 test에서 기본/최적화 프롬프트를 비교합니다.
    개선은 보장되지 않습니다. 동일 모델이 생성과 평가를 맡으므로 실제 품질은 사람이 따로 검토하세요.
    중단되면 같은 설정으로 다시 실행하면 저장된 LLM 응답을 재사용합니다.
    """,
    ),
    cell(
        "code",
        """
    result = run_experiment(config, prepared=prepared)
    metrics = result["metrics"]
    display(pd.DataFrame({name: metrics[name] for name in ["baseline", "optimized", "delta"]}))
    print("선택된 반복:", metrics["best_iteration"], "종료 사유:", metrics["stop_reason"])
    """,
        "run",
    ),
    cell(
        "code",
        """
    import matplotlib.pyplot as plt

    history = pd.DataFrame(result["optimization"].history)
    ax = history.plot(x="iteration", y=["train_score", "validation_score"], marker="o", figsize=(8, 4))
    ax.axhline(config.threshold, color="gray", linestyle="--", label="threshold")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Normalized weighted score")
    ax.set_xticks(history["iteration"])
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(config.output_dir) / "score_history.png", dpi=160)
    plt.show()
    print(result["optimization"].best_prompt)
    """,
        "plot",
    ),
    cell(
        "markdown",
        """
    ## 4. 새 리뷰에 적용

    최적화한 프롬프트와 같은 RAG 인덱스를 사용합니다. 새 리뷰는 결과 비교에 포함하지 않습니다.
    """,
    ),
    cell(
        "code",
        """
    from scrable import Review

    new_review = Review(id="new-example", app_id=config.app_id,
                        review="My food never arrived, but the app says it was delivered.")
    answer, sources = result["scorer"].generate(result["optimization"].best_prompt, new_review)
    print(answer)
    print("검색 출처:", [s.source for s in sources])
    """,
        "inference",
    ),
    cell(
        "code",
        """
    import shutil

    output_dir = Path(config.output_dir)
    print("저장 위치:", output_dir.resolve())
    for path in sorted(output_dir.glob("*")):
        if path.is_file():
            print("-", path.name)
    # 캐시(큰 파일)는 제외하고 결과만 묶습니다.
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        export = Path(temp) / "results"
        export.mkdir()
        for path in output_dir.iterdir():
            if path.is_file() and path.suffix in {".json", ".txt", ".png"}:
                shutil.copy2(path, export / path.name)
        archive = shutil.make_archive(str(output_dir / "results"), "zip", export)
    print("결과 ZIP:", archive)
    """,
        "export",
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, item in enumerate(cells):
    item["id"] = f"scrable-{i:02d}"
Path(__file__).resolve().parents[1].joinpath("train.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
