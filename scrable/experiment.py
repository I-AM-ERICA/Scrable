"""Notebook and CLI share the same complete experiment entry point."""

import argparse
import json
import platform
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .config import Config
from .data import download_reviews, knowledge_documents, load_custom_reviews, load_documents, split_reviews
from .io import digest, write_json
from .llm import build_llms
from .optimizer import PromptOptimizer
from .pipeline import PromptGenerator, ScoredResponseGenerator, average_score
from .prompts import BASE_PROMPT
from .rag import build_retriever
from .types import CRITERIA


def prepare_data(config):
    if config.custom_data:
        rows = load_custom_reviews(config.custom_data)
        provenance = {
            "path": config.custom_data,
            "sha256": digest([asdict(r) for r in rows]),
            "source": "user-supplied JSONL",
        }
    else:
        rows, provenance = download_reviews(config)
    splits = split_reviews(rows, config)
    documents = knowledge_documents(splits["knowledge"])
    if config.documents_dir:
        documents += load_documents(config.documents_dir, config.app_id)
    if not documents:
        raise ValueError(
            "No knowledge documents: supply documents_dir or reference replies in the knowledge split"
        )
    return splits, documents, provenance


def summarize(feedback):
    return {
        "overall": average_score(feedback),
        "count": len(feedback),
        **{
            criterion: sum((f.scores[criterion] - 1) / 4 for f in feedback) / len(feedback)
            for criterion in CRITERIA
        },
    }


def package_versions():
    result = {"python": platform.python_version()}
    for package in (
        "scrable",
        "torch",
        "transformers",
        "sentence-transformers",
        "openai",
        "numpy",
        "langchain-text-splitters",
        "azure-search-documents",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            pass
    return result


def run_experiment(config, prepared=None, llms=None, retriever=None):
    """Run download -> RAG -> optimize -> held-out test -> export.

    Optional object injection makes orchestration testable without model/API calls.
    No test double is used by the CLI or notebook's normal execution path.
    """
    directory = Path(config.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    splits, documents, provenance = prepared if prepared is not None else prepare_data(config)
    manifest = {
        "config": config.to_dict(),
        "provenance": provenance,
        "splits": {name: [asdict(r) for r in rows] for name, rows in splits.items()},
        "documents": [asdict(d) for d in documents],
    }
    signature = digest(manifest)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["signature"] != signature:
            raise ValueError(
                "output_dir contains a different experiment. Choose a new output_dir to avoid mixing runs."
            )
    write_json(manifest_path, {**manifest, "signature": signature, "versions": package_versions()})
    if retriever is None:
        retriever = build_retriever(config, documents)
    if llms is None:
        llms = build_llms(config)
    response_llm, judge_llm, prompt_llm = llms
    scorer = ScoredResponseGenerator(response_llm, judge_llm, retriever, config)
    optimizer = PromptOptimizer(scorer, PromptGenerator(prompt_llm, config), config)
    result = optimizer.run(splits["train"], splits["validation"])
    # Test is never supplied to PromptGen, early stopping or prompt selection.
    baseline = scorer(BASE_PROMPT, splits["test"], description="Held-out test: baseline")
    optimized = scorer(result.best_prompt, splits["test"], description="Held-out test: selected prompt")
    metrics = {"baseline": summarize(baseline), "optimized": summarize(optimized)}
    metrics["delta"] = {k: metrics["optimized"][k] - metrics["baseline"][k] for k in ("overall", *CRITERIA)}
    metrics.update(
        best_iteration=result.best_iteration,
        stop_reason=result.stop_reason,
        selection="validation overall score; earliest prompt on ties",
        evaluation="LLM-as-a-Judge, not human evaluation",
        rag_fingerprint=retriever.fingerprint,
        llm_usage=[
            {
                "calls": getattr(llm, "calls", None),
                "cache_hits": getattr(llm, "hits", None),
                "backend": getattr(getattr(llm, "backend", llm), "identity", None),
            }
            for llm in {id(llm): llm for llm in llms}.values()
        ],
    )
    write_json(directory / "metrics.json", metrics)
    write_json(
        directory / "test_predictions.json",
        {
            "baseline": [f.to_dict() for f in baseline],
            "optimized": [f.to_dict() for f in optimized],
        },
    )
    write_json(directory / "result.json", asdict(result))
    print(f"Saved results to {directory.resolve()}", flush=True)
    return {"optimization": result, "metrics": metrics, "scorer": scorer, "splits": splits}


def main():
    parser = argparse.ArgumentParser(description="SCRABLE prompt optimization (no gradient training)")
    parser.add_argument("--config", help="JSON Config file; defaults to HyperCLOVA 1.5B demo")
    parser.add_argument("--output-dir")
    parser.add_argument("--prepare-only", action="store_true", help="Download and validate data only")
    args = parser.parse_args()
    config = Config.load(args.config) if args.config else Config()
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.prepare_only:
        splits, docs, provenance = prepare_data(config)
        print(
            json.dumps(
                {
                    "splits": {k: len(v) for k, v in splits.items()},
                    "documents": len(docs),
                    "provenance": provenance,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    result = run_experiment(config)
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
