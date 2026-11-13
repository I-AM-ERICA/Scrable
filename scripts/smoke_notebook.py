"""Execute all workflow cells with explicit test doubles; no model quality claim.

Installation and authentication are skipped ONLY in this temporary test notebook.
The shipped train.ipynb is never modified and uses real models by default.
"""

import json
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
notebook = nbformat.read(ROOT / "train.ipynb", as_version=4)
nbformat.validate(notebook)
with tempfile.TemporaryDirectory(prefix="scrable-notebook-") as temporary:
    output = str(Path(temporary) / "results")
    bootstrap = """
import os, sys
from pathlib import Path
from scrable import Config
import scrable.experiment as experiment
from scrable.data import knowledge_documents, split_reviews
from scrable.rag import HybridRetriever
from tests.helpers import reviews, ScriptedLLM, TestEmbeddings
from IPython.display import display
import pandas as pd
import matplotlib
matplotlib.use("Agg")

def fixture_data(config):
    splits = split_reviews(reviews(), config)
    return splits, knowledge_documents(splits["knowledge"]), {"source": "TEST FIXTURE, NOT REAL DATA"}

def fixture_llms(config):
    llm = ScriptedLLM()
    return llm, llm, llm

experiment.prepare_data = fixture_data
experiment.build_llms = fixture_llms
experiment.build_retriever = lambda config, docs: HybridRetriever(
    docs, TestEmbeddings(), Path(config.output_dir) / "rag")
print("TEST-ONLY notebook workflow: synthetic fixtures and scripted LLM")
"""
    for cell in notebook.cells:
        tags = cell.metadata.get("tags", [])
        if "install" in tags:
            cell.source = bootstrap
        elif "preflight" in tags:
            cell.source = 'print("TEST: model authentication intentionally skipped")'
        elif "config" in tags:
            cell.source = (
                "config = Config(train_size=4, validation_size=2, test_size=2, knowledge_size=8, "
                f"app_id='app', max_iterations=1, threshold=1, output_dir={json.dumps(output)})"
            )
    client = NotebookClient(
        notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    )
    # Avoid a user's unrelated default Python kernel.
    client.create_kernel_manager()
    client.km.kernel_spec.argv = [
        __import__("sys").executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client.execute()
    assert (Path(output) / "metrics.json").is_file()
    assert (Path(output) / "score_history.png").is_file()
    assert (Path(output) / "results.zip").is_file()
    print("Notebook workflow passed: optimization, test comparison, plot, inference and ZIP export.")
