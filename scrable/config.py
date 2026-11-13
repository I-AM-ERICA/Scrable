import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Config:
    backend: str = "local"
    model: str = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
    judge_model: str | None = None
    prompt_model: str | None = None
    device: str = "auto"
    seed: int = 42
    train_size: int = 10
    validation_size: int = 4
    test_size: int = 4
    knowledge_size: int = 64
    max_iterations: int = 2  # Number of revisions, excluding baseline (iteration 0).
    threshold: float = 0.95
    lowest_fraction: float = 0.30
    random_fraction: float = 0.10
    dataset_revision: str = "49068c54a4551304208aea80749f1a4d5ab893d2"
    app_id: str = "com.ubercab.eats"
    download_limit: int = 3000  # Eligible, unique reviews from a bounded source prefix.
    max_scan_lines: int = 100000
    data_dir: str = "data"
    output_dir: str = "outputs/demo"
    custom_data: str | None = None
    documents_dir: str | None = None
    rag_backend: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    azure_embedding_model: str = "text-embedding-ada-002"
    azure_index_prefix: str = "scrable"
    chunk_tokens: int = 500
    chunk_overlap: int = 50
    top_k: int = 4
    context_tokens: int = 8192
    response_tokens: int = 256
    judge_tokens: int = 256
    prompt_tokens: int = 768
    json_attempts: int = 3

    def __post_init__(self):
        if self.backend not in {"local", "openai"}:
            raise ValueError("backend must be local or openai")
        if self.rag_backend not in {"local", "azure"}:
            raise ValueError("rag_backend must be local or azure")
        for name in (
            "train_size",
            "validation_size",
            "test_size",
            "knowledge_size",
            "download_limit",
            "max_scan_lines",
            "top_k",
            "chunk_tokens",
            "context_tokens",
            "response_tokens",
            "judge_tokens",
            "prompt_tokens",
            "json_attempts",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.max_iterations) is not int or self.max_iterations < 0:
            raise ValueError("max_iterations must be >= 0")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold is normalized to [0, 1]")
        if not 0 < self.lowest_fraction <= 1 or not 0 <= self.random_fraction <= 1:
            raise ValueError("invalid selection fractions")
        if self.lowest_fraction + self.random_fraction > 1:
            raise ValueError("selection fractions must sum to <= 1")
        if not 0 <= self.chunk_overlap < self.chunk_tokens:
            raise ValueError("chunk_overlap must be smaller than chunk_tokens")
        if max(self.response_tokens, self.judge_tokens, self.prompt_tokens) >= self.context_tokens:
            raise ValueError("output budget must be smaller than context_tokens")
        if not self.model.strip() or not self.app_id.strip():
            raise ValueError("model and app_id cannot be empty")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self):
        return asdict(self)
