import json

import numpy as np

from scrable.types import Review


class TestEmbeddings:
    """Deterministic test fixture, not a semantic model."""

    __test__ = False
    identity = "test-only-bag-of-keywords-v1"

    def encode(self, texts):
        return np.array(
            [[1 + text.lower().count(word) for word in ("food", "refund", "delivery")] for text in texts],
            dtype=np.float32,
        )


class ScriptedLLM:
    """Test-only fixture. Never used by production defaults."""

    identity = {"backend": "test-only"}

    def __init__(self):
        self.messages = []

    def complete(self, messages, max_tokens):
        self.messages.append(messages)
        payload = json.loads(messages[-1]["content"])
        if "criterion" in payload:
            score = 4 if "improved" in payload["prediction"] else 2
            return json.dumps(
                {
                    "score": score,
                    "reason": "Fixture rating, not a real model judgment.",
                    "suggestions": ["Address the specific delivery problem."],
                }
            )
        if "current_prompt" in payload:
            return json.dumps(
                {"prompt": "improved: Respond to the specific concern using only documented facts."}
            )
        return "improved reply" if "improved:" in messages[0]["content"] else "baseline reply"


def reviews(count=30):
    return [
        Review(
            str(i),
            f"Food delivery review number {i} was delayed",
            "app",
            f"REFERENCE-ONLY-{i}: Please provide delivery details.",
        )
        for i in range(count)
    ]
