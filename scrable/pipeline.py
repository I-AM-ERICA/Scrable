import json
import math
import random

from .llm import structured_call
from .prompts import GROUNDING_RULES, JUDGE_SYSTEM, PROMPT_GEN_SYSTEM, RUBRICS
from .rag import context_payload
from .types import CRITERIA, Feedback


class ScoredResponseGenerator:
    def __init__(self, response_llm, judge_llm, retriever, config):
        self.response_llm = response_llm
        self.judge_llm = judge_llm
        self.retriever = retriever
        self.config = config

    def generate(self, prompt, review):
        chunks = self.retriever.retrieve(review.review, review.app_id, self.config.top_k)
        # Expert reference is deliberately absent from response-generation inputs.
        payload = {
            "app_id": review.app_id,
            "review": review.review,
            "retrieved_documents": context_payload(chunks),
        }
        prediction = self.response_llm.complete(
            [
                {"role": "system", "content": prompt + "\n\n" + GROUNDING_RULES},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            max_tokens=self.config.response_tokens,
        )
        if not prediction.strip():
            raise ValueError("Empty generated response")
        return prediction, chunks

    def judge(self, review, prediction, chunks):
        scores, rationales, suggestions = {}, {}, []
        for criterion in CRITERIA:
            payload = {
                "criterion": criterion,
                "rubric": RUBRICS[criterion],
                "app_id": review.app_id,
                "review": review.review,
                "prediction": prediction,
            }
            # Match paper's evaluation setup: only accuracy sees the expert answer.
            if criterion == "accuracy":
                payload.update(
                    expert_response=review.expert_response, retrieved_documents=context_payload(chunks)
                )
            result = structured_call(
                self.judge_llm,
                [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                validate_judgment,
                self.config.judge_tokens,
                self.config.json_attempts,
            )
            scores[criterion] = result["score"]
            rationales[criterion] = result["reason"]
            suggestions.extend(f"[{criterion}] {item}" for item in result["suggestions"])
        return Feedback(
            review.id,
            review.review,
            review.app_id,
            prediction,
            scores,
            suggestions,
            rationales,
            [c.id for c in chunks],
        )

    def __call__(self, prompt, reviews, description="ScoredResponseGen"):
        from tqdm.auto import tqdm

        if not reviews:
            raise ValueError("ScoredResponseGen requires nonempty reviews")
        feedback = []
        for review in tqdm(reviews, desc=description):
            prediction, chunks = self.generate(prompt, review)
            feedback.append(self.judge(review, prediction, chunks))
        return feedback


def validate_judgment(value):
    if set(value) != {"score", "reason", "suggestions"}:
        raise ValueError("Required keys: score, reason, suggestions")
    if type(value["score"]) is not int or not 1 <= value["score"] <= 5:
        raise ValueError("score must be an integer in [1, 5]")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("reason must be a nonempty string")
    if not isinstance(value["suggestions"], list) or not all(
        isinstance(x, str) and x.strip() for x in value["suggestions"]
    ):
        raise ValueError("suggestions must be an array of nonempty strings")
    return value


def average_score(feedback):
    if not feedback:
        raise ValueError("Cannot average empty feedback")
    return sum(f.score for f in feedback) / len(feedback)


def identify_ir(feedback, lowest_fraction=0.30, random_fraction=0.10, seed=42):
    """Ceil fractions for small datasets; random cases exclude the lowest set."""
    if not feedback:
        raise ValueError("IdentifyIR requires nonempty feedback")
    if not 0 < lowest_fraction <= 1 or not 0 <= random_fraction <= 1:
        raise ValueError("Invalid selection fractions")
    if lowest_fraction + random_fraction > 1:
        raise ValueError("Selection fractions exceed 1")
    if len({f.review_id for f in feedback}) != len(feedback):
        raise ValueError("Duplicate review IDs in feedback")
    ordered = sorted(feedback, key=lambda f: (f.score, f.review_id))
    count = math.ceil(len(ordered) * lowest_fraction)
    lowest, remaining = ordered[:count], ordered[count:]
    random_count = min(len(remaining), math.ceil(len(ordered) * random_fraction))
    extra = random.Random(seed).sample(remaining, random_count)
    return lowest + extra


class PromptGenerator:
    def __init__(self, llm, config):
        self.llm, self.config = llm, config

    def __call__(self, current_prompt, selected):
        # Retain every selected case while bounding each example's length.
        payload = {
            "current_prompt": current_prompt,
            "feedback": [
                {
                    "review": f.review[:600],
                    "prediction": f.prediction[:800],
                    "score": f.score,
                    "suggestions": [s[:300] for s in f.suggestions],
                    "scores": f.scores,
                }
                for f in selected
            ],
        }
        result = structured_call(
            self.llm,
            [
                {"role": "system", "content": PROMPT_GEN_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            validate_prompt,
            self.config.prompt_tokens,
            self.config.json_attempts,
        )
        return result["prompt"].strip()


def validate_prompt(value):
    if set(value) != {"prompt"} or not isinstance(value["prompt"], str):
        raise ValueError("Expected exactly one string field: prompt")
    if not 20 <= len(value["prompt"].strip()) <= 6000:
        raise ValueError("Prompt must contain 20 to 6000 characters")
    return value
