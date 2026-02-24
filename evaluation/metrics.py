from __future__ import annotations

import random
import re
import string
from typing import Dict, List, Sequence, Tuple

import numpy as np


def normalize_answer(text: str) -> str:
    def lower(s: str) -> str:
        return s.lower()

    def remove_punc(s: str) -> str:
        table = str.maketrans("", "", string.punctuation)
        return s.translate(table)

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    return white_space_fix(remove_articles(remove_punc(lower(str(text)))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(tok), gold_tokens.count(tok)) for tok in common)
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em_f1(prediction: str, gold_answer) -> Tuple[float, float]:
    if isinstance(gold_answer, list):
        candidates = [(exact_match_score(prediction, g), f1_score(prediction, g)) for g in gold_answer]
        return max(candidates, key=lambda x: x[1]) if candidates else (0.0, 0.0)
    gold = str(gold_answer)
    return exact_match_score(prediction, gold), f1_score(prediction, gold)


def summarize_records(records: List[Dict], abstain_text: str = "모르겠습니다.") -> Dict[str, float]:
    if not records:
        return {
            "sample_count": 0.0,
            "em": 0.0,
            "f1": 0.0,
            "hallucination_rate": 0.0,
            "coverage": 0.0,
            "selective_accuracy": 0.0,
            "latency_ms": 0.0,
            "checker_eval_count": 0.0,
            "checker_parse_fail_count": 0.0,
            "checker_parse_success_rate": 0.0,
        }

    em_values = [float(r.get("em", 0.0)) for r in records]
    f1_values = [float(r.get("f1", 0.0)) for r in records]
    latency_values = [float(r.get("latency_ms", 0.0)) for r in records]

    is_correct = [int(r.get("is_correct", 0)) for r in records]
    is_abstain = [int(r.get("is_abstain", 0)) for r in records]

    total = len(records)
    answered = total - sum(is_abstain)
    wrong_non_abstain = sum(1 for c, a in zip(is_correct, is_abstain) if c == 0 and a == 0)
    correct_answered = sum(1 for c, a in zip(is_correct, is_abstain) if c == 1 and a == 0)

    checker_records = [
        r
        for r in records
        if str(r.get("checker_name", "none")).strip().lower() not in {"", "none", "없음"}
    ]
    checker_eval_count = len(checker_records)
    checker_parse_fail_count = sum(
        1
        for r in checker_records
        if isinstance(r.get("checker_meta"), dict) and str(r.get("checker_meta", {}).get("파싱오류", "")).strip() != ""
    )
    checker_parse_success_rate = (
        float((checker_eval_count - checker_parse_fail_count) / checker_eval_count) if checker_eval_count > 0 else 0.0
    )

    return {
        "sample_count": float(total),
        "em": float(np.mean(em_values)),
        "f1": float(np.mean(f1_values)),
        "hallucination_rate": float(wrong_non_abstain / total),
        "coverage": float(answered / total),
        "selective_accuracy": float(correct_answered / answered) if answered > 0 else 0.0,
        "latency_ms": float(np.mean(latency_values)),
        "checker_eval_count": float(checker_eval_count),
        "checker_parse_fail_count": float(checker_parse_fail_count),
        "checker_parse_success_rate": checker_parse_success_rate,
    }


def paired_bootstrap_test(
    baseline_values: Sequence[float],
    candidate_values: Sequence[float],
    n_samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    if len(baseline_values) != len(candidate_values):
        raise ValueError("부트스트랩 비교는 동일 길이의 두 시퀀스가 필요합니다.")

    n = len(baseline_values)
    if n == 0:
        return {"observed_diff": 0.0, "p_value": 1.0, "ci_low": 0.0, "ci_high": 0.0}

    base = np.array(baseline_values, dtype=float)
    cand = np.array(candidate_values, dtype=float)
    observed = float(np.mean(cand - base))

    rng = random.Random(seed)
    diffs: List[float] = []
    extreme = 0

    for _ in range(int(n_samples)):
        idx = [rng.randrange(n) for _ in range(n)]
        diff = float(np.mean((cand - base)[idx]))
        diffs.append(diff)
        if abs(diff) >= abs(observed):
            extreme += 1

    alpha = 1.0 - float(confidence_level)
    low_q = 100 * (alpha / 2)
    high_q = 100 * (1.0 - alpha / 2)

    return {
        "observed_diff": observed,
        "p_value": float((extreme + 1) / (int(n_samples) + 1)),
        "ci_low": float(np.percentile(diffs, low_q)),
        "ci_high": float(np.percentile(diffs, high_q)),
    }
