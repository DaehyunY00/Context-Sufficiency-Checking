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
    """
    Paired permutation(sign-flip) test + permutation-based CI.
    기존 부트스트랩 기반 극단값 계산의 p-value 왜곡을 방지한다.
    """
    if len(baseline_values) != len(candidate_values):
        raise ValueError("부트스트랩 비교는 동일 길이의 두 시퀀스가 필요합니다.")

    n = len(baseline_values)
    if n == 0:
        return {"observed_diff": 0.0, "p_value": 1.0, "ci_low": 0.0, "ci_high": 0.0}

    base = np.array(baseline_values, dtype=float)
    cand = np.array(candidate_values, dtype=float)
    delta = cand - base
    observed = float(np.mean(delta))

    rng = random.Random(seed)
    diffs: List[float] = []
    extreme = 0

    for _ in range(int(n_samples)):
        signs = np.array([1.0 if rng.random() < 0.5 else -1.0 for _ in range(n)], dtype=float)
        diff = float(np.mean(delta * signs))
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


def bootstrap_mean_ci(
    values: Sequence[float],
    n_samples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    arr = np.array(list(values), dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(int(n_samples)):
        idx = [rng.randrange(n) for _ in range(n)]
        means.append(float(np.mean(arr[idx])))

    alpha = 1.0 - float(confidence_level)
    low_q = 100 * (alpha / 2)
    high_q = 100 * (1.0 - alpha / 2)

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if n > 1 else 0.0,
        "ci_low": float(np.percentile(means, low_q)),
        "ci_high": float(np.percentile(means, high_q)),
    }


def bootstrap_mean_diff_ci(
    baseline_values: Sequence[float],
    candidate_values: Sequence[float],
    n_samples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    if len(baseline_values) != len(candidate_values):
        raise ValueError("평균 차이 부트스트랩은 동일 길이의 두 시퀀스가 필요합니다.")

    base = np.array(list(baseline_values), dtype=float)
    cand = np.array(list(candidate_values), dtype=float)
    n = len(base)
    if n == 0:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    rng = random.Random(seed)
    diffs: List[float] = []
    delta = cand - base
    for _ in range(int(n_samples)):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(float(np.mean(delta[idx])))

    alpha = 1.0 - float(confidence_level)
    low_q = 100 * (alpha / 2)
    high_q = 100 * (1.0 - alpha / 2)
    return {
        "mean_diff": float(np.mean(delta)),
        "ci_low": float(np.percentile(diffs, low_q)),
        "ci_high": float(np.percentile(diffs, high_q)),
    }


def _to_numpy_binary(y_true: Sequence[int], y_score: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    yt = np.array([1 if int(v) > 0 else 0 for v in y_true], dtype=int)
    ys = np.array([float(v) for v in y_score], dtype=float)
    if yt.shape[0] != ys.shape[0]:
        raise ValueError("라벨과 스코어 길이가 다릅니다.")
    ys = np.clip(ys, 0.0, 1.0)
    return yt, ys


def build_reliability_bins(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    yt, yp = _to_numpy_binary(y_true, y_prob)
    if len(yt) == 0:
        return []

    n_bins = max(2, int(n_bins))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: List[Dict[str, float]] = []
    for i in range(n_bins):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == n_bins - 1:
            mask = (yp >= lo) & (yp <= hi)
        else:
            mask = (yp >= lo) & (yp < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "bin_index": float(i),
                    "bin_left": lo,
                    "bin_right": hi,
                    "count": 0.0,
                    "avg_confidence": 0.0,
                    "empirical_accuracy": 0.0,
                    "abs_gap": 0.0,
                }
            )
            continue

        avg_conf = float(np.mean(yp[mask]))
        acc = float(np.mean(yt[mask]))
        bins.append(
            {
                "bin_index": float(i),
                "bin_left": lo,
                "bin_right": hi,
                "count": float(count),
                "avg_confidence": avg_conf,
                "empirical_accuracy": acc,
                "abs_gap": abs(avg_conf - acc),
            }
        )
    return bins


def expected_calibration_error(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 10,
) -> float:
    bins = build_reliability_bins(y_true=y_true, y_prob=y_prob, n_bins=n_bins)
    total = sum(float(b["count"]) for b in bins)
    if total <= 0:
        return 0.0
    return float(sum((float(b["count"]) / total) * float(b["abs_gap"]) for b in bins))


def brier_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    yt, yp = _to_numpy_binary(y_true, y_prob)
    if len(yt) == 0:
        return 0.0
    return float(np.mean((yp - yt) ** 2))


def _roc_points(y_true: np.ndarray, y_score: np.ndarray) -> List[Dict[str, float]]:
    thresholds = sorted(set(float(v) for v in y_score), reverse=True)
    points: List[Dict[str, float]] = []
    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    if pos == 0 or neg == 0:
        return points

    for th in thresholds:
        pred = (y_score >= th).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        tpr = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        points.append({"threshold": th, "tpr": float(tpr), "fpr": float(fpr)})

    points.append({"threshold": 0.0, "tpr": 1.0, "fpr": 1.0})
    points.append({"threshold": 1.0, "tpr": 0.0, "fpr": 0.0})
    points = sorted(points, key=lambda x: (x["fpr"], x["tpr"]))
    return points


def _pr_points(y_true: np.ndarray, y_score: np.ndarray) -> List[Dict[str, float]]:
    thresholds = sorted(set(float(v) for v in y_score), reverse=True)
    points: List[Dict[str, float]] = []
    pos = int((y_true == 1).sum())
    if pos == 0:
        return points

    for th in thresholds:
        pred = (y_score >= th).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        points.append({"threshold": th, "precision": float(precision), "recall": float(recall)})

    points.append({"threshold": 0.0, "precision": float(pos / max(1, len(y_true))), "recall": 1.0})
    points.append({"threshold": 1.0, "precision": 1.0, "recall": 0.0})
    points = sorted(points, key=lambda x: x["recall"])
    return points


def _auc_xy(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))


def roc_pr_diagnostics(
    y_true: Sequence[int],
    y_score: Sequence[float],
) -> Dict[str, object]:
    yt, ys = _to_numpy_binary(y_true, y_score)
    if len(yt) == 0:
        return {"auroc": 0.0, "auprc": 0.0, "roc_points": [], "pr_points": []}

    roc_pts = _roc_points(yt, ys)
    pr_pts = _pr_points(yt, ys)

    if roc_pts:
        auroc = _auc_xy([p["fpr"] for p in roc_pts], [p["tpr"] for p in roc_pts])
    else:
        auroc = 0.0

    if pr_pts:
        auprc = _auc_xy([p["recall"] for p in pr_pts], [p["precision"] for p in pr_pts])
    else:
        auprc = 0.0

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "roc_points": roc_pts,
        "pr_points": pr_pts,
    }


def risk_coverage_diagnostics(
    y_true: Sequence[int],
    y_score: Sequence[float],
) -> Dict[str, object]:
    """
    선택적 예측 관점의 Risk-Coverage 곡선과 AURC를 계산한다.
    - coverage: score 임계값 이상으로 수용된 비율
    - risk: 수용 집합에서의 1 - accuracy
    """
    yt, ys = _to_numpy_binary(y_true, y_score)
    if len(yt) == 0:
        return {"aurc": 0.0, "risk_coverage_points": []}

    thresholds = sorted(set(float(v) for v in ys), reverse=True)
    points: List[Dict[str, float]] = [
        {
            "threshold": 1.0 + 1e-12,
            "coverage": 0.0,
            "accuracy": 1.0,
            "risk": 0.0,
        }
    ]

    for th in thresholds:
        selected = ys >= th
        selected_count = int(selected.sum())
        if selected_count <= 0:
            continue
        coverage = float(selected_count / max(1, len(yt)))
        accuracy = float(np.mean(yt[selected]))
        risk = float(1.0 - accuracy)
        points.append(
            {
                "threshold": float(th),
                "coverage": coverage,
                "accuracy": accuracy,
                "risk": risk,
            }
        )

    if points and points[-1]["coverage"] < 1.0:
        full_acc = float(np.mean(yt))
        points.append(
            {
                "threshold": 0.0,
                "coverage": 1.0,
                "accuracy": full_acc,
                "risk": float(1.0 - full_acc),
            }
        )

    points = sorted(points, key=lambda x: (x["coverage"], -x["threshold"]))
    aurc = _auc_xy([p["coverage"] for p in points], [p["risk"] for p in points])
    return {"aurc": float(aurc), "risk_coverage_points": points}
