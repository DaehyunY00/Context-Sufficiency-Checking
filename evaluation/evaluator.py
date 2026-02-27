from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from evaluation.metrics import (
    brier_score,
    build_reliability_bins,
    bootstrap_mean_ci,
    bootstrap_mean_diff_ci,
    expected_calibration_error,
    paired_bootstrap_test,
    risk_coverage_diagnostics,
    roc_pr_diagnostics,
    summarize_records,
)


def save_jsonl(records: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_summary_row(summary: Dict, run_name: str, checker: str, strategy: str) -> Dict:
    checker_norm = str(checker).strip().lower()
    has_checker = checker_norm not in {"", "none", "없음"}
    return {
        "실험명": run_name,
        "체커": checker,
        "전략": strategy,
        "샘플수": int(summary.get("sample_count", 0)),
        "EM": float(summary.get("em", 0.0)),
        "F1": float(summary.get("f1", 0.0)),
        "환각률": float(summary.get("hallucination_rate", 0.0)),
        "커버리지": float(summary.get("coverage", 0.0)),
        "선택적정확도": float(summary.get("selective_accuracy", 0.0)),
        "평균지연(ms)": float(summary.get("latency_ms", 0.0)),
        "평균지연(ms,warmup제외)": "",
        "지연표준편차(ms)": "",
        "지연P50(ms)": "",
        "지연P95(ms)": "",
        "실행장치": "",
        "CPU평균지연(ms)": "",
        "MPS평균지연(ms)": "",
        "체커판정수": int(summary.get("checker_eval_count", 0)) if has_checker else "",
        "체커파싱실패수": int(summary.get("checker_parse_fail_count", 0)) if has_checker else "",
        "체커파싱성공률": float(summary.get("checker_parse_success_rate", 0.0)) if has_checker else "",
        "AURC": "",
        "CSC_Temperature": "",
        "CSC_ECE_before": "",
        "CSC_ECE_after": "",
        "CSC_ECE": "",
        "CSC_Brier_before": "",
        "CSC_Brier_after": "",
        "CSC_Brier": "",
        "CSC_AUROC": "",
        "CSC_AUPRC": "",
        "CSC-정답상관_Pearsonr": "",
        "CSC-정답상관_Spearmanrho": "",
        "검색점수-충분성상관": "",
        "검색점수-충분성_Pearsonr": "",
        "검색점수-충분성_Spearmanrho": "",
        "검색점수-충분성_MI": "",
        "고검색-불충분비율": "",
        "저검색-충분비율": "",
    }


def attach_significance(
    row: Dict,
    baseline_records: List[Dict],
    candidate_records: List[Dict],
    n_samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict:
    base_em = [float(r.get("em", 0.0)) for r in baseline_records]
    cand_em = [float(r.get("em", 0.0)) for r in candidate_records]
    base_f1 = [float(r.get("f1", 0.0)) for r in baseline_records]
    cand_f1 = [float(r.get("f1", 0.0)) for r in candidate_records]

    em_test = paired_bootstrap_test(base_em, cand_em, n_samples=n_samples, confidence_level=confidence_level, seed=seed)
    f1_test = paired_bootstrap_test(base_f1, cand_f1, n_samples=n_samples, confidence_level=confidence_level, seed=seed)

    out = dict(row)
    out["EM_차이"] = em_test["observed_diff"]
    out["EM_p값"] = em_test["p_value"]
    out["F1_차이"] = f1_test["observed_diff"]
    out["F1_p값"] = f1_test["p_value"]
    return out


def _safe_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    x = np.array(list(xs), dtype=float)
    y = np.array(list(ys), dtype=float)
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return 0.0
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _rank_array(values: Sequence[float]) -> np.ndarray:
    arr = np.array(list(values), dtype=float)
    n = len(arr)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def _safe_spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    x_rank = _rank_array(xs)
    y_rank = _rank_array(ys)
    if len(x_rank) == 0 or len(y_rank) == 0 or len(x_rank) != len(y_rank):
        return 0.0
    return _safe_corr(x_rank.tolist(), y_rank.tolist())


def _quantile_bin(values: Sequence[float], n_bins: int = 10) -> np.ndarray:
    arr = np.array(list(values), dtype=float)
    if len(arr) == 0:
        return np.array([], dtype=int)
    n_bins = max(2, int(n_bins))
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(arr, qs)
    edges = np.unique(edges)
    if len(edges) <= 1:
        return np.zeros(len(arr), dtype=int)
    bins = np.digitize(arr, edges[1:-1], right=False)
    return bins.astype(int)


def _mutual_information_discrete(x_bin: Sequence[int], y_bin: Sequence[int]) -> float:
    xb = np.array(list(x_bin), dtype=int)
    yb = np.array(list(y_bin), dtype=int)
    if len(xb) == 0 or len(yb) == 0 or len(xb) != len(yb):
        return 0.0

    n = len(xb)
    x_vals = sorted(set(int(v) for v in xb.tolist()))
    y_vals = sorted(set(int(v) for v in yb.tolist()))
    if len(x_vals) <= 1 or len(y_vals) <= 1:
        return 0.0

    mi = 0.0
    for xv in x_vals:
        px = float((xb == xv).sum()) / n
        if px <= 0:
            continue
        for yv in y_vals:
            py = float((yb == yv).sum()) / n
            if py <= 0:
                continue
            pxy = float(((xb == xv) & (yb == yv)).sum()) / n
            if pxy <= 0:
                continue
            mi += pxy * float(np.log(pxy / (px * py)))
    return float(max(0.0, mi))


def _setup_matplotlib_korean_font(plt) -> None:
    """
    macOS/리눅스 환경에서 한글 폰트 경고를 줄이기 위한 안전 설정.
    """
    try:
        from matplotlib import font_manager

        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in ["AppleGothic", "NanumGothic", "Malgun Gothic", "Arial Unicode MS"]:
            if name in available:
                plt.rcParams["font.family"] = name
                break
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _safe_logit(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _safe_sigmoid(logits: np.ndarray) -> np.ndarray:
    z = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _binary_nll(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = np.array(y_true, dtype=float)
    p = np.clip(np.array(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _fit_temperature_scaling(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    t_min: float = 0.25,
    t_max: float = 5.0,
    n_grid: int = 81,
) -> Tuple[float, float]:
    y = np.array(list(y_true), dtype=int)
    p = np.array(list(y_prob), dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 1.0, _binary_nll(y_true=y, y_prob=p) if len(y) else 0.0

    logits = _safe_logit(p)
    ts = np.exp(np.linspace(np.log(max(1e-6, t_min)), np.log(max(t_min + 1e-6, t_max)), int(n_grid)))
    best_t = 1.0
    best_nll = float("inf")
    for t in ts:
        scaled = _safe_sigmoid(logits / float(t))
        nll = _binary_nll(y_true=y, y_prob=scaled)
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t, best_nll


def _apply_temperature_scaling(y_prob: Sequence[float], temperature: float) -> np.ndarray:
    p = np.array(list(y_prob), dtype=float)
    logits = _safe_logit(p)
    t = max(1e-6, float(temperature))
    return _safe_sigmoid(logits / t)


def latency_quality_analysis(
    records: List[Dict],
    warmup_drop: int = 5,
    hist_bins: int = 20,
) -> Dict[str, object]:
    if not records:
        return {
            "count": 0,
            "warmup_drop": int(warmup_drop),
            "mean_ms": 0.0,
            "mean_ms_wo_warmup": 0.0,
            "std_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "devices": {},
            "histogram": [],
        }

    lat = np.array([float(r.get("latency_ms", 0.0)) for r in records], dtype=float)
    warmup_drop = max(0, int(warmup_drop))
    lat_wo = lat[warmup_drop:] if len(lat) > warmup_drop else lat

    device_values: Dict[str, List[float]] = {}
    for r in records:
        dev = str(r.get("generator_device", "unknown")).strip().lower() or "unknown"
        device_values.setdefault(dev, []).append(float(r.get("latency_ms", 0.0)))

    device_summary = {
        dev: {
            "count": len(vals),
            "mean_ms": float(np.mean(vals)) if vals else 0.0,
            "std_ms": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }
        for dev, vals in device_values.items()
    }

    hist_bins = max(5, int(hist_bins))
    counts, edges = np.histogram(lat, bins=hist_bins)
    histogram = []
    for i in range(len(counts)):
        histogram.append(
            {
                "bin_left": float(edges[i]),
                "bin_right": float(edges[i + 1]),
                "count": int(counts[i]),
            }
        )

    return {
        "count": int(len(lat)),
        "warmup_drop": warmup_drop,
        "mean_ms": float(np.mean(lat)),
        "mean_ms_wo_warmup": float(np.mean(lat_wo)) if len(lat_wo) > 0 else 0.0,
        "std_ms": float(np.std(lat, ddof=1)) if len(lat) > 1 else 0.0,
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "devices": device_summary,
        "histogram": histogram,
    }


def checker_quality_analysis(
    records: List[Dict],
    n_bins: int = 10,
    retrieval_high_q: float = 0.8,
    retrieval_low_q: float = 0.2,
    temperature_scaling: bool = True,
    temperature_val_ratio: float = 0.3,
    temperature_seed: int = 42,
    temperature_min_samples: int = 50,
) -> Dict[str, object]:
    scored_records = [
        r
        for r in records
        if str(r.get("checker_name", "none")).strip().lower() not in {"", "none", "없음"}
        and float(r.get("checker_score", -1.0)) >= 0.0
    ]

    if not scored_records:
        return {
            "score_count": 0,
            "aurc": 0.0,
            "ece": 0.0,
            "ece_before": 0.0,
            "ece_after": 0.0,
            "brier": 0.0,
            "brier_before": 0.0,
            "brier_after": 0.0,
            "auroc": 0.0,
            "auprc": 0.0,
            "temperature_scaling_enabled": bool(temperature_scaling),
            "temperature_scaling_applied": False,
            "temperature": 1.0,
            "temperature_val_ratio": float(temperature_val_ratio),
            "calibration_eval_count": 0,
            "score_correct_corr": 0.0,
            "retrieval_checker_corr": 0.0,
            "retrieval_checker_pearson": 0.0,
            "retrieval_checker_spearman": 0.0,
            "high_retrieval_insufficient_rate": 0.0,
            "low_retrieval_sufficient_rate": 0.0,
            "high_retrieval_threshold": 0.0,
            "low_retrieval_threshold": 0.0,
            "high_retrieval_count": 0,
            "low_retrieval_count": 0,
            "reliability_bins": [],
            "reliability_bins_before": [],
            "reliability_bins_after": [],
            "roc_points": [],
            "pr_points": [],
            "risk_coverage_points": [],
            "retrieval_checker_scatter": [],
            "score_density_bins": [],
            "score_threshold_hint": 0.5,
        }

    scores = [float(np.clip(float(r.get("checker_score", 0.0)), 0.0, 1.0)) for r in scored_records]
    y_correct = [int(r.get("is_correct", 0)) for r in scored_records]
    y_answerable = [int(r.get("oracle_answerable", r.get("is_correct", 0))) for r in scored_records]

    score_arr_all = np.array(scores, dtype=float)
    y_arr_all = np.array(y_answerable, dtype=int)

    calibrated_scores_all = score_arr_all.copy()
    fitted_temperature = 1.0
    calibration_applied = False
    eval_idx = np.arange(len(score_arr_all))

    unique_classes = set(int(x) for x in y_arr_all.tolist())
    can_calibrate = (
        bool(temperature_scaling)
        and len(score_arr_all) >= int(max(2, temperature_min_samples))
        and len(unique_classes) >= 2
    )
    if can_calibrate:
        ratio = float(np.clip(temperature_val_ratio, 0.05, 0.9))
        n = len(score_arr_all)
        n_val = int(round(n * ratio))
        n_val = max(1, min(n - 1, n_val))
        rng = np.random.default_rng(int(temperature_seed))
        perm = np.arange(n)
        rng.shuffle(perm)
        val_idx = perm[:n_val]
        test_idx = perm[n_val:]

        y_val = y_arr_all[val_idx]
        if len(set(int(x) for x in y_val.tolist())) >= 2:
            fitted_temperature, _ = _fit_temperature_scaling(
                y_true=y_val.tolist(),
                y_prob=score_arr_all[val_idx].tolist(),
            )
            calibrated_scores_all = _apply_temperature_scaling(
                y_prob=score_arr_all.tolist(),
                temperature=fitted_temperature,
            )
            calibration_applied = True
            if len(test_idx) > 0 and len(set(int(x) for x in y_arr_all[test_idx].tolist())) >= 2:
                eval_idx = test_idx

    y_eval = y_arr_all[eval_idx]
    raw_eval = score_arr_all[eval_idx]
    cal_eval = calibrated_scores_all[eval_idx]
    raw_eval_list = [float(x) for x in raw_eval.tolist()]
    cal_eval_list = [float(x) for x in cal_eval.tolist()]
    y_eval_list = [int(x) for x in y_eval.tolist()]

    reliability_bins_before = build_reliability_bins(y_true=y_eval_list, y_prob=raw_eval_list, n_bins=n_bins)
    reliability_bins_after = build_reliability_bins(y_true=y_eval_list, y_prob=cal_eval_list, n_bins=n_bins)
    ece_before = expected_calibration_error(y_true=y_eval_list, y_prob=raw_eval_list, n_bins=n_bins)
    ece_after = expected_calibration_error(y_true=y_eval_list, y_prob=cal_eval_list, n_bins=n_bins)
    brier_before = brier_score(y_true=y_eval_list, y_prob=raw_eval_list)
    brier_after = brier_score(y_true=y_eval_list, y_prob=cal_eval_list)

    roc_pr = roc_pr_diagnostics(y_true=y_answerable, y_score=scores)
    risk_cov = risk_coverage_diagnostics(y_true=y_answerable, y_score=scores)

    retrieval_scores = [
        float(r.get("initial_max_retrieval_score", max(r.get("retrieved_scores", [0.0]) or [0.0])))
        for r in scored_records
    ]
    checker_scores = [float(x) for x in calibrated_scores_all.tolist()]
    pearson = _safe_corr(retrieval_scores, checker_scores)
    spearman = _safe_spearman(retrieval_scores, checker_scores)
    retrieval_bins = _quantile_bin(retrieval_scores, n_bins=10)
    suff_label = [
        1 if str(r.get("checker_label", "")).strip().upper() == "SUFFICIENT" else 0
        for r in scored_records
    ]
    score_bins = _quantile_bin(checker_scores, n_bins=10)
    retrieval_suff_mi = _mutual_information_discrete(retrieval_bins.tolist(), score_bins.tolist())
    retrieval_suff_label_mi = _mutual_information_discrete(retrieval_bins.tolist(), suff_label)
    score_correct_pearson = _safe_corr(checker_scores, y_correct)
    score_correct_spearman = _safe_spearman(checker_scores, y_correct)

    threshold_candidates: List[float] = []
    for r in scored_records:
        meta = r.get("checker_meta", {})
        if not isinstance(meta, dict):
            continue
        if "threshold" in meta:
            try:
                threshold_candidates.append(float(meta.get("threshold")))
                continue
            except (TypeError, ValueError):
                pass
        if "confidence_threshold" in meta:
            try:
                threshold_candidates.append(float(meta.get("confidence_threshold")))
                continue
            except (TypeError, ValueError):
                pass
        if "기준_커버리지" in meta:
            try:
                threshold_candidates.append(float(meta.get("기준_커버리지")))
                continue
            except (TypeError, ValueError):
                pass
        if "임계값" in meta:
            try:
                th = float(meta.get("임계값"))
                checker_name = str(r.get("checker_name", "")).strip().lower()
                if checker_name == "self_consistency":
                    threshold_candidates.append(1.0 - th)
                else:
                    threshold_candidates.append(th)
            except (TypeError, ValueError):
                pass
    score_threshold_hint = float(np.median(threshold_candidates)) if threshold_candidates else 0.5

    score_arr = np.array(checker_scores, dtype=float)
    oracle_suf = np.array(y_answerable, dtype=int)
    edges = np.linspace(0.0, 1.0, 21)
    suff_scores = score_arr[oracle_suf == 1]
    ins_scores = score_arr[oracle_suf == 0]
    suff_hist, _ = np.histogram(suff_scores, bins=edges)
    ins_hist, _ = np.histogram(ins_scores, bins=edges)
    bin_width = edges[1] - edges[0]
    score_density_bins = []
    for i in range(len(suff_hist)):
        left = float(edges[i])
        right = float(edges[i + 1])
        suff_count = int(suff_hist[i])
        ins_count = int(ins_hist[i])
        suff_density = (
            float(suff_count / max(1, len(suff_scores)) / max(1e-12, bin_width))
            if len(suff_scores) > 0
            else 0.0
        )
        ins_density = (
            float(ins_count / max(1, len(ins_scores)) / max(1e-12, bin_width))
            if len(ins_scores) > 0
            else 0.0
        )
        score_density_bins.append(
            {
                "bin_left": left,
                "bin_right": right,
                "sufficient_count": suff_count,
                "insufficient_count": ins_count,
                "sufficient_density": suff_density,
                "insufficient_density": ins_density,
            }
        )

    high_q = float(np.quantile(retrieval_scores, retrieval_high_q))
    low_q = float(np.quantile(retrieval_scores, retrieval_low_q))
    high_group = [
        r for r, rs in zip(scored_records, retrieval_scores) if float(rs) >= high_q
    ]
    low_group = [
        r for r, rs in zip(scored_records, retrieval_scores) if float(rs) <= low_q
    ]
    high_insufficient_rate = (
        float(
            np.mean(
                [
                    1.0 if str(x.get("checker_label", "")).strip().upper() == "INSUFFICIENT" else 0.0
                    for x in high_group
                ]
            )
        )
        if high_group
        else 0.0
    )
    low_sufficient_rate = (
        float(
            np.mean(
                [
                    1.0 if str(x.get("checker_label", "")).strip().upper() == "SUFFICIENT" else 0.0
                    for x in low_group
                ]
            )
        )
        if low_group
        else 0.0
    )

    scatter_rows = [
        {
            "retrieval_score": float(rs),
            "checker_score": float(cs),
            "oracle_answerable": int(ans),
            "is_correct": int(corr),
        }
        for rs, cs, ans, corr in zip(retrieval_scores, checker_scores, y_answerable, y_correct)
    ]

    return {
        "score_count": len(scores),
        "aurc": float(risk_cov["aurc"]),
        "ece": float(ece_after),
        "ece_before": float(ece_before),
        "ece_after": float(ece_after),
        "brier": float(brier_after),
        "brier_before": float(brier_before),
        "brier_after": float(brier_after),
        "auroc": float(roc_pr["auroc"]),
        "auprc": float(roc_pr["auprc"]),
        "temperature_scaling_enabled": bool(temperature_scaling),
        "temperature_scaling_applied": bool(calibration_applied),
        "temperature": float(fitted_temperature),
        "temperature_val_ratio": float(temperature_val_ratio),
        "calibration_eval_count": int(len(eval_idx)),
        "score_correct_corr": score_correct_pearson,
        "score_correct_pearson": score_correct_pearson,
        "score_correct_spearman": score_correct_spearman,
        "retrieval_checker_corr": pearson,
        "retrieval_checker_pearson": pearson,
        "retrieval_checker_spearman": spearman,
        "retrieval_sufficiency_mi": retrieval_suff_mi,
        "retrieval_sufficiency_label_mi": retrieval_suff_label_mi,
        "high_retrieval_insufficient_rate": high_insufficient_rate,
        "low_retrieval_sufficient_rate": low_sufficient_rate,
        "high_retrieval_threshold": high_q,
        "low_retrieval_threshold": low_q,
        "high_retrieval_count": len(high_group),
        "low_retrieval_count": len(low_group),
        "reliability_bins": reliability_bins_after,
        "reliability_bins_before": reliability_bins_before,
        "reliability_bins_after": reliability_bins_after,
        "roc_points": roc_pr["roc_points"],
        "pr_points": roc_pr["pr_points"],
        "risk_coverage_points": risk_cov["risk_coverage_points"],
        "retrieval_checker_scatter": scatter_rows,
        "score_density_bins": score_density_bins,
        "score_threshold_hint": score_threshold_hint,
    }


def save_checker_artifacts(
    analysis: Dict[str, object],
    output_dir: Path,
    run_name: str,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: Dict[str, str] = {}

    rel_before_path = output_dir / f"{run_name}_reliability_before.csv"
    rel_after_path = output_dir / f"{run_name}_reliability_after.csv"
    rel_path = output_dir / f"{run_name}_reliability.csv"
    roc_path = output_dir / f"{run_name}_roc.csv"
    pr_path = output_dir / f"{run_name}_pr.csv"
    rc_path = output_dir / f"{run_name}_risk_coverage.csv"
    scatter_path = output_dir / f"{run_name}_retrieval_checker_scatter.csv"
    score_density_path = output_dir / f"{run_name}_score_density.csv"

    _save_dict_rows_csv(
        rows=list(analysis.get("reliability_bins_before", analysis.get("reliability_bins", []))),
        path=rel_before_path,
        fieldnames=["bin_index", "bin_left", "bin_right", "count", "avg_confidence", "empirical_accuracy", "abs_gap"],
    )
    _save_dict_rows_csv(
        rows=list(analysis.get("reliability_bins_after", analysis.get("reliability_bins", []))),
        path=rel_after_path,
        fieldnames=["bin_index", "bin_left", "bin_right", "count", "avg_confidence", "empirical_accuracy", "abs_gap"],
    )
    _save_dict_rows_csv(
        rows=list(analysis.get("reliability_bins_after", analysis.get("reliability_bins", []))),
        path=rel_path,
        fieldnames=["bin_index", "bin_left", "bin_right", "count", "avg_confidence", "empirical_accuracy", "abs_gap"],
    )
    _save_dict_rows_csv(rows=list(analysis.get("roc_points", [])), path=roc_path, fieldnames=["threshold", "fpr", "tpr"])
    _save_dict_rows_csv(
        rows=list(analysis.get("pr_points", [])),
        path=pr_path,
        fieldnames=["threshold", "recall", "precision"],
    )
    _save_dict_rows_csv(
        rows=list(analysis.get("risk_coverage_points", [])),
        path=rc_path,
        fieldnames=["threshold", "coverage", "accuracy", "risk"],
    )
    _save_dict_rows_csv(
        rows=list(analysis.get("retrieval_checker_scatter", [])),
        path=scatter_path,
        fieldnames=["retrieval_score", "checker_score", "oracle_answerable", "is_correct"],
    )
    _save_dict_rows_csv(
        rows=list(analysis.get("score_density_bins", [])),
        path=score_density_path,
        fieldnames=[
            "bin_left",
            "bin_right",
            "sufficient_count",
            "insufficient_count",
            "sufficient_density",
            "insufficient_density",
        ],
    )
    saved_paths["reliability_before_csv"] = str(rel_before_path)
    saved_paths["reliability_after_csv"] = str(rel_after_path)
    saved_paths["reliability_csv"] = str(rel_path)
    saved_paths["roc_csv"] = str(roc_path)
    saved_paths["pr_csv"] = str(pr_path)
    saved_paths["risk_coverage_csv"] = str(rc_path)
    saved_paths["retrieval_checker_scatter_csv"] = str(scatter_path)
    saved_paths["score_density_csv"] = str(score_density_path)

    try:
        import matplotlib.pyplot as plt
        _setup_matplotlib_korean_font(plt)

        rel_before_png = output_dir / f"{run_name}_reliability_before.png"
        rel_after_png = output_dir / f"{run_name}_reliability_after.png"
        rel_png = output_dir / f"{run_name}_reliability.png"
        roc_png = output_dir / f"{run_name}_roc.png"
        pr_png = output_dir / f"{run_name}_pr.png"
        rc_png = output_dir / f"{run_name}_risk_coverage.png"
        scatter_png = output_dir / f"{run_name}_retrieval_checker_scatter.png"
        score_density_png = output_dir / f"{run_name}_score_density.png"

        bins_before = list(analysis.get("reliability_bins_before", analysis.get("reliability_bins", [])))
        bins_after = list(analysis.get("reliability_bins_after", analysis.get("reliability_bins", [])))
        if bins_before:
            plt.figure(figsize=(5.6, 5.0))
            xs = [float(b["avg_confidence"]) for b in bins_before if float(b["count"]) > 0]
            ys = [float(b["empirical_accuracy"]) for b in bins_before if float(b["count"]) > 0]
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            if xs and ys:
                plt.plot(xs, ys, marker="o")
            plt.xlabel("예측 확률(체커 점수)")
            plt.ylabel("실제 Answerable 비율")
            plt.title("Calibration Before Temperature Scaling")
            plt.tight_layout()
            plt.savefig(rel_before_png, dpi=160)
            plt.close()
            saved_paths["reliability_before_png"] = str(rel_before_png)

        if bins_after:
            plt.figure(figsize=(5.6, 5.0))
            xs = [float(b["avg_confidence"]) for b in bins_after if float(b["count"]) > 0]
            ys = [float(b["empirical_accuracy"]) for b in bins_after if float(b["count"]) > 0]
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            if xs and ys:
                plt.plot(xs, ys, marker="o")
            plt.xlabel("예측 확률(체커 점수)")
            plt.ylabel("실제 Answerable 비율")
            plt.title("Calibration After Temperature Scaling")
            plt.tight_layout()
            plt.savefig(rel_after_png, dpi=160)
            plt.close()
            saved_paths["reliability_after_png"] = str(rel_after_png)

            # backward compatibility
            plt.figure(figsize=(5.6, 5.0))
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            if xs and ys:
                plt.plot(xs, ys, marker="o")
            plt.xlabel("예측 확률(체커 점수)")
            plt.ylabel("실제 Answerable 비율")
            plt.title("Calibration Reliability Diagram")
            plt.tight_layout()
            plt.savefig(rel_png, dpi=160)
            plt.close()
            saved_paths["reliability_png"] = str(rel_png)

        roc_points = list(analysis.get("roc_points", []))
        if roc_points:
            plt.figure(figsize=(5.6, 5.0))
            fprs = [float(p["fpr"]) for p in roc_points]
            tprs = [float(p["tpr"]) for p in roc_points]
            plt.plot(fprs, tprs, marker=".")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.tight_layout()
            plt.savefig(roc_png, dpi=160)
            plt.close()
            saved_paths["roc_png"] = str(roc_png)

        pr_points = list(analysis.get("pr_points", []))
        if pr_points:
            plt.figure(figsize=(5.6, 5.0))
            recalls = [float(p["recall"]) for p in pr_points]
            precisions = [float(p["precision"]) for p in pr_points]
            plt.plot(recalls, precisions, marker=".")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("PR Curve")
            plt.tight_layout()
            plt.savefig(pr_png, dpi=160)
            plt.close()
            saved_paths["pr_png"] = str(pr_png)

        rc_points = list(analysis.get("risk_coverage_points", []))
        if rc_points:
            plt.figure(figsize=(5.6, 5.0))
            covers = [float(p["coverage"]) for p in rc_points]
            risks = [float(p["risk"]) for p in rc_points]
            plt.plot(covers, risks, marker=".")
            plt.xlabel("Coverage")
            plt.ylabel("Risk (1-Accuracy)")
            plt.title("Risk-Coverage Curve")
            plt.tight_layout()
            plt.savefig(rc_png, dpi=160)
            plt.close()
            saved_paths["risk_coverage_png"] = str(rc_png)

        scatter = list(analysis.get("retrieval_checker_scatter", []))
        if scatter:
            plt.figure(figsize=(5.6, 5.0))
            xs = [float(p["retrieval_score"]) for p in scatter]
            ys = [float(p["checker_score"]) for p in scatter]
            cs = [int(p["oracle_answerable"]) for p in scatter]
            plt.scatter(xs, ys, c=cs, cmap="coolwarm", alpha=0.45, s=14)
            th = float(analysis.get("score_threshold_hint", 0.5))
            plt.axhline(th, linestyle="--", color="gray", linewidth=1.0, label=f"CSC boundary={th:.2f}")
            hq = analysis.get("high_retrieval_threshold", None)
            lq = analysis.get("low_retrieval_threshold", None)
            if hq is not None:
                plt.axvline(float(hq), linestyle=":", color="#264653", linewidth=1.0, label="high-q")
            if lq is not None:
                plt.axvline(float(lq), linestyle=":", color="#f4a261", linewidth=1.0, label="low-q")
            plt.xlabel("Retrieval Score")
            plt.ylabel("Sufficiency Score")
            plt.title("Retrieval vs Sufficiency Scatter (Decision Boundaries)")
            plt.legend(loc="best", fontsize=8)
            plt.tight_layout()
            plt.savefig(scatter_png, dpi=160)
            plt.close()
            saved_paths["retrieval_checker_scatter_png"] = str(scatter_png)

        density_bins = list(analysis.get("score_density_bins", []))
        if density_bins:
            centers = [
                (float(b.get("bin_left", 0.0)) + float(b.get("bin_right", 0.0))) / 2.0
                for b in density_bins
            ]
            suff_d = [float(b.get("sufficient_density", 0.0)) for b in density_bins]
            ins_d = [float(b.get("insufficient_density", 0.0)) for b in density_bins]
            th = float(analysis.get("score_threshold_hint", 0.5))
            plt.figure(figsize=(5.8, 4.2))
            plt.plot(centers, suff_d, label="Oracle Answerable=1", color="#2a9d8f")
            plt.plot(centers, ins_d, label="Oracle Answerable=0", color="#e76f51")
            plt.axvline(th, linestyle="--", color="gray", label=f"threshold={th:.2f}")
            plt.xlabel("CSC Score")
            plt.ylabel("Density")
            plt.title("CSC Score Density by Oracle Answerability")
            plt.legend()
            plt.tight_layout()
            plt.savefig(score_density_png, dpi=160)
            plt.close()
            saved_paths["score_density_png"] = str(score_density_png)
    except Exception:
        # matplotlib 미설치 또는 렌더링 실패 시 CSV만 저장
        pass

    return saved_paths


def save_latency_artifacts(
    analysis: Dict[str, object],
    output_dir: Path,
    run_name: str,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, str] = {}

    hist_csv = output_dir / f"{run_name}_latency_histogram.csv"
    dev_csv = output_dir / f"{run_name}_latency_devices.csv"

    _save_dict_rows_csv(
        rows=list(analysis.get("histogram", [])),
        path=hist_csv,
        fieldnames=["bin_left", "bin_right", "count"],
    )
    device_rows = []
    for dev, info in dict(analysis.get("devices", {})).items():
        device_rows.append(
            {
                "device": dev,
                "count": int(info.get("count", 0)),
                "mean_ms": float(info.get("mean_ms", 0.0)),
                "std_ms": float(info.get("std_ms", 0.0)),
            }
        )
    _save_dict_rows_csv(rows=device_rows, path=dev_csv, fieldnames=["device", "count", "mean_ms", "std_ms"])
    saved["latency_histogram_csv"] = str(hist_csv)
    saved["latency_devices_csv"] = str(dev_csv)

    try:
        import matplotlib.pyplot as plt
        _setup_matplotlib_korean_font(plt)

        hist_png = output_dir / f"{run_name}_latency_histogram.png"
        data = list(analysis.get("histogram", []))
        if data:
            centers = [
                (float(x.get("bin_left", 0.0)) + float(x.get("bin_right", 0.0))) / 2.0
                for x in data
            ]
            counts = [int(x.get("count", 0)) for x in data]
            widths = [
                max(1e-6, float(x.get("bin_right", 0.0)) - float(x.get("bin_left", 0.0)))
                for x in data
            ]
            plt.figure(figsize=(5.8, 4.0))
            plt.bar(centers, counts, width=widths, alpha=0.75)
            plt.xlabel("Latency (ms)")
            plt.ylabel("Count")
            plt.title("Latency Histogram")
            plt.tight_layout()
            plt.savefig(hist_png, dpi=160)
            plt.close()
            saved["latency_histogram_png"] = str(hist_png)
    except Exception:
        pass

    return saved


def _save_dict_rows_csv(rows: List[Dict], path: Path, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def save_summary_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = [
        "메서드키",
        "실험명",
        "시드",
        "시드수",
        "체커",
        "전략",
        "샘플수",
        "EM",
        "F1",
        "환각률",
        "커버리지",
        "선택적정확도",
        "평균지연(ms)",
        "평균지연(ms,warmup제외)",
        "지연표준편차(ms)",
        "지연P50(ms)",
        "지연P95(ms)",
        "실행장치",
        "CPU평균지연(ms)",
        "MPS평균지연(ms)",
        "체커판정수",
        "체커파싱실패수",
        "체커파싱성공률",
        "AURC",
        "CSC_Temperature",
        "CSC_ECE_before",
        "CSC_ECE_after",
        "CSC_ECE",
        "CSC_Brier_before",
        "CSC_Brier_after",
        "CSC_Brier",
        "CSC_AUROC",
        "CSC_AUPRC",
        "CSC-정답상관_Pearsonr",
        "CSC-정답상관_Spearmanrho",
        "검색점수-충분성상관",
        "검색점수-충분성_Pearsonr",
        "검색점수-충분성_Spearmanrho",
        "검색점수-충분성_MI",
        "고검색-불충분비율",
        "저검색-충분비율",
        "EM_차이",
        "EM_p값",
        "F1_차이",
        "F1_p값",
        "EM_평균±표준편차",
        "EM_CI95",
        "F1_평균±표준편차",
        "F1_CI95",
        "환각률_평균±표준편차",
        "환각률_CI95",
        "커버리지_평균±표준편차",
        "커버리지_CI95",
        "EM_차이_CI95",
        "F1_차이_CI95",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def make_markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "결과가 없습니다."

    headers = [
        "실험명",
        "체커",
        "전략",
        "EM",
        "EM_CI95",
        "F1",
        "F1_CI95",
        "환각률",
        "환각률_CI95",
        "커버리지",
        "커버리지_CI95",
        "선택적정확도",
        "평균지연(ms)",
        "평균지연(ms,warmup제외)",
        "AURC",
        "CSC_Temperature",
        "CSC_ECE_before",
        "CSC_ECE_after",
        "CSC_ECE",
        "CSC_Brier_before",
        "CSC_Brier_after",
        "CSC_Brier",
        "CSC_AUROC",
        "CSC_AUPRC",
        "CSC-정답상관_Pearsonr",
        "CSC-정답상관_Spearmanrho",
        "검색점수-충분성_Pearsonr",
        "검색점수-충분성_Spearmanrho",
        "검색점수-충분성_MI",
        "고검색-불충분비율",
        "저검색-충분비율",
        "EM_p값",
        "F1_p값",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("실험명", "")),
                    str(row.get("체커", "")),
                    str(row.get("전략", "")),
                    _fmt_metric_or_pm(row, "EM"),
                    _fmt_ci(row.get("EM_CI95", "")),
                    _fmt_metric_or_pm(row, "F1"),
                    _fmt_ci(row.get("F1_CI95", "")),
                    _fmt_metric_or_pm(row, "환각률"),
                    _fmt_ci(row.get("환각률_CI95", "")),
                    _fmt_metric_or_pm(row, "커버리지"),
                    _fmt_ci(row.get("커버리지_CI95", "")),
                    _fmt_metric(row.get("선택적정확도", "")),
                    _fmt_latency(row.get("평균지연(ms)", "")),
                    _fmt_latency(row.get("평균지연(ms,warmup제외)", "")),
                    _fmt_metric(row.get("AURC", "")),
                    _fmt_metric(row.get("CSC_Temperature", "")),
                    _fmt_metric(row.get("CSC_ECE_before", "")),
                    _fmt_metric(row.get("CSC_ECE_after", "")),
                    _fmt_metric(row.get("CSC_ECE", "")),
                    _fmt_metric(row.get("CSC_Brier_before", "")),
                    _fmt_metric(row.get("CSC_Brier_after", "")),
                    _fmt_metric(row.get("CSC_Brier", "")),
                    _fmt_metric(row.get("CSC_AUROC", "")),
                    _fmt_metric(row.get("CSC_AUPRC", "")),
                    _fmt_metric(row.get("CSC-정답상관_Pearsonr", "")),
                    _fmt_metric(row.get("CSC-정답상관_Spearmanrho", "")),
                    _fmt_metric(row.get("검색점수-충분성_Pearsonr", row.get("검색점수-충분성상관", ""))),
                    _fmt_metric(row.get("검색점수-충분성_Spearmanrho", "")),
                    _fmt_metric(row.get("검색점수-충분성_MI", "")),
                    _fmt_metric(row.get("고검색-불충분비율", "")),
                    _fmt_metric(row.get("저검색-충분비율", "")),
                    _fmt_p(row.get("EM_p값", "")),
                    _fmt_p(row.get("F1_p값", "")),
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def save_markdown(markdown_text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_text, encoding="utf-8")


def summarize_for_report(
    records: List[Dict],
    run_name: str,
    checker: str,
    strategy: str,
    abstain_text: str,
    calibration_bins: int = 10,
    calibration_temperature_scaling: bool = True,
    calibration_val_ratio: float = 0.3,
    calibration_seed: int = 42,
    calibration_min_samples: int = 50,
) -> Tuple[Dict, Dict[str, object]]:
    summary = summarize_records(records, abstain_text=abstain_text)
    row = build_summary_row(summary, run_name=run_name, checker=checker, strategy=strategy)

    checker_norm = str(checker).strip().lower()
    if checker_norm not in {"", "none", "없음"}:
        analysis = checker_quality_analysis(
            records=records,
            n_bins=calibration_bins,
            temperature_scaling=calibration_temperature_scaling,
            temperature_val_ratio=calibration_val_ratio,
            temperature_seed=calibration_seed,
            temperature_min_samples=calibration_min_samples,
        )
        row["AURC"] = float(analysis["aurc"])
        row["CSC_Temperature"] = float(analysis.get("temperature", 1.0))
        row["CSC_ECE_before"] = float(analysis.get("ece_before", analysis.get("ece", 0.0)))
        row["CSC_ECE_after"] = float(analysis.get("ece_after", analysis.get("ece", 0.0)))
        row["CSC_ECE"] = float(analysis["ece"])
        row["CSC_Brier_before"] = float(analysis.get("brier_before", analysis.get("brier", 0.0)))
        row["CSC_Brier_after"] = float(analysis.get("brier_after", analysis.get("brier", 0.0)))
        row["CSC_Brier"] = float(analysis["brier"])
        row["CSC_AUROC"] = float(analysis["auroc"])
        row["CSC_AUPRC"] = float(analysis["auprc"])
        row["CSC-정답상관_Pearsonr"] = float(analysis["score_correct_pearson"])
        row["CSC-정답상관_Spearmanrho"] = float(analysis["score_correct_spearman"])
        row["검색점수-충분성상관"] = float(analysis["retrieval_checker_corr"])
        row["검색점수-충분성_Pearsonr"] = float(analysis["retrieval_checker_pearson"])
        row["검색점수-충분성_Spearmanrho"] = float(analysis["retrieval_checker_spearman"])
        row["검색점수-충분성_MI"] = float(analysis["retrieval_sufficiency_mi"])
        row["고검색-불충분비율"] = float(analysis["high_retrieval_insufficient_rate"])
        row["저검색-충분비율"] = float(analysis["low_retrieval_sufficient_rate"])
        return row, analysis
    return row, {}


def aggregate_seed_rows(
    per_seed_rows: List[Dict],
    baseline_key: str,
    metrics: Sequence[str] = ("EM", "F1", "환각률", "커버리지"),
    extra_mean_metrics: Sequence[str] = (
        "선택적정확도",
        "평균지연(ms)",
        "평균지연(ms,warmup제외)",
        "지연표준편차(ms)",
        "지연P50(ms)",
        "지연P95(ms)",
        "CPU평균지연(ms)",
        "MPS평균지연(ms)",
        "AURC",
        "CSC_Temperature",
        "CSC_ECE_before",
        "CSC_ECE_after",
        "CSC_ECE",
        "CSC_Brier_before",
        "CSC_Brier_after",
        "CSC_Brier",
        "CSC_AUROC",
        "CSC_AUPRC",
        "CSC-정답상관_Pearsonr",
        "CSC-정답상관_Spearmanrho",
        "검색점수-충분성상관",
        "검색점수-충분성_Pearsonr",
        "검색점수-충분성_Spearmanrho",
        "검색점수-충분성_MI",
        "고검색-불충분비율",
        "저검색-충분비율",
    ),
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for row in per_seed_rows:
        key = str(row.get("메서드키", row.get("실험명", "")))
        grouped.setdefault(key, []).append(row)

    baseline_rows = grouped.get(baseline_key, [])
    aggregated: List[Dict] = []
    for key, rows in grouped.items():
        base = dict(rows[0])
        base["실험명"] = key
        base["시드"] = ""
        base["시드수"] = len(rows)
        base["EM_p값"] = ""
        base["F1_p값"] = ""
        for metric in metrics:
            values = [float(r.get(metric, 0.0)) for r in rows]
            ci = bootstrap_mean_ci(values, n_samples=n_bootstrap, confidence_level=confidence_level, seed=seed)
            base[metric] = ci["mean"]
            base[f"{metric}_평균±표준편차"] = f"{ci['mean']:.4f} ± {ci['std']:.4f}"
            base[f"{metric}_CI95"] = f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"

        for metric in extra_mean_metrics:
            vals: List[float] = []
            for row in rows:
                value = row.get(metric, "")
                if value == "":
                    continue
                try:
                    vals.append(float(value))
                except (TypeError, ValueError):
                    continue
            if vals:
                base[metric] = float(np.mean(vals))

        if baseline_rows and key != baseline_key and len(rows) == len(baseline_rows):
            base_em = [float(r.get("EM", 0.0)) for r in baseline_rows]
            cand_em = [float(r.get("EM", 0.0)) for r in rows]
            base_f1 = [float(r.get("F1", 0.0)) for r in baseline_rows]
            cand_f1 = [float(r.get("F1", 0.0)) for r in rows]

            em_diff = bootstrap_mean_diff_ci(
                baseline_values=base_em,
                candidate_values=cand_em,
                n_samples=n_bootstrap,
                confidence_level=confidence_level,
                seed=seed,
            )
            f1_diff = bootstrap_mean_diff_ci(
                baseline_values=base_f1,
                candidate_values=cand_f1,
                n_samples=n_bootstrap,
                confidence_level=confidence_level,
                seed=seed,
            )
            base["EM_차이_CI95"] = f"[{em_diff['ci_low']:.4f}, {em_diff['ci_high']:.4f}]"
            base["F1_차이_CI95"] = f"[{f1_diff['ci_low']:.4f}, {f1_diff['ci_high']:.4f}]"
        else:
            base["EM_차이_CI95"] = ""
            base["F1_차이_CI95"] = ""

        aggregated.append(base)

    return sorted(aggregated, key=lambda x: str(x.get("실험명", "")))


def _fmt_p(value) -> str:
    if value == "":
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_metric(value) -> str:
    if value == "":
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_latency(value) -> str:
    if value == "":
        return "-"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_metric_or_pm(row: Dict, key: str) -> str:
    pm_key = f"{key}_평균±표준편차"
    if pm_key in row and str(row.get(pm_key, "")).strip():
        return str(row.get(pm_key))
    return _fmt_metric(row.get(key, ""))


def _fmt_ci(value) -> str:
    text = str(value).strip()
    return text if text else "-"
