from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import sys
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from evaluation.evaluator import aggregate_seed_rows, make_markdown_table, save_markdown, save_summary_csv
from evaluation.metrics import bootstrap_mean_ci, paired_bootstrap_test
from pipeline import RAGPipeline, load_config


def _parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation 실험 실행 (3-seed 통계/Calibration/ROC 포함)")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--threshold-sweep", type=str, default="0.2,0.35,0.5,0.65")
    parser.add_argument("--k-sweep", type=str, default="3,5,7")
    parser.add_argument("--run-name", type=str, default="ablations_v3")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--autorater-preflight-samples", type=int, default=20)
    parser.add_argument("--autorater-min-parse-success", type=float, default=0.30)
    parser.add_argument("--autorater-force-run", action="store_true")
    return parser.parse_args()


def _autorater_preflight(pipeline: RAGPipeline, n_samples: int) -> Dict:
    n = max(0, min(int(n_samples), len(pipeline.examples)))
    if n == 0:
        return {
            "표본수": 0,
            "파싱성공수": 0,
            "파싱실패수": 0,
            "파싱성공률": 0.0,
            "파싱방식분포": {},
            "라벨분포": {},
        }

    checker = pipeline._build_checker("autorater", checker_overrides=None)
    parse_fail = 0
    parse_method_counts: Counter = Counter()
    label_counts: Counter = Counter()

    for sample in pipeline.examples[:n]:
        docs = pipeline._retrieve(sample["question"], pipeline.k_initial)
        contexts = [d["text"] for d in docs]
        label, _, meta = checker.predict(sample["question"], contexts)

        label_counts[str(label)] += 1
        if isinstance(meta, dict):
            parse_method = str(meta.get("파싱방식", "unknown")).strip() or "unknown"
            parse_method_counts[parse_method] += 1
            if "파싱오류" in meta:
                parse_fail += 1
        else:
            parse_fail += 1
            parse_method_counts["meta_missing"] += 1

    parse_success = n - parse_fail
    return {
        "표본수": n,
        "파싱성공수": parse_success,
        "파싱실패수": parse_fail,
        "파싱성공률": parse_success / max(1, n),
        "파싱방식분포": dict(parse_method_counts),
        "라벨분포": dict(label_counts),
    }


def _method_key(
    checker: str | None,
    strategy: str,
    th: float | None = None,
    k: int | None = None,
    uncertainty_metric: str | None = None,
    sc_n: int | None = None,
) -> str:
    if checker is None and strategy == "baseline" and k is None:
        return "baseline"
    if strategy == "uncertainty_abstain":
        metric = str(uncertainty_metric or "avg_token_prob").strip().lower()
        return f"uncertainty_abstain_{metric}"
    if checker == "self_consistency" and sc_n is not None:
        return f"self_consistency_n{int(sc_n)}_{strategy}"
    if k is not None:
        return f"baseline_k={k}"
    if th is not None:
        return f"heuristic_abstain_th={th:.2f}"
    return f"{checker}_{strategy}"


def _run_seed(
    seed: int,
    args: argparse.Namespace,
    config: Dict,
    checkers: List[str],
) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    config = dict(config)
    config["run"] = dict(config.get("run", {}))
    config["run"]["seed"] = int(seed)
    if args.max_questions is not None:
        config.setdefault("dataset", {})["max_questions"] = int(args.max_questions)
    if args.output_dir is not None:
        config["run"]["output_dir"] = args.output_dir

    pipeline = RAGPipeline(config=config, project_root=ROOT)
    rows: List[Dict] = []
    records_by_key: Dict[str, List[Dict]] = {}
    base_run_name = f"{args.run_name}_seed{seed}_baseline"
    base_row, baseline_records, _ = pipeline.run_experiment(
        run_name=base_run_name,
        strategy_mode="baseline",
        checker_name=None,
    )
    base_row["메서드키"] = "baseline"
    base_row["시드"] = seed
    rows.append(base_row)
    records_by_key["baseline"] = baseline_records

    unc_cfg = config.get("strategy", {}).get("uncertainty", {})
    uncertainty_runs = [
        (
            "avg_token_prob",
            float(unc_cfg.get("logprob_threshold", unc_cfg.get("threshold", 0.20))),
        ),
        (
            "entropy_confidence",
            float(unc_cfg.get("entropy_threshold", unc_cfg.get("threshold", 0.20))),
        ),
    ]
    for metric, threshold in uncertainty_runs:
        run_name = f"{args.run_name}_seed{seed}_uncertainty_{metric}_abstain"
        row, records, _ = pipeline.run_experiment(
            run_name=run_name,
            strategy_mode="uncertainty_abstain",
            checker_name=None,
            strategy_overrides={"metric": metric, "threshold": threshold},
            baseline_records=baseline_records,
        )
        key = _method_key(checker=None, strategy="uncertainty_abstain", uncertainty_metric=metric)
        row["전략"] = f"uncertainty_abstain({metric},th={threshold:.3f})"
        row["메서드키"] = key
        row["시드"] = seed
        rows.append(row)
        records_by_key[key] = records

    for checker in checkers:
        for strategy in ["abstain", "reretrieve"]:
            run_name = f"{args.run_name}_seed{seed}_{checker}_{strategy}"
            row, records, _ = pipeline.run_experiment(
                run_name=run_name,
                strategy_mode=strategy,
                checker_name=checker,
                baseline_records=baseline_records,
            )
            key = _method_key(checker=checker, strategy=strategy)
            row["메서드키"] = key
            row["시드"] = seed
            rows.append(row)
            records_by_key[key] = records

    # Self-consistency 안정성 분석용 추가 실행 (예: n=3 vs 기본 n=5)
    sc_cfg = config.get("sufficiency", {}).get("self_consistency", {})
    sc_stab_cfg = sc_cfg.get("stability", {})
    sc_default_n = int(sc_cfg.get("n_samples", 5))
    sc_stability_enabled = bool(sc_stab_cfg.get("enabled", True))
    sc_compare_samples = _parse_int_list(",".join(str(x) for x in sc_stab_cfg.get("compare_samples", [3, 5])))
    if sc_stability_enabled and "self_consistency" in checkers:
        for n in sc_compare_samples:
            if int(n) == int(sc_default_n):
                continue
            run_name = f"{args.run_name}_seed{seed}_self_consistency_n{int(n)}_abstain"
            row, records, _ = pipeline.run_experiment(
                run_name=run_name,
                strategy_mode="abstain",
                checker_name="self_consistency",
                checker_overrides={"n_samples": int(n)},
                baseline_records=baseline_records,
            )
            key = _method_key(checker="self_consistency", strategy="abstain", sc_n=int(n))
            row["전략"] = f"abstain(n={int(n)})"
            row["메서드키"] = key
            row["시드"] = seed
            rows.append(row)
            records_by_key[key] = records

    for th in _parse_float_list(args.threshold_sweep):
        run_name = f"{args.run_name}_seed{seed}_heuristic_th_{str(th).replace('.', '_')}"
        row, records, _ = pipeline.run_experiment(
            run_name=run_name,
            strategy_mode="abstain",
            checker_name="heuristic",
            checker_overrides={"min_coverage_ratio": th},
            baseline_records=baseline_records,
        )
        key = _method_key(checker="heuristic", strategy="abstain", th=th)
        row["전략"] = f"abstain(th={th:.2f})"
        row["메서드키"] = key
        row["시드"] = seed
        rows.append(row)
        records_by_key[key] = records

    for k in _parse_int_list(args.k_sweep):
        run_name = f"{args.run_name}_seed{seed}_k_{k}"
        row, records, _ = pipeline.run_experiment(
            run_name=run_name,
            strategy_mode="baseline",
            checker_name=None,
            k_initial=k,
            baseline_records=baseline_records,
        )
        key = _method_key(checker=None, strategy="baseline", k=k)
        row["전략"] = f"baseline(k={k})"
        row["메서드키"] = key
        row["시드"] = seed
        rows.append(row)
        records_by_key[key] = records

    return rows, records_by_key


def _attach_aggregated_pvalues(
    agg_rows: List[Dict],
    records_by_key: Dict[str, List[Dict]],
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
) -> List[Dict]:
    out_rows: List[Dict] = []
    baseline_records = records_by_key.get("baseline", [])
    base_em = [float(r.get("em", 0.0)) for r in baseline_records]
    base_f1 = [float(r.get("f1", 0.0)) for r in baseline_records]

    for row in agg_rows:
        key = str(row.get("메서드키", row.get("실험명", "")))
        new_row = dict(row)

        if key == "baseline":
            new_row["EM_차이"] = ""
            new_row["F1_차이"] = ""
            new_row["EM_p값"] = ""
            new_row["F1_p값"] = ""
            out_rows.append(new_row)
            continue

        if not baseline_records or key not in records_by_key:
            new_row["EM_p값"] = ""
            new_row["F1_p값"] = ""
            out_rows.append(new_row)
            continue

        cand_records = records_by_key[key]
        cand_em = [float(r.get("em", 0.0)) for r in cand_records]
        cand_f1 = [float(r.get("f1", 0.0)) for r in cand_records]

        if len(cand_em) != len(base_em):
            new_row["EM_p값"] = ""
            new_row["F1_p값"] = ""
            out_rows.append(new_row)
            continue

        em_test = paired_bootstrap_test(
            baseline_values=base_em,
            candidate_values=cand_em,
            n_samples=n_bootstrap,
            confidence_level=confidence_level,
            seed=seed,
        )
        f1_test = paired_bootstrap_test(
            baseline_values=base_f1,
            candidate_values=cand_f1,
            n_samples=n_bootstrap,
            confidence_level=confidence_level,
            seed=seed,
        )
        new_row["EM_차이"] = em_test["observed_diff"]
        new_row["F1_차이"] = f1_test["observed_diff"]
        new_row["EM_p값"] = em_test["p_value"]
        new_row["F1_p값"] = f1_test["p_value"]
        out_rows.append(new_row)

    return out_rows


def _build_retrieval_sufficiency_markdown(rows: List[Dict]) -> str:
    checker_rows = [r for r in rows if str(r.get("체커", "")).strip().lower() not in {"", "none", "없음"}]
    if not checker_rows:
        return "체커 기반 실험이 없어 sufficiency-vs-retrieval 분석을 생략했습니다."

    lines = [
        "### Sufficiency vs Retrieval Quality 분리 분석",
        "",
        "| 방법 | 고검색-불충분비율 | 저검색-충분비율 | Pearson r | Spearman ρ | MI(retrieval;sufficiency) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in checker_rows:
        lines.append(
            "| "
            f"{row.get('실험명','')} | "
            f"{float(row.get('고검색-불충분비율', 0.0)):.4f} | "
            f"{float(row.get('저검색-충분비율', 0.0)):.4f} | "
            f"{float(row.get('검색점수-충분성_Pearsonr', row.get('검색점수-충분성상관', 0.0))):.4f} | "
            f"{float(row.get('검색점수-충분성_Spearmanrho', 0.0)):.4f} | "
            f"{float(row.get('검색점수-충분성_MI', 0.0)):.4f} |"
        )
    return "\n".join(lines)


def _build_uncertainty_comparison_markdown(rows: List[Dict]) -> str:
    preferred_keys = {
        "baseline",
        "heuristic_abstain",
        "uncertainty_abstain_avg_token_prob",
        "uncertainty_abstain_entropy_confidence",
    }
    selected = []
    for row in rows:
        key = str(row.get("메서드키", ""))
        if key in preferred_keys:
            selected.append(row)

    if not selected:
        return "불확실성 baseline 비교 대상 행이 없어 표 생성을 생략했습니다."

    lines = [
        "### 불확실성 기반 Abstention 비교",
        "",
        "| Method | F1 | Hallucination | AURC |",
        "|---|---:|---:|---:|",
    ]
    for row in selected:
        aurc_val = row.get("AURC", "")
        aurc_txt = f"{float(aurc_val):.4f}" if aurc_val != "" else "-"
        lines.append(
            "| "
            f"{row.get('실험명', '')} | "
            f"{float(row.get('F1', 0.0)):.4f} | "
            f"{float(row.get('환각률', 0.0)):.4f} | "
            f"{aurc_txt} |"
        )
    return "\n".join(lines)


def _to_float(value) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_calibration_before_after_markdown(rows: List[Dict]) -> str:
    checker_rows = [r for r in rows if str(r.get("체커", "")).strip().lower() not in {"", "none", "없음"}]
    if not checker_rows:
        return "Calibration 전/후 비교 대상이 없어 생략했습니다."

    lines = [
        "### Calibration 전/후 비교 (Temperature Scaling)",
        "",
        "| Method | ECE(before) | ECE(after) | ΔECE(after-before) | Temperature |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in checker_rows:
        ece_b = _to_float(row.get("CSC_ECE_before"))
        ece_a = _to_float(row.get("CSC_ECE_after"))
        temp = _to_float(row.get("CSC_Temperature"))
        delta = (ece_a - ece_b) if (ece_a is not None and ece_b is not None) else None
        ece_b_txt = f"{ece_b:.4f}" if ece_b is not None else "-"
        ece_a_txt = f"{ece_a:.4f}" if ece_a is not None else "-"
        delta_txt = f"{delta:.4f}" if delta is not None else "-"
        temp_txt = f"{temp:.4f}" if temp is not None else "-"
        lines.append(f"| {row.get('실험명','')} | {ece_b_txt} | {ece_a_txt} | {delta_txt} | {temp_txt} |")
    return "\n".join(lines)


def _build_aurc_main_markdown(rows: List[Dict]) -> str:
    row_map = {str(r.get("메서드키", "")): r for r in rows}
    order = [
        "baseline",
        "heuristic_abstain",
        "self_consistency_abstain",
        "uncertainty_abstain_avg_token_prob",
        "uncertainty_abstain_entropy_confidence",
    ]
    selected = [row_map[k] for k in order if k in row_map]
    if not selected:
        return "AURC 메인 비교 대상이 없어 생략했습니다."

    baseline_row = row_map.get("baseline")
    baseline_aurc = _to_float(baseline_row.get("AURC")) if baseline_row else None
    baseline_name = "baseline"
    if baseline_aurc is None:
        alt = row_map.get("uncertainty_abstain_avg_token_prob")
        alt_aurc = _to_float(alt.get("AURC")) if alt else None
        if alt_aurc is not None:
            baseline_aurc = alt_aurc
            baseline_name = "uncertainty_abstain_avg_token_prob(대체)"

    lines = [
        "### AURC 메인 비교",
        "",
        f"- ΔAURC 기준: `{baseline_name}`",
        "",
        "| Method | AURC↓ | ΔAURC vs Baseline |",
        "|---|---:|---:|",
    ]
    for row in selected:
        key = str(row.get("메서드키", ""))
        aurc = _to_float(row.get("AURC"))
        if aurc is None:
            aurc_txt = "-"
            delta_txt = "-"
        else:
            aurc_txt = f"{aurc:.4f}"
            delta_txt = f"{(aurc - baseline_aurc):.4f}" if baseline_aurc is not None else "-"
        if key == "baseline" and baseline_aurc is None:
            delta_txt = "-"
        lines.append(f"| {row.get('실험명','')} | {aurc_txt} | {delta_txt} |")
    return "\n".join(lines)


def _dataset_display_name(dataset_name: str) -> str:
    name = str(dataset_name or "").strip().lower()
    if name in {"hotpotqa", "hotpot_qa"}:
        return "HotpotQA"
    if name in {"2wiki", "2wikimultihopqa", "2wiki_multihop_qa"}:
        return "2WikiMultiHopQA"
    if name in {"natural_questions", "nq"}:
        return "Natural Questions"
    return dataset_name or "Dataset"


def _build_tau_sweep_main_markdown(rows: List[Dict], threshold_list: List[float], dataset_name: str) -> str:
    ds = _dataset_display_name(dataset_name)
    row_map = {str(r.get("메서드키", "")): r for r in rows}
    lines = [
        f"### Threshold τ Sweep ({ds} 기준: F1/환각률/AURC)",
        "",
        "| τ | F1 | Hallucination | AURC |",
        "|---:|---:|---:|---:|",
    ]
    has = False
    for th in sorted(set(float(x) for x in threshold_list)):
        key = f"heuristic_abstain_th={th:.2f}"
        row = row_map.get(key)
        if row is None:
            continue
        has = True
        f1 = _to_float(row.get("F1"))
        hall = _to_float(row.get("환각률"))
        aurc = _to_float(row.get("AURC"))
        f1_txt = f"{f1:.4f}" if f1 is not None else "-"
        hall_txt = f"{hall:.4f}" if hall is not None else "-"
        aurc_txt = f"{aurc:.4f}" if aurc is not None else "-"
        lines.append(f"| {th:.2f} | {f1_txt} | {hall_txt} | {aurc_txt} |")
    if not has:
        return "τ sweep 결과가 없어 메인 표 생성을 생략했습니다."
    return "\n".join(lines)


def _self_consistency_n_from_key(key: str, records: List[Dict]) -> int:
    if key == "self_consistency_abstain":
        if records:
            meta = records[0].get("checker_meta", {})
            if isinstance(meta, dict):
                samples = meta.get("샘플답변", [])
                if isinstance(samples, list) and samples:
                    return int(len(samples))
        return 5
    m = re.search(r"self_consistency_n(\d+)_abstain", key)
    if m:
        return int(m.group(1))
    return -1


def _build_self_consistency_stability_markdown(
    rows: List[Dict],
    records_by_key: Dict[str, List[Dict]],
) -> str:
    row_map = {str(r.get("메서드키", "")): r for r in rows}
    sc_keys: List[str] = []
    for key in row_map.keys():
        if key == "self_consistency_abstain" or key.startswith("self_consistency_n"):
            if key.endswith("_abstain"):
                sc_keys.append(key)
    sc_keys = [k for k in sc_keys if k in records_by_key]
    if not sc_keys:
        return "Self-consistency n 비교 대상이 없어 생략했습니다."

    rows_out: List[Dict] = []
    for key in sc_keys:
        recs = records_by_key.get(key, [])
        n_val = _self_consistency_n_from_key(key, recs)
        if n_val <= 0:
            continue
        disagree = []
        scores = []
        for r in recs:
            scores.append(float(r.get("checker_score", 0.0)))
            meta = r.get("checker_meta", {})
            if isinstance(meta, dict):
                try:
                    disagree.append(float(meta.get("불일치도", 0.0)))
                except (TypeError, ValueError):
                    pass
        src = row_map[key]
        ci = bootstrap_mean_ci(scores, n_samples=1000, confidence_level=0.95, seed=42) if scores else {
            "mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0
        }
        dis_ci = bootstrap_mean_ci(disagree, n_samples=1000, confidence_level=0.95, seed=42) if disagree else {
            "mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0
        }
        rows_out.append(
            {
                "n": int(n_val),
                "f1": float(src.get("F1", 0.0)),
                "hall": float(src.get("환각률", 0.0)),
                "cov": float(src.get("커버리지", 0.0)),
                "score_std": float(ci["std"]),
                "disagree_mean": float(dis_ci["mean"]),
                "disagree_std": float(dis_ci["std"]),
            }
        )

    if not rows_out:
        return "Self-consistency n 비교 결과가 없어 생략했습니다."
    rows_out = sorted(rows_out, key=lambda x: x["n"])
    lines = [
        "### Self-consistency 안정성 분석 (n=3 vs n=5)",
        "",
        "| n_samples | F1 | Hallucination | Coverage | Checker score std | Disagreement mean | Disagreement std |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in rows_out:
        lines.append(
            f"| {x['n']} | {x['f1']:.4f} | {x['hall']:.4f} | {x['cov']:.4f} | "
            f"{x['score_std']:.4f} | {x['disagree_mean']:.4f} | {x['disagree_std']:.4f} |"
        )

    n3 = next((x for x in rows_out if x["n"] == 3), None)
    n5 = next((x for x in rows_out if x["n"] == 5), None)
    if n3 and n5:
        def _pct_reduction(old: float, new: float) -> float:
            if abs(old) < 1e-12:
                return 0.0
            return (old - new) / abs(old) * 100.0

        score_red = _pct_reduction(n3["score_std"], n5["score_std"])
        dis_red = _pct_reduction(n3["disagree_std"], n5["disagree_std"])
        lines.append("")
        lines.append(
            f"- 분산 감소(n=3→5): score std {score_red:.2f}% 감소, "
            f"disagreement std {dis_red:.2f}% 감소"
        )
    return "\n".join(lines)


def _build_latency_device_markdown(
    rows: List[Dict],
    records_by_key: Dict[str, List[Dict]],
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
) -> str:
    row_map = {str(r.get("메서드키", "")): r for r in rows}
    key_order = [
        "baseline",
        "heuristic_abstain",
        "self_consistency_abstain",
        "uncertainty_abstain_avg_token_prob",
        "uncertainty_abstain_entropy_confidence",
    ]
    for key in sorted(row_map.keys()):
        if key.startswith("self_consistency_n") and key.endswith("_abstain"):
            key_order.append(key)

    selected = [k for k in key_order if k in row_map and k in records_by_key]
    if not selected:
        return "Latency 분리 분석 대상이 없어 생략했습니다."

    lines = [
        "### Latency 분석 (CPU/MPS 분리, warm-up 제거, 95% CI)",
        "",
        "| Method | warmup제외 평균(ms) | 전체지연 평균 95% CI | 지연 표준편차(ms) | CPU 평균(ms) | MPS 평균(ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in selected:
        row = row_map[key]
        recs = records_by_key[key]
        lat_all = [float(r.get("latency_ms", 0.0)) for r in recs]
        ci = bootstrap_mean_ci(
            lat_all,
            n_samples=int(n_bootstrap),
            confidence_level=float(confidence_level),
            seed=int(seed),
        ) if lat_all else {"ci_low": 0.0, "ci_high": 0.0, "std": 0.0}

        cpu_vals = [float(r.get("latency_ms", 0.0)) for r in recs if str(r.get("generator_device", "")).lower() == "cpu"]
        mps_vals = [float(r.get("latency_ms", 0.0)) for r in recs if str(r.get("generator_device", "")).lower() == "mps"]
        cpu_mean = sum(cpu_vals) / len(cpu_vals) if cpu_vals else None
        mps_mean = sum(mps_vals) / len(mps_vals) if mps_vals else None

        warmup_mean = _to_float(row.get("평균지연(ms,warmup제외)"))
        std_ms = _to_float(row.get("지연표준편차(ms)"))
        ci_txt = f"[{float(ci['ci_low']):.1f}, {float(ci['ci_high']):.1f}]"
        warm_txt = f"{warmup_mean:.1f}" if warmup_mean is not None else "-"
        std_txt = f"{std_ms:.1f}" if std_ms is not None else "-"
        cpu_txt = f"{cpu_mean:.1f}" if cpu_mean is not None else "-"
        mps_txt = f"{mps_mean:.1f}" if mps_mean is not None else "-"
        lines.append(f"| {row.get('실험명','')} | {warm_txt} | {ci_txt} | {std_txt} | {cpu_txt} | {mps_txt} |")
    return "\n".join(lines)


def _build_csc_accuracy_correlation_markdown(rows: List[Dict]) -> str:
    checker_rows = [r for r in rows if str(r.get("체커", "")).strip().lower() not in {"", "none", "없음"}]
    if not checker_rows:
        return "CSC-정답 상관 분석 대상이 없어 생략했습니다."

    lines = [
        "### CSC 점수 vs 최종 정답 상관 분석",
        "",
        "| 방법 | Pearson r | Spearman ρ |",
        "|---|---:|---:|",
    ]
    for row in checker_rows:
        lines.append(
            "| "
            f"{row.get('실험명','')} | "
            f"{float(row.get('CSC-정답상관_Pearsonr', 0.0)):.4f} | "
            f"{float(row.get('CSC-정답상관_Spearmanrho', 0.0)):.4f} |"
        )
    return "\n".join(lines)


def _sample_cost(record: Dict, w_h: float, w_a: float, w_l: float, latency_scale_ms: float) -> float:
    is_correct = int(record.get("is_correct", 0))
    is_abstain = int(record.get("is_abstain", 0))
    latency = float(record.get("latency_ms", 0.0))
    hallucination = 1 if (is_correct == 0 and is_abstain == 0) else 0
    latency_term = latency / max(1e-6, float(latency_scale_ms))
    return float(w_h * hallucination + w_a * is_abstain + w_l * latency_term)


def _build_policy_optimization_markdown(
    agg_rows: List[Dict],
    records_by_key: Dict[str, List[Dict]],
    threshold_list: List[float],
    config: Dict,
) -> str:
    policy_cfg = config.get("strategy", {}).get("policy_optimization", {})
    if not bool(policy_cfg.get("enabled", True)):
        return "정책 최적화 분석이 비활성화되어 생략했습니다."

    w_h = float(policy_cfg.get("hallucination_weight", 1.0))
    w_a = float(policy_cfg.get("abstain_weight", 0.3))
    w_l = float(policy_cfg.get("latency_weight", 0.0))
    latency_scale = float(policy_cfg.get("latency_scale_ms", 1000.0))

    boot_cfg = config.get("evaluation", {}).get("bootstrap", {})
    n_boot = int(boot_cfg.get("n_samples", 1000))
    conf = float(boot_cfg.get("confidence_level", 0.95))
    seed = int(config.get("run", {}).get("seed", 42))

    row_map = {str(r.get("메서드키", "")): r for r in agg_rows}
    eval_rows: List[Dict] = []
    for th in threshold_list:
        key = f"heuristic_abstain_th={th:.2f}"
        if key not in records_by_key or key not in row_map:
            continue
        costs = [_sample_cost(r, w_h=w_h, w_a=w_a, w_l=w_l, latency_scale_ms=latency_scale) for r in records_by_key[key]]
        if not costs:
            continue
        ci = bootstrap_mean_ci(costs, n_samples=n_boot, confidence_level=conf, seed=seed)
        src = row_map[key]
        eval_rows.append(
            {
                "threshold": th,
                "cost_mean": float(ci["mean"]),
                "cost_ci_low": float(ci["ci_low"]),
                "cost_ci_high": float(ci["ci_high"]),
                "f1": float(src.get("F1", 0.0)),
                "hall": float(src.get("환각률", 0.0)),
                "cov": float(src.get("커버리지", 0.0)),
            }
        )

    if not eval_rows:
        return "정책 최적화용 threshold sweep 결과가 없어 생략했습니다."

    best = min(eval_rows, key=lambda x: x["cost_mean"])
    lines = [
        "### 정책 최적화 (Threshold τ Sweep)",
        "",
        f"- 비용함수: `L = {w_h:.3f}*Hallucination + {w_a:.3f}*Abstain + {w_l:.3f}*(Latency/{latency_scale:.1f}ms)`",
        f"- 최적 임계값: `τ* = {best['threshold']:.2f}` (평균비용={best['cost_mean']:.4f}, 95% CI=[{best['cost_ci_low']:.4f}, {best['cost_ci_high']:.4f}])",
        "",
        "| τ | 비용 평균 | 비용 95% CI | F1 | 환각률 | 커버리지 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(eval_rows, key=lambda x: x["threshold"]):
        lines.append(
            f"| {r['threshold']:.2f} | {r['cost_mean']:.4f} | [{r['cost_ci_low']:.4f}, {r['cost_ci_high']:.4f}] | "
            f"{r['f1']:.4f} | {r['hall']:.4f} | {r['cov']:.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_name = str(config.get("dataset", {}).get("name", "dataset"))

    effective_max_questions = int(
        args.max_questions if args.max_questions is not None else config.get("dataset", {}).get("max_questions", 0)
    )
    if 0 < effective_max_questions < 500:
        print(
            f"[경고] 현재 max_questions={effective_max_questions} 입니다. "
            "유의성 검정 안정성을 위해 500 이상을 권장합니다."
        )

    seeds = _parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("--seeds는 최소 1개 이상 필요합니다. 예: --seeds 42,43,44")
    print(f"[설정] 반복 시드: {seeds}")

    checkers = ["heuristic", "autorater", "self_consistency"]
    entailment_enabled = bool(config.get("sufficiency", {}).get("entailment", {}).get("enabled", False))
    if entailment_enabled:
        checkers.append("entailment")

    if "autorater" in checkers and int(args.autorater_preflight_samples) > 0:
        preview_cfg = dict(config)
        preview_cfg["run"] = dict(config.get("run", {}))
        preview_cfg["run"]["seed"] = int(seeds[0])
        if args.max_questions is not None:
            preview_cfg.setdefault("dataset", {})["max_questions"] = int(args.max_questions)
        if args.output_dir is not None:
            preview_cfg["run"]["output_dir"] = args.output_dir
        preview_pipeline = RAGPipeline(config=preview_cfg, project_root=ROOT)

        threshold = max(0.0, min(1.0, float(args.autorater_min_parse_success)))
        try:
            preflight = _autorater_preflight(preview_pipeline, n_samples=int(args.autorater_preflight_samples))
            print(
                "[autorater 사전점검] "
                f"표본={preflight['표본수']}, 파싱성공={preflight['파싱성공수']}, "
                f"파싱실패={preflight['파싱실패수']}, 파싱성공률={preflight['파싱성공률']:.3f}"
            )
            print(f"[autorater 사전점검] 파싱방식분포={preflight['파싱방식분포']}")
            if preflight["파싱성공률"] < threshold and not bool(args.autorater_force_run):
                print("[autorater 사전점검] 임계값 미만이므로 autorater 실험을 제외합니다.")
                checkers = [c for c in checkers if c != "autorater"]
        except Exception as exc:
            if bool(args.autorater_force_run):
                print(f"[autorater 사전점검 경고] 점검 실패했지만 강제 실행합니다: {exc}")
            else:
                print(f"[autorater 사전점검] 점검 실패로 autorater 실험을 제외합니다: {exc}")
                checkers = [c for c in checkers if c != "autorater"]

    all_rows: List[Dict] = []
    records_all: Dict[str, List[Dict]] = {}

    for seed in seeds:
        print(f"\n[시드 실행] seed={seed}")
        seed_rows, seed_records = _run_seed(seed=seed, args=args, config=config, checkers=checkers)
        all_rows.extend(seed_rows)
        for key, records in seed_records.items():
            records_all.setdefault(key, []).extend(records)

    output_dir = Path(
        args.output_dir if args.output_dir else config.get("run", {}).get("output_dir", str(ROOT / "outputs"))
    )
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_csv = output_dir / f"{args.run_name}_per_seed_summary.csv"
    save_summary_csv(all_rows, per_seed_csv)
    print(f"[저장] 시드별 요약 CSV: {per_seed_csv}")

    boot_cfg = config.get("evaluation", {}).get("bootstrap", {})
    agg_rows = aggregate_seed_rows(
        per_seed_rows=all_rows,
        baseline_key="baseline",
        n_bootstrap=int(boot_cfg.get("n_samples", 2000)),
        confidence_level=float(boot_cfg.get("confidence_level", 0.95)),
        seed=int(seeds[0]),
    )
    agg_rows = _attach_aggregated_pvalues(
        agg_rows=agg_rows,
        records_by_key=records_all,
        n_bootstrap=int(boot_cfg.get("n_samples", 2000)),
        confidence_level=float(boot_cfg.get("confidence_level", 0.95)),
        seed=int(seeds[0]),
    )
    agg_md = make_markdown_table(agg_rows)

    agg_csv_path = output_dir / f"{args.run_name}_summary.csv"
    agg_md_path = output_dir / f"{args.run_name}_summary.md"
    save_summary_csv(agg_rows, agg_csv_path)
    save_markdown(agg_md, agg_md_path)
    print(f"[저장] 집계 요약 CSV: {agg_csv_path}")
    print(f"[저장] 집계 요약 Markdown: {agg_md_path}")

    calibration_md = _build_calibration_before_after_markdown(agg_rows)
    calibration_md_path = output_dir / f"{args.run_name}_calibration_before_after.md"
    save_markdown(calibration_md, calibration_md_path)
    print(f"[저장] Calibration 전/후 비교표: {calibration_md_path}")

    aurc_main_md = _build_aurc_main_markdown(agg_rows)
    aurc_main_md_path = output_dir / f"{args.run_name}_aurc_main.md"
    save_markdown(aurc_main_md, aurc_main_md_path)
    print(f"[저장] AURC 메인 비교표: {aurc_main_md_path}")

    retrieval_analysis = _build_retrieval_sufficiency_markdown(agg_rows)
    retrieval_md_path = output_dir / f"{args.run_name}_retrieval_vs_sufficiency.md"
    save_markdown(retrieval_analysis, retrieval_md_path)
    print(f"[저장] Sufficiency vs Retrieval 분석: {retrieval_md_path}")

    uncertainty_cmp = _build_uncertainty_comparison_markdown(agg_rows)
    uncertainty_md_path = output_dir / f"{args.run_name}_uncertainty_comparison.md"
    save_markdown(uncertainty_cmp, uncertainty_md_path)
    print(f"[저장] Uncertainty 비교표: {uncertainty_md_path}")

    corr_md = _build_csc_accuracy_correlation_markdown(agg_rows)
    corr_md_path = output_dir / f"{args.run_name}_csc_accuracy_correlation.md"
    save_markdown(corr_md, corr_md_path)
    print(f"[저장] CSC-정답 상관 분석: {corr_md_path}")

    tau_main_md = _build_tau_sweep_main_markdown(
        agg_rows,
        _parse_float_list(args.threshold_sweep),
        dataset_name=dataset_name,
    )
    tau_main_md_path = output_dir / f"{args.run_name}_tau_sweep_main.md"
    save_markdown(tau_main_md, tau_main_md_path)
    print(f"[저장] τ sweep 메인 비교표: {tau_main_md_path}")

    sc_stability_md = _build_self_consistency_stability_markdown(agg_rows, records_all)
    sc_stability_md_path = output_dir / f"{args.run_name}_self_consistency_stability.md"
    save_markdown(sc_stability_md, sc_stability_md_path)
    print(f"[저장] self-consistency 안정성 분석: {sc_stability_md_path}")

    latency_md = _build_latency_device_markdown(
        rows=agg_rows,
        records_by_key=records_all,
        n_bootstrap=int(boot_cfg.get("n_samples", 2000)),
        confidence_level=float(boot_cfg.get("confidence_level", 0.95)),
        seed=int(seeds[0]),
    )
    latency_md_path = output_dir / f"{args.run_name}_latency_device_ci.md"
    save_markdown(latency_md, latency_md_path)
    print(f"[저장] 지연 분석 요약표(CPU/MPS, CI): {latency_md_path}")

    policy_opt = _build_policy_optimization_markdown(
        agg_rows=agg_rows,
        records_by_key=records_all,
        threshold_list=_parse_float_list(args.threshold_sweep),
        config=config,
    )
    policy_md_path = output_dir / f"{args.run_name}_policy_optimization.md"
    save_markdown(policy_opt, policy_md_path)
    print(f"[저장] 정책 최적화 분석: {policy_md_path}")

    print("\n[최종 결과] 3-seed 집계 한글 마크다운 표")
    print(agg_md)
    print("\n" + calibration_md)
    print("\n" + aurc_main_md)
    print("\n" + retrieval_analysis)
    print("\n" + uncertainty_cmp)
    print("\n" + corr_md)
    print("\n" + tau_main_md)
    print("\n" + sc_stability_md)
    print("\n" + latency_md)
    print("\n" + policy_opt)


if __name__ == "__main__":
    main()
