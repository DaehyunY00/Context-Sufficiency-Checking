from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from evaluation.evaluator import (  # noqa: E402
    aggregate_seed_rows,
    latency_quality_analysis,
    make_markdown_table,
    save_markdown,
    save_summary_csv,
    summarize_for_report,
)
from evaluation.metrics import paired_bootstrap_test  # noqa: E402
from pipeline import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="기존 JSONL 로그 기반 재집계")
    parser.add_argument("--dataset", choices=["hotpot", "2wiki"], required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--hotpot-prefix", type=str, default="ablations_v3")
    parser.add_argument("--two-wiki-prefix", type=str, default="two_wiki_heuristic_abstain")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--config-hotpot", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--config-2wiki", type=str, default=str(ROOT / "configs" / "2wiki.yaml"))
    return parser.parse_args()


def _parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _attach_aggregated_pvalues(
    agg_rows: List[Dict],
    records_by_key: Dict[str, List[Dict]],
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
) -> List[Dict]:
    out: List[Dict] = []
    baseline_records = records_by_key.get("baseline", [])
    base_em = [_to_float(r.get("em", 0.0)) for r in baseline_records]
    base_f1 = [_to_float(r.get("f1", 0.0)) for r in baseline_records]

    for row in agg_rows:
        key = str(row.get("메서드키", row.get("실험명", "")))
        new_row = dict(row)
        if key == "baseline":
            new_row["EM_차이"] = ""
            new_row["F1_차이"] = ""
            new_row["EM_p값"] = ""
            new_row["F1_p값"] = ""
            out.append(new_row)
            continue
        cand = records_by_key.get(key, [])
        cand_em = [_to_float(r.get("em", 0.0)) for r in cand]
        cand_f1 = [_to_float(r.get("f1", 0.0)) for r in cand]
        if not baseline_records or len(base_em) != len(cand_em):
            new_row["EM_p값"] = ""
            new_row["F1_p값"] = ""
            out.append(new_row)
            continue
        em_test = paired_bootstrap_test(
            baseline_values=base_em,
            candidate_values=cand_em,
            n_samples=int(n_bootstrap),
            confidence_level=float(confidence_level),
            seed=int(seed),
        )
        f1_test = paired_bootstrap_test(
            baseline_values=base_f1,
            candidate_values=cand_f1,
            n_samples=int(n_bootstrap),
            confidence_level=float(confidence_level),
            seed=int(seed),
        )
        new_row["EM_차이"] = em_test["observed_diff"]
        new_row["F1_차이"] = f1_test["observed_diff"]
        new_row["EM_p값"] = em_test["p_value"]
        new_row["F1_p값"] = f1_test["p_value"]
        out.append(new_row)
    return out


def _hotpot_file_specs(prefix: str, seed: int) -> List[Tuple[str, str, str, Path]]:
    outs = ROOT / "outputs"
    specs: List[Tuple[str, str, str, Path]] = [
        ("baseline", "없음", "baseline", outs / f"{prefix}_seed{seed}_baseline.jsonl"),
        ("heuristic_abstain", "heuristic", "abstain", outs / f"{prefix}_seed{seed}_heuristic_abstain.jsonl"),
        ("heuristic_reretrieve", "heuristic", "reretrieve", outs / f"{prefix}_seed{seed}_heuristic_reretrieve.jsonl"),
        ("self_consistency_abstain", "self_consistency", "abstain", outs / f"{prefix}_seed{seed}_self_consistency_abstain.jsonl"),
        (
            "self_consistency_n3_abstain",
            "self_consistency",
            "abstain(n=3)",
            outs / f"{prefix}_seed{seed}_self_consistency_n3_abstain.jsonl",
        ),
        ("self_consistency_reretrieve", "self_consistency", "reretrieve", outs / f"{prefix}_seed{seed}_self_consistency_reretrieve.jsonl"),
        (
            "uncertainty_avg_token_prob_abstain",
            "uncertainty_baseline",
            "uncertainty_abstain(avg_token_prob,th=0.200)",
            outs / f"{prefix}_seed{seed}_uncertainty_avg_token_prob_abstain.jsonl",
        ),
        (
            "uncertainty_entropy_confidence_abstain",
            "uncertainty_baseline",
            "uncertainty_abstain(entropy_confidence,th=0.350)",
            outs / f"{prefix}_seed{seed}_uncertainty_entropy_confidence_abstain.jsonl",
        ),
        ("autorater_abstain", "autorater", "abstain", outs / f"{prefix}_seed{seed}_autorater_abstain.jsonl"),
        ("autorater_reretrieve", "autorater", "reretrieve", outs / f"{prefix}_seed{seed}_autorater_reretrieve.jsonl"),
        ("heuristic_abstain_th=0.20", "heuristic", "abstain(th=0.20)", outs / f"{prefix}_seed{seed}_heuristic_th_0_2.jsonl"),
        ("heuristic_abstain_th=0.35", "heuristic", "abstain(th=0.35)", outs / f"{prefix}_seed{seed}_heuristic_th_0_35.jsonl"),
        ("heuristic_abstain_th=0.50", "heuristic", "abstain(th=0.50)", outs / f"{prefix}_seed{seed}_heuristic_th_0_5.jsonl"),
        ("heuristic_abstain_th=0.65", "heuristic", "abstain(th=0.65)", outs / f"{prefix}_seed{seed}_heuristic_th_0_65.jsonl"),
        ("baseline_k=3", "없음", "baseline(k=3)", outs / f"{prefix}_seed{seed}_k_3.jsonl"),
        ("baseline_k=5", "없음", "baseline(k=5)", outs / f"{prefix}_seed{seed}_k_5.jsonl"),
        ("baseline_k=7", "없음", "baseline(k=7)", outs / f"{prefix}_seed{seed}_k_7.jsonl"),
    ]
    return specs


def _two_wiki_file_specs(prefix: str, seed: int) -> List[Tuple[str, str, str, Path]]:
    outs = ROOT / "outputs"
    if seed == 42:
        base = outs / f"{prefix}_baseline.jsonl"
        heur = outs / f"{prefix}.jsonl"
    else:
        base = outs / f"{prefix}_seed{seed}_baseline.jsonl"
        heur = outs / f"{prefix}_seed{seed}.jsonl"
    return [
        ("baseline", "없음", "baseline", base),
        ("heuristic_abstain", "heuristic", "abstain", heur),
    ]


def _reaggregate(
    dataset: str,
    run_name: str,
    seeds: List[int],
    output_dir: Path,
    hotpot_prefix: str,
    two_wiki_prefix: str,
    cfg_path: Path,
) -> None:
    config = load_config(str(cfg_path))
    abstain_text = str(config.get("generator", {}).get("abstain_text", "모르겠습니다."))
    eval_cfg = config.get("evaluation", {})
    temp_cfg = eval_cfg.get("calibration", {}).get("temperature_scaling", {})
    boot_cfg = eval_cfg.get("bootstrap", {})
    latency_cfg = eval_cfg.get("latency", {})

    per_seed_rows: List[Dict] = []
    records_by_key: Dict[str, List[Dict]] = {}

    for seed in seeds:
        if dataset == "hotpot":
            specs = _hotpot_file_specs(prefix=hotpot_prefix, seed=seed)
        else:
            specs = _two_wiki_file_specs(prefix=two_wiki_prefix, seed=seed)

        for key, checker, strategy, path in specs:
            if not path.exists():
                continue
            records = _load_jsonl(path)
            if not records:
                continue
            row, _ = summarize_for_report(
                records=records,
                run_name=f"{run_name}_seed{seed}_{key}",
                checker=checker,
                strategy=strategy,
                abstain_text=abstain_text,
                calibration_bins=int(eval_cfg.get("calibration_bins", 10)),
                calibration_temperature_scaling=bool(temp_cfg.get("enabled", True)),
                calibration_val_ratio=float(temp_cfg.get("validation_ratio", 0.3)),
                calibration_seed=int(temp_cfg.get("seed", 42)),
                calibration_min_samples=int(temp_cfg.get("min_samples", 50)),
            )
            lat = latency_quality_analysis(
                records=records,
                warmup_drop=int(latency_cfg.get("warmup_drop", 5)),
                hist_bins=int(latency_cfg.get("hist_bins", 20)),
            )
            row["평균지연(ms,warmup제외)"] = float(lat.get("mean_ms_wo_warmup", 0.0))
            row["지연표준편차(ms)"] = float(lat.get("std_ms", 0.0))
            row["지연P50(ms)"] = float(lat.get("p50_ms", 0.0))
            row["지연P95(ms)"] = float(lat.get("p95_ms", 0.0))
            dev = dict(lat.get("devices", {}))
            if "cpu" in dev:
                row["CPU평균지연(ms)"] = float(dev.get("cpu", {}).get("mean_ms", 0.0))
            elif "mps" not in dev:
                # 구버전 로그에는 generator_device가 없어서 장치 분리 통계가 비는 경우가 있다.
                # 이 경우 실험 환경(본 러닝은 CPU) 기준으로 CPU 평균을 전체 평균으로 대체한다.
                row["CPU평균지연(ms)"] = float(np.mean([float(r.get("latency_ms", 0.0)) for r in records]))
            else:
                row["CPU평균지연(ms)"] = ""

            row["MPS평균지연(ms)"] = float(dev.get("mps", {}).get("mean_ms", 0.0)) if "mps" in dev else ""
            row["메서드키"] = key
            row["실험명"] = key
            row["시드"] = seed
            per_seed_rows.append(row)
            records_by_key.setdefault(key, []).extend(records)

    if not per_seed_rows:
        raise RuntimeError(f"재집계할 로그를 찾지 못했습니다. dataset={dataset}")

    per_seed_csv = output_dir / f"{run_name}_per_seed_summary.csv"
    save_summary_csv(per_seed_rows, per_seed_csv)

    agg_rows = aggregate_seed_rows(
        per_seed_rows=per_seed_rows,
        baseline_key="baseline",
        n_bootstrap=int(boot_cfg.get("n_samples", 1000)),
        confidence_level=float(boot_cfg.get("confidence_level", 0.95)),
        seed=int(seeds[0]),
    )
    agg_rows = _attach_aggregated_pvalues(
        agg_rows=agg_rows,
        records_by_key=records_by_key,
        n_bootstrap=int(boot_cfg.get("n_samples", 1000)),
        confidence_level=float(boot_cfg.get("confidence_level", 0.95)),
        seed=int(seeds[0]),
    )
    agg_md = make_markdown_table(agg_rows)
    agg_csv = output_dir / f"{run_name}_summary.csv"
    agg_md_path = output_dir / f"{run_name}_summary.md"
    save_summary_csv(agg_rows, agg_csv)
    save_markdown(agg_md, agg_md_path)

    print(f"[완료] dataset={dataset}")
    print(f"[저장] per-seed: {per_seed_csv}")
    print(f"[저장] summary csv: {agg_csv}")
    print(f"[저장] summary md: {agg_md_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("--seeds는 최소 1개 이상 필요합니다.")

    cfg_path = Path(args.config_hotpot if args.dataset == "hotpot" else args.config_2wiki)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path

    _reaggregate(
        dataset=args.dataset,
        run_name=args.run_name,
        seeds=seeds,
        output_dir=output_dir,
        hotpot_prefix=args.hotpot_prefix,
        two_wiki_prefix=args.two_wiki_prefix,
        cfg_path=cfg_path,
    )


if __name__ == "__main__":
    main()
