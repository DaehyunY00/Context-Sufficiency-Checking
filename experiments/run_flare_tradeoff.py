from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from evaluation.evaluator import aggregate_seed_rows, make_markdown_table, save_markdown, save_summary_csv
from pipeline import RAGPipeline, load_config


def _parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLARE-lite vs CSC trade-off 비교 (대규모 샘플)")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--run-name", type=str, default="flare_tradeoff")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument(
        "--max-questions",
        type=int,
        default=3000,
        help="0 이하이면 전체 split 사용, 양수면 해당 개수로 제한",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-heuristic", action="store_true")
    parser.add_argument("--skip-flare", action="store_true")
    return parser.parse_args()


def _apply_seed_config(base_cfg: Dict, seed: int, max_questions: int, output_dir: str | None) -> Dict:
    cfg = deepcopy(base_cfg)
    cfg.setdefault("run", {})["seed"] = int(seed)
    if output_dir is not None:
        cfg.setdefault("run", {})["output_dir"] = str(output_dir)
    if int(max_questions) > 0:
        cfg.setdefault("dataset", {})["max_questions"] = int(max_questions)
    return cfg


def main() -> None:
    args = parse_args()
    seeds = _parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("--seeds는 최소 1개 이상 필요합니다.")

    base_cfg = load_config(args.config)
    if args.output_dir is not None:
        base_cfg.setdefault("run", {})["output_dir"] = str(args.output_dir)
    output_dir = Path(str(base_cfg.get("run", {}).get("output_dir", "outputs")))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: List[Dict] = []
    for seed in seeds:
        cfg = _apply_seed_config(base_cfg, seed=seed, max_questions=int(args.max_questions), output_dir=str(output_dir))
        pipeline = RAGPipeline(config=cfg, project_root=ROOT)

        prefix = f"{args.run_name}_seed{seed}"
        base_row, base_records, _ = pipeline.run_experiment(
            run_name=f"{prefix}_baseline",
            strategy_mode="baseline",
            checker_name=None,
        )
        base_row["메서드키"] = "baseline"
        base_row["시드"] = int(seed)
        per_seed_rows.append(base_row)

        if not bool(args.skip_heuristic):
            heur_row, _, _ = pipeline.run_experiment(
                run_name=f"{prefix}_heuristic_abstain",
                strategy_mode="abstain",
                checker_name="heuristic",
                baseline_records=base_records,
            )
            heur_row["메서드키"] = "heuristic_abstain"
            heur_row["시드"] = int(seed)
            per_seed_rows.append(heur_row)

        if not bool(args.skip_flare):
            flare_row, _, _ = pipeline.run_experiment(
                run_name=f"{prefix}_flare_lite",
                strategy_mode="flare_lite",
                checker_name=None,
                baseline_records=base_records,
            )
            flare_row["메서드키"] = "flare_lite"
            flare_row["시드"] = int(seed)
            per_seed_rows.append(flare_row)

    agg_rows = aggregate_seed_rows(
        per_seed_rows=per_seed_rows,
        baseline_key="baseline",
        n_bootstrap=int(base_cfg.get("evaluation", {}).get("bootstrap", {}).get("n_samples", 1000)),
        confidence_level=float(base_cfg.get("evaluation", {}).get("bootstrap", {}).get("confidence_level", 0.95)),
        seed=int(seeds[0]),
    )

    summary_csv = output_dir / f"{args.run_name}_summary.csv"
    summary_md = output_dir / f"{args.run_name}_summary.md"
    save_summary_csv(agg_rows, summary_csv)
    save_markdown(make_markdown_table(agg_rows), summary_md)

    focus_keys = {"baseline", "heuristic_abstain", "flare_lite"}
    lines = [
        "### FLARE-lite vs CSC Trade-off",
        "",
        "| Method | F1 | Hallucination | Coverage | Latency(ms) | AURC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in agg_rows:
        key = str(row.get("메서드키", ""))
        if key not in focus_keys:
            continue
        lines.append(
            "| {name} | {f1:.4f} | {hall:.4f} | {cov:.4f} | {lat:.1f} | {aurc} |".format(
                name=str(row.get("실험명", key)),
                f1=float(row.get("F1", 0.0)),
                hall=float(row.get("환각률", 0.0)),
                cov=float(row.get("커버리지", 0.0)),
                lat=float(row.get("지연시간(ms)", 0.0)),
                aurc=(
                    f"{float(row.get('AURC', 0.0)):.4f}"
                    if str(row.get("AURC", "")) != ""
                    else "-"
                ),
            )
        )

    tradeoff_md = output_dir / f"{args.run_name}_tradeoff.md"
    save_markdown("\n".join(lines), tradeoff_md)

    print(f"[저장] 요약 CSV: {summary_csv}")
    print(f"[저장] 요약 Markdown: {summary_md}")
    print(f"[저장] Trade-off 비교표: {tradeoff_md}")


if __name__ == "__main__":
    main()

