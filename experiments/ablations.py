from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from evaluation.evaluator import make_markdown_table, save_markdown, save_summary_csv
from pipeline import RAGPipeline, load_config


def _parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation 실험 실행")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--threshold-sweep", type=str, default="0.2,0.35,0.5,0.65")
    parser.add_argument("--k-sweep", type=str, default="3,5,7")
    parser.add_argument("--run-name", type=str, default="ablations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.max_questions is not None:
        config.setdefault("dataset", {})["max_questions"] = int(args.max_questions)
    if args.output_dir is not None:
        config.setdefault("run", {})["output_dir"] = args.output_dir

    pipeline = RAGPipeline(config=config, project_root=ROOT)
    rows = []

    # 1) baseline
    base_row, baseline_records, _ = pipeline.run_experiment(
        run_name=f"{args.run_name}_baseline",
        strategy_mode="baseline",
        checker_name=None,
    )
    rows.append(base_row)

    # 2) baseline vs 각 checker + abstain/reretrieve
    checkers = ["heuristic", "autorater", "self_consistency"]
    entailment_enabled = bool(config.get("sufficiency", {}).get("entailment", {}).get("enabled", False))
    if entailment_enabled:
        checkers.append("entailment")

    for checker in checkers:
        for strategy in ["abstain", "reretrieve"]:
            run_name = f"{args.run_name}_{checker}_{strategy}"
            row, _, _ = pipeline.run_experiment(
                run_name=run_name,
                strategy_mode=strategy,
                checker_name=checker,
                baseline_records=baseline_records,
            )
            rows.append(row)

    # 3) threshold sweep (heuristic)
    for th in _parse_float_list(args.threshold_sweep):
        run_name = f"{args.run_name}_heuristic_th_{str(th).replace('.', '_')}"
        row, _, _ = pipeline.run_experiment(
            run_name=run_name,
            strategy_mode="abstain",
            checker_name="heuristic",
            checker_overrides={
                "min_coverage_ratio": th,
            },
            baseline_records=baseline_records,
        )
        row["전략"] = f"abstain(th={th:.2f})"
        rows.append(row)

    # 4) k sweep (baseline)
    for k in _parse_int_list(args.k_sweep):
        run_name = f"{args.run_name}_k_{k}"
        row, _, _ = pipeline.run_experiment(
            run_name=run_name,
            strategy_mode="baseline",
            checker_name=None,
            k_initial=k,
            baseline_records=baseline_records,
        )
        row["전략"] = f"baseline(k={k})"
        rows.append(row)

    markdown = make_markdown_table(rows)
    report_cfg = config.get("report", {})

    if bool(report_cfg.get("save_csv", True)):
        csv_path = pipeline.output_dir / f"{args.run_name}_summary.csv"
        save_summary_csv(rows, csv_path)
        print(f"[저장] CSV 요약: {csv_path}")

    if bool(report_cfg.get("save_markdown", True)):
        md_path = pipeline.output_dir / f"{args.run_name}_summary.md"
        save_markdown(markdown, md_path)
        print(f"[저장] Markdown 요약: {md_path}")

    print("\n[최종 결과] Ablation 한글 마크다운 표")
    print(markdown)


if __name__ == "__main__":
    main()
