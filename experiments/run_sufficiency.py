from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from evaluation.evaluator import make_markdown_table, save_markdown, save_summary_csv
from pipeline import RAGPipeline, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sufficiency 전략 실행")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--run-name", type=str, default="sufficiency_run")
    parser.add_argument("--checker", type=str, default=None, choices=["heuristic", "autorater", "self_consistency", "entailment"])
    parser.add_argument("--strategy", type=str, default=None, choices=["abstain", "reretrieve", "hybrid", "uncertainty_abstain"])
    parser.add_argument(
        "--uncertainty-metric",
        type=str,
        default=None,
        choices=["avg_token_prob", "entropy_confidence", "avg_logprob"],
    )
    parser.add_argument("--uncertainty-threshold", type=float, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.max_questions is not None:
        config.setdefault("dataset", {})["max_questions"] = int(args.max_questions)
    if args.output_dir is not None:
        config.setdefault("run", {})["output_dir"] = args.output_dir
    if args.seed is not None:
        config.setdefault("run", {})["seed"] = int(args.seed)

    checker = args.checker or str(config.get("sufficiency", {}).get("checker", "heuristic"))
    strategy = args.strategy or str(config.get("strategy", {}).get("mode", "abstain"))
    strategy_overrides = {}
    if args.uncertainty_metric is not None:
        strategy_overrides["metric"] = args.uncertainty_metric
    if args.uncertainty_threshold is not None:
        strategy_overrides["threshold"] = float(args.uncertainty_threshold)

    pipeline = RAGPipeline(config=config, project_root=ROOT)

    baseline_name = f"{args.run_name}_baseline"
    base_row, base_records, _ = pipeline.run_experiment(
        run_name=baseline_name,
        strategy_mode="baseline",
        checker_name=None,
    )

    cand_row, _, paths = pipeline.run_experiment(
        run_name=args.run_name,
        strategy_mode=strategy,
        checker_name=(None if strategy == "uncertainty_abstain" else checker),
        strategy_overrides=(strategy_overrides if strategy_overrides else None),
        baseline_records=base_records,
    )

    rows = [base_row, cand_row]
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

    print("\n[결과표] Baseline vs Sufficiency")
    print(markdown)
    print(f"\n[저장] 샘플 JSONL: {paths['jsonl_path']}")


if __name__ == "__main__":
    main()
