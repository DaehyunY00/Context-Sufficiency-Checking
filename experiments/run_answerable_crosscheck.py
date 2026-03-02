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
    parser = argparse.ArgumentParser(
        description="Answerable 정의 교차검증(gold_containment vs entailment) 다중 시드 실행"
    )
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--run-name", type=str, default="answerable_crosscheck")
    parser.add_argument("--checker", type=str, default="heuristic", choices=["heuristic", "self_consistency", "autorater", "entailment"])
    parser.add_argument("--strategy", type=str, default="abstain", choices=["abstain", "reretrieve", "hybrid"])
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--entail-model", type=str, default="cross-encoder/nli-distilroberta-base")
    parser.add_argument("--entail-threshold", type=float, default=0.6)
    return parser.parse_args()


def _apply_common_overrides(cfg: Dict, *, seed: int, max_questions: int | None, output_dir: str | None) -> Dict:
    out = deepcopy(cfg)
    out.setdefault("run", {})["seed"] = int(seed)
    if output_dir is not None:
        out.setdefault("run", {})["output_dir"] = str(output_dir)
    if max_questions is not None:
        out.setdefault("dataset", {})["max_questions"] = int(max_questions)
    return out


def main() -> None:
    args = parse_args()
    seeds = _parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("--seeds는 최소 1개 이상 필요합니다.")

    base_cfg = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(str(base_cfg.get("run", {}).get("output_dir", "outputs")))
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = ["gold_containment", "entailment"]
    all_agg_rows: List[Dict] = []

    for mode in modes:
        print(f"\n[교차검증] Answerable 모드={mode} 실행 시작")
        per_seed_rows: List[Dict] = []

        for seed in seeds:
            cfg = _apply_common_overrides(
                cfg=base_cfg,
                seed=seed,
                max_questions=args.max_questions,
                output_dir=str(output_dir),
            )
            answerable_cfg = cfg.setdefault("evaluation", {}).setdefault("answerable", {})
            answerable_cfg["mode"] = mode
            if mode == "entailment":
                answerable_cfg["entail_model_name"] = str(args.entail_model)
                answerable_cfg["entail_prob_threshold"] = float(args.entail_threshold)

            pipeline = RAGPipeline(config=cfg, project_root=ROOT)
            prefix = f"{args.run_name}_{mode}_seed{seed}"

            base_row, base_records, _ = pipeline.run_experiment(
                run_name=f"{prefix}_baseline",
                strategy_mode="baseline",
                checker_name=None,
            )
            base_row["메서드키"] = "baseline"
            base_row["시드"] = seed
            per_seed_rows.append(base_row)

            cand_row, _, _ = pipeline.run_experiment(
                run_name=f"{prefix}_{args.checker}_{args.strategy}",
                strategy_mode=args.strategy,
                checker_name=args.checker,
                baseline_records=base_records,
            )
            cand_row["메서드키"] = f"{args.checker}_{args.strategy}"
            cand_row["시드"] = seed
            per_seed_rows.append(cand_row)

        agg_rows = aggregate_seed_rows(
            per_seed_rows=per_seed_rows,
            baseline_key="baseline",
            n_bootstrap=int(base_cfg.get("evaluation", {}).get("bootstrap", {}).get("n_samples", 2000)),
            confidence_level=float(base_cfg.get("evaluation", {}).get("bootstrap", {}).get("confidence_level", 0.95)),
            seed=int(seeds[0]),
        )
        for row in agg_rows:
            row["실험명"] = f"{mode}::{row.get('실험명', '')}"
            row["메서드키"] = f"{mode}::{row.get('메서드키', row.get('실험명', ''))}"
        all_agg_rows.extend(agg_rows)

    summary_csv = output_dir / f"{args.run_name}_summary.csv"
    summary_md = output_dir / f"{args.run_name}_summary.md"
    save_summary_csv(all_agg_rows, summary_csv)
    save_markdown(make_markdown_table(all_agg_rows), summary_md)
    print(f"[저장] 요약 CSV: {summary_csv}")
    print(f"[저장] 요약 Markdown: {summary_md}")

    # 본문/부록 삽입용 간단 비교표
    compare_lines = [
        "| Answerable 정의 | 메서드 | F1 | 환각률 | 커버리지 | AURC | CSC_AUROC | CSC_ECE_after |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_agg_rows:
        method_key = str(row.get("메서드키", ""))
        if "baseline" in method_key:
            continue
        mode = "entailment" if method_key.startswith("entailment::") else "gold_containment"
        method = str(row.get("체커", ""))
        compare_lines.append(
            "| {mode} | {method} | {f1:.4f} | {hall:.4f} | {cov:.4f} | {aurc:.4f} | {auroc:.4f} | {ece:.4f} |".format(
                mode=mode,
                method=method,
                f1=float(row.get("F1", 0.0)),
                hall=float(row.get("환각률", 0.0)),
                cov=float(row.get("커버리지", 0.0)),
                aurc=float(row.get("AURC", 0.0)) if str(row.get("AURC", "")) != "" else 0.0,
                auroc=float(row.get("CSC_AUROC", 0.0)) if str(row.get("CSC_AUROC", "")) != "" else 0.0,
                ece=float(row.get("CSC_ECE_after", 0.0)) if str(row.get("CSC_ECE_after", "")) != "" else 0.0,
            )
        )

    compare_md = output_dir / f"{args.run_name}_compare.md"
    compare_md.write_text("\n".join(compare_lines) + "\n", encoding="utf-8")
    print(f"[저장] 비교 Markdown: {compare_md}")


if __name__ == "__main__":
    main()
