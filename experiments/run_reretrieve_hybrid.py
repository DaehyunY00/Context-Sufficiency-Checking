from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from evaluation.evaluator import aggregate_seed_rows, make_markdown_table, save_markdown, save_summary_csv
from pipeline import RAGPipeline, load_config


def _parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heuristic Reretrieve/Hybrid(3→6→9) 전용 실행")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--k-initial", type=int, default=3)
    parser.add_argument("--k-reretrieve", type=int, default=6)
    parser.add_argument("--k-reretrieve-second", type=int, default=9)
    return parser.parse_args()


def _row_with_meta(row: Dict, seed: int, method_key: str, strategy_label: str) -> Dict:
    out = dict(row)
    out["메서드키"] = method_key
    out["시드"] = int(seed)
    out["체커"] = "heuristic"
    out["전략"] = strategy_label
    return out


def main() -> None:
    args = parse_args()
    seeds = _parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("--seeds는 최소 1개 이상 필요합니다.")

    all_rows: List[Dict] = []
    records_all: Dict[str, List[Dict]] = {}

    for seed in seeds:
        config = load_config(args.config)
        config.setdefault("run", {})["seed"] = int(seed)
        if args.output_dir is not None:
            config.setdefault("run", {})["output_dir"] = str(args.output_dir)
        if args.max_questions is not None:
            config.setdefault("dataset", {})["max_questions"] = int(args.max_questions)

        retrieval_cfg = config.setdefault("retrieval", {})
        retrieval_cfg["top_k_initial"] = int(args.k_initial)
        retrieval_cfg["top_k_reretrieve"] = int(args.k_reretrieve)

        print(
            f"[시드 실행] seed={seed} | k: {args.k_initial}->{args.k_reretrieve}->{args.k_reretrieve_second}"
        )
        pipeline = RAGPipeline(config=config, project_root=ROOT)

        # Heuristic + Reretrieve (3->6->9)
        rerun_name = f"{args.run_name}_seed{seed}_heuristic_reretrieve"
        rerow, rerecords, _ = pipeline.run_experiment(
            run_name=rerun_name,
            strategy_mode="reretrieve",
            checker_name="heuristic",
            strategy_overrides={"k_reretrieve_second": int(args.k_reretrieve_second)},
            k_initial=int(args.k_initial),
            k_reretrieve=int(args.k_reretrieve),
        )
        rekey = f"heuristic_reretrieve_{args.k_initial}_{args.k_reretrieve}_{args.k_reretrieve_second}"
        all_rows.append(_row_with_meta(rerow, seed=seed, method_key=rekey, strategy_label="reretrieve(3->6->9)"))
        records_all.setdefault(rekey, []).extend(rerecords)

        # Heuristic + Hybrid (3->6->9, insufficient then abstain)
        hyrun_name = f"{args.run_name}_seed{seed}_heuristic_hybrid"
        hyrow, hyrecords, _ = pipeline.run_experiment(
            run_name=hyrun_name,
            strategy_mode="hybrid",
            checker_name="heuristic",
            strategy_overrides={"k_reretrieve_second": int(args.k_reretrieve_second)},
            k_initial=int(args.k_initial),
            k_reretrieve=int(args.k_reretrieve),
        )
        hykey = f"heuristic_hybrid_{args.k_initial}_{args.k_reretrieve}_{args.k_reretrieve_second}"
        all_rows.append(_row_with_meta(hyrow, seed=seed, method_key=hykey, strategy_label="hybrid(3->6->9)"))
        records_all.setdefault(hykey, []).extend(hyrecords)

    output_dir = Path(
        args.output_dir if args.output_dir else load_config(args.config).get("run", {}).get("output_dir", str(ROOT / "outputs"))
    )
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_csv = output_dir / f"{args.run_name}_per_seed_summary.csv"
    save_summary_csv(all_rows, per_seed_csv)

    boot_cfg = load_config(args.config).get("evaluation", {}).get("bootstrap", {})
    agg_rows = aggregate_seed_rows(
        per_seed_rows=all_rows,
        baseline_key=f"heuristic_reretrieve_{args.k_initial}_{args.k_reretrieve}_{args.k_reretrieve_second}",
        n_bootstrap=int(boot_cfg.get("n_samples", 1000)),
        confidence_level=float(boot_cfg.get("confidence_level", 0.95)),
        seed=int(seeds[0]),
    )
    summary_csv = output_dir / f"{args.run_name}_summary.csv"
    summary_md = output_dir / f"{args.run_name}_summary.md"
    save_summary_csv(agg_rows, summary_csv)
    save_markdown(make_markdown_table(agg_rows), summary_md)

    print(f"[저장] 시드별 요약: {per_seed_csv}")
    print(f"[저장] 집계 요약 CSV: {summary_csv}")
    print(f"[저장] 집계 요약 MD: {summary_md}")


if __name__ == "__main__":
    main()
