from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys
from typing import List

ROOT = Path(__file__).resolve().parents[1]


def _parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Llama3-8B-Instruct 기준 10-seed 실험 스위트 실행기"
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="configs/llama3_8b_hotpot.yaml,configs/llama3_8b_2wiki.yaml,configs/llama3_8b_musique.yaml,configs/llama3_8b_strategyqa.yaml",
        help="실행할 config 파일 목록(콤마 구분)",
    )
    parser.add_argument("--run-prefix", type=str, default="llama3_seed10")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--checkers", type=str, default="heuristic,autorater,self_consistency")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--include-bm25-threshold-baseline", action="store_true")
    parser.add_argument("--include-random-matched-baseline", action="store_true")
    parser.add_argument("--skip-answerable-crosscheck", action="store_true")
    parser.add_argument("--skip-retriever-generalization", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run(cmd: List[str], dry_run: bool) -> None:
    print("[실행명령] " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    args = parse_args()
    cfg_paths = [str((ROOT / c).resolve()) if not Path(c).is_absolute() else c for c in _parse_csv(args.configs)]
    out_dir = str((ROOT / args.output_dir).resolve()) if not Path(args.output_dir).is_absolute() else args.output_dir

    for cfg in cfg_paths:
        dataset_tag = Path(cfg).stem.replace("llama3_8b_", "")
        run_name = f"{args.run_prefix}_{dataset_tag}"
        cmd = [
            sys.executable,
            str(ROOT / "experiments" / "ablations.py"),
            "--config",
            cfg,
            "--run-name",
            run_name,
            "--seeds",
            args.seeds,
            "--checkers",
            args.checkers,
            "--output-dir",
            out_dir,
        ]
        if args.max_questions is not None:
            cmd += ["--max-questions", str(args.max_questions)]
        if args.include_bm25_threshold_baseline:
            cmd += ["--include-bm25-threshold-baseline"]
        if args.include_random_matched_baseline:
            cmd += ["--include-random-matched-baseline"]
        _run(cmd, dry_run=bool(args.dry_run))

    # 교차검증/리트리버 일반화는 Hotpot config를 기준으로 실행
    hotpot_cfg = next((c for c in cfg_paths if "hotpot" in Path(c).stem.lower()), "")
    if hotpot_cfg and not args.skip_answerable_crosscheck:
        cmd = [
            sys.executable,
            str(ROOT / "experiments" / "run_answerable_crosscheck.py"),
            "--config",
            hotpot_cfg,
            "--run-name",
            f"{args.run_prefix}_answerable_cross_hotpot",
            "--checker",
            "heuristic",
            "--strategy",
            "abstain",
            "--seeds",
            args.seeds,
            "--output-dir",
            out_dir,
        ]
        if args.max_questions is not None:
            cmd += ["--max-questions", str(args.max_questions)]
        _run(cmd, dry_run=bool(args.dry_run))

    if hotpot_cfg and not args.skip_retriever_generalization:
        cmd = [
            sys.executable,
            str(ROOT / "experiments" / "run_retriever_generalization.py"),
            "--config",
            hotpot_cfg,
            "--run-name",
            f"{args.run_prefix}_retriever_generalization_hotpot",
            "--checker",
            "heuristic",
            "--strategy",
            "abstain",
            "--seeds",
            args.seeds,
            "--output-dir",
            out_dir,
        ]
        if args.max_questions is not None:
            cmd += ["--max-questions", str(args.max_questions)]
        _run(cmd, dry_run=bool(args.dry_run))

    print("[완료] Llama3-8B 10-seed 실험 스위트 명령 실행 완료")


if __name__ == "__main__":
    main()

