from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from evaluation.evaluator import save_markdown, save_summary_csv
from pipeline import RAGPipeline, load_config


def _parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _parse_str_list(text: str) -> List[str]:
    return [tok.strip() for tok in str(text).split(",") if tok.strip()]


def _dataset_label(config: Dict) -> str:
    name = str(config.get("dataset", {}).get("name", "")).lower().strip()
    if "hotpot" in name:
        return "HotpotQA"
    if "2wiki" in name:
        return "2WikiMultiHopQA"
    return name or "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heuristic+Abstain k-sweep (k=3,5,7) 다중 시드 실행")
    parser.add_argument(
        "--configs",
        type=str,
        default=f"{ROOT / 'configs' / 'default.yaml'},{ROOT / 'configs' / '2wiki.yaml'}",
        help="콤마 구분 설정 파일 경로",
    )
    parser.add_argument("--run-name", type=str, default="ablation_v8_k_sweep_heuristic_abstain")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--k-values", type=str, default="3,5,7")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def _aggregate(per_seed_rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(per_seed_rows)
    grouped = []
    for (dataset, k), g in df.groupby(["데이터셋", "k"], sort=True):
        f1_vals = g["F1"].astype(float).to_numpy()
        hall_vals = g["환각률"].astype(float).to_numpy()
        cov_vals = g["커버리지"].astype(float).to_numpy()
        lat_vals = g["평균지연(ms)"].astype(float).to_numpy()
        grouped.append(
            {
                "Dataset": str(dataset),
                "k": int(k),
                "F1": float(np.mean(f1_vals)),
                "F1_std": float(np.std(f1_vals, ddof=1)) if len(f1_vals) > 1 else 0.0,
                "Hallucination": float(np.mean(hall_vals)),
                "Hallucination_std": float(np.std(hall_vals, ddof=1)) if len(hall_vals) > 1 else 0.0,
                "Coverage": float(np.mean(cov_vals)),
                "Coverage_std": float(np.std(cov_vals, ddof=1)) if len(cov_vals) > 1 else 0.0,
                "Latency(ms)": float(np.mean(lat_vals)),
                "Latency_std": float(np.std(lat_vals, ddof=1)) if len(lat_vals) > 1 else 0.0,
            }
        )
    return pd.DataFrame(grouped).sort_values(["Dataset", "k"]).reset_index(drop=True)


def _to_markdown_table(df: pd.DataFrame) -> str:
    lines = [
        "| Dataset | k | F1 | Hallucination | Coverage | Latency(ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            "| {ds} | {k} | {f1:.4f} ± {f1s:.4f} | {h:.4f} ± {hs:.4f} | {c:.4f} ± {cs:.4f} | {l:.1f} ± {ls:.1f} |".format(
                ds=str(r["Dataset"]),
                k=int(r["k"]),
                f1=float(r["F1"]),
                f1s=float(r["F1_std"]),
                h=float(r["Hallucination"]),
                hs=float(r["Hallucination_std"]),
                c=float(r["Coverage"]),
                cs=float(r["Coverage_std"]),
                l=float(r["Latency(ms)"]),
                ls=float(r["Latency_std"]),
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_paths = [Path(p) for p in _parse_str_list(args.configs)]
    seeds = _parse_int_list(args.seeds)
    k_values = _parse_int_list(args.k_values)
    if not config_paths:
        raise ValueError("--configs는 최소 1개 이상 필요합니다.")
    if not seeds:
        raise ValueError("--seeds는 최소 1개 이상 필요합니다.")
    if not k_values:
        raise ValueError("--k-values는 최소 1개 이상 필요합니다.")

    per_seed_rows: List[Dict] = []
    output_dir: Path | None = None

    for cfg_path in config_paths:
        base_cfg = load_config(str(cfg_path))
        dataset = _dataset_label(base_cfg)
        dataset_slug = dataset.lower().replace(" ", "_")

        for seed in seeds:
            config = load_config(str(cfg_path))
            config.setdefault("run", {})["seed"] = int(seed)
            if args.output_dir is not None:
                config["run"]["output_dir"] = str(args.output_dir)

            pipeline = RAGPipeline(config=config, project_root=ROOT)
            if output_dir is None:
                output_dir = pipeline.output_dir

            for k in k_values:
                run_name = f"{args.run_name}_{dataset_slug}_seed{seed}_k{k}_heuristic_abstain"
                print(f"[실행] dataset={dataset} seed={seed} k={k}")
                row, _, _ = pipeline.run_experiment(
                    run_name=run_name,
                    strategy_mode="abstain",
                    checker_name="heuristic",
                    k_initial=int(k),
                )
                row["데이터셋"] = dataset
                row["k"] = int(k)
                row["메서드키"] = f"heuristic_abstain_k={int(k)}"
                row["시드"] = int(seed)
                per_seed_rows.append(row)

    if output_dir is None:
        output_dir = Path(args.output_dir if args.output_dir else (ROOT / "outputs"))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_csv = output_dir / f"{args.run_name}_per_seed_summary.csv"
    save_summary_csv(per_seed_rows, per_seed_csv)

    agg_df = _aggregate(per_seed_rows)
    agg_csv = output_dir / f"{args.run_name}_summary.csv"
    agg_md = output_dir / f"{args.run_name}_summary.md"
    agg_df.to_csv(agg_csv, index=False)
    save_markdown(_to_markdown_table(agg_df), agg_md)

    print(f"[저장] 시드별 요약: {per_seed_csv}")
    print(f"[저장] 집계 요약 CSV: {agg_csv}")
    print(f"[저장] 집계 요약 MD: {agg_md}")


if __name__ == "__main__":
    main()
