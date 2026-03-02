from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from evaluation.metrics import roc_pr_diagnostics
from pipeline import RAGPipeline, load_config


def _parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _dataset_label(config: Dict) -> str:
    return str(config.get("dataset", {}).get("name", "unknown"))


def _prepare_cases(pipeline: RAGPipeline) -> List[Tuple[str, List[str], int]]:
    cases: List[Tuple[str, List[str], int]] = []
    for sample in pipeline.examples:
        docs = pipeline._retrieve(sample["question"], pipeline.k_initial)
        contexts = [d["text"] for d in docs]
        oracle_answerable, _ = pipeline._estimate_oracle_answerable(
            question=sample["question"],
            gold_answer=sample["gold_answer"],
            contexts=contexts,
        )
        cases.append((sample["question"], contexts, int(oracle_answerable)))
    return cases


def _evaluate_variant(
    pipeline: RAGPipeline,
    cases: List[Tuple[str, List[str], int]],
    variant: str,
) -> Dict:
    checker = pipeline._build_checker("heuristic", {"variant": variant})

    y_true: List[int] = []
    y_score: List[float] = []
    latency_ms: List[float] = []
    n = len(cases)

    for idx, (question, contexts, oracle_answerable) in enumerate(cases, start=1):
        ts = time.perf_counter()
        _, score, _ = checker.predict(question, contexts)
        latency_ms.append((time.perf_counter() - ts) * 1000.0)
        y_true.append(int(oracle_answerable))
        y_score.append(float(np.clip(float(score), 0.0, 1.0)))
        if idx % 100 == 0:
            print(f"[진행] heuristic-{variant}: {idx}/{n}")

    diag = roc_pr_diagnostics(y_true=y_true, y_score=y_score)
    return {
        "auroc": float(diag.get("auroc", 0.0)),
        "auprc": float(diag.get("auprc", 0.0)),
        "checker_latency_ms": float(np.mean(latency_ms)) if latency_ms else 0.0,
        "샘플수": n,
    }


def _aggregate_rows(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str], List[Dict]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["variant"]))
        grouped.setdefault(key, []).append(row)

    out: List[Dict] = []
    for (dataset, variant), vals in sorted(grouped.items()):
        aurocs = np.array([float(v["auroc"]) for v in vals], dtype=float)
        lat = np.array([float(v["checker_latency_ms"]) for v in vals], dtype=float)
        out.append(
            {
                "dataset": dataset,
                "variant": variant,
                "seed_count": len(vals),
                "auroc_mean": float(np.mean(aurocs)),
                "auroc_std": float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else 0.0,
                "latency_mean_ms": float(np.mean(lat)),
                "latency_std_ms": float(np.std(lat, ddof=1)) if len(lat) > 1 else 0.0,
            }
        )
    return out


def _build_compact_table(agg_rows: List[Dict]) -> List[Dict]:
    by_variant: Dict[str, Dict[str, Dict]] = {}
    for row in agg_rows:
        by_variant.setdefault(str(row["variant"]), {})[str(row["dataset"]).lower()] = row

    def _pick_dataset(row_map: Dict[str, Dict], needle: str) -> Dict:
        for k, v in row_map.items():
            if needle in k:
                return v
        return {}

    compact: List[Dict] = []
    variants = ["h1", "h2", "h3"]
    for variant in variants:
        row_map = by_variant.get(variant, {})
        hotpot = _pick_dataset(row_map, "hotpot")
        wiki = _pick_dataset(row_map, "2wiki")
        hotpot_lat = float(hotpot.get("latency_mean_ms", 0.0))
        wiki_lat = float(wiki.get("latency_mean_ms", 0.0))
        compact.append(
            {
                "Estimator variant": variant.upper(),
                "AUROC(Hotpot)": (
                    f"{float(hotpot.get('auroc_mean', 0.0)):.4f} ± {float(hotpot.get('auroc_std', 0.0)):.4f}"
                    if hotpot
                    else "-"
                ),
                "AUROC(2Wiki)": (
                    f"{float(wiki.get('auroc_mean', 0.0)):.4f} ± {float(wiki.get('auroc_std', 0.0)):.4f}"
                    if wiki
                    else "-"
                ),
                "_hotpot_latency_ms": hotpot_lat,
                "_wiki_latency_ms": wiki_lat,
            }
        )

    h1_row = next((r for r in compact if r["Estimator variant"] == "H1"), None)
    h1_hotpot = float(h1_row.get("_hotpot_latency_ms", 0.0)) if h1_row else 0.0
    h1_wiki = float(h1_row.get("_wiki_latency_ms", 0.0)) if h1_row else 0.0

    for row in compact:
        delta_hotpot = float(row.get("_hotpot_latency_ms", 0.0)) - h1_hotpot
        delta_wiki = float(row.get("_wiki_latency_ms", 0.0)) - h1_wiki
        row["Latency추가(ms)"] = f"Hotpot {delta_hotpot:+.3f} / 2Wiki {delta_wiki:+.3f}"
        row.pop("_hotpot_latency_ms", None)
        row.pop("_wiki_latency_ms", None)

    return compact


def _save_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_markdown(rows: List[Dict]) -> str:
    if not rows:
        return "결과가 없습니다."
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heuristic 변형(H1/H2/H3) AUROC/지연 비교")
    parser.add_argument("--hotpot-config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--wiki-config", type=str, default=str(ROOT / "configs" / "2wiki.yaml"))
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default="heuristic_variant_ablation_v1")
    parser.add_argument("--max-questions-hotpot", type=int, default=None)
    parser.add_argument("--max-questions-2wiki", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_int_list(args.seeds)
    variants = ["h1", "h2", "h3"]
    per_seed_rows: List[Dict] = []

    configs = [
        ("hotpot", Path(args.hotpot_config), args.max_questions_hotpot),
        ("2wiki", Path(args.wiki_config), args.max_questions_2wiki),
    ]

    for dataset_tag, cfg_path, max_q in configs:
        for seed in seeds:
            config = load_config(cfg_path)
            config.setdefault("run", {})["seed"] = int(seed)
            if args.output_dir is not None:
                config.setdefault("run", {})["output_dir"] = str(args.output_dir)
            if max_q is not None:
                config.setdefault("dataset", {})["max_questions"] = int(max_q)

            print(f"[실행] dataset={dataset_tag} seed={seed}")
            pipeline = RAGPipeline(config=config, project_root=ROOT)
            cases = _prepare_cases(pipeline)
            dataset_name = _dataset_label(config)
            print(f"[정보] dataset={dataset_name} seed={seed} 샘플수={len(cases)}")

            for variant in variants:
                print(f"[실행] {dataset_name} seed={seed} variant={variant}")
                result = _evaluate_variant(pipeline=pipeline, cases=cases, variant=variant)
                per_seed_rows.append(
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "variant": variant,
                        "auroc": result["auroc"],
                        "auprc": result["auprc"],
                        "checker_latency_ms": result["checker_latency_ms"],
                        "샘플수": result["샘플수"],
                    }
                )

    output_root = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(load_config(args.hotpot_config).get("run", {}).get("output_dir", "outputs"))
    )
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    per_seed_csv = output_root / f"{args.run_name}_per_seed.csv"
    _save_csv(per_seed_rows, per_seed_csv)

    agg_rows = _aggregate_rows(per_seed_rows)
    agg_csv = output_root / f"{args.run_name}_aggregate.csv"
    _save_csv(agg_rows, agg_csv)

    compact_rows = _build_compact_table(agg_rows)
    compact_csv = output_root / f"{args.run_name}_table.csv"
    _save_csv(compact_rows, compact_csv)

    markdown = _build_markdown(compact_rows)
    md_path = output_root / f"{args.run_name}_table.md"
    md_path.write_text(markdown, encoding="utf-8")

    print(f"[저장] 시드별 결과: {per_seed_csv}")
    print(f"[저장] 집계 결과: {agg_csv}")
    print(f"[저장] 보고용 표 CSV: {compact_csv}")
    print(f"[저장] 보고용 표 MD: {md_path}")
    print("\n[결과표] Heuristic 변형 비교")
    print(markdown)


if __name__ == "__main__":
    main()
