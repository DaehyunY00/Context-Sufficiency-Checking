from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from evaluation.metrics import roc_pr_diagnostics
from pipeline import RAGPipeline, load_config


def _parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


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
    headers = ["T", "Diversity", "Var(g_SC)", "AUROC"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['T']:.1f} | "
            f"{row['Diversity']:.4f} | "
            f"{row['Var(g_SC)']:.6f} | "
            f"{row['AUROC']:.4f} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-consistency 온도 민감도 분석")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temps", type=str, default="0.3,0.5,0.7,1.0")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--max-questions", type=int, default=120)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default="self_consistency_temp_sensitivity_hotpot_seed42")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temps = _parse_float_list(args.temps)

    config = load_config(args.config)
    config.setdefault("run", {})["seed"] = int(args.seed)
    config.setdefault("dataset", {})["name"] = "hotpotqa"
    if args.max_questions is not None:
        config.setdefault("dataset", {})["max_questions"] = int(args.max_questions)
    if args.output_dir is not None:
        config.setdefault("run", {})["output_dir"] = str(args.output_dir)

    pipeline = RAGPipeline(config=config, project_root=ROOT)
    cases = _prepare_cases(pipeline)
    print(f"[정보] HotpotQA seed={args.seed} 샘플수={len(cases)}")

    rows: List[Dict] = []
    for temp in temps:
        print(f"[실행] Self-consistency T={temp:.1f} n={args.n_samples}")
        checker = pipeline._build_checker(
            "self_consistency",
            {"temperature": float(temp), "n_samples": int(args.n_samples)},
        )

        scores: List[float] = []
        labels: List[int] = []
        diversity_values: List[float] = []

        for idx, (question, contexts, oracle_answerable) in enumerate(cases, start=1):
            _, score, meta = checker.predict(question, contexts)
            scores.append(float(np.clip(float(score), 0.0, 1.0)))
            labels.append(int(oracle_answerable))

            norm_answers = []
            if isinstance(meta, dict):
                raw = meta.get("정규화답변", [])
                if isinstance(raw, list):
                    norm_answers = [str(x) for x in raw]
            unique_ratio = len(set(norm_answers)) / max(1, len(norm_answers))
            diversity_values.append(float(unique_ratio))

            if idx % 20 == 0:
                print(f"[진행] T={temp:.1f}: {idx}/{len(cases)}")

        diag = roc_pr_diagnostics(y_true=labels, y_score=scores)
        rows.append(
            {
                "T": float(temp),
                "Diversity": float(np.mean(diversity_values)) if diversity_values else 0.0,
                "Var(g_SC)": float(np.var(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "AUROC": float(diag.get("auroc", 0.0)),
                "샘플수": len(cases),
                "seed": int(args.seed),
                "n_samples": int(args.n_samples),
            }
        )

    output_root = Path(config.get("run", {}).get("output_dir", "outputs"))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / f"{args.run_name}.csv"
    _save_csv(rows, csv_path)
    md_text = _build_markdown(rows)
    md_path = output_root / f"{args.run_name}.md"
    md_path.write_text(md_text, encoding="utf-8")

    print(f"[저장] CSV: {csv_path}")
    print(f"[저장] MD: {md_path}")
    print("\n[결과표] Self-consistency Temperature 민감도")
    print(md_text)


if __name__ == "__main__":
    main()
