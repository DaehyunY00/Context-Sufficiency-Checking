from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from evaluation.evaluator import aggregate_seed_rows, make_markdown_table, save_markdown, save_summary_csv
from pipeline import RAGPipeline, load_config


@dataclass
class RetrieverSetting:
    name: str
    overrides: Dict


def _parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="리트리버 일반화 검증(e5 vs DPR vs ColBERT-근사)")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--run-name", type=str, default="retriever_generalization")
    parser.add_argument("--checker", type=str, default="heuristic", choices=["heuristic", "self_consistency", "autorater", "entailment"])
    parser.add_argument("--strategy", type=str, default="abstain", choices=["abstain", "reretrieve", "hybrid"])
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-colbert", action="store_true")
    return parser.parse_args()


def _apply_common_overrides(cfg: Dict, *, seed: int, max_questions: int | None, output_dir: str | None) -> Dict:
    out = deepcopy(cfg)
    out.setdefault("run", {})["seed"] = int(seed)
    if output_dir is not None:
        out.setdefault("run", {})["output_dir"] = str(output_dir)
    if max_questions is not None:
        out.setdefault("dataset", {})["max_questions"] = int(max_questions)
    return out


def _settings(skip_colbert: bool) -> List[RetrieverSetting]:
    base = [
        RetrieverSetting(
            name="e5_small",
            overrides={
                "model_type": "sentence_transformer",
                "embed_model": "intfloat/e5-small-v2",
                "query_model_name": None,
                "corpus_model_name": None,
            },
        ),
        RetrieverSetting(
            name="dpr_nq",
            overrides={
                "model_type": "dpr",
                "embed_model": "facebook/dpr-question_encoder-single-nq-base",
                "query_model_name": "facebook/dpr-question_encoder-single-nq-base",
                "corpus_model_name": "facebook/dpr-ctx_encoder-single-nq-base",
                "max_length": 256,
            },
        ),
    ]
    if not skip_colbert:
        base.append(
            RetrieverSetting(
                name="colbertv2_approx",
                overrides={
                    "model_type": "sentence_transformer",
                    "embed_model": "colbert-ir/colbertv2.0",
                    "query_model_name": None,
                    "corpus_model_name": None,
                },
            )
        )
    return base


def main() -> None:
    args = parse_args()
    seeds = _parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("--seeds는 최소 1개 이상 필요합니다.")

    base_cfg = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(str(base_cfg.get("run", {}).get("output_dir", "outputs")))
    output_dir.mkdir(parents=True, exist_ok=True)

    all_agg_rows: List[Dict] = []
    failures: List[Dict] = []

    for setting in _settings(skip_colbert=bool(args.skip_colbert)):
        print(f"\n[일반화] 리트리버={setting.name} 실행 시작")
        per_seed_rows: List[Dict] = []
        failed = False

        for seed in seeds:
            cfg = _apply_common_overrides(
                cfg=base_cfg,
                seed=seed,
                max_questions=args.max_questions,
                output_dir=str(output_dir),
            )
            cfg.setdefault("retrieval", {}).update(deepcopy(setting.overrides))

            try:
                pipeline = RAGPipeline(config=cfg, project_root=ROOT)
                prefix = f"{args.run_name}_{setting.name}_seed{seed}"

                base_row, base_records, _ = pipeline.run_experiment(
                    run_name=f"{prefix}_baseline",
                    strategy_mode="baseline",
                    checker_name=None,
                )
                base_row["메서드키"] = f"{setting.name}__baseline"
                base_row["시드"] = seed
                per_seed_rows.append(base_row)

                cand_row, _, _ = pipeline.run_experiment(
                    run_name=f"{prefix}_{args.checker}_{args.strategy}",
                    strategy_mode=args.strategy,
                    checker_name=args.checker,
                    baseline_records=base_records,
                )
                cand_row["메서드키"] = f"{setting.name}__{args.checker}_{args.strategy}"
                cand_row["시드"] = seed
                per_seed_rows.append(cand_row)
            except Exception as exc:  # pragma: no cover
                failures.append(
                    {
                        "리트리버": setting.name,
                        "시드": seed,
                        "오류": str(exc),
                    }
                )
                print(f"[경고] 리트리버={setting.name}, 시드={seed} 실패: {exc}")
                failed = True
                break

        if failed or not per_seed_rows:
            continue

        agg_rows = aggregate_seed_rows(
            per_seed_rows=per_seed_rows,
            baseline_key=f"{setting.name}__baseline",
            n_bootstrap=int(base_cfg.get("evaluation", {}).get("bootstrap", {}).get("n_samples", 2000)),
            confidence_level=float(base_cfg.get("evaluation", {}).get("bootstrap", {}).get("confidence_level", 0.95)),
            seed=int(seeds[0]),
        )
        for row in agg_rows:
            row["실험명"] = f"{setting.name}::{row.get('실험명', '')}"
            row["메서드키"] = f"{setting.name}::{row.get('메서드키', row.get('실험명', ''))}"
        all_agg_rows.extend(agg_rows)

    summary_csv = output_dir / f"{args.run_name}_summary.csv"
    summary_md = output_dir / f"{args.run_name}_summary.md"
    save_summary_csv(all_agg_rows, summary_csv)
    save_markdown(make_markdown_table(all_agg_rows), summary_md)
    print(f"[저장] 요약 CSV: {summary_csv}")
    print(f"[저장] 요약 Markdown: {summary_md}")

    compare_lines = [
        "| 리트리버 | 메서드 | F1 | 환각률 | 커버리지 | AURC | CSC_AUROC | 체커파싱성공률 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_agg_rows:
        key = str(row.get("메서드키", ""))
        if "baseline" in key:
            continue
        retriever_name = key.split("::", 1)[0]
        compare_lines.append(
            "| {retr} | {method} | {f1:.4f} | {hall:.4f} | {cov:.4f} | {aurc:.4f} | {auroc:.4f} | {parse:.4f} |".format(
                retr=retriever_name,
                method=str(row.get("체커", "")),
                f1=float(row.get("F1", 0.0)),
                hall=float(row.get("환각률", 0.0)),
                cov=float(row.get("커버리지", 0.0)),
                aurc=float(row.get("AURC", 0.0)) if str(row.get("AURC", "")) != "" else 0.0,
                auroc=float(row.get("CSC_AUROC", 0.0)) if str(row.get("CSC_AUROC", "")) != "" else 0.0,
                parse=float(row.get("체커파싱성공률", 0.0)) if str(row.get("체커파싱성공률", "")) != "" else 0.0,
            )
        )
    compare_md = output_dir / f"{args.run_name}_compare.md"
    compare_md.write_text("\n".join(compare_lines) + "\n", encoding="utf-8")
    print(f"[저장] 비교 Markdown: {compare_md}")

    if failures:
        failure_md = output_dir / f"{args.run_name}_failures.md"
        lines = ["| 리트리버 | 시드 | 오류 |", "|---|---:|---|"]
        for f in failures:
            lines.append(f"| {f['리트리버']} | {f['시드']} | {f['오류']} |")
        failure_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[저장] 실패 로그: {failure_md}")


if __name__ == "__main__":
    main()
