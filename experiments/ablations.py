from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Dict, List

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
    parser.add_argument("--autorater-preflight-samples", type=int, default=20)
    parser.add_argument("--autorater-min-parse-success", type=float, default=0.30)
    parser.add_argument("--autorater-force-run", action="store_true")
    return parser.parse_args()


def _autorater_preflight(pipeline: RAGPipeline, n_samples: int) -> Dict:
    n = max(0, min(int(n_samples), len(pipeline.examples)))
    if n == 0:
        return {
            "표본수": 0,
            "파싱성공수": 0,
            "파싱실패수": 0,
            "파싱성공률": 0.0,
            "파싱방식분포": {},
            "라벨분포": {},
        }

    checker = pipeline._build_checker("autorater", checker_overrides=None)
    parse_fail = 0
    parse_method_counts: Counter = Counter()
    label_counts: Counter = Counter()

    for sample in pipeline.examples[:n]:
        docs = pipeline._retrieve(sample["question"], pipeline.k_initial)
        contexts = [d["text"] for d in docs]
        label, _, meta = checker.predict(sample["question"], contexts)

        label_counts[str(label)] += 1

        if isinstance(meta, dict):
            parse_method = str(meta.get("파싱방식", "unknown")).strip() or "unknown"
            parse_method_counts[parse_method] += 1
            if "파싱오류" in meta:
                parse_fail += 1
        else:
            parse_fail += 1
            parse_method_counts["meta_missing"] += 1

    parse_success = n - parse_fail
    return {
        "표본수": n,
        "파싱성공수": parse_success,
        "파싱실패수": parse_fail,
        "파싱성공률": parse_success / max(1, n),
        "파싱방식분포": dict(parse_method_counts),
        "라벨분포": dict(label_counts),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.max_questions is not None:
        config.setdefault("dataset", {})["max_questions"] = int(args.max_questions)
    if args.output_dir is not None:
        config.setdefault("run", {})["output_dir"] = args.output_dir

    effective_max_questions = int(config.get("dataset", {}).get("max_questions", 0))
    if 0 < effective_max_questions < 500:
        print(
            f"[경고] 현재 max_questions={effective_max_questions} 입니다. "
            "유의성 검정 안정성을 위해 500 이상을 권장합니다."
        )

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

    if "autorater" in checkers and int(args.autorater_preflight_samples) > 0:
        threshold = max(0.0, min(1.0, float(args.autorater_min_parse_success)))
        try:
            preflight = _autorater_preflight(pipeline=pipeline, n_samples=int(args.autorater_preflight_samples))
            print(
                "[autorater 사전점검] "
                f"표본={preflight['표본수']}, 파싱성공={preflight['파싱성공수']}, "
                f"파싱실패={preflight['파싱실패수']}, 파싱성공률={preflight['파싱성공률']:.3f}"
            )
            print(f"[autorater 사전점검] 파싱방식분포={preflight['파싱방식분포']}")
            print(f"[autorater 사전점검] 라벨분포={preflight['라벨분포']}")

            if preflight["파싱성공률"] < threshold and not bool(args.autorater_force_run):
                print(
                    "[autorater 사전점검] 파싱성공률이 임계값 미만이라 autorater 실험을 건너뜁니다. "
                    "--autorater-force-run 옵션으로 강제 실행할 수 있습니다."
                )
                checkers = [c for c in checkers if c != "autorater"]
        except Exception as exc:
            if bool(args.autorater_force_run):
                print(f"[autorater 사전점검 경고] 점검 실패했지만 강제 실행합니다: {exc}")
            else:
                print(f"[autorater 사전점검] 점검 실패로 autorater 실험을 건너뜁니다: {exc}")
                checkers = [c for c in checkers if c != "autorater"]

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
