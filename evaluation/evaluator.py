from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from evaluation.metrics import paired_bootstrap_test, summarize_records


def save_jsonl(records: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_summary_row(summary: Dict, run_name: str, checker: str, strategy: str) -> Dict:
    checker_norm = str(checker).strip().lower()
    has_checker = checker_norm not in {"", "none", "없음"}
    return {
        "실험명": run_name,
        "체커": checker,
        "전략": strategy,
        "샘플수": int(summary.get("sample_count", 0)),
        "EM": float(summary.get("em", 0.0)),
        "F1": float(summary.get("f1", 0.0)),
        "환각률": float(summary.get("hallucination_rate", 0.0)),
        "커버리지": float(summary.get("coverage", 0.0)),
        "선택적정확도": float(summary.get("selective_accuracy", 0.0)),
        "평균지연(ms)": float(summary.get("latency_ms", 0.0)),
        "체커판정수": int(summary.get("checker_eval_count", 0)) if has_checker else "",
        "체커파싱실패수": int(summary.get("checker_parse_fail_count", 0)) if has_checker else "",
        "체커파싱성공률": float(summary.get("checker_parse_success_rate", 0.0)) if has_checker else "",
    }


def attach_significance(
    row: Dict,
    baseline_records: List[Dict],
    candidate_records: List[Dict],
    n_samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict:
    base_em = [float(r.get("em", 0.0)) for r in baseline_records]
    cand_em = [float(r.get("em", 0.0)) for r in candidate_records]
    base_f1 = [float(r.get("f1", 0.0)) for r in baseline_records]
    cand_f1 = [float(r.get("f1", 0.0)) for r in candidate_records]

    em_test = paired_bootstrap_test(base_em, cand_em, n_samples=n_samples, confidence_level=confidence_level, seed=seed)
    f1_test = paired_bootstrap_test(base_f1, cand_f1, n_samples=n_samples, confidence_level=confidence_level, seed=seed)

    out = dict(row)
    out["EM_차이"] = em_test["observed_diff"]
    out["EM_p값"] = em_test["p_value"]
    out["F1_차이"] = f1_test["observed_diff"]
    out["F1_p값"] = f1_test["p_value"]
    return out


def save_summary_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = [
        "실험명",
        "체커",
        "전략",
        "샘플수",
        "EM",
        "F1",
        "환각률",
        "커버리지",
        "선택적정확도",
        "평균지연(ms)",
        "체커판정수",
        "체커파싱실패수",
        "체커파싱성공률",
        "EM_차이",
        "EM_p값",
        "F1_차이",
        "F1_p값",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def make_markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "결과가 없습니다."

    headers = [
        "실험명",
        "체커",
        "전략",
        "EM",
        "F1",
        "환각률",
        "커버리지",
        "선택적정확도",
        "평균지연(ms)",
        "체커파싱성공률",
        "체커파싱실패수",
        "EM_p값",
        "F1_p값",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("실험명", "")),
                    str(row.get("체커", "")),
                    str(row.get("전략", "")),
                    f"{float(row.get('EM', 0.0)):.4f}",
                    f"{float(row.get('F1', 0.0)):.4f}",
                    f"{float(row.get('환각률', 0.0)):.4f}",
                    f"{float(row.get('커버리지', 0.0)):.4f}",
                    f"{float(row.get('선택적정확도', 0.0)):.4f}",
                    f"{float(row.get('평균지연(ms)', 0.0)):.1f}",
                    _fmt_ratio(row.get("체커파싱성공률", "")),
                    _fmt_int(row.get("체커파싱실패수", "")),
                    _fmt_p(row.get("EM_p값", "")),
                    _fmt_p(row.get("F1_p값", "")),
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def save_markdown(markdown_text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_text, encoding="utf-8")


def summarize_for_report(records: List[Dict], run_name: str, checker: str, strategy: str, abstain_text: str) -> Dict:
    summary = summarize_records(records, abstain_text=abstain_text)
    return build_summary_row(summary, run_name=run_name, checker=checker, strategy=strategy)


def _fmt_p(value) -> str:
    if value == "":
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_ratio(value) -> str:
    if value == "":
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_int(value) -> str:
    if value == "":
        return "-"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)
