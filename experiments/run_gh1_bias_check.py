from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from evaluation.metrics import normalize_answer, roc_pr_diagnostics
from sufficiency.heuristic import KeywordCoverageChecker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="gH1 vs gH1' 편향 점검: gold_containment/human_judgment 라벨 AUROC 비교"
    )
    parser.add_argument("--jsonl", type=str, required=True, help="실험 JSONL 로그")
    parser.add_argument("--manual-csv", type=str, required=True, help="human_judgment 포함 수동 검증 CSV")
    parser.add_argument("--output-csv", type=str, default=str(ROOT / "outputs" / "gh1_bias_check.csv"))
    parser.add_argument("--output-md", type=str, default=str(ROOT / "outputs" / "gh1_bias_check.md"))
    return parser.parse_args()


def _load_jsonl(path: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("question_id", "")).strip()
            if qid:
                out[qid] = row
    return out


def _load_manual(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _gold_tokens(gold_answer) -> set[str]:
    values: List[str] = []
    if isinstance(gold_answer, list):
        values = [str(x) for x in gold_answer if str(x).strip()]
    else:
        values = [str(gold_answer)]
    out: set[str] = set()
    for v in values:
        out.update(normalize_answer(v).split())
    return {t for t in out if t}


def _gh1_scores(question: str, contexts: List[str], answer_tokens: set[str]) -> Tuple[float, float]:
    checker = KeywordCoverageChecker(variant="h1")
    keywords = checker._extract_keywords(question)  # noqa: SLF001
    ctx_join = " ".join([str(c).lower() for c in contexts])
    if not keywords:
        return 0.0, 0.0

    hits = sum(1 for kw in keywords if kw in ctx_join)
    g_h1 = float(hits / max(1, len(keywords)))

    filtered = [kw for kw in keywords if normalize_answer(kw) not in answer_tokens]
    if not filtered:
        g_h1_prime = 0.0
    else:
        hits_prime = sum(1 for kw in filtered if kw in ctx_join)
        g_h1_prime = float(hits_prime / max(1, len(filtered)))
    return g_h1, g_h1_prime


def _safe_int01(text: str) -> int | None:
    raw = str(text).strip().lower()
    if raw in {"1", "true", "yes", "y"}:
        return 1
    if raw in {"0", "false", "no", "n"}:
        return 0
    return None


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    jsonl_map = _load_jsonl(Path(args.jsonl))
    manual_rows = _load_manual(Path(args.manual_csv))

    y_gold: List[int] = []
    y_human: List[int] = []
    s_h1_gold: List[float] = []
    s_h1p_gold: List[float] = []
    s_h1_human: List[float] = []
    s_h1p_human: List[float] = []

    for m in manual_rows:
        qid = str(m.get("question_id", "")).strip()
        if not qid or qid not in jsonl_map:
            continue
        rec = jsonl_map[qid]
        question = str(rec.get("question", ""))
        contexts = rec.get("initial_contexts", rec.get("retrieved_contexts", []))
        if not isinstance(contexts, list):
            contexts = [str(contexts)]
        answer_tokens = _gold_tokens(rec.get("gold_answer", ""))
        h1, h1p = _gh1_scores(question=question, contexts=contexts, answer_tokens=answer_tokens)

        gold_label = int(rec.get("oracle_answerable", 0))
        y_gold.append(gold_label)
        s_h1_gold.append(h1)
        s_h1p_gold.append(h1p)

        human_label = _safe_int01(m.get("human_judgment", ""))
        if human_label is not None:
            y_human.append(human_label)
            s_h1_human.append(h1)
            s_h1p_human.append(h1p)

    rows_out: List[Dict] = []
    if y_gold:
        au_h1 = float(roc_pr_diagnostics(y_true=y_gold, y_score=s_h1_gold).get("auroc", 0.0))
        au_h1p = float(roc_pr_diagnostics(y_true=y_gold, y_score=s_h1p_gold).get("auroc", 0.0))
        rows_out.append(
            {
                "label_set": "gold_containment",
                "n": len(y_gold),
                "gH1_AUROC": au_h1,
                "gH1_prime_AUROC": au_h1p,
                "delta(gH1'-gH1)": au_h1p - au_h1,
            }
        )

    if y_human:
        au_h1 = float(roc_pr_diagnostics(y_true=y_human, y_score=s_h1_human).get("auroc", 0.0))
        au_h1p = float(roc_pr_diagnostics(y_true=y_human, y_score=s_h1p_human).get("auroc", 0.0))
        rows_out.append(
            {
                "label_set": "human_judgment",
                "n": len(y_human),
                "gH1_AUROC": au_h1,
                "gH1_prime_AUROC": au_h1p,
                "delta(gH1'-gH1)": au_h1p - au_h1,
            }
        )

    out_csv = Path(args.output_csv)
    out_md = Path(args.output_md)
    _write_csv(out_csv, rows_out)

    md_lines = [
        "### gH1 편향 점검 (gold_containment vs human_judgment)",
        "",
        "| Label set | n | gH1 AUROC | gH1' AUROC | Δ(gH1'-gH1) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows_out:
        md_lines.append(
            "| {label} | {n} | {a:.4f} | {b:.4f} | {d:+.4f} |".format(
                label=row["label_set"],
                n=int(row["n"]),
                a=float(row["gH1_AUROC"]),
                b=float(row["gH1_prime_AUROC"]),
                d=float(row["delta(gH1'-gH1)"]),
            )
        )
    if not rows_out:
        md_lines.append("| (데이터 부족) | 0 | - | - | - |")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[저장] CSV: {out_csv}")
    print(f"[저장] Markdown: {out_md}")
    if not y_human:
        print("[주의] human_judgment(0/1) 라벨이 없어 human 기준 AUROC는 계산되지 않았습니다.")


if __name__ == "__main__":
    main()

