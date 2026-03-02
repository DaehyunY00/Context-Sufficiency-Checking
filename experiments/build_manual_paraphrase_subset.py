from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
import sys
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="수동 검증용 paraphrase 후보 subset CSV 생성기"
    )
    parser.add_argument("--jsonl", type=str, required=True, help="실험 JSONL 로그 경로")
    parser.add_argument("--output-csv", type=str, default=str(ROOT / "outputs" / "manual_paraphrase_subset.csv"))
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _gold_to_text(gold) -> str:
    if isinstance(gold, list):
        return " || ".join([str(x) for x in gold if str(x).strip()])
    return str(gold)


def main() -> None:
    args = parse_args()
    src = Path(args.jsonl)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(src)
    if not rows:
        raise ValueError(f"입력 JSONL이 비어 있습니다: {src}")

    # paraphrase 후보 우선: EM=0이면서 F1>0
    para = [r for r in rows if float(r.get("em", 0.0)) < 1.0 and float(r.get("f1", 0.0)) > 0.0]
    others = [r for r in rows if r not in para]

    rng = random.Random(int(args.seed))
    rng.shuffle(para)
    rng.shuffle(others)

    n = max(1, int(args.sample_size))
    n_para = min(len(para), max(1, int(n * 0.6)))
    picked = para[:n_para] + others[: max(0, n - n_para)]
    rng.shuffle(picked)
    picked = picked[:n]

    fieldnames = [
        "question_id",
        "question",
        "gold_answer",
        "final_answer",
        "em",
        "f1",
        "oracle_answerable",
        "paraphrase_candidate",
        "retrieved_context_preview",
        "human_judgment",  # 수동 라벨 입력: 0/1
        "비고",
    ]

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in picked:
            contexts = r.get("initial_contexts", r.get("retrieved_contexts", []))
            if isinstance(contexts, list):
                preview = " ".join([str(x) for x in contexts[:2]])
            else:
                preview = str(contexts)
            preview = preview[:800]

            writer.writerow(
                {
                    "question_id": str(r.get("question_id", "")),
                    "question": str(r.get("question", "")),
                    "gold_answer": _gold_to_text(r.get("gold_answer", "")),
                    "final_answer": str(r.get("final_answer", "")),
                    "em": float(r.get("em", 0.0)),
                    "f1": float(r.get("f1", 0.0)),
                    "oracle_answerable": int(r.get("oracle_answerable", 0)),
                    "paraphrase_candidate": int(float(r.get("em", 0.0)) < 1.0 and float(r.get("f1", 0.0)) > 0.0),
                    "retrieved_context_preview": preview,
                    "human_judgment": "",
                    "비고": "",
                }
            )

    print(f"[저장] 수동 검증 subset CSV: {out}")
    print(f"[정보] 총 {len(picked)}개 샘플 (paraphrase 후보 우선추출)")


if __name__ == "__main__":
    main()

