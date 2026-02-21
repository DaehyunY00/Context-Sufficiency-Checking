from __future__ import annotations

from typing import Dict, List

import faiss
import numpy as np


class FaissParagraphIndex:
    """문단 검색용 FAISS CPU 인덱스."""

    def __init__(self, metric: str = "inner_product") -> None:
        metric = metric.lower()
        if metric not in {"inner_product", "l2"}:
            raise ValueError(f"지원하지 않는 metric: {metric}")
        self.metric = metric
        self.index: faiss.Index | None = None
        self.payloads: List[Dict] = []

    def build(self, embeddings: np.ndarray, payloads: List[Dict]) -> None:
        if len(embeddings) != len(payloads):
            raise ValueError("임베딩 수와 payload 수가 다릅니다.")
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")

        dim = int(embeddings.shape[1])
        self.index = faiss.IndexFlatIP(dim) if self.metric == "inner_product" else faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.payloads = payloads

    def search(self, query_embeddings: np.ndarray, top_k: int) -> List[List[Dict]]:
        if self.index is None:
            raise RuntimeError("인덱스가 생성되지 않았습니다. build()를 먼저 호출하세요.")
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype("float32")

        safe_k = min(max(int(top_k), 1), int(self.index.ntotal))
        distances, indices = self.index.search(query_embeddings, safe_k)

        all_rows: List[List[Dict]] = []
        for dist_row, idx_row in zip(distances, indices):
            row: List[Dict] = []
            for score, idx in zip(dist_row.tolist(), idx_row.tolist()):
                if idx < 0:
                    continue
                meta = self.payloads[idx]
                row.append(
                    {
                        "doc_id": meta.get("doc_id", str(idx)),
                        "title": meta.get("title", ""),
                        "text": meta.get("text", ""),
                        "score": float(score),
                    }
                )
            all_rows.append(row)
        return all_rows
