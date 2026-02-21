from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    """로컬 임베딩 생성기(MPS/CPU 우선)."""

    def __init__(
        self,
        model_name: str,
        device_preference: Sequence[str] | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = self._resolve_device(device_preference or ["mps", "cpu"])
        self.model = SentenceTransformer(model_name, device=self.device)
        self._is_e5 = "e5" in model_name.lower()

    @staticmethod
    def _resolve_device(device_preference: Sequence[str]) -> str:
        for cand in [str(x).lower().strip() for x in device_preference]:
            if cand == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            if cand == "cuda" and torch.cuda.is_available():
                return "cuda"
            if cand == "cpu":
                return "cpu"
        return "cpu"

    def _encode(self, texts: Iterable[str]) -> np.ndarray:
        text_list = list(texts)
        if not text_list:
            dim = self.model.get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype="float32")

        embeddings = self.model.encode(
            text_list,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")

    def encode_corpus(self, texts: Iterable[str]) -> np.ndarray:
        if self._is_e5:
            return self._encode([f"passage: {t}" for t in texts])
        return self._encode(texts)

    def encode_queries(self, texts: Iterable[str]) -> np.ndarray:
        if self._is_e5:
            return self._encode([f"query: {t}" for t in texts])
        return self._encode(texts)

    def embedding_dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())
