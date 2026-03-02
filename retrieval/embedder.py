from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


class LocalEmbedder:
    """로컬 임베딩 생성기(MPS/CPU 우선).

    지원 백엔드:
    - sentence_transformer: 단일 인코더 (e5/bge 등)
    - dpr: dual-encoder (query/document 분리)
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "sentence_transformer",
        query_model_name: str | None = None,
        corpus_model_name: str | None = None,
        device_preference: Sequence[str] | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        max_length: int = 256,
    ) -> None:
        self.model_name = model_name
        self.model_type = str(model_type).strip().lower()
        self.query_model_name = str(query_model_name).strip() if query_model_name else ""
        self.corpus_model_name = str(corpus_model_name).strip() if corpus_model_name else ""
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.max_length = max(8, int(max_length))
        self.device = self._resolve_device(device_preference or ["mps", "cpu"])
        self._is_e5 = "e5" in model_name.lower()

        self.model = None
        self.query_tokenizer = None
        self.corpus_tokenizer = None
        self.query_model = None
        self.corpus_model = None

        if self.model_type in {"sentence_transformer", "st", "sbert", "colbert"}:
            # colbert는 late-interaction 구현이 아닌 임베딩 근사로 동작한다.
            self.model = SentenceTransformer(model_name, device=self.device)
            self.model_type = "sentence_transformer"
        elif self.model_type == "dpr":
            q_model = self.query_model_name or "facebook/dpr-question_encoder-single-nq-base"
            c_model = self.corpus_model_name or "facebook/dpr-ctx_encoder-single-nq-base"
            self.query_model_name = q_model
            self.corpus_model_name = c_model

            self.query_tokenizer = AutoTokenizer.from_pretrained(q_model)
            self.corpus_tokenizer = AutoTokenizer.from_pretrained(c_model)
            self.query_model = AutoModel.from_pretrained(q_model).to(self.device).eval()
            self.corpus_model = AutoModel.from_pretrained(c_model).to(self.device).eval()
            self._is_e5 = False
        else:
            raise ValueError(
                f"지원하지 않는 retrieval.model_type: {self.model_type}. "
                "지원: sentence_transformer, dpr"
            )

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
        if self.model is None:
            raise RuntimeError("SentenceTransformer 모델이 초기화되지 않았습니다.")

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

    def _encode_dpr(self, texts: Iterable[str], *, encode_query: bool) -> np.ndarray:
        if self.query_model is None or self.corpus_model is None or self.query_tokenizer is None or self.corpus_tokenizer is None:
            raise RuntimeError("DPR 인코더가 초기화되지 않았습니다.")

        text_list = list(texts)
        if not text_list:
            dim = int(self.query_model.config.hidden_size)
            return np.zeros((0, dim), dtype="float32")

        tokenizer = self.query_tokenizer if encode_query else self.corpus_tokenizer
        model = self.query_model if encode_query else self.corpus_model

        all_embs: list[np.ndarray] = []
        for i in range(0, len(text_list), self.batch_size):
            batch = text_list[i : i + self.batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            else:
                emb = outputs.last_hidden_state[:, 0, :]

            if self.normalize_embeddings:
                emb = F.normalize(emb, p=2, dim=1)
            all_embs.append(emb.detach().cpu().numpy().astype("float32"))

        return np.concatenate(all_embs, axis=0) if all_embs else np.zeros((0, int(self.query_model.config.hidden_size)), dtype="float32")

    def encode_corpus(self, texts: Iterable[str]) -> np.ndarray:
        if self.model_type == "dpr":
            return self._encode_dpr(texts, encode_query=False)
        if self._is_e5:
            return self._encode([f"passage: {t}" for t in texts])
        return self._encode(texts)

    def encode_queries(self, texts: Iterable[str]) -> np.ndarray:
        if self.model_type == "dpr":
            return self._encode_dpr(texts, encode_query=True)
        if self._is_e5:
            return self._encode([f"query: {t}" for t in texts])
        return self._encode(texts)

    def embedding_dim(self) -> int:
        if self.model_type == "dpr":
            if self.query_model is None:
                return 0
            return int(self.query_model.config.hidden_size)
        if self.model is None:
            return 0
        return int(self.model.get_sentence_embedding_dimension())
