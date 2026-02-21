from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm

from evaluation.evaluator import attach_significance, save_jsonl, summarize_for_report
from evaluation.metrics import compute_em_f1
from generator.model import LocalHFTextGenerator
from retrieval.embedder import LocalEmbedder
from retrieval.index import FaissParagraphIndex
from retrieval.utils import build_corpus_from_examples, extract_hotpot_context_paragraphs
from sufficiency.base import INSUFFICIENT
from sufficiency.entailment_checker import EntailmentChecker
from sufficiency.heuristic import KeywordCoverageChecker
from sufficiency.llm_checker import LLMAutoraterChecker
from sufficiency.self_consistency import SelfConsistencyChecker


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def load_config(config_path: str | Path) -> Dict:
    """YAML 설정 로드."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def probe_mps_status() -> Dict[str, str | bool]:
    """현재 파이썬 환경에서 MPS 사용 가능 여부를 점검한다."""
    built = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built())
    available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    error = ""

    if built and not available:
        try:
            _ = torch.ones(1, device="mps")
        except Exception as exc:  # pragma: no cover
            error = str(exc)

    return {
        "torch_version": str(torch.__version__),
        "mps_built": built,
        "mps_available": available,
        "probe_error": error,
    }


def _resolve_hotpot_dataset(dataset_name: str) -> Tuple[str, str]:
    name = str(dataset_name).lower().strip()
    if name in {"hotpotqa", "hotpot_qa"}:
        return "hotpot_qa", "distractor"
    raise ValueError(f"현재는 HotpotQA만 지원합니다. 입력값: {dataset_name}")


def load_hotpot_examples(config: Dict) -> List[Dict]:
    """HotpotQA subset 로딩."""
    dataset_cfg = config.get("dataset", {})
    run_cfg = config.get("run", {})

    hf_name, hf_conf = _resolve_hotpot_dataset(dataset_cfg.get("name", "hotpotqa"))
    split = str(dataset_cfg.get("split", "validation"))
    max_questions = int(dataset_cfg.get("max_questions", 500))
    seed = int(run_cfg.get("seed", 42))

    print(f"[데이터] {hf_name}/{hf_conf} {split} split 로딩 중...")
    ds = load_dataset(hf_name, hf_conf, split=split)

    if 0 < max_questions < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_questions))

    examples: List[Dict] = []
    for ex in ds:
        examples.append(
            {
                "question_id": str(ex.get("id", ex.get("_id", len(examples)))),
                "question": str(ex.get("question", "")),
                "gold_answer": ex.get("answer", ""),
                "contexts": extract_hotpot_context_paragraphs(ex),
            }
        )

    print(f"[데이터] 사용 질문 수: {len(examples)}")
    return examples


class RAGPipeline:
    """RAG + 문맥 충분성 검사 통합 파이프라인."""

    def __init__(self, config: Dict, project_root: Optional[Path] = None) -> None:
        self.config = config
        self.project_root = project_root or Path(__file__).resolve().parent

        run_cfg = config.get("run", {})
        self.seed = int(run_cfg.get("seed", 42))
        set_global_seed(self.seed)

        output_dir = Path(run_cfg.get("output_dir", "outputs"))
        self.output_dir = output_dir if output_dir.is_absolute() else self.project_root / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_jsonl = bool(run_cfg.get("log_jsonl", True))
        self.device_preference: Sequence[str] = run_cfg.get("device_preference", ["mps", "cpu"])
        self.force_mps = bool(run_cfg.get("force_mps", False))

        mps_status = probe_mps_status()
        if "mps" in [str(x).lower().strip() for x in self.device_preference]:
            print(
                f"[장치 점검] torch={mps_status['torch_version']} | "
                f"mps_built={mps_status['mps_built']} | mps_available={mps_status['mps_available']}"
            )
            if not mps_status["mps_available"]:
                print("[경고] 현재 환경에서 MPS(GPU)를 사용할 수 없어 CPU로 폴백합니다.")
                if mps_status["probe_error"]:
                    print(f"[경고] MPS 점검 오류: {mps_status['probe_error']}")
                if self.force_mps:
                    raise RuntimeError(
                        "run.force_mps=true 이지만 MPS 사용이 불가능합니다. "
                        "PyTorch/MacOS 조합을 업데이트하거나 가상환경을 재설정하세요."
                    )

        self.examples = load_hotpot_examples(config)

        dataset_cfg = config.get("dataset", {})
        chunk_cfg = dataset_cfg.get("corpus_chunk", {})
        chunk_size = int(chunk_cfg.get("chunk_size", 300))
        chunk_overlap = int(chunk_cfg.get("chunk_overlap", 50))

        print("[인덱스] 코퍼스 구성 중...")
        self.corpus = build_corpus_from_examples(
            self.examples,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        print(f"[인덱스] 문단 청크 수: {len(self.corpus)}")

        retrieval_cfg = config.get("retrieval", {})
        index_type = str(retrieval_cfg.get("index_type", "faiss_cpu")).lower().strip()
        if index_type != "faiss_cpu":
            raise ValueError(f"현재 지원하는 인덱스 타입은 faiss_cpu 뿐입니다. 입력값: {index_type}")
        self.k_initial = int(retrieval_cfg.get("top_k_initial", 3))
        self.k_reretrieve = int(retrieval_cfg.get("top_k_reretrieve", 6))

        print("[임베딩] 임베딩 모델 로딩/인코딩 중...")
        self.embedder = LocalEmbedder(
            model_name=str(retrieval_cfg.get("embed_model", "intfloat/e5-small-v2")),
            device_preference=self.device_preference,
            batch_size=int(retrieval_cfg.get("batch_size", 32)),
            normalize_embeddings=bool(retrieval_cfg.get("normalize_embeddings", True)),
        )
        print(f"[장치] 임베딩 장치: {self.embedder.device}")

        corpus_embeddings = self.embedder.encode_corpus([x["text"] for x in self.corpus])

        self.index = FaissParagraphIndex(metric="inner_product")
        self.index.build(corpus_embeddings, self.corpus)
        print("[인덱스] FAISS CPU 인덱스 준비 완료")

        generator_cfg = config.get("generator", {})
        self.abstain_text = str(generator_cfg.get("abstain_text", "모르겠습니다."))
        self.prompt_style = str(generator_cfg.get("prompt_style", "qa_short_ko"))

        print("[생성기] 생성 모델 로딩 중...")
        self.generator = LocalHFTextGenerator(
            model_name=str(generator_cfg.get("model_name", "google/flan-t5-large")),
            device_preference=self.device_preference,
            max_input_length=int(generator_cfg.get("max_input_length", 1024)),
            max_new_tokens=int(generator_cfg.get("max_new_tokens", 32)),
            temperature=float(generator_cfg.get("temperature", 0.2)),
            do_sample=bool(generator_cfg.get("do_sample", False)),
            num_threads=generator_cfg.get("num_threads"),
        )
        print(f"[장치] 생성 장치: {self.generator.device}")

        self.checker_cache: Dict[str, object] = {}

    def _retrieve(self, question: str, top_k: int) -> List[Dict]:
        query_emb = self.embedder.encode_queries([question])
        return self.index.search(query_emb, top_k=top_k)[0]

    def _build_checker(self, checker_name: Optional[str], checker_overrides: Optional[Dict] = None):
        if checker_name is None:
            return None

        checker_name = str(checker_name).lower().strip()
        checker_overrides = checker_overrides or {}
        key = f"{checker_name}::{checker_overrides}"
        if key in self.checker_cache:
            return self.checker_cache[key]

        suff_cfg = self.config.get("sufficiency", {})

        if checker_name == "heuristic":
            heur_cfg = suff_cfg.get("heuristic", {})
            checker = KeywordCoverageChecker(
                min_keyword_hits=int(checker_overrides.get("min_keyword_hits", heur_cfg.get("min_keyword_hits", 2))),
                min_coverage_ratio=float(
                    checker_overrides.get("min_coverage_ratio", heur_cfg.get("min_coverage_ratio", 0.35))
                ),
            )

        elif checker_name == "autorater":
            auto_cfg = self.config.get("autorater", {})
            if not bool(auto_cfg.get("enabled", True)):
                raise ValueError("autorater.enabled=false 상태입니다. 설정을 true로 바꾸거나 checker를 변경하세요.")
            model_name = str(checker_overrides.get("model_name", auto_cfg.get("model_name", "google/flan-t5-base")))

            if model_name == self.generator.model_name:
                backend = self.generator
            else:
                backend = LocalHFTextGenerator(
                    model_name=model_name,
                    device_preference=self.device_preference,
                    max_input_length=int(auto_cfg.get("max_input_length", 1024)),
                    max_new_tokens=int(auto_cfg.get("max_new_tokens", 64)),
                    temperature=float(auto_cfg.get("temperature", 0.0)),
                    do_sample=bool(auto_cfg.get("do_sample", False)),
                    num_threads=self.config.get("generator", {}).get("num_threads"),
                )

            prompt_path_cfg = str(auto_cfg.get("prompt_template", "templates/autorater_ko.txt"))
            prompt_path = Path(prompt_path_cfg)
            if not prompt_path.is_absolute():
                prompt_path = self.project_root / prompt_path

            checker = LLMAutoraterChecker(
                text_generator=backend,
                prompt_template_path=prompt_path,
                max_new_tokens=int(checker_overrides.get("max_new_tokens", auto_cfg.get("max_new_tokens", 64))),
                temperature=float(checker_overrides.get("temperature", auto_cfg.get("temperature", 0.0))),
                do_sample=bool(checker_overrides.get("do_sample", auto_cfg.get("do_sample", False))),
                parse_fail_policy=str(auto_cfg.get("parse_fail_policy", "insufficient")),
                confidence_threshold=float(checker_overrides.get("confidence_threshold", 0.0)),
            )

        elif checker_name == "self_consistency":
            sc_cfg = suff_cfg.get("self_consistency", {})
            checker = SelfConsistencyChecker(
                generator=self.generator,
                n_samples=int(checker_overrides.get("n_samples", sc_cfg.get("n_samples", 5))),
                temperature=float(checker_overrides.get("temperature", sc_cfg.get("temperature", 0.7))),
                disagreement_threshold=float(
                    checker_overrides.get("disagreement_threshold", sc_cfg.get("disagreement_threshold", 0.6))
                ),
                abstain_text=self.abstain_text,
                prompt_style=self.prompt_style,
            )

        elif checker_name == "entailment":
            ent_cfg = suff_cfg.get("entailment", {})
            checker = EntailmentChecker(
                model_name=str(checker_overrides.get("model_name", ent_cfg.get("model_name", "roberta-base-mnli"))),
                sufficient_if_entail_prob_ge=float(
                    checker_overrides.get(
                        "sufficient_if_entail_prob_ge",
                        ent_cfg.get("sufficient_if_entail_prob_ge", 0.6),
                    )
                ),
                device_preference=self.device_preference,
            )
        else:
            raise ValueError(f"지원하지 않는 checker: {checker_name}")

        self.checker_cache[key] = checker
        return checker

    def _generate_answer(self, question: str, retrieved_docs: List[Dict]) -> str:
        contexts = [d["text"] for d in retrieved_docs]
        gen_cfg = self.config.get("generator", {})
        return self.generator.generate_answer(
            question=question,
            contexts=contexts,
            abstain_text=self.abstain_text,
            prompt_style=self.prompt_style,
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 32)),
            temperature=float(gen_cfg.get("temperature", 0.2)),
            do_sample=bool(gen_cfg.get("do_sample", False)),
        )

    def _run_single(
        self,
        question: str,
        checker,
        strategy_mode: str,
        k_initial: int,
        k_reretrieve: int,
    ) -> Dict:
        strategy_mode = str(strategy_mode).lower().strip()

        initial_docs = self._retrieve(question, k_initial)
        initial_contexts = [d["text"] for d in initial_docs]

        checker_label = "SKIP"
        checker_score = -1.0
        checker_meta: Dict = {}

        if checker is not None:
            checker_label, checker_score, checker_meta = checker.predict(question, initial_contexts)

        final_docs = initial_docs
        strategy_used = "baseline"

        if strategy_mode == "baseline" or checker is None:
            answer = self._generate_answer(question, final_docs)
            strategy_used = "baseline"

        elif strategy_mode == "abstain":
            if checker_label == INSUFFICIENT:
                answer = self.abstain_text
                strategy_used = "abstain"
            else:
                answer = self._generate_answer(question, final_docs)
                strategy_used = "initial_generate"

        elif strategy_mode == "reretrieve":
            if checker_label == INSUFFICIENT:
                final_docs = self._retrieve(question, k_reretrieve)
                strategy_used = "reretrieve_generate"
            else:
                strategy_used = "initial_generate"
            answer = self._generate_answer(question, final_docs)

        elif strategy_mode == "hybrid":
            if checker_label == INSUFFICIENT:
                final_docs = self._retrieve(question, k_reretrieve)
                label2, score2, meta2 = checker.predict(question, [d["text"] for d in final_docs])
                checker_label = label2
                checker_score = score2
                checker_meta = {"1차": checker_meta, "2차": meta2}
                hybrid_cfg = self.config.get("strategy", {}).get("hybrid", {})
                reretrieve_then_abstain = bool(hybrid_cfg.get("reretrieve_then_abstain", True))
                if label2 == INSUFFICIENT and reretrieve_then_abstain:
                    answer = self.abstain_text
                    strategy_used = "hybrid_reretrieve_abstain"
                else:
                    answer = self._generate_answer(question, final_docs)
                    strategy_used = (
                        "hybrid_reretrieve_generate"
                        if label2 != INSUFFICIENT
                        else "hybrid_reretrieve_generate_forced"
                    )
            else:
                answer = self._generate_answer(question, final_docs)
                strategy_used = "hybrid_initial_generate"
        else:
            raise ValueError(f"지원하지 않는 전략: {strategy_mode}")

        return {
            "answer": answer,
            "checker_label": checker_label,
            "checker_score": float(checker_score),
            "checker_meta": checker_meta,
            "strategy_used": strategy_used,
            "retrieved_doc_ids": [d["doc_id"] for d in final_docs],
            "retrieved_scores": [float(d["score"]) for d in final_docs],
        }

    def run_experiment(
        self,
        run_name: str,
        strategy_mode: str = "baseline",
        checker_name: Optional[str] = None,
        checker_overrides: Optional[Dict] = None,
        k_initial: Optional[int] = None,
        k_reretrieve: Optional[int] = None,
        baseline_records: Optional[List[Dict]] = None,
    ) -> Tuple[Dict, List[Dict], Dict]:
        k_initial = int(k_initial if k_initial is not None else self.k_initial)
        k_reretrieve = int(k_reretrieve if k_reretrieve is not None else self.k_reretrieve)
        checker = self._build_checker(checker_name, checker_overrides)

        print(
            f"[실험 시작] 이름={run_name} | 전략={strategy_mode} | 체커={checker_name or '없음'} | "
            f"k초기={k_initial}, k재검색={k_reretrieve}"
        )

        records: List[Dict] = []
        for idx, sample in enumerate(tqdm(self.examples, desc=f"{run_name} 실행", leave=False), start=1):
            ts = time.perf_counter()
            out = self._run_single(
                question=sample["question"],
                checker=checker,
                strategy_mode=strategy_mode,
                k_initial=k_initial,
                k_reretrieve=k_reretrieve,
            )
            latency_ms = (time.perf_counter() - ts) * 1000.0

            em, f1 = compute_em_f1(out["answer"], sample["gold_answer"])
            is_abstain = int(str(out["answer"]).strip() == self.abstain_text)
            is_correct = int(em == 1.0)

            record = {
                "question_id": sample["question_id"],
                "question": sample["question"],
                "gold_answer": sample["gold_answer"],
                "retrieved_doc_ids": out["retrieved_doc_ids"],
                "retrieved_scores": out["retrieved_scores"],
                "checker_name": checker_name or "none",
                "checker_label": out["checker_label"],
                "checker_score": out["checker_score"],
                "strategy_used": out["strategy_used"],
                "final_answer": out["answer"],
                "is_correct": is_correct,
                "is_abstain": is_abstain,
                "latency_ms": latency_ms,
                "em": em,
                "f1": f1,
                "checker_meta": out["checker_meta"],
            }
            records.append(record)

            if idx % 20 == 0:
                print(f"[진행] {run_name}: {idx}/{len(self.examples)}")

        row = summarize_for_report(
            records=records,
            run_name=run_name,
            checker=checker_name or "없음",
            strategy=strategy_mode,
            abstain_text=self.abstain_text,
        )

        if baseline_records is not None:
            boot_cfg = self.config.get("evaluation", {}).get("bootstrap", {})
            if bool(boot_cfg.get("enabled", True)):
                row = attach_significance(
                    row=row,
                    baseline_records=baseline_records,
                    candidate_records=records,
                    n_samples=int(boot_cfg.get("n_samples", 1000)),
                    confidence_level=float(boot_cfg.get("confidence_level", 0.95)),
                    seed=self.seed,
                )
            else:
                row.update({"EM_차이": "", "EM_p값": "", "F1_차이": "", "F1_p값": ""})
        else:
            row.update({"EM_차이": "", "EM_p값": "", "F1_차이": "", "F1_p값": ""})

        jsonl_path = self.output_dir / f"{run_name}.jsonl"
        if self.log_jsonl:
            save_jsonl(records, jsonl_path)
            print(f"[저장] JSONL: {jsonl_path}")

        return row, records, {"jsonl_path": str(jsonl_path)}
