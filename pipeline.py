from __future__ import annotations

import os
import platform
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm

from evaluation.evaluator import (
    attach_significance,
    latency_quality_analysis,
    save_checker_artifacts,
    save_jsonl,
    save_latency_artifacts,
    summarize_for_report,
)
from evaluation.metrics import compute_em_f1, normalize_answer
from generator.model import LocalHFTextGenerator
from retrieval.embedder import LocalEmbedder
from retrieval.index import FaissParagraphIndex
from retrieval.utils import build_corpus_from_examples, extract_context_paragraphs
from sufficiency.base import INSUFFICIENT, SUFFICIENT
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


def configure_hf_cache(run_cfg: Dict) -> Dict[str, str]:
    """
    HuggingFace 캐시 경로를 로컬 워크스페이스 하위로 고정한다.
    권한 문제로 ~/.cache 잠금 파일 생성이 실패하는 경우를 방지한다.
    """
    cache_root_raw = str(run_cfg.get("hf_cache_dir", ".hf_cache")).strip()
    cache_root = Path(cache_root_raw).expanduser()
    if not cache_root.is_absolute():
        cache_root = Path.cwd() / cache_root

    datasets_cache = cache_root / "datasets"
    transformers_cache = cache_root / "transformers"
    datasets_cache.mkdir(parents=True, exist_ok=True)
    transformers_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)

    return {
        "hf_home": str(cache_root),
        "hf_datasets_cache": str(datasets_cache),
        "hf_transformers_cache": str(transformers_cache),
    }


def probe_mps_status() -> Dict[str, str | bool]:
    """현재 파이썬 환경에서 MPS 사용 가능 여부를 점검한다."""
    macos_version = str(platform.mac_ver()[0] or "").strip()
    py_version = str(platform.python_version() or "").strip()
    built = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built())
    available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    error = ""
    hint = ""

    if built and not available:
        try:
            _ = torch.ones(1, device="mps")
        except Exception as exc:  # pragma: no cover
            error = str(exc)
        if macos_version:
            try:
                major = int(macos_version.split(".")[0])
            except ValueError:
                major = 0
            if major >= 13:
                if "dev" in str(torch.__version__).lower():
                    hint = (
                        "개발 버전(PyTorch nightly)에서 MPS 감지 이슈가 있을 수 있습니다. "
                        "안정 버전(torch/torchvision/torchaudio 동일 라인) 재설치를 권장합니다."
                    )
                else:
                    hint = (
                        "MPS 미활성은 macOS/파이썬/torch 조합 문제일 수 있습니다. "
                        "Python 3.11~3.12 기반 새 환경에서 torch 안정 버전 재설치를 권장합니다."
                    )
        try:
            py_major, py_minor = [int(x) for x in py_version.split(".")[:2]]
            if py_major >= 3 and py_minor >= 13:
                hint = (
                    "현재 Python 3.13 환경에서는 MPS가 비활성인 경우가 있습니다. "
                    "Python 3.12 환경에서 torch 안정 버전으로 재설치해 점검하세요."
                )
        except Exception:
            pass

    return {
        "torch_version": str(torch.__version__),
        "python_version": py_version,
        "macos_version": macos_version,
        "mps_built": built,
        "mps_available": available,
        "probe_error": error,
        "diagnosis_hint": hint,
    }


def _resolve_dataset_source(dataset_name: str) -> Tuple[str, Optional[str]]:
    name = str(dataset_name).lower().strip()
    if name in {"hotpotqa", "hotpot_qa"}:
        return "hotpot_qa", "distractor"
    if name in {"2wikimultihopqa", "2wiki", "2wiki_multihop_qa"}:
        # 허깅페이스 공개 미러가 여러 개 존재하므로, 기본값은 대표 경로를 사용한다.
        return "scholarly-shadows-syndicate/2wikimultihopqa", None
    if name in {"natural_questions", "nq", "naturalquestions"}:
        return "natural_questions", "default"
    if name in {"musique", "musique_qa"}:
        # 기본 경로(필요 시 configs/*.yaml의 hf_name으로 오버라이드 가능)
        return "dgslibisey/MuSiQue", None
    if name in {"strategyqa", "strategy_qa"}:
        # 기본 경로(필요 시 configs/*.yaml의 hf_name으로 오버라이드 가능)
        return "ChilleD/StrategyQA", None
    raise ValueError(
        "지원하지 않는 dataset.name 입니다. "
        "기본 지원: hotpotqa, 2wikimultihopqa, natural_questions, musique, strategyqa "
        f"(입력값: {dataset_name})"
    )


def _extract_question(example: Dict) -> str:
    if isinstance(example.get("question"), dict):
        return str(example.get("question", {}).get("text", ""))
    if "question" in example:
        return str(example.get("question", ""))
    # 일부 StrategyQA/MuSiQue 미러는 input/query 키를 사용한다.
    if "input" in example:
        return str(example.get("input", ""))
    if "query" in example:
        return str(example.get("query", ""))
    return ""


def _extract_nq_document_tokens(example: Dict) -> List[str]:
    document = example.get("document", {})
    if not isinstance(document, dict):
        return []

    tokens_obj = document.get("tokens", {})
    tokens: List[str] = []
    html_flags: List[bool] = []

    if isinstance(tokens_obj, dict):
        raw_tokens = tokens_obj.get("token", tokens_obj.get("tokens", []))
        raw_html = tokens_obj.get("is_html", [])
        if isinstance(raw_tokens, list):
            tokens = [str(t) for t in raw_tokens]
        if isinstance(raw_html, list):
            html_flags = [bool(x) for x in raw_html]
    elif isinstance(tokens_obj, list):
        for item in tokens_obj:
            if isinstance(item, dict):
                tokens.append(str(item.get("token", item.get("text", ""))))
                html_flags.append(bool(item.get("is_html", False)))
            else:
                tokens.append(str(item))
                html_flags.append(False)

    if tokens and len(html_flags) == len(tokens):
        tokens = [tok for tok, is_html in zip(tokens, html_flags) if not is_html]
    return [t for t in tokens if str(t).strip()]


def _iter_nq_annotations(example: Dict) -> List[Dict]:
    """
    NQ annotations를 샘플 단위 dict 리스트로 정규화한다.
    - list 포맷: 그대로 사용
    - dict-of-lists 포맷: 인덱스별 dict로 변환
    """
    annotations = example.get("annotations")
    if isinstance(annotations, list):
        return [a for a in annotations if isinstance(a, dict)]
    if not isinstance(annotations, dict):
        return []

    max_len = 0
    for value in annotations.values():
        if isinstance(value, list):
            max_len = max(max_len, len(value))
    if max_len <= 0:
        return []

    normalized: List[Dict] = []
    for idx in range(max_len):
        ann_item: Dict = {}
        for key, value in annotations.items():
            if isinstance(value, list):
                ann_item[key] = value[idx] if idx < len(value) else None
            else:
                ann_item[key] = value
        normalized.append(ann_item)
    return normalized


def _extract_nq_yes_no_answer(raw_value) -> str:
    """
    HF natural_questions 포맷의 yes/no 라벨(-1/0/1 또는 문자열)을 정규화한다.
    일반적으로 -1=없음, 0=no, 1=yes로 기록된다.
    """
    try:
        value_int = int(raw_value)
        if value_int == 1:
            return "yes"
        if value_int == 0:
            return "no"
        return ""
    except (TypeError, ValueError):
        pass

    value_str = str(raw_value).strip().upper()
    if value_str == "YES":
        return "yes"
    if value_str == "NO":
        return "no"
    return ""


def _extract_nq_short_answers(example: Dict) -> List[str]:
    annotations = _iter_nq_annotations(example)
    if not annotations:
        return []

    doc_tokens = _extract_nq_document_tokens(example)
    answers: List[str] = []

    for ann in annotations:
        yes_no = _extract_nq_yes_no_answer(ann.get("yes_no_answer", ""))
        if yes_no:
            answers.append(yes_no)

        short_answers_raw = ann.get("short_answers", [])
        if isinstance(short_answers_raw, dict):
            short_answers = [short_answers_raw]
        elif isinstance(short_answers_raw, list):
            short_answers = short_answers_raw
        else:
            short_answers = []

        for sa in short_answers:
            if isinstance(sa, str):
                text = sa.strip()
                if text:
                    answers.append(text)
                continue
            if not isinstance(sa, dict):
                continue

            text_field = sa.get("text", "")
            if isinstance(text_field, list):
                for text_item in text_field:
                    clean = str(text_item).strip()
                    if clean:
                        answers.append(clean)
            else:
                clean = str(text_field).strip()
                if clean:
                    answers.append(clean)

            start_field = sa.get("start_token", sa.get("start", -1))
            end_field = sa.get("end_token", sa.get("end", -1))
            if isinstance(start_field, list) and isinstance(end_field, list):
                span_pairs = zip(start_field, end_field)
            else:
                span_pairs = [(start_field, end_field)]

            for start_raw, end_raw in span_pairs:
                try:
                    start = int(start_raw)
                    end = int(end_raw)
                except (TypeError, ValueError):
                    continue
                if 0 <= start < end <= len(doc_tokens):
                    span = " ".join(doc_tokens[start:end]).strip()
                    if span:
                        answers.append(span)

    uniq: List[str] = []
    seen = set()
    for answer in answers:
        key = normalize_answer(answer)
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(str(answer))
    return uniq


def _extract_nq_long_answers(example: Dict) -> List[str]:
    annotations = _iter_nq_annotations(example)
    if not annotations:
        return []

    doc_tokens = _extract_nq_document_tokens(example)
    if not doc_tokens:
        return []

    answers: List[str] = []
    for ann in annotations:
        long_answer = ann.get("long_answer", {})
        if not isinstance(long_answer, dict):
            continue
        try:
            start = int(long_answer.get("start_token", -1))
            end = int(long_answer.get("end_token", -1))
        except (TypeError, ValueError):
            continue
        if 0 <= start < end <= len(doc_tokens):
            span = " ".join(doc_tokens[start:end]).strip()
            if span:
                answers.append(span)

    uniq: List[str] = []
    seen = set()
    for answer in answers:
        key = normalize_answer(answer)
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(str(answer))
    return uniq


def _is_empty_gold_answer(gold_answer) -> bool:
    if gold_answer is None:
        return True
    if isinstance(gold_answer, list):
        return len([x for x in gold_answer if str(x).strip()]) == 0
    return str(gold_answer).strip() == ""


def _extract_gold_answer(example: Dict):
    # StrategyQA 계열 yes/no (bool) 정규화
    if isinstance(example.get("answer"), bool):
        return "yes" if bool(example.get("answer")) else "no"
    if isinstance(example.get("target"), bool):
        return "yes" if bool(example.get("target")) else "no"

    nq_answers = _extract_nq_short_answers(example)
    if nq_answers:
        return nq_answers
    nq_long_answers = _extract_nq_long_answers(example)
    if nq_long_answers:
        return nq_long_answers

    if "answer" in example:
        answer = example.get("answer")
        if isinstance(answer, dict):
            if "text" in answer:
                text = answer.get("text")
                if isinstance(text, list):
                    return [str(x) for x in text if str(x).strip()]
                return str(text)
            return str(answer)
        return answer

    if "answers" in example:
        answers = example.get("answers")
        if isinstance(answers, dict):
            text = answers.get("text", [])
            if isinstance(text, list):
                clean = [str(x) for x in text if str(x).strip()]
                if clean:
                    return clean
        if isinstance(answers, list):
            clean = [str(x) for x in answers if str(x).strip()]
            if clean:
                return clean
        return answers

    if "answer_text" in example:
        return str(example.get("answer_text", ""))

    if "answers_text" in example:
        answers_text = example.get("answers_text")
        if isinstance(answers_text, list):
            clean = [str(x) for x in answers_text if str(x).strip()]
            if clean:
                return clean
        if str(answers_text).strip():
            return str(answers_text)

    return str(example.get("answer", ""))


def _extract_question_id(example: Dict, fallback_idx: int) -> str:
    for key in ["id", "_id", "question_id", "qid", "uid"]:
        if key in example and str(example.get(key, "")).strip():
            return str(example.get(key))
    return str(fallback_idx)


def load_examples(config: Dict) -> List[Dict]:
    """설정된 데이터셋 subset 로딩."""
    dataset_cfg = config.get("dataset", {})
    run_cfg = config.get("run", {})
    dataset_name = str(dataset_cfg.get("name", "hotpotqa"))

    hf_name_override = dataset_cfg.get("hf_name")
    hf_conf_override = dataset_cfg.get("hf_config")
    if hf_name_override:
        hf_name = str(hf_name_override)
        hf_conf = str(hf_conf_override) if hf_conf_override is not None else None
    else:
        hf_name, hf_conf = _resolve_dataset_source(dataset_name)

    split = str(dataset_cfg.get("split", "validation"))
    max_questions = int(dataset_cfg.get("max_questions", 500))
    seed = int(run_cfg.get("seed", 42))
    drop_empty_gold = bool(dataset_cfg.get("drop_empty_gold_answers", False))
    cache_paths = configure_hf_cache(run_cfg)

    display_name = f"{hf_name}/{hf_conf}" if hf_conf else hf_name
    print(f"[데이터] {display_name} {split} split 로딩 중...")
    print(f"[캐시] HF 데이터 캐시: {cache_paths['hf_datasets_cache']}")
    if hf_conf is None:
        ds = load_dataset(hf_name, split=split, cache_dir=cache_paths["hf_datasets_cache"])
    else:
        ds = load_dataset(hf_name, hf_conf, split=split, cache_dir=cache_paths["hf_datasets_cache"])

    if 0 < max_questions < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_questions))

    examples: List[Dict] = []
    skipped_no_context = 0
    skipped_empty_gold = 0
    for idx, ex in enumerate(ds):
        question = _extract_question(ex)
        contexts = extract_context_paragraphs(ex, dataset_name=dataset_name)
        gold_answer = _extract_gold_answer(ex)
        if not contexts:
            skipped_no_context += 1
            continue
        if drop_empty_gold and _is_empty_gold_answer(gold_answer):
            skipped_empty_gold += 1
            continue
        examples.append(
            {
                "question_id": _extract_question_id(ex, fallback_idx=idx),
                "question": question,
                "gold_answer": gold_answer,
                "contexts": contexts,
            }
        )

    print(f"[데이터] 사용 질문 수: {len(examples)}")
    if skipped_no_context > 0:
        print(f"[데이터] 문맥 추출 실패로 제외된 샘플 수: {skipped_no_context}")
    if skipped_empty_gold > 0:
        print(f"[데이터] 빈 정답 샘플 제외 수: {skipped_empty_gold}")
    if not examples:
        raise ValueError(
            "데이터셋에서 문맥을 추출한 샘플이 0개입니다. "
            "dataset.name/hf_name/hf_config 및 컨텍스트 필드 매핑을 확인하세요."
        )
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
                f"python={mps_status.get('python_version', 'unknown')} | "
                f"macOS={mps_status['macos_version'] or 'unknown'} | "
                f"mps_built={mps_status['mps_built']} | mps_available={mps_status['mps_available']}"
            )
            if not mps_status["mps_available"]:
                print("[경고] 현재 환경에서 MPS(GPU)를 사용할 수 없어 CPU로 폴백합니다.")
                if mps_status["probe_error"]:
                    print(f"[경고] MPS 점검 오류: {mps_status['probe_error']}")
                if mps_status["diagnosis_hint"]:
                    print(f"[경고] 진단 힌트: {mps_status['diagnosis_hint']}")
                if self.force_mps:
                    raise RuntimeError(
                        "run.force_mps=true 이지만 MPS 사용이 불가능합니다. "
                        "PyTorch/MacOS 조합을 업데이트하거나 가상환경을 재설정하세요."
                    )

        self.examples = load_examples(config)

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
        retrieval_model_type = str(retrieval_cfg.get("model_type", "sentence_transformer")).strip().lower()
        self.embedder = LocalEmbedder(
            model_name=str(retrieval_cfg.get("embed_model", "intfloat/e5-small-v2")),
            model_type=retrieval_model_type,
            query_model_name=retrieval_cfg.get("query_model_name"),
            corpus_model_name=retrieval_cfg.get("corpus_model_name"),
            device_preference=self.device_preference,
            batch_size=int(retrieval_cfg.get("batch_size", 32)),
            normalize_embeddings=bool(retrieval_cfg.get("normalize_embeddings", True)),
            max_length=int(retrieval_cfg.get("max_length", 256)),
        )
        print(f"[장치] 임베딩 장치: {self.embedder.device}")
        if retrieval_model_type == "dpr":
            print(
                "[임베딩] DPR 듀얼 인코더 사용: "
                f"query={self.embedder.query_model_name}, corpus={self.embedder.corpus_model_name}"
            )
        else:
            print(f"[임베딩] 임베딩 백엔드: {retrieval_model_type}")

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
            torch_dtype=generator_cfg.get("torch_dtype"),
            low_cpu_mem_usage=bool(generator_cfg.get("low_cpu_mem_usage", True)),
            trust_remote_code=bool(generator_cfg.get("trust_remote_code", False)),
            backend=str(generator_cfg.get("backend", "transformers")),
            allow_backend_fallback=bool(generator_cfg.get("allow_backend_fallback", True)),
            ollama_url=str(generator_cfg.get("ollama_url", "http://localhost:11434/api/generate")),
            ollama_timeout_sec=int(generator_cfg.get("ollama_timeout_sec", 180)),
        )
        print(f"[장치] 생성 장치: {self.generator.runtime_device_label}")

        self.checker_cache: Dict[str, object] = {}
        eval_cfg = self.config.get("evaluation", {})
        answerable_cfg = eval_cfg.get("answerable", {})
        self.answerable_mode = str(answerable_cfg.get("mode", "gold_containment")).strip().lower()
        self.answerable_entail_threshold = float(answerable_cfg.get("entail_prob_threshold", 0.6))
        self.answerable_entail_model_name = str(
            answerable_cfg.get("entail_model_name", "cross-encoder/nli-distilroberta-base")
        )
        self.answerable_entailment_checker: Optional[EntailmentChecker] = None
        if self.answerable_mode == "entailment":
            print(
                f"[평가정의] Answerable 판정={self.answerable_mode} "
                f"(모델={self.answerable_entail_model_name}, 임계값={self.answerable_entail_threshold})"
            )
            self.answerable_entailment_checker = EntailmentChecker(
                model_name=self.answerable_entail_model_name,
                sufficient_if_entail_prob_ge=self.answerable_entail_threshold,
                device_preference=self.device_preference,
            )
        else:
            self.answerable_mode = "gold_containment"
            print("[평가정의] Answerable 판정=gold_containment (gold 답변 문자열 포함 여부)")

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
                    checker_overrides.get("min_coverage_ratio", heur_cfg.get("min_coverage_ratio", 0.5))
                ),
                variant=str(checker_overrides.get("variant", heur_cfg.get("variant", "h1"))),
            )

        elif checker_name == "autorater":
            auto_cfg = self.config.get("autorater", {})
            if not bool(auto_cfg.get("enabled", True)):
                raise ValueError("autorater.enabled=false 상태입니다. 설정을 true로 바꾸거나 checker를 변경하세요.")
            model_name = str(
                checker_overrides.get("model_name", auto_cfg.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct"))
            )

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
                    torch_dtype=auto_cfg.get("torch_dtype"),
                    low_cpu_mem_usage=bool(auto_cfg.get("low_cpu_mem_usage", True)),
                    trust_remote_code=bool(auto_cfg.get("trust_remote_code", False)),
                    backend=str(auto_cfg.get("backend", self.config.get("generator", {}).get("backend", "transformers"))),
                    allow_backend_fallback=bool(auto_cfg.get("allow_backend_fallback", self.config.get("generator", {}).get("allow_backend_fallback", True))),
                    ollama_url=str(auto_cfg.get("ollama_url", self.config.get("generator", {}).get("ollama_url", "http://localhost:11434/api/generate"))),
                    ollama_timeout_sec=int(auto_cfg.get("ollama_timeout_sec", self.config.get("generator", {}).get("ollama_timeout_sec", 180))),
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
                max_parse_retries=int(checker_overrides.get("max_parse_retries", auto_cfg.get("max_parse_retries", 1))),
                max_context_chars=int(checker_overrides.get("max_context_chars", auto_cfg.get("max_context_chars", 1800))),
            )

        elif checker_name == "self_consistency":
            sc_cfg = suff_cfg.get("self_consistency", {})
            checker = SelfConsistencyChecker(
                generator=self.generator,
                n_samples=int(checker_overrides.get("n_samples", sc_cfg.get("n_samples", 3))),
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
                model_name=str(
                    checker_overrides.get(
                        "model_name",
                        ent_cfg.get("model_name", "cross-encoder/nli-distilroberta-base"),
                    )
                ),
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

    @staticmethod
    def _get_gold_candidates(gold_answer) -> List[str]:
        if isinstance(gold_answer, list):
            cands = [normalize_answer(str(x)) for x in gold_answer if str(x).strip()]
        else:
            cands = [normalize_answer(str(gold_answer))]
        return [c for c in cands if c]

    def _estimate_oracle_answerable(self, question: str, gold_answer, contexts: List[str]) -> Tuple[int, Dict]:
        merged = normalize_answer(" ".join([str(x) for x in contexts]))
        candidates = self._get_gold_candidates(gold_answer)
        if not merged or not candidates:
            return 0, {"mode": self.answerable_mode, "score": 0.0, "matched_answer": ""}

        if self.answerable_mode == "entailment":
            checker = self.answerable_entailment_checker
            if checker is None:
                return 0, {
                    "mode": self.answerable_mode,
                    "score": 0.0,
                    "matched_answer": "",
                    "error": "entailment checker not initialized",
                }

            premise = " ".join([str(c) for c in contexts])[:2400]
            best_score = 0.0
            best_answer = ""
            for cand in candidates:
                hypothesis = f"The answer to the question '{question}' is '{cand}'."
                prob = float(checker.score_entailment(premise=premise, hypothesis=hypothesis))
                if prob > best_score:
                    best_score = prob
                    best_answer = cand
            label = int(best_score >= self.answerable_entail_threshold)
            return label, {
                "mode": self.answerable_mode,
                "score": best_score,
                "matched_answer": best_answer,
                "threshold": self.answerable_entail_threshold,
            }

        # default: gold answer containment in retrieved contexts
        for cand in candidates:
            if cand in merged:
                return 1, {"mode": "gold_containment", "score": 1.0, "matched_answer": cand}
        return 0, {"mode": "gold_containment", "score": 0.0, "matched_answer": ""}

    def _run_single(
        self,
        question: str,
        gold_answer,
        checker,
        strategy_mode: str,
        k_initial: int,
        k_reretrieve: int,
        strategy_overrides: Optional[Dict] = None,
    ) -> Dict:
        strategy_mode = str(strategy_mode).lower().strip()
        strategy_overrides = strategy_overrides or {}
        k_reretrieve_second_raw = strategy_overrides.get("k_reretrieve_second", None)
        k_reretrieve_second: Optional[int] = None
        if k_reretrieve_second_raw not in (None, "", 0):
            try:
                parsed_k2 = int(k_reretrieve_second_raw)
                if parsed_k2 > int(k_reretrieve):
                    k_reretrieve_second = parsed_k2
            except (TypeError, ValueError):
                k_reretrieve_second = None

        initial_docs = self._retrieve(question, k_initial)
        initial_contexts = [d["text"] for d in initial_docs]
        initial_doc_ids = [d["doc_id"] for d in initial_docs]
        initial_scores = [float(d["score"]) for d in initial_docs]
        oracle_answerable, oracle_meta = self._estimate_oracle_answerable(
            question=question,
            gold_answer=gold_answer,
            contexts=initial_contexts,
        )

        checker_label = "SKIP"
        checker_score = -1.0
        checker_meta: Dict = {}
        uncertainty_meta: Dict = {}

        if checker is not None:
            checker_label, checker_score, checker_meta = checker.predict(question, initial_contexts)

        final_docs = initial_docs
        strategy_used = "baseline"
        answer = self.abstain_text

        if strategy_mode == "baseline":
            answer = self._generate_answer(question, final_docs)
            strategy_used = "baseline"

        elif strategy_mode == "uncertainty_abstain":
            unc_cfg = dict(self.config.get("strategy", {}).get("uncertainty", {}))
            unc_cfg.update(strategy_overrides)
            metric = str(unc_cfg.get("metric", "avg_token_prob")).strip().lower()
            threshold = float(unc_cfg.get("threshold", 0.20))
            gen_cfg = self.config.get("generator", {})

            out = self.generator.generate_answer_with_uncertainty(
                question=question,
                contexts=initial_contexts,
                abstain_text=self.abstain_text,
                prompt_style=self.prompt_style,
                max_new_tokens=int(gen_cfg.get("max_new_tokens", 32)),
                temperature=float(gen_cfg.get("temperature", 0.2)),
                do_sample=bool(gen_cfg.get("do_sample", False)),
            )
            raw_answer = str(out.get("text", self.abstain_text))
            if metric == "entropy_confidence":
                confidence = float(out.get("entropy_confidence", 0.0))
            elif metric == "avg_logprob":
                confidence = float(max(0.0, min(1.0, np.exp(float(out.get("avg_token_logprob", -20.0))))))
            else:
                metric = "avg_token_prob"
                confidence = float(out.get("avg_token_prob", 0.0))
            confidence = float(max(0.0, min(1.0, confidence)))
            checker_label = SUFFICIENT if confidence >= threshold else INSUFFICIENT
            checker_score = confidence
            checker_meta = {
                "mode": "uncertainty_baseline",
                "metric": metric,
                "threshold": threshold,
            }
            uncertainty_meta = {
                "metric": metric,
                "threshold": threshold,
                "avg_token_logprob": float(out.get("avg_token_logprob", -20.0)),
                "avg_token_prob": float(out.get("avg_token_prob", 0.0)),
                "avg_token_entropy": float(out.get("avg_token_entropy", 0.0)),
                "entropy_confidence": float(out.get("entropy_confidence", 0.0)),
                "token_count": int(out.get("token_count", 0)),
            }
            if checker_label == INSUFFICIENT:
                answer = self.abstain_text
                strategy_used = f"uncertainty_abstain({metric})"
            else:
                answer = raw_answer
                strategy_used = f"uncertainty_generate({metric})"

        elif strategy_mode in {"bm25_score_threshold", "retrieval_score_threshold_abstain"}:
            # BM25 계열 점수 임계값 baseline.
            # 현재 파이프라인이 dense/BM25를 모두 지원할 수 있으므로, top-k 점수로부터
            # query-local confidence(softmax top1 prob)를 계산해 임계값과 비교한다.
            bm25_cfg = self.config.get("strategy", {}).get("bm25_threshold", {})
            th = float(strategy_overrides.get("threshold", bm25_cfg.get("threshold", 0.50)))
            metric = str(strategy_overrides.get("score_metric", bm25_cfg.get("score_metric", "softmax_top1"))).strip().lower()
            scores = np.array([float(s) for s in initial_scores], dtype=float)
            if scores.size <= 0:
                confidence = 0.0
            else:
                if metric == "sigmoid_max":
                    confidence = float(1.0 / (1.0 + np.exp(-float(np.max(scores)))))
                else:
                    # default: softmax top1 probability
                    shifted = scores - float(np.max(scores))
                    exp_scores = np.exp(shifted)
                    confidence = float(exp_scores[0] / max(1e-12, float(np.sum(exp_scores))))
                    metric = "softmax_top1"

            checker_score = float(max(0.0, min(1.0, confidence)))
            checker_label = SUFFICIENT if checker_score >= th else INSUFFICIENT
            checker_meta = {
                "mode": "bm25_score_threshold",
                "metric": metric,
                "threshold": th,
                "top1_score_raw": float(initial_scores[0]) if initial_scores else 0.0,
                "topk_scores_raw": [float(x) for x in initial_scores],
            }
            if checker_label == INSUFFICIENT:
                answer = self.abstain_text
                strategy_used = "bm25_threshold_abstain"
            else:
                answer = self._generate_answer(question, final_docs)
                strategy_used = "bm25_threshold_generate"

        elif strategy_mode == "random_abstain":
            random_cfg = self.config.get("strategy", {}).get("random_abstain", {})
            rate = float(strategy_overrides.get("abstain_rate", random_cfg.get("abstain_rate", 0.1)))
            rate = float(max(0.0, min(1.0, rate)))
            # 재현성을 위해 질문과 시드를 이용한 deterministic pseudo-random score 사용
            rnd = random.Random(f"{self.seed}::{question}").random()
            checker_score = float(1.0 - rnd)
            checker_label = INSUFFICIENT if rnd < rate else SUFFICIENT
            checker_meta = {
                "mode": "random_abstain",
                "abstain_rate": rate,
                "random_draw": float(rnd),
            }
            if checker_label == INSUFFICIENT:
                answer = self.abstain_text
                strategy_used = "random_abstain"
            else:
                answer = self._generate_answer(question, final_docs)
                strategy_used = "random_generate"

        elif strategy_mode == "flare_lite":
            # FLARE 경량 근사: 저신뢰 생성 시 질의 확장 재검색 후 재생성
            flare_cfg = dict(self.config.get("strategy", {}).get("flare", {}))
            flare_cfg.update(strategy_overrides)
            metric = str(flare_cfg.get("metric", "avg_token_prob")).strip().lower()
            conf_th = float(flare_cfg.get("confidence_threshold", 0.35))
            max_rounds = max(1, int(flare_cfg.get("max_rounds", 2)))
            k_growth = max(1, int(flare_cfg.get("k_growth", 3)))
            query_with_draft = bool(flare_cfg.get("query_with_draft", True))
            abstain_on_low_conf = bool(flare_cfg.get("abstain_on_low_conf", False))

            gen_cfg = self.config.get("generator", {})
            round_metas: List[Dict] = []
            current_docs = list(initial_docs)
            current_question = str(question)

            def _pick_confidence(out_dict: Dict, m: str) -> float:
                if m == "entropy_confidence":
                    return float(out_dict.get("entropy_confidence", 0.0))
                if m == "avg_logprob":
                    return float(max(0.0, min(1.0, np.exp(float(out_dict.get("avg_token_logprob", -20.0))))))
                return float(out_dict.get("avg_token_prob", 0.0))

            last_out: Dict = {"text": self.abstain_text}
            last_conf = 0.0
            used_rounds = 0
            for round_idx in range(max_rounds):
                used_rounds = round_idx + 1
                current_contexts = [d["text"] for d in current_docs]
                out = self.generator.generate_answer_with_uncertainty(
                    question=current_question,
                    contexts=current_contexts,
                    abstain_text=self.abstain_text,
                    prompt_style=self.prompt_style,
                    max_new_tokens=int(gen_cfg.get("max_new_tokens", 32)),
                    temperature=float(gen_cfg.get("temperature", 0.2)),
                    do_sample=bool(gen_cfg.get("do_sample", False)),
                )
                conf = float(max(0.0, min(1.0, _pick_confidence(out, metric))))
                last_out = out
                last_conf = conf
                round_metas.append(
                    {
                        "round": round_idx + 1,
                        "k": len(current_docs),
                        "confidence": conf,
                        "metric": metric,
                        "draft_answer": str(out.get("text", ""))[:160],
                    }
                )

                # 충분히 신뢰되면 조기 종료
                if conf >= conf_th:
                    break

                # 마지막 라운드면 더 이상 재검색하지 않음
                if round_idx >= max_rounds - 1:
                    break

                next_k = int(k_reretrieve + (k_growth * round_idx))
                next_k = max(next_k, int(k_reretrieve))
                draft = str(out.get("text", "")).strip()
                if query_with_draft and draft and draft != self.abstain_text:
                    next_query = f"{question} {draft}"
                else:
                    next_query = str(question)
                current_docs = self._retrieve(next_query, next_k)
                current_question = str(question)

            final_docs = current_docs
            checker_score = float(last_conf)
            checker_label = SUFFICIENT if checker_score >= conf_th else INSUFFICIENT
            checker_meta = {
                "mode": "flare_lite",
                "metric": metric,
                "confidence_threshold": conf_th,
                "max_rounds": max_rounds,
                "k_growth": k_growth,
                "query_with_draft": query_with_draft,
                "abstain_on_low_conf": abstain_on_low_conf,
                "rounds_used": used_rounds,
                "rounds": round_metas,
            }
            uncertainty_meta = {
                "metric": metric,
                "threshold": conf_th,
                "avg_token_logprob": float(last_out.get("avg_token_logprob", -20.0)),
                "avg_token_prob": float(last_out.get("avg_token_prob", 0.0)),
                "avg_token_entropy": float(last_out.get("avg_token_entropy", 0.0)),
                "entropy_confidence": float(last_out.get("entropy_confidence", 0.0)),
                "token_count": int(last_out.get("token_count", 0)),
            }

            if checker_label == INSUFFICIENT and abstain_on_low_conf:
                answer = self.abstain_text
                strategy_used = "flare_lite_abstain"
            else:
                answer = str(last_out.get("text", self.abstain_text)).strip() or self.abstain_text
                strategy_used = "flare_lite_generate"

        elif checker is None:
            # checker 없이 abstain/reretrieve/hybrid를 호출하면 baseline으로 간주
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
                stage_meta: Dict[str, Dict] = {"1차": checker_meta}
                if checker is not None:
                    label2, score2, meta2 = checker.predict(question, [d["text"] for d in final_docs])
                    checker_label = label2
                    checker_score = score2
                    stage_meta["2차"] = meta2
                    if label2 == INSUFFICIENT and k_reretrieve_second is not None:
                        final_docs = self._retrieve(question, k_reretrieve_second)
                        label3, score3, meta3 = checker.predict(question, [d["text"] for d in final_docs])
                        checker_label = label3
                        checker_score = score3
                        stage_meta["3차"] = meta3
                        strategy_used = "reretrieve2_generate"
                    else:
                        strategy_used = "reretrieve1_generate"
                    checker_meta = stage_meta
            else:
                strategy_used = "initial_generate"
            answer = self._generate_answer(question, final_docs)

        elif strategy_mode == "hybrid":
            if checker_label == INSUFFICIENT:
                final_docs = self._retrieve(question, k_reretrieve)
                label2, score2, meta2 = checker.predict(question, [d["text"] for d in final_docs])
                stage_meta: Dict[str, Dict] = {"1차": checker_meta, "2차": meta2}
                checker_label = label2
                checker_score = score2
                reretrieve_stage = "reretrieve1"
                if label2 == INSUFFICIENT and k_reretrieve_second is not None:
                    final_docs = self._retrieve(question, k_reretrieve_second)
                    label3, score3, meta3 = checker.predict(question, [d["text"] for d in final_docs])
                    checker_label = label3
                    checker_score = score3
                    stage_meta["3차"] = meta3
                    reretrieve_stage = "reretrieve2"
                checker_meta = stage_meta
                hybrid_cfg = self.config.get("strategy", {}).get("hybrid", {})
                reretrieve_then_abstain = bool(hybrid_cfg.get("reretrieve_then_abstain", True))
                if checker_label == INSUFFICIENT and reretrieve_then_abstain:
                    answer = self.abstain_text
                    strategy_used = f"hybrid_{reretrieve_stage}_abstain"
                else:
                    answer = self._generate_answer(question, final_docs)
                    strategy_used = (
                        f"hybrid_{reretrieve_stage}_generate"
                        if checker_label != INSUFFICIENT
                        else f"hybrid_{reretrieve_stage}_generate_forced"
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
            "initial_retrieved_doc_ids": initial_doc_ids,
            "initial_retrieved_scores": initial_scores,
            "initial_contexts": initial_contexts,
            "retrieved_doc_ids": [d["doc_id"] for d in final_docs],
            "retrieved_scores": [float(d["score"]) for d in final_docs],
            "retrieved_contexts": [str(d.get("text", "")) for d in final_docs],
            "oracle_answerable": int(oracle_answerable),
            "oracle_answerable_meta": oracle_meta,
            "uncertainty_meta": uncertainty_meta,
        }

    def run_experiment(
        self,
        run_name: str,
        strategy_mode: str = "baseline",
        checker_name: Optional[str] = None,
        checker_overrides: Optional[Dict] = None,
        strategy_overrides: Optional[Dict] = None,
        k_initial: Optional[int] = None,
        k_reretrieve: Optional[int] = None,
        baseline_records: Optional[List[Dict]] = None,
    ) -> Tuple[Dict, List[Dict], Dict]:
        k_initial = int(k_initial if k_initial is not None else self.k_initial)
        k_reretrieve = int(k_reretrieve if k_reretrieve is not None else self.k_reretrieve)
        strategy_mode = str(strategy_mode).lower().strip()
        effective_checker_name = checker_name
        if checker_name is None and strategy_mode == "uncertainty_abstain":
            effective_checker_name = "uncertainty_baseline"
        if checker_name is None and strategy_mode in {"bm25_score_threshold", "retrieval_score_threshold_abstain"}:
            effective_checker_name = "bm25_score_threshold"
        if checker_name is None and strategy_mode == "random_abstain":
            effective_checker_name = "random_abstain"
        if checker_name is None and strategy_mode == "flare_lite":
            effective_checker_name = "flare_lite"
        checker = self._build_checker(checker_name, checker_overrides)
        run_cfg = self.config.get("run", {})
        progress_every = int(run_cfg.get("progress_every", 20))
        progress_every_env = str(os.environ.get("CS_PROGRESS_EVERY", "")).strip()
        if progress_every_env:
            try:
                progress_every = max(1, int(progress_every_env))
            except ValueError:
                pass

        print(
            f"[실험 시작] 이름={run_name} | 전략={strategy_mode} | 체커={effective_checker_name or '없음'} | "
            f"k초기={k_initial}, k재검색={k_reretrieve}"
        )

        records: List[Dict] = []
        for idx, sample in enumerate(tqdm(self.examples, desc=f"{run_name} 실행", leave=False), start=1):
            ts = time.perf_counter()
            out = self._run_single(
                question=sample["question"],
                gold_answer=sample["gold_answer"],
                checker=checker,
                strategy_mode=strategy_mode,
                k_initial=k_initial,
                k_reretrieve=k_reretrieve,
                strategy_overrides=strategy_overrides,
            )
            latency_ms = (time.perf_counter() - ts) * 1000.0

            em, f1 = compute_em_f1(out["answer"], sample["gold_answer"])
            is_abstain = int(str(out["answer"]).strip() == self.abstain_text)
            is_correct = int(em == 1.0)

            record = {
                "question_id": sample["question_id"],
                "question": sample["question"],
                "gold_answer": sample["gold_answer"],
                "initial_retrieved_doc_ids": out["initial_retrieved_doc_ids"],
                "initial_retrieved_scores": out["initial_retrieved_scores"],
                "initial_contexts": out.get("initial_contexts", []),
                "initial_max_retrieval_score": max(out["initial_retrieved_scores"]) if out["initial_retrieved_scores"] else 0.0,
                "initial_mean_retrieval_score": (
                    float(np.mean(out["initial_retrieved_scores"])) if out["initial_retrieved_scores"] else 0.0
                ),
                "retrieved_doc_ids": out["retrieved_doc_ids"],
                "retrieved_scores": out["retrieved_scores"],
                "retrieved_contexts": out.get("retrieved_contexts", []),
                "checker_name": effective_checker_name or "none",
                "checker_label": out["checker_label"],
                "checker_score": out["checker_score"],
                "estimated_answerable_prob": float(out["checker_score"]) if float(out["checker_score"]) >= 0 else -1.0,
                "oracle_answerable": int(out["oracle_answerable"]),
                "oracle_answerable_mode": str(out.get("oracle_answerable_meta", {}).get("mode", self.answerable_mode)),
                "oracle_answerable_score": float(out.get("oracle_answerable_meta", {}).get("score", -1.0)),
                "oracle_answerable_matched_answer": str(
                    out.get("oracle_answerable_meta", {}).get("matched_answer", "")
                ),
                "strategy_used": out["strategy_used"],
                "final_answer": out["answer"],
                "is_correct": is_correct,
                "is_abstain": is_abstain,
                "latency_ms": latency_ms,
                "em": em,
                "f1": f1,
                "checker_meta": out["checker_meta"],
                "uncertainty_metric": str(out.get("uncertainty_meta", {}).get("metric", "")),
                "uncertainty_threshold": out.get("uncertainty_meta", {}).get("threshold", ""),
                "generation_avg_token_logprob": out.get("uncertainty_meta", {}).get("avg_token_logprob", ""),
                "generation_avg_token_prob": out.get("uncertainty_meta", {}).get("avg_token_prob", ""),
                "generation_avg_token_entropy": out.get("uncertainty_meta", {}).get("avg_token_entropy", ""),
                "generation_entropy_confidence": out.get("uncertainty_meta", {}).get("entropy_confidence", ""),
                "embedder_device": str(self.embedder.device),
                "generator_device": str(getattr(self.generator, "runtime_device_label", str(self.generator.device))),
            }
            records.append(record)

            if progress_every > 0 and (idx == 1 or idx % progress_every == 0):
                print(
                    f"[진행] {run_name}: {idx}/{len(self.examples)} | "
                    f"질문ID={sample['question_id']} | 지연={latency_ms:.1f}ms"
                )

        eval_cfg = self.config.get("evaluation", {})
        calibration_bins = int(eval_cfg.get("calibration_bins", 10))
        temp_cfg = eval_cfg.get("calibration", {}).get("temperature_scaling", {})
        row, checker_analysis = summarize_for_report(
            records=records,
            run_name=run_name,
            checker=effective_checker_name or "없음",
            strategy=strategy_mode,
            abstain_text=self.abstain_text,
            calibration_bins=calibration_bins,
            calibration_temperature_scaling=bool(temp_cfg.get("enabled", True)),
            calibration_val_ratio=float(temp_cfg.get("validation_ratio", 0.3)),
            calibration_seed=int(temp_cfg.get("seed", self.seed)),
            calibration_min_samples=int(temp_cfg.get("min_samples", 50)),
        )

        latency_cfg = self.config.get("evaluation", {}).get("latency", {})
        latency_analysis = latency_quality_analysis(
            records=records,
            warmup_drop=int(latency_cfg.get("warmup_drop", 5)),
            hist_bins=int(latency_cfg.get("hist_bins", 20)),
        )
        row["평균지연(ms,warmup제외)"] = float(latency_analysis.get("mean_ms_wo_warmup", 0.0))
        row["지연표준편차(ms)"] = float(latency_analysis.get("std_ms", 0.0))
        row["지연P50(ms)"] = float(latency_analysis.get("p50_ms", 0.0))
        row["지연P95(ms)"] = float(latency_analysis.get("p95_ms", 0.0))
        dev_map = dict(latency_analysis.get("devices", {}))
        row["CPU평균지연(ms)"] = float(dev_map.get("cpu", {}).get("mean_ms", 0.0)) if "cpu" in dev_map else ""
        row["MPS평균지연(ms)"] = float(dev_map.get("mps", {}).get("mean_ms", 0.0)) if "mps" in dev_map else ""
        row["실행장치"] = ",".join(sorted(dev_map.keys()))
        if str(effective_checker_name or "").strip().lower() not in {"", "none", "없음"}:
            print(
                "[체커 진단] "
                f"판정수={row.get('체커판정수', '-')}, "
                f"파싱실패수={row.get('체커파싱실패수', '-')}, "
                f"파싱성공률={row.get('체커파싱성공률', '-')}, "
                f"ECE(before→after)={row.get('CSC_ECE_before', '-')}→{row.get('CSC_ECE_after', '-')}, "
                f"AUROC={row.get('CSC_AUROC', '-')}, T={row.get('CSC_Temperature', '-')}"
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

        artifact_paths: Dict[str, str] = {"jsonl_path": str(jsonl_path)}
        latency_dir = self.output_dir / "지연분석"
        latency_saved = save_latency_artifacts(
            analysis=latency_analysis,
            output_dir=latency_dir,
            run_name=run_name,
        )
        artifact_paths.update(latency_saved)
        print(f"[저장] 지연 분석 산출물: {latency_dir}")
        if checker_analysis:
            curve_dir = self.output_dir / "진단곡선"
            saved = save_checker_artifacts(analysis=checker_analysis, output_dir=curve_dir, run_name=run_name)
            artifact_paths.update(saved)
            print(f"[저장] 체커 진단 산출물: {curve_dir}")

        return row, records, artifact_paths
