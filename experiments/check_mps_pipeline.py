from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from generator.model import LocalHFTextGenerator
from pipeline import load_config
from retrieval.embedder import LocalEmbedder
from sufficiency.entailment_checker import EntailmentChecker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPS 기반 파이프라인 장치 점검(임베딩/생성/NLI)")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--skip-entailment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_cfg = cfg.get("run", {})
    device_pref = run_cfg.get("device_preference", ["mps", "cpu"])

    print("[점검] torch")
    print(f"- torch={torch.__version__}")
    print(f"- mps_built={bool(hasattr(torch.backends, 'mps') and torch.backends.mps.is_built())}")
    print(f"- mps_available={bool(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())}")
    print(f"- device_preference={device_pref}")

    retrieval_cfg = cfg.get("retrieval", {})
    embedder = LocalEmbedder(
        model_name=str(retrieval_cfg.get("embed_model", "intfloat/e5-small-v2")),
        model_type=str(retrieval_cfg.get("model_type", "sentence_transformer")),
        query_model_name=retrieval_cfg.get("query_model_name"),
        corpus_model_name=retrieval_cfg.get("corpus_model_name"),
        device_preference=device_pref,
        batch_size=1,
        normalize_embeddings=bool(retrieval_cfg.get("normalize_embeddings", True)),
        max_length=int(retrieval_cfg.get("max_length", 256)),
    )
    _ = embedder.encode_queries(["MPS 장치 점검 질문"])
    print("[점검] 임베딩")
    print(f"- embedder_device={embedder.device}")

    gen_cfg = cfg.get("generator", {})
    generator = LocalHFTextGenerator(
        model_name=str(gen_cfg.get("model_name", "google/flan-t5-large")),
        device_preference=device_pref,
        max_input_length=int(gen_cfg.get("max_input_length", 1024)),
        max_new_tokens=8,
        temperature=0.0,
        do_sample=False,
        num_threads=gen_cfg.get("num_threads"),
        torch_dtype=gen_cfg.get("torch_dtype"),
        low_cpu_mem_usage=bool(gen_cfg.get("low_cpu_mem_usage", True)),
        trust_remote_code=bool(gen_cfg.get("trust_remote_code", False)),
        backend=str(gen_cfg.get("backend", "transformers")),
        allow_backend_fallback=bool(gen_cfg.get("allow_backend_fallback", True)),
        ollama_url=str(gen_cfg.get("ollama_url", "http://localhost:11434/api/generate")),
        ollama_timeout_sec=int(gen_cfg.get("ollama_timeout_sec", 180)),
    )
    _ = generator.generate_from_prompt("질문: 한국의 수도는?\n답:", max_new_tokens=4, temperature=0.0, do_sample=False)
    print("[점검] 생성")
    print(f"- generator_backend={generator.backend}")
    print(f"- generator_device={generator.runtime_device_label}")

    if not bool(args.skip_entailment):
        ent_cfg = cfg.get("sufficiency", {}).get("entailment", {})
        checker = EntailmentChecker(
            model_name=str(ent_cfg.get("model_name", "cross-encoder/nli-distilroberta-base")),
            sufficient_if_entail_prob_ge=float(ent_cfg.get("sufficient_if_entail_prob_ge", 0.6)),
            device_preference=device_pref,
        )
        _ = checker.score_entailment(
            premise="Seoul is the capital city of South Korea.",
            hypothesis="The answer to the question 'What is the capital of South Korea?' is 'Seoul'.",
        )
        print("[점검] NLI")
        print(f"- entailment_model={checker.model_name}")
        print(f"- entailment_device={checker.device}")

    print("\n결론: 위 장치 로그를 기준으로 MPS 활용 여부를 확인하세요.")


if __name__ == "__main__":
    main()

