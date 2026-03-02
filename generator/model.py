from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

from generator.utils import build_qa_prompt, clean_generated_text


@dataclass
class GenerationParams:
    max_new_tokens: int = 32
    temperature: float = 0.2
    do_sample: bool = False


class LocalHFTextGenerator:
    """HuggingFace 생성 모델 래퍼(MPS/CPU 우선)."""

    def __init__(
        self,
        model_name: str,
        device_preference: Sequence[str] | None = None,
        max_input_length: int = 1024,
        max_new_tokens: int = 32,
        temperature: float = 0.2,
        do_sample: bool = False,
        num_threads: Optional[int] = None,
        torch_dtype: Optional[str] = None,
        low_cpu_mem_usage: bool = True,
        trust_remote_code: bool = False,
        backend: str = "transformers",
        allow_backend_fallback: bool = True,
        ollama_url: str = "http://localhost:11434/api/generate",
        ollama_timeout_sec: int = 180,
    ) -> None:
        self.model_name = model_name
        self.max_input_length = int(max_input_length)
        self.backend = str(backend).strip().lower() or "transformers"
        if self.backend not in {"transformers", "auto", "ollama"}:
            raise ValueError(f"지원하지 않는 generator.backend: {backend}")
        self.allow_backend_fallback = bool(allow_backend_fallback)
        self.ollama_url = str(ollama_url).strip() or "http://localhost:11434/api/generate"
        self.ollama_timeout_sec = int(max(5, ollama_timeout_sec))

        self.device = torch.device(self._resolve_device(device_preference or ["mps", "cpu"]))
        self.default_params = GenerationParams(
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            do_sample=bool(do_sample),
        )

        self.tokenizer = None
        self.model = None
        self.is_encoder_decoder = False

        if num_threads is not None and num_threads > 0:
            torch.set_num_threads(int(num_threads))

        if self.backend == "ollama":
            self._check_ollama_available()
            return

        try:
            self._init_transformers_model(
                trust_remote_code=bool(trust_remote_code),
                low_cpu_mem_usage=bool(low_cpu_mem_usage),
                torch_dtype=torch_dtype,
            )
            self.backend = "transformers"
        except Exception as exc:
            if self.backend == "auto" and self.allow_backend_fallback:
                self.backend = "ollama"
                self._check_ollama_available()
                print(
                    "[생성기] transformers 로딩 실패로 ollama backend로 전환합니다: "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                raise

    @property
    def runtime_device_label(self) -> str:
        if self.backend == "ollama":
            return "metal(ollama)"
        return str(self.device)

    def _init_transformers_model(
        self,
        trust_remote_code: bool,
        low_cpu_mem_usage: bool,
        torch_dtype: Optional[str],
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=bool(trust_remote_code),
        )
        config = AutoConfig.from_pretrained(
            self.model_name,
            trust_remote_code=bool(trust_remote_code),
        )
        self.is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))

        model_kwargs: Dict = {
            "low_cpu_mem_usage": bool(low_cpu_mem_usage),
            "trust_remote_code": bool(trust_remote_code),
        }
        parsed_dtype = self._parse_torch_dtype(torch_dtype)
        if parsed_dtype is not None:
            # transformers 최신 버전 권장 키워드(dtype) 우선 사용
            model_kwargs["dtype"] = parsed_dtype

        model_cls = AutoModelForSeq2SeqLM if self.is_encoder_decoder else AutoModelForCausalLM
        try:
            self.model = model_cls.from_pretrained(self.model_name, **model_kwargs)
        except TypeError as exc:
            # 구버전 호환: dtype 미지원 시 torch_dtype로 재시도
            if parsed_dtype is None or "dtype" not in str(exc):
                raise
            legacy_kwargs = dict(model_kwargs)
            legacy_kwargs.pop("dtype", None)
            legacy_kwargs["torch_dtype"] = parsed_dtype
            self.model = model_cls.from_pretrained(self.model_name, **legacy_kwargs)

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device)
        self.model.eval()

    def _check_ollama_available(self) -> None:
        if requests is None:
            raise RuntimeError("ollama backend 사용을 위해 requests 패키지가 필요합니다.")
        tags_url = self.ollama_url.replace("/api/generate", "/api/tags")
        try:
            resp = requests.get(tags_url, timeout=5)
            resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"ollama 서버 접근 실패: {tags_url} ({exc})\n"
                "대안: `ollama serve` 실행 또는 generator.backend를 transformers로 변경"
            ) from exc

    @staticmethod
    def _parse_torch_dtype(dtype_name: Optional[str]):
        if dtype_name is None:
            return None
        key = str(dtype_name).strip().lower()
        if key in {"", "none", "auto"}:
            return None
        if key in {"float16", "fp16", "half"}:
            return torch.float16
        if key in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if key in {"float32", "fp32"}:
            return torch.float32
        raise ValueError(f"지원하지 않는 torch_dtype: {dtype_name}")

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

    def _fallback_to_cpu(self) -> None:
        if self.backend != "transformers":
            return
        if self.device.type == "cpu":
            return
        self.model.to("cpu")
        self.device = torch.device("cpu")

    def _build_generate_kwargs(
        self,
        *,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        with_scores: bool = False,
    ) -> Dict:
        kwargs: Dict = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if bool(do_sample):
            kwargs["temperature"] = float(max(1e-5, temperature))
        else:
            # deterministic 생성 시 sampling 관련 경고(top_p ignored) 방지
            kwargs["temperature"] = 1.0
            kwargs["top_p"] = 1.0

        if with_scores:
            kwargs["return_dict_in_generate"] = True
            kwargs["output_scores"] = True
        return kwargs

    @staticmethod
    def _estimate_text_confidence(text: str) -> float:
        answer = str(text or "").strip().lower()
        if not answer:
            return 0.05
        if "모르겠습니다" in answer or "알 수 없" in answer:
            return 0.05
        if len(answer.split()) <= 2:
            return 0.45
        return 0.65

    def _ollama_generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        seed: Optional[int],
    ) -> Dict:
        if requests is None:
            raise RuntimeError("ollama backend 사용을 위해 requests 패키지가 필요합니다.")

        options: Dict[str, float | int] = {"num_predict": int(max_new_tokens)}
        options["temperature"] = float(temperature if do_sample else 0.0)
        if seed is not None:
            options["seed"] = int(seed)

        payload = {
            "model": self.model_name,
            "prompt": str(prompt),
            "stream": False,
            "options": options,
        }
        resp = requests.post(
            self.ollama_url,
            json=payload,
            timeout=self.ollama_timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json() if resp.text else {}
        if not isinstance(data, dict):
            data = {}
        return data

    def generate_from_prompt(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> str:
        if seed is not None:
            set_seed(int(seed))

        params = self.default_params
        max_new_tokens = int(max_new_tokens if max_new_tokens is not None else params.max_new_tokens)
        temperature = float(temperature if temperature is not None else params.temperature)
        do_sample = bool(do_sample if do_sample is not None else params.do_sample)

        if self.backend == "ollama":
            data = self._ollama_generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                seed=seed,
            )
            text = clean_generated_text(str(data.get("response", "")))
            return text

        if temperature <= 0:
            do_sample = False
            temperature = 1.0

        if self.tokenizer is None or self.model is None:
            raise RuntimeError("transformers backend 초기화 실패: tokenizer/model이 없습니다.")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self._build_generate_kwargs(
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                    ),
                )
        except RuntimeError as exc:
            if self.device.type != "mps":
                raise exc
            self._fallback_to_cpu()
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self._build_generate_kwargs(
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                    ),
                )

        if self.is_encoder_decoder:
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            prompt_len = inputs["input_ids"].shape[1]
            decoded = self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        return clean_generated_text(decoded)

    def generate_from_prompt_with_stats(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, float | int | str]:
        if seed is not None:
            set_seed(int(seed))

        params = self.default_params
        max_new_tokens = int(max_new_tokens if max_new_tokens is not None else params.max_new_tokens)
        temperature = float(temperature if temperature is not None else params.temperature)
        do_sample = bool(do_sample if do_sample is not None else params.do_sample)

        if self.backend == "ollama":
            data = self._ollama_generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                seed=seed,
            )
            text = clean_generated_text(str(data.get("response", "")))
            token_count = int(data.get("eval_count", 0) or 0)
            if token_count <= 0:
                token_count = max(1, len(text.split()))
            conf = float(max(0.0, min(1.0, self._estimate_text_confidence(text))))
            avg_logprob = float(math.log(max(conf, 1e-8)))
            avg_entropy_norm = float(max(0.0, min(1.0, 1.0 - conf)))
            return {
                "text": text,
                "token_count": int(token_count),
                "avg_token_logprob": avg_logprob,
                "avg_token_prob": conf,
                "avg_token_entropy": float(avg_entropy_norm),
                "avg_token_entropy_norm": avg_entropy_norm,
                "entropy_confidence": conf,
            }

        if temperature <= 0:
            do_sample = False
            temperature = 1.0

        if self.tokenizer is None or self.model is None:
            raise RuntimeError("transformers backend 초기화 실패: tokenizer/model이 없습니다.")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        def _run_generate() -> object:
            with torch.no_grad():
                return self.model.generate(
                    **inputs,
                    **self._build_generate_kwargs(
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                        with_scores=True,
                    ),
                )

        try:
            generated = _run_generate()
        except RuntimeError as exc:
            if self.device.type != "mps":
                raise exc
            self._fallback_to_cpu()
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            generated = _run_generate()

        sequences = generated.sequences[0]
        step_scores = list(generated.scores or [])
        generated_token_count = len(step_scores)
        if generated_token_count > 0:
            generated_token_ids = sequences[-generated_token_count:]
        else:
            generated_token_ids = sequences.new_zeros((0,), dtype=sequences.dtype)

        if self.is_encoder_decoder:
            decoded = self.tokenizer.decode(sequences, skip_special_tokens=True)
        else:
            prompt_len = inputs["input_ids"].shape[1]
            decoded = self.tokenizer.decode(sequences[prompt_len:], skip_special_tokens=True)
        text = clean_generated_text(decoded)

        token_logprobs = []
        token_entropies = []
        token_entropy_norms = []
        for tok, logits in zip(generated_token_ids.tolist(), step_scores):
            step_logits = logits[0]
            log_probs = torch.log_softmax(step_logits, dim=-1)
            probs = torch.exp(log_probs)
            token_logprobs.append(float(log_probs[int(tok)].item()))

            entropy = float((-(probs * log_probs)).sum().item())
            vocab_size = int(step_logits.shape[-1])
            max_entropy = max(1e-12, math.log(max(2, vocab_size)))
            token_entropies.append(entropy)
            token_entropy_norms.append(float(entropy / max_entropy))

        if token_logprobs:
            avg_logprob = float(sum(token_logprobs) / len(token_logprobs))
            avg_prob = float(max(0.0, min(1.0, math.exp(avg_logprob))))
            avg_entropy = float(sum(token_entropies) / len(token_entropies))
            avg_entropy_norm = float(sum(token_entropy_norms) / len(token_entropy_norms))
            entropy_conf = float(max(0.0, min(1.0, 1.0 - avg_entropy_norm)))
        else:
            avg_logprob = -20.0
            avg_prob = 0.0
            avg_entropy = 0.0
            avg_entropy_norm = 1.0
            entropy_conf = 0.0

        return {
            "text": text,
            "token_count": int(len(token_logprobs)),
            "avg_token_logprob": avg_logprob,
            "avg_token_prob": avg_prob,
            "avg_token_entropy": avg_entropy,
            "avg_token_entropy_norm": avg_entropy_norm,
            "entropy_confidence": entropy_conf,
        }

    def generate_answer_with_uncertainty(
        self,
        question: str,
        contexts: Iterable[str],
        abstain_text: str = "모르겠습니다.",
        prompt_style: str = "qa_short_ko",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, float | int | str]:
        if prompt_style != "qa_short_ko":
            raise ValueError(f"지원하지 않는 prompt_style: {prompt_style}")

        prompt = build_qa_prompt(question=question, contexts=contexts, abstain_text=abstain_text)
        out = self.generate_from_prompt_with_stats(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            seed=seed,
        )
        if not str(out.get("text", "")).strip():
            out["text"] = abstain_text
        return out

    def generate_answer(
        self,
        question: str,
        contexts: Iterable[str],
        abstain_text: str = "모르겠습니다.",
        prompt_style: str = "qa_short_ko",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> str:
        if prompt_style != "qa_short_ko":
            raise ValueError(f"지원하지 않는 prompt_style: {prompt_style}")

        prompt = build_qa_prompt(question=question, contexts=contexts, abstain_text=abstain_text)
        output = self.generate_from_prompt(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            seed=seed,
        )
        return output if output else abstain_text
