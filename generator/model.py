from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

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
    ) -> None:
        self.model_name = model_name
        self.max_input_length = int(max_input_length)
        self.device = torch.device(self._resolve_device(device_preference or ["mps", "cpu"]))
        self.default_params = GenerationParams(
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            do_sample=bool(do_sample),
        )

        if num_threads is not None and num_threads > 0:
            torch.set_num_threads(int(num_threads))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))

        if self.is_encoder_decoder:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device)
        self.model.eval()

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
        if self.device.type == "cpu":
            return
        self.model.to("cpu")
        self.device = torch.device("cpu")

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

        if temperature <= 0:
            do_sample = False
            temperature = 1.0

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
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        except RuntimeError as exc:
            if self.device.type != "mps":
                raise exc
            self._fallback_to_cpu()
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
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

        if temperature <= 0:
            do_sample = False
            temperature = 1.0

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
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
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
