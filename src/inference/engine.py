"""vLLM-based inference engine for Nemotron model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import InferenceConfig

logger = logging.getLogger(__name__)


class NemotronEngine:
    """Wrapper around vLLM for batch inference.

    Handles both merged model checkpoints and base models with LoRA adapters.
    Heavy imports (vllm, torch) are deferred to __init__ so the module can be
    imported without GPU access (e.g. for CLI parsing / testing).
    """

    def __init__(
        self,
        model_path: str,
        config: InferenceConfig,
        adapter_path: Optional[str] = None,
    ):
        """Initialize vLLM engine.

        Args:
            model_path: Path or HuggingFace ID for the base / merged model.
            config: InferenceConfig with engine parameters.
            adapter_path: Optional path to a LoRA adapter directory.
                If provided, the base model is loaded and the adapter is applied.
        """
        from vllm import LLM

        self.config = config
        self.model_path = model_path
        self.adapter_path = adapter_path

        llm_kwargs: Dict[str, Any] = {
            "model": model_path,
            "trust_remote_code": True,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "dtype": "bfloat16",
            "max_model_len": config.max_new_tokens * 2,  # room for prompt + completion
        }

        if adapter_path is not None:
            llm_kwargs["enable_lora"] = True
            logger.info(f"Loading base model '{model_path}' with LoRA adapter '{adapter_path}'")
        else:
            logger.info(f"Loading merged model from '{model_path}'")

        self.llm = LLM(**llm_kwargs)
        logger.info("vLLM engine initialised successfully")

    def _build_sampling_params(self, **overrides: Any):
        """Build vLLM SamplingParams from config with optional overrides."""
        from vllm import SamplingParams

        params = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_new_tokens,
        }
        params.update(overrides)
        return SamplingParams(**params)

    def generate_batch(
        self,
        prompts: list[str],
        num_generations: int = 1,
        **sampling_overrides: Any,
    ) -> list[list[str]]:
        """Generate num_generations responses per prompt via vLLM.

        Expands prompts so that each prompt appears num_generations times,
        then reshapes outputs back to list-of-lists.

        Args:
            prompts: List of input prompts.
            num_generations: Number of independent generations per prompt.
            **sampling_overrides: Override temperature, top_p, max_tokens, etc.

        Returns:
            List of lists: result[i] contains num_generations responses
            for prompts[i].
        """
        from vllm import LoRARequest

        sampling_params = self._build_sampling_params(**sampling_overrides)

        # Expand prompts: each prompt repeated num_generations times
        expanded = [p for p in prompts for _ in range(num_generations)]

        logger.info(
            f"Generating {len(expanded)} total completions "
            f"({len(prompts)} prompts x {num_generations} generations), "
            f"temperature={sampling_params.temperature}, "
            f"max_tokens={sampling_params.max_tokens}"
        )

        generate_kwargs: Dict[str, Any] = {
            "prompts": expanded,
            "sampling_params": sampling_params,
        }

        if self.adapter_path is not None:
            generate_kwargs["lora_request"] = LoRARequest(
                "adapter", 1, self.adapter_path
            )

        outputs = self.llm.generate(**generate_kwargs)

        # Extract text from outputs
        texts = [output.outputs[0].text for output in outputs]

        # Reshape: group by prompt
        results: list[list[str]] = []
        for i in range(len(prompts)):
            start = i * num_generations
            end = start + num_generations
            results.append(texts[start:end])

        return results

    def generate_single(self, prompt: str, **kwargs: Any) -> str:
        """Single generation for debugging / interactive use.

        Args:
            prompt: Input prompt string.
            **kwargs: Forwarded to generate_batch as sampling overrides.

        Returns:
            Single response string.
        """
        results = self.generate_batch([prompt], num_generations=1, **kwargs)
        return results[0][0]
