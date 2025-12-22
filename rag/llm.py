from typing import Dict
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)
from langchain_huggingface import HuggingFacePipeline


class LLMModel:
    """Wraps a HuggingFace causal LM with a generation pipeline and LangChain wrapper."""

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        max_new_tokens: int = 512,
        device_map: str = "auto",
        nf4: bool = True,
        temperature: float = 0.1,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.temperature = temperature

        if nf4:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_type=torch.float16,
            )
        else:
            nf4_config = None

        torch.cuda.empty_cache()
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, quantization_config=nf4_config, device_map=self.device_map
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self._pipeline = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        self._hf_pipeline = HuggingFacePipeline(
            pipeline=self._pipeline, model_kwargs={"temperature": self.temperature}
        )

    def generate(self, prompt: str, **kwargs) -> str:
        # Uses the LangChain wrapper's invoke for consistency
        return self._hf_pipeline.invoke(prompt)

    @property
    def raw_pipeline(self):
        return self._pipeline

    @property
    def tokenizer(self):
        return self._tokenizer

