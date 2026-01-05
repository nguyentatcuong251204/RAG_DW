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
        

        # Automatic CUDA detection
        if not torch.cuda.is_available():
            nf4 = False
            self.device_map = "cpu"
            print("CUDA not available. Falling back to CPU and disabling NF4 quantization.")

        if nf4:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_type=torch.float16,
                llm_int8_enable_fp16_cpu_offload=True
            )
        else:
            nf4_config = None

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            quantization_config=nf4_config, 
            device_map=self.device_map,
            offload_folder="./offload",
            torch_dtype="auto",
            cache_dir="D:/hf_cache"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Cấu hình pipeline với các tham số sinh văn bản
        # Configure generation pipeline with safer decoding defaults
        # We keep max_new_tokens and temperature but also set typical sampling params
        self._pipeline = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self._tokenizer.eos_token_id,
            temperature=self.temperature,
            do_sample=True if self.temperature > 0 else False,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
        )

        self._hf_pipeline = HuggingFacePipeline(pipeline=self._pipeline)
        # Additional defaults for generate() that can be overridden per-call
        self.default_generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": 50,
            "top_p": 0.95,
            "repetition_penalty": 1.1,
        }

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from the model.

        Accepts generation kwargs (e.g., max_new_tokens, temperature, top_k, top_p, repetition_penalty)
        which override the defaults configured for the pipeline.
        """
        gen_kwargs = {**self.default_generation_kwargs, **kwargs}
        # The HuggingFacePipeline wrapper supports passing kwargs via invoke()
        try:
            out = self._hf_pipeline.invoke(prompt, **gen_kwargs)
            return out if isinstance(out, str) else str(out)
        except TypeError:
            # Some wrappers expect a single argument call, fall back to direct callable
            out = self._pipeline(prompt, **gen_kwargs)
            # pipeline returns list of dicts
            if isinstance(out, list) and len(out) > 0 and "generated_text" in out[0]:
                return out[0]["generated_text"]
            return str(out)

    @property
    def llm(self):
        """Returns the LangChain compatible LLM object."""
        return self._hf_pipeline


