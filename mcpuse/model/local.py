from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import threading
from .base import BaseLLM

class TransformersLLM(BaseLLM):
    def __init__(self, model_path: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        # set pad_token for generation (required by some models)
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def stream_generate(self, user_input: str, max_new_tokens: int = 512, temperature: float = 0.7):
        # Build chat prompt using tokenizer's chat template
        # If your model doesn't support chat template, fall back to plain text
        try:
            messages = [{"role": "user", "content": user_input.strip()}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = user_input  # fallback: plain prompt

        # Encode prompt as input IDs
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

        # Setup streaming
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        gen_kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for token in streamer:
            yield token

    def generate(self, user_input: str, max_new_tokens: int = 512):
        response = ""
        for token in self.stream_generate(user_input, max_new_tokens=max_new_tokens):
            response += token
        return response
