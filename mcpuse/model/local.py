from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
torch._dynamo.config.suppress_errors = True
import threading
from .base import BaseLLM

class TransformersLLM(BaseLLM):
    def __init__(self, model_path, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def stream_generate(self, prompt, max_new_tokens=512, temperature=0.7):
        messages = [{"role": "user", "content": prompt.strip()}]
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt_text = prompt

        input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)

        thread = threading.Thread(
            target=self.model.generate,
            kwargs={
                "input_ids": input_ids,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "streamer": streamer,
            }
        )
        thread.start()
        for token in streamer:
            yield token

    def generate(self, user_input: str, max_new_tokens: int = 512):
        response = ""
        for token in self.stream_generate(user_input, max_new_tokens=max_new_tokens):
            response += token
        return response
