from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
torch._dynamo.config.suppress_errors = True
import threading
from .base import BaseLLM

class CodeLLM(BaseLLM):
    def __init__(self, model_path, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def stream_generate(self, code_input, max_new_tokens=512):
        input_ids = self.tokenizer(code_input, return_tensors="pt").input_ids.to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)

        thread = threading.Thread(
            target=self.model.generate,
            kwargs={"input_ids": input_ids, "max_new_tokens": max_new_tokens, "streamer": streamer}
        )
        thread.start()

        for token in streamer:
            yield token

    def generate(self, code_input, max_new_tokens=512):
        return ''.join(self.stream_generate(code_input, max_new_tokens))
