from transformers import PreTrainedTokenizer
from typing import List, Dict

class ChatContext:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_tokens: int = 8192):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.sessions: Dict[str, List[Dict]] = {}  # session_id -> list of messages

    def get(self, session_id: str) -> List[Dict]:
        return self.sessions.get(session_id, [])

    def append(self, session_id: str, message: Dict):
        history = self.sessions.get(session_id, [])
        history.append(message)
        trimmed = self.truncate_to_fit(history)
        self.sessions[session_id] = trimmed

    def truncate_to_fit(self, messages: List[Dict]) -> List[Dict]:
        while True:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            token_len = len(self.tokenizer(prompt).input_ids)
            if token_len < self.max_tokens or len(messages) <= 2:
                return messages
            # Remove oldest user+assistant pair
            messages = messages[2:]
