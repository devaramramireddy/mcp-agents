import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


class VisionLLM:
    def __init__(self, model_path, device="cuda", max_input_tokens=8192):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto" if device != "cpu" else "cpu",
            torch_dtype="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = device
        self.max_input_tokens = max_input_tokens

    def truncate_messages(self, messages):
        """
        Trim oldest turns if token length exceeds max_input_tokens.
        """
        while True:
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            tokenized = self.processor.tokenizer(prompt, return_tensors="pt")
            input_len = tokenized["input_ids"].shape[-1]
            if input_len < self.max_input_tokens or len(messages) <= 2:
                return messages
            messages = messages[2:]  # drop oldest user + assistant

    def build_messages(self, image, current_prompt, session_vision_history=None):
        """
        Build a multi-turn messages list with chat image-text format.
        """
        messages = []

        if session_vision_history:
            messages.extend(session_vision_history)

        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": current_prompt}
            ]
        })

        return self.truncate_messages(messages)

    def extract_images_from_messages(self, messages):
        images = []
        for msg in messages:
            for c in msg.get("content", []):
                if c.get("type") == "image":
                    img = c["image"]
                    if isinstance(img, str):
                        img = Image.open(img).convert("RGB")
                    images.append(img)
        return images

    def generate(self, image_files, prompt_text, session_vision_history=None, max_new_tokens=256):
        image_path = image_files[0] if isinstance(image_files, list) else image_files
        image = Image.open(image_path).convert("RGB")

        messages = self.build_messages(image, prompt_text, session_vision_history)

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = self.extract_images_from_messages(messages)
        video_inputs = None

        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
            response = self.processor.batch_decode(trimmed, skip_special_tokens=True)
            return response[0]
