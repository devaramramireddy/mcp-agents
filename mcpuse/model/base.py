class BaseLLM:
    def generate(self, prompt, **kwargs):
        raise NotImplementedError("Implement 'generate' in subclass.")
