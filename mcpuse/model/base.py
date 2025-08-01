class BaseLLM:
    def generate(self, input_data, **options):
        raise NotImplementedError("Subclasses must implement the generate method.")
