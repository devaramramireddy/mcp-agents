from mcpuse.model.local import TransformersLLM

def test_transformers_loader():
    llm = TransformersLLM(model_path="gpt2", device="cpu")
    out = llm.generate("The capital of France is")
    assert isinstance(out, str)

if __name__ == "__main__":
    test_transformers_loader()
    print("test_transformers_loader passed")
