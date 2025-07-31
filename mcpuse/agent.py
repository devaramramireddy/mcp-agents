class MCPAgent:
    def __init__(self, llm, client, max_steps=30):
        self.llm = llm
        self.client = client
        self.max_steps = max_steps

    async def run(self, query):
        # Example simple agent: directly use LLM's completion
        return self.llm.generate(query)
