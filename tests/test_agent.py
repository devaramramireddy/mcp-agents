import asyncio
from mcpuse.agent import MCPAgent
from mcpuse.client import MCPClient
from mcpuse.model.local import TransformersLLM

async def test_agent():
    llm = TransformersLLM(model_path="gpt2", device="cpu")
    client = MCPClient()
    agent = MCPAgent(llm, client)
    result = await agent.run("Say hello!")
    assert isinstance(result, str)

if __name__ == "__main__":
    asyncio.run(test_agent())
