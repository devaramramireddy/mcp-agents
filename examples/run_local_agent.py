import asyncio
from mcpuse.agent import MCPAgent
from mcpuse.client import MCPClient
from mcpuse.model.local import TransformersLLM
from mcpuse.utils.utils import load_yaml

async def main():
    model_cfg = load_yaml("configs/model_config.yaml")
    model_path = model_cfg.get("model_path", "gpt2")
    device = model_cfg.get("device", "cuda")

    llm = TransformersLLM(model_path=model_path, device=device)
    client = MCPClient("configs/mcp_servers.json")
    agent = MCPAgent(llm=llm, client=client)
    query = "What's an interesting fact about deep sea creatures?"
    result = await agent.run(query)
    print("Agent result:\n", result)

if __name__ == "__main__":
    asyncio.run(main())
