import chainlit as cl
from mcpuse.agent import MCPAgent
from mcpuse.client import MCPClient
from mcpuse.model.local import TransformersLLM
from mcpuse.utils import load_yaml

# Load config
model_cfg = load_yaml("configs/model_config.yaml")
model_path = model_cfg.get("model_path", "meta-llama/Llama-2-7b-chat-hf")  # Replace with your Llama model path
device = model_cfg.get("device", "cuda")

# Initialize model and client
llm = TransformersLLM(model_path=model_path, device=device)
client = MCPClient("configs/mcp_servers.json")
agent = MCPAgent(llm=llm, client=client)

def remove_echo(user_input: str, model_response: str) -> str:
    # Clean up if the model starts by repeating the user question
    # Use startswith, but account for spacing
    cleaned = model_response
    if cleaned.strip().lower().startswith(user_input.strip().lower()):
        cleaned = cleaned[len(user_input):].lstrip()
    return cleaned

@cl.on_message
async def on_message(message: cl.Message):
    user_input = message.content
    msg = cl.Message(content="")

    response = ""
    for token in llm.stream_generate(user_input):
        response += token
        await msg.stream_token(token)

    msg.content = response
    await msg.update()
    # OR, in future, stream tokens as Hugging Face pipeline adds streaming for chat messages

