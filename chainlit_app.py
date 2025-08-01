import uuid
import asyncio
from PIL import Image
import chainlit as cl
import re
from mcpuse.utils.utils import load_yaml, remove_echo
from mcpuse.agent import MCPAgent
from mcpuse.client import MCPClient
from mcpuse.utils.context_manager import ChatContext
from mcpuse.utils.db import save_message, save_feedback

from mcpuse.model.local import TransformersLLM
from mcpuse.model.code import CodeLLM
from mcpuse.model.vision import VisionLLM
from mcpuse.tools.websearch import extract_top_results, summarize_results

cfg = load_yaml("configs/model_config.yaml")
device = cfg.get("device", "cuda")
save_chat = cfg.get("save_chat", False)

text_llm = TransformersLLM(cfg["text_model_path"], device=device)
code_llm = CodeLLM(cfg["code_model_path"], device=device)
vision_llm = VisionLLM(cfg["vision_model_path"], device=device)

agent = MCPAgent(text_agent=text_llm, code_agent=code_llm, vision_agent=vision_llm)
client = MCPClient()
text_context = ChatContext(tokenizer=text_llm.tokenizer, max_tokens=8192)

@cl.on_chat_start
async def on_start():
    username = "anonymous"
    session_id = f"{username}_{uuid.uuid4().hex}"
    cl.user_session.set("username", username)
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("vision_history", [])
    cl.user_session.set("last_image_files", None)
    await cl.Message(content=f"✅ Chat session started!").send()

def stream_sections(text):
    # Smartly splits multi-section markdown answers for chunked streaming
    sections = re.split(r'(### .+|^[-*] .+|^\d+\..+)', text, flags=re.MULTILINE)
    # Combine headings with following text for natural chunks
    chunks = []
    i = 0
    while i < len(sections):
        if i + 1 < len(sections) and (sections[i].startswith('### ') or sections[i].startswith('- ') or re.match(r'^\d+\.', sections[i])):
            chunks.append((sections[i] + '\n' + sections[i+1]).strip())
            i += 2
        else:
            if sections[i].strip():
                chunks.append(sections[i].strip())
            i += 1
    return [c for c in chunks if c]

def ensure_images_loaded(vision_history):
    """
    Convert any string or path 'image' fields in chat history to PIL.Image.
    """
    fixed = []
    for turn in vision_history:
        turn_copy = dict(turn)
        content = []
        for item in turn_copy.get("content", []):
            entry = dict(item)
            if entry.get("type") == "image":
                img = entry["image"]
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                entry["image"] = img
            content.append(entry)
        turn_copy["content"] = content
        fixed.append(turn_copy)
    return fixed

@cl.on_message
async def on_message(message: cl.Message):
    user_input = message.content.strip()
    uploaded_files = message.elements if hasattr(message, "elements") else []
    image_files = [f.path for f in uploaded_files if getattr(f, "mime", "").startswith("image/")]
    msg = cl.Message(content="")
    session_id = cl.user_session.get("session_id")
    username = cl.user_session.get("username", "anonymous")
    vision_history = cl.user_session.get("vision_history", [])

    # === VISION INPUT: NEW IMAGE ===
    if image_files:
        img_obj = Image.open(image_files[0]).convert("RGB")
        cl.user_session.set("last_image_files", image_files)
        vision_history.append({
            "role": "user",
            "content": [
                {"type": "image", "image": img_obj},
                {"type": "text", "text": user_input}
            ]
        })
        cleaned_history = ensure_images_loaded(vision_history[:-1])
        result = vision_llm.generate(
            image_files=image_files,
            prompt_text=user_input,
            session_vision_history=cleaned_history
        )
        vision_history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": result}]
        })
        cl.user_session.set("vision_history", vision_history)

        if save_chat:
            save_message(session_id, "user", user_input, username)
            save_message(session_id, "assistant", result, username)

        await simulate_sectioned_stream(result, msg)
        await send_feedback_actions()
        return

    # === VISION FOLLOW-UP: NO IMAGE, USE PREVIOUS ===
    if not image_files and cl.user_session.get("last_image_files"):
        image_files = cl.user_session.get("last_image_files")
        img_obj = Image.open(image_files[0]).convert("RGB")
        vision_history.append({
            "role": "user",
            "content": [
                {"type": "image", "image": img_obj},
                {"type": "text", "text": user_input}
            ]
        })
        cleaned_history = ensure_images_loaded(vision_history[:-1])
        result = vision_llm.generate(
            image_files=image_files,
            prompt_text=user_input,
            session_vision_history=cleaned_history
        )
        vision_history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": result}]
        })
        cl.user_session.set("vision_history", vision_history)

        if save_chat:
            save_message(session_id, "user", user_input, username)
            save_message(session_id, "assistant", result, username)

        await simulate_sectioned_stream(result, msg)
        await send_feedback_actions()
        return

    # === TEXT-ONLY ===
    history = text_context.get(session_id)
    history.append({"role": "user", "type": "text", "content": user_input})
    history = text_context.truncate_to_fit(history)
    prompt = text_llm.tokenizer.apply_chat_template(
        [{"role": h["role"], "content": h["content"]} for h in history],
        tokenize=False,
        add_generation_prompt=True
    )
    response = ""
    for token in text_llm.stream_generate(prompt):
        response += token
        await msg.stream_token(token)
    clean_response = remove_echo(user_input, response)
    history.append({"role": "assistant", "type": "text", "content": clean_response})
    text_context.sessions[session_id] = history

    msg.content = clean_response
    await msg.update()

    if save_chat:
        save_message(session_id, "user", user_input, username)
        save_message(session_id, "assistant", clean_response, username)
    await send_feedback_actions()

async def simulate_sectioned_stream(full_text, msg, delay=0.2):
    sections = stream_sections(full_text)
    for chunk in sections:
        await msg.stream_token(chunk + "\n\n")
        await asyncio.sleep(delay)
    await msg.update()

async def send_feedback_actions():
    await cl.Message(
        content="Was this helpful?",
        actions=[
            cl.Action(name="feedback_yes", label="Yes 👍", payload={}),
            cl.Action(name="feedback_no", label="No 👎", payload={}),
            cl.Action(name="web_search", label="Web Search 🔍", payload={})
        ]
    ).send()

@cl.action_callback("feedback_yes")
async def on_feedback_yes(action):
    session_id = cl.user_session.get("session_id", "unknown_session")
    save_feedback(session_id, 1)
    await cl.Message(content="Thanks for your feedback ✅").send()

@cl.action_callback("feedback_no")
async def on_feedback_no(action):
    session_id = cl.user_session.get("session_id", "unknown_session")
    save_feedback(session_id, 0)
    await cl.Message(content="Thank you for your feedback.").send()

@cl.action_callback("web_search")
async def on_web_search(action):
    session_id = cl.user_session.get("session_id", "unknown_session")
    username = cl.user_session.get("username", "anonymous")
    history = text_context.get(session_id)
    query = next((h["content"] for h in reversed(history) if h["role"] == "user"), None)
    if not query:
        await cl.Message(content="❌ No question found.").send()
        return
    await cl.Message(content=f"🔍 Searching for: {query}").send()
    results = extract_top_results(query)
    summary, refs = summarize_results(results, text_llm)
    reply = f"{summary}\n\n🔗 Sources:\n" + "\n".join(f"- {r}" for r in refs)
    if save_chat:
        save_message(session_id, "tool:websearch", reply, username)
    await cl.Message(content=reply).send()
