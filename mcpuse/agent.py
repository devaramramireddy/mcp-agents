class MCPAgent:
    def __init__(self, text_agent, code_agent=None, vision_agent=None):
        self.text_agent = text_agent
        self.code_agent = code_agent
        self.vision_agent = vision_agent

    async def run(self, user_input, image=None, code=False):
        if image and self.vision_agent:
            return self.vision_agent.generate(image, prompt=user_input)
        elif code and self.code_agent:
            return self.code_agent.generate(user_input)
        else:
            return self.text_agent.generate(user_input)
