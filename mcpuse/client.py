import json
import os

class MCPClient:
    def __init__(self, config_path="configs/mcp_servers.json"):
        self.config = None
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)

    def get_config(self):
        return self.config

    def query_tool(self, tool_name, input_data):
        # Simulate tool call — real logic can be added later
        return f"Tool[{tool_name}] received: {input_data}"
