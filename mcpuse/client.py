import json
import os

class MCPClient:
    def __init__(self, config_path="configs/mcp_servers.json"):
        # For demo: just load config. Add MCP HTTP/Websocket logic as needed.
        if not os.path.exists(config_path):
            self.config = None
        else:
            with open(config_path, "r") as f:
                self.config = json.load(f)

    def get_config(self):
        return self.config

    def query_tool(self, tool_name, input_data):
        # Mock tool usage
        return f"Tool[{tool_name}] received: {input_data}"
