






# def convert_mcp_tool_to_openai_tool(mcp_tool) -> dict:
#     """
#     MCP tool -> OpenRouter compatible (OpenAI tool) schema
#     """
#     params_schema = getattr(mcp_tool, "inputSchema", None) or {"type": "object", "properties": {}}

#     # Safety: OpenAI expects "parameters" to be a JSON Schema object.
#     # Ensure it has a type.
#     if "type" not in params_schema:
#         params_schema["type"] = "object"

#     return {
#         "type": "function",
#         "function": {
#             "name": mcp_tool.name,
#             "description": getattr(mcp_tool, "description", "") or "",
#             "parameters": params_schema,
#         },
#     }



# class MCPToolManager:
#     """ 
#     Helper class to manage MCP tools
    
#     """
#     def __init__(self, aio_auth_key: str):
#         self.aio_auth_key = aio_auth_key
#         self.jwt_token: Optional[str] = None
#         self.headers: Optional[dict] = None

#         self.exit_stack = AsyncExitStack()
#         self.session: Optional[ClientSession] = None

#         self.llm = OpenAI(
#             api_key=OPENROUTER_API_KEY,
#             base_url="https://openrouter.ai/api/v1",
#         )




from datetime import datetime
import os, json, asyncio
from contextlib import AsyncExitStack
import requests
from requests.auth import HTTPBasicAuth

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


LOGIN_URL = "https://api.aio.eresearch.unimelb.edu.au/login"
MCP_URL = "https://mcp.aio.eresearch.unimelb.edu.au/mcp"
NAME = "aio_mcp_toolset_v2"


def mcp_tool_to_openai_tool(tool):
    

    if"required" not in tool.inputSchema:
        tool.inputSchema["required"] = []
    converted_tool = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": tool.inputSchema["properties"],
                "required": tool.inputSchema["required"],
            }
        }
    }

    return converted_tool


def get_jwt(auth_key: str) -> str:
    res = requests.post(LOGIN_URL, auth=HTTPBasicAuth("apikey", auth_key), timeout=15)
    if not res.ok:
        raise RuntimeError(f"Auth failed: {res.status_code} {res.text}")
    return res.text.strip()




async def main():
    auth_key = os.environ["AIO_AUTH_KEY"]
    out_path = os.environ.get("TOOLSET_OUT", f"toolsets/{NAME}.json")

    jwt = get_jwt(auth_key)
    headers = {"Authorization": f"Bearer {jwt}"}

    exit_stack = AsyncExitStack()
    try:
        transport = await exit_stack.enter_async_context(streamablehttp_client(MCP_URL, headers=headers))
        read_stream, write_stream, _ = transport
        session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))

        await session.initialize()
        resp = await session.list_tools()

        tools = [mcp_tool_to_openai_tool(t) for t in resp.tools]
        mcp_raw_tools = []
        for t in resp.tools:
            if hasattr(t, "model_dump"):
                mcp_raw_tools.append(t.model_dump())
            elif hasattr(t, "dict"):
                mcp_raw_tools.append(t.dict())
            else:
                mcp_raw_tools.append({
                    "name": getattr(t, "name", None),
                    "description": getattr(t, "description", None),
                    "inputSchema": getattr(t, "inputSchema", None),
                    "outputSchema": getattr(t, "outputSchema", None),
                    "_meta": getattr(t, "_meta", None),
                })

        toolset = {
            "toolset_id": NAME,
            "generated_at_utc":datetime.now().isoformat(),
            "notes": (
                    "mcp_raw preserves MCP tool definitions including outputSchema/_meta. "
                    "openai_tools is derived for OpenRouter/OpenAI tool calling."
                ),
            "mcp_server": MCP_URL,
            
            "openai_tools": {
                "tools": tools,
            },
            "mcp_raw": {
                "tools": mcp_raw_tools,
            },

        }

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(toolset, f, ensure_ascii=False, indent=2)

        print(f" dumped {len(tools)} tools -> {out_path}")

    finally:
        await exit_stack.aclose()


if __name__ == "__main__":
    asyncio.run(main())