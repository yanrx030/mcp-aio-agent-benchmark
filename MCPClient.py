import asyncio
from contextlib import AsyncExitStack
import json
import time

from eval_logger import JSONLLogger, utc_now_iso
from mcp_result_analyzer import analyze_tool_result
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.metadata_utils import get_display_name


from typing import Any, Optional
import requests
from requests.auth import HTTPBasicAuth



from openai import OpenAI


class MCPClient:
    """
    A client class for interacting with the AIRED MCP server.
    This class manages the connection and communication with the AIRED MCP server.
    """

    LOGIN_URL = "https://api.aio.eresearch.unimelb.edu.au/login"
    MCP_URL = "https://mcp.aio.eresearch.unimelb.edu.au/mcp"
    
    def __init__(self, auth_key: str, openrouter_api_key: str, openrouter_model: str):
        self.auth_key = auth_key
        self.logger = JSONLLogger()
        self.session_id = self.logger.new_session_id()

        self.jwt_token: Optional[str] = None
        self.headers: Optional[dict[str, str]] = None
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()


        # OpenRouter (OpenAI-compatible) client
        self.api_key = openrouter_api_key
        self.model = openrouter_model
        self.llm = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )


        # Conversation memory for LLM side
        self.messages: list[dict[str, Any]] = []

    
    def prepare_headers(self):
        if not self.jwt_token:
            raise Exception("JWT token not available. Please authenticate first.")
        self.headers = {
            "Authorization": f"Bearer {self.jwt_token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }

    def authenticate(self):
        url = self.LOGIN_URL
        res = requests.post(url, auth=HTTPBasicAuth('apikey', self.auth_key),timeout=15)
        if res.ok:
            # Assumes the API returns the raw JWT as plain text
            self.jwt_token = res.text
            self.prepare_headers()
            print("Authentication successful.")
        else:
            raise RuntimeError(f"Authentication failed: {res.status_code} {res.text}")
        
    def _get_openai_tool_names(self) -> set[str]:
        """Extract tool names from cached OpenAI tools."""
        if not hasattr(self, "openai_tools_all"):
            raise RuntimeError("Tools not loaded. Call load_tools(toolset_path) first.")
        names: set[str] = set()
        for t in self.openai_tools_all:
            fn = t.get("function") or {}
            name = fn.get("name")
            if name:
                names.add(name)
        return names
    

    async def connect_to_server(self):
        if not self.headers:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        transport = await self.exit_stack.enter_async_context(
        streamablehttp_client(self.MCP_URL, headers=self.headers))
        read_stream, write_stream, _ = transport

        self.session = await self.exit_stack.enter_async_context(
        ClientSession(read_stream, write_stream))
        await self.session.initialize()

        # check available tools 
        live = await self.session.list_tools()
        live_names = {t.name for t in live.tools if getattr(t, "name", None)}

        # 2) loaded names from toolset
        loaded_names = self._get_openai_tool_names()

        missing_in_loaded = sorted(list(live_names - loaded_names))
        extra_in_loaded = sorted(list(loaded_names - live_names))

        ok = (len(missing_in_loaded) == 0 and len(extra_in_loaded) == 0)
        if ok:
            print("Connected to MCP server. All tools in loaded toolset are available on server.")
            return True
        else:
            print("toolset mismatch with server")
            return False


    async def cleanup(self):
        await self.exit_stack.aclose()

    def load_tools(self, toolset_path: str) -> None:
        """
        Load a frozen toolset file (dual-track):
          - openai_tools.tools: directly usable for OpenRouter/OpenAI tool calling
          - mcp_raw.tools: preserves original MCP tool definitions (incl outputSchema/_meta)

        After loading:
          self.toolset_id
          self.openai_tools_all
          self.openai_tool_map  (name -> openai tool dict) #use name to look up tool def
          self.mcp_raw_tools_all
          self.mcp_raw_tool_map (name -> raw tool dict) #use name to look up mcp raw tool def
        """
        with open(toolset_path, "r", encoding="utf-8") as f:
            toolset = json.load(f)

        # Basic validations
        self.toolset_id = toolset.get("toolset_id") or "unknown_toolset"
        openai_tools = toolset.get("openai_tools", {}).get("tools")
        mcp_raw_tools = toolset.get("mcp_raw", {}).get("tools")

        if not isinstance(openai_tools, list) or len(openai_tools) == 0:
            raise RuntimeError(f"Invalid toolset: openai_tools.tools missing/empty in {toolset_path}")
        if not isinstance(mcp_raw_tools, list) or len(mcp_raw_tools) == 0:
            raise RuntimeError(f"Invalid toolset: mcp_raw.tools missing/empty in {toolset_path}")

        # Cache lists
        self.openai_tools_all = openai_tools
        self.mcp_raw_tools_all = mcp_raw_tools

        # Build name -> tool maps
        self.openai_tool_map = {}
        for t in openai_tools:
            # expected shape:
            # {"type":"function","function":{"name":"...","description":"...","parameters":{...}}}
            name = (t.get("function") or {}).get("name")
            if not name:
                continue
            self.openai_tool_map[name] = t

        self.mcp_raw_tool_map = {}
        for t in mcp_raw_tools:
            # expected raw shape includes: {"name": "...", "description": "...", "inputSchema":..., "outputSchema":...}
            name = t.get("name")
            if not name:
                continue
            self.mcp_raw_tool_map[name] = t


    async def process_query(self, query: str) -> str:
        """Process a user query and log query-level and tool-level behavior."""
        max_steps = 10  # safety to prevent infinite loops; adjust as needed
        if not self.session:
            raise RuntimeError("Not connected to MCP server. Call connect_to_server() first.")

        query_record = self.logger.build_query_run(
            query_id=self.logger.new_query_id(),
            session_id=self.session_id,
            user_query=query,
            model=self.model,
            toolset_id=getattr(self, "toolset_id", None),
        )
        query_id = query_record.query_id
        tool_call_count = 0
        final_answer = ""
        tools_used: list[str] = []
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

        def add_usage(resp: Any) -> None:
            nonlocal input_tokens, output_tokens, total_tokens
            usage = getattr(resp, "usage", None)
            if not usage:
                return
            input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
        
        try:
            # System prompt: keep it short; you can later replace with your benchmark prompt template.
            if not self.messages:
                self.messages.append(
                    {
                        "role": "system",
                        "content": (
                            "You are a tool-using assistant. "
                            "When you need server data, call the provided tools with valid JSON arguments. "
                            "If helpful, briefly state your next action in one short sentence before using tools."
                        ),
                    }
                )
            self.messages.append({"role": "user", "content": query})

            # Step loop: model may call tool(s), then we execute, then model may call again, etc.
            for step in range(1, max_steps + 1):
                resp = self.llm.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.openai_tools_all
                )
                add_usage(resp)

                if not resp or not getattr(resp, "choices", None):
                    raise RuntimeError(f"LLM returned no choices for model={self.model}")


                msg = resp.choices[0].message
                # Save assistant message to history (important for tool_call_id linking)
                self.messages.append(msg.model_dump())

                # If no tool calls, we're done.
                if not msg.tool_calls:
                    final_answer = msg.content or ""
                    return final_answer

                # Execute all tool calls in this message
                for tc in msg.tool_calls:
                    tool_call_count += 1
                    tool_name = tc.function.name
                    if tool_name not in tools_used:
                        tools_used.append(tool_name)
                    raw_args = tc.function.arguments or "{}"
                    tool_record = self.logger.build_tool_call(
                        query_id=query_id,
                        tool_name=tool_name,
                        step_index=step,
                        tool_call_id=tc.id,
                        raw_arguments=raw_args,
                        execution_attempted=False,
                        transport_success=False,
                        mcp_is_error=None,
                        payload_has_error=None,
                        execution_success=False,
                    )

                    # Parse tool arguments
                    try:
                        tool_args = json.loads(raw_args) if raw_args else {}
                        tool_record.parsed_arguments = tool_args
                        tool_record.parse_success = True
                    except Exception as e:
                        tool_record.parse_success = False
                        tool_record.schema_valid = False
                        tool_record.constraint_valid = False
                        tool_record.ipa_pass = False
                        tool_record.output_schema_valid = None
                        tool_record.failure_stage = "parse"
                        tool_record.error_type = type(e).__name__
                        tool_record.error_message = str(e)

                        err_text = f"Tool arguments JSON parse failed for {tool_name}: {e}. raw={raw_args}"
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": tool_name,
                                "content": err_text,
                            }
                        )
                        self.logger.log_tool_call(tool_record)
                        continue

                    # Schema/constraint validation is not wired yet; keep the fields explicit in the logs.
                    tool_record.schema_valid = None
                    tool_record.constraint_valid = None
                    tool_record.ipa_pass = None

                    # Call MCP tool
                    t0 = time.time()
                    tool_record.execution_attempted = True
                    try:
                        result = await self.session.call_tool(tool_name, tool_args)
                        tool_record.transport_success = True
                        tool_record.latency_ms = int((time.time() - t0) * 1000)
                        analyzed = analyze_tool_result(result)
                        tool_record.response_summary = analyzed["response_summary"]
                        tool_record.content_blocks = analyzed["content_blocks"]
                        tool_record.structured_content = analyzed["structured_content"]
                        tool_record.mcp_is_error = analyzed["mcp_is_error"]
                        tool_record.payload_has_error = analyzed["payload_has_error"]
                        tool_record.execution_success = analyzed["execution_success"]
                        tool_record.failure_stage = analyzed["failure_stage"]
                        tool_record.http_status_code = analyzed["http_status_code"]
                        tool_record.output_schema_valid = analyzed["output_schema_valid"]
                        tool_record.server_error_message = analyzed["server_error_message"]

                        if not tool_record.execution_success:
                            tool_record.error_type = (
                                "ToolOutputValidationError"
                                if tool_record.failure_stage == "output_validation"
                                else "ToolServerError"
                            )
                            tool_record.error_message = (
                                tool_record.server_error_message or tool_record.response_summary
                            )

                        # Append tool result to messages
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": tool_name,
                                "content": tool_record.response_summary,
                            }
                        )
                    except Exception as e:
                        tool_record.latency_ms = int((time.time() - t0) * 1000)
                        tool_record.transport_success = False
                        tool_record.mcp_is_error = None
                        tool_record.execution_success = False
                        tool_record.failure_stage = "execution"
                        tool_record.error_type = type(e).__name__
                        tool_record.error_message = str(e)

                        # Tool execution error
                        err_text = f"MCP tool call failed: {tool_name} args={tool_args} error={e}"
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": tool_name,
                                "content": err_text,
                            }
                        )
                    finally:
                        self.logger.log_tool_call(tool_record)

            # If we hit max_steps, ask model to summarize anyway (no more tool calling).
            final = self.llm.chat.completions.create(
                model=self.model,
                messages=self.messages,
            )
            add_usage(final)
            if not final or not getattr(final, "choices", None):
                raise RuntimeError(
                    f"LLM returned no choices during final summarization for model={self.model}"
                )
            final_msg = final.choices[0].message
            self.messages.append(final_msg.model_dump())
            final_answer = final_msg.content or ""
            return final_answer
        except Exception:
            raise
        finally:
            query_record.end_time = utc_now_iso()
            query_record.final_answer = final_answer
            query_record.total_tool_calls = tool_call_count
            query_record.tools_used = tools_used
            query_record.input_tokens = input_tokens
            query_record.output_tokens = output_tokens
            query_record.total_tokens = total_tokens
            self.logger.log_query_run(query_record)


    # async def display_tools(self):
    #     """Display available tools with human-readable names"""
    #     tools_response = await self.session.list_tools()

    #     for tool in tools_response.tools:
    #         # get_display_name() returns the title if available, otherwise the name
    #         display_name = get_display_name(tool)
    #         print(f"Tool: {display_name}")
    #         if tool.description:
    #             print(f"   {tool.description}")

    def get_tool_parameters(self, tool_name: str) -> dict:
        """Retrieve the parameters for a specific tool."""
        if not hasattr(self, "mcp_raw_tool_map"):
            raise RuntimeError("Tools not loaded. Call load_tools(toolset_path) first.")

        tool = self.mcp_raw_tool_map.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in the loaded toolset.")

        input_schema = tool.get("inputSchema", {}).get("properties", {})
        parameters = {
            param: {
                "description": details.get("description", "No description available"),
                "type": details.get("type", "unknown")
            }
            for param, details in input_schema.items()
        }
        return parameters











