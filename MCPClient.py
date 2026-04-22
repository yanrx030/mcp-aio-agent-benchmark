import asyncio
from contextlib import AsyncExitStack
import json
import time

from constraint_validator import has_explicit_constraints, validate_tool_arguments
from eval_logger import JSONLLogger, utc_now_iso
from mcp_result_analyzer import analyze_tool_result
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.metadata_utils import get_display_name


from typing import Any, Optional
import requests
from requests.auth import HTTPBasicAuth



from openai import OpenAI
from prompts import prompts


class MCPAuthenticationError(RuntimeError):
    """Raised when MCP authentication fails or token is invalid/expired."""


class MCPTransportError(RuntimeError):
    """Raised when MCP transport/connectivity fails."""


def _exception_text(exc: BaseException) -> str:
    parts: list[str] = []
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        next_exc = current.__cause__ or current.__context__
        current = next_exc if isinstance(next_exc, BaseException) else None
    return " | ".join(parts).lower()


def is_authentication_error(exc: BaseException) -> bool:
    if isinstance(exc, MCPAuthenticationError):
        return True
    text = _exception_text(exc)
    auth_markers = (
        "401",
        "unauthorized",
        "forbidden",
        "not authenticated",
        "authentication failed",
        "invalid token",
        "token expired",
        "jwt",
        "credential",
    )
    return any(marker in text for marker in auth_markers)


def is_transport_error(exc: BaseException) -> bool:
    if isinstance(exc, MCPTransportError):
        return True
    text = _exception_text(exc)
    transport_markers = (
        "connecterror",
        "connection",
        "all connection attempts failed",
        "timed out",
        "timeout",
        "network",
        "transport",
        "temporarily unavailable",
    )
    return any(marker in text for marker in transport_markers)


class JwtTokenManager:
    """Process-wide JWT cache/refresh manager for concurrent workers."""

    def __init__(self, auth_key: str, login_url: str) -> None:
        self.auth_key = auth_key
        self.login_url = login_url
        self._jwt_token: str | None = None
        self._lock = asyncio.Lock()
        self._login_count = 0
        self._refresh_count = 0

    @property
    def login_count(self) -> int:
        return self._login_count

    @property
    def refresh_count(self) -> int:
        return self._refresh_count

    async def initialize(self) -> str:
        return await self.get_token()

    async def get_token(self) -> str:
        async with self._lock:
            if self._jwt_token:
                return self._jwt_token
            token = await asyncio.to_thread(
                MCPClient.request_jwt_token,
                self.auth_key,
                self.login_url,
            )
            self._jwt_token = token
            self._login_count += 1
            print("Authentication successful.")
            return token

    async def refresh_if_stale_or_forced(
        self,
        *,
        force: bool = False,
        failed_token: str | None = None,
    ) -> str:
        async with self._lock:
            if self._jwt_token and not force:
                return self._jwt_token
            # Another worker already refreshed; reuse latest token.
            if force and failed_token and self._jwt_token and self._jwt_token != failed_token:
                return self._jwt_token

            had_token = self._jwt_token is not None
            token = await asyncio.to_thread(
                MCPClient.request_jwt_token,
                self.auth_key,
                self.login_url,
            )
            self._jwt_token = token
            self._login_count += 1
            if had_token:
                self._refresh_count += 1
                print("JWT refreshed.")
            else:
                print("Authentication successful.")
            return token


class MCPClient:
    """
    A client class for interacting with the AIRED MCP server.
    This class manages the connection and communication with the AIRED MCP server.
    """

    LOGIN_URL = "https://api.aio.eresearch.unimelb.edu.au/login"
    MCP_URL = "https://mcp.aio.eresearch.unimelb.edu.au/mcp"
    
    def __init__(
        self,
        auth_key: str,
        openrouter_api_key: str,
        openrouter_model: str,
        openrouter_params: dict[str, Any] | None = None,
        logger: JSONLLogger | None = None,
        jwt_token: str | None = None,
    ):
        self.auth_key = auth_key
        self.logger = logger or JSONLLogger()
        self.session_id = self.logger.new_session_id()

        self.jwt_token: Optional[str] = None
        self.headers: Optional[dict[str, str]] = None
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()


        # OpenRouter (OpenAI-compatible) client
        self.api_key = openrouter_api_key
        self.model = openrouter_model
        self.openrouter_params = dict(openrouter_params or {})
        self.llm = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )


        # Conversation memory for LLM side
        self.messages: list[dict[str, Any]] = []
        self.system_prompt = prompts.MAIN_SYSTEM_PROMPT.strip()
        if jwt_token:
            self.set_jwt_token(jwt_token)

    
    def prepare_headers(self):
        if not self.jwt_token:
            raise MCPAuthenticationError("JWT token not available. Please authenticate first.")
        self.headers = {
            "Authorization": f"Bearer {self.jwt_token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }

    @staticmethod
    def request_jwt_token(auth_key: str, login_url: str | None = None) -> str:
        url = login_url or MCPClient.LOGIN_URL
        res = requests.post(url, auth=HTTPBasicAuth("apikey", auth_key), timeout=15)
        if not res.ok:
            raise MCPAuthenticationError(f"Authentication failed: {res.status_code} {res.text}")
        return res.text

    def set_jwt_token(self, jwt_token: str) -> None:
        self.jwt_token = jwt_token
        self.prepare_headers()

    def authenticate(self) -> str:
        token = self.request_jwt_token(self.auth_key, self.LOGIN_URL)
        self.set_jwt_token(token)
        print("Authentication successful.")
        return token
        
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

    def _build_completion_kwargs(
        self,
        messages: list[dict[str, Any]],
        *,
        include_tools: bool,
    ) -> dict[str, Any]:
        completion_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if include_tools:
            completion_kwargs["tools"] = self.openai_tools_all

        params = dict(self.openrouter_params)
        if not include_tools:
            params.pop("tool_choice", None)
            params.pop("parallel_tool_calls", None)

        direct_kwargs = {
            "temperature",
            "top_p",
            "max_tokens",
            "frequency_penalty",
            "presence_penalty",
            "stream",
            "tool_choice",
            "parallel_tool_calls",
        }
        extra_body: dict[str, Any] = {}

        for key, value in params.items():
            if value is None:
                continue
            if key in direct_kwargs:
                completion_kwargs[key] = value
            else:
                extra_body[key] = value

        if extra_body:
            completion_kwargs["extra_body"] = extra_body
        return completion_kwargs

    async def _create_chat_completion(self, **kwargs: Any) -> Any:
        # OpenAI client is synchronous; move network I/O off the event loop.
        return await asyncio.to_thread(
            self.llm.chat.completions.create,
            **kwargs,
        )
    

    async def connect_to_server(self):
        if not self.headers:
            raise MCPAuthenticationError("Not authenticated. Call authenticate() or set_jwt_token() first.")
        try:
            transport = await self.exit_stack.enter_async_context(
                streamablehttp_client(self.MCP_URL, headers=self.headers)
            )
            read_stream, write_stream, _ = transport

            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self.session.initialize()

            # check available tools
            live = await self.session.list_tools()
            live_names = {t.name for t in live.tools if getattr(t, "name", None)}

            # loaded names from toolset
            loaded_names = self._get_openai_tool_names()

            missing_in_loaded = sorted(list(live_names - loaded_names))
            extra_in_loaded = sorted(list(loaded_names - live_names))

            ok = (len(missing_in_loaded) == 0 and len(extra_in_loaded) == 0)
            if ok:
                print("Connected to MCP server. All tools in loaded toolset are available on server.")
                return True

            print("toolset mismatch with server")
            return False
        except Exception as exc:
            if is_authentication_error(exc):
                raise MCPAuthenticationError(f"MCP authentication failed during connect: {exc}") from exc
            if is_transport_error(exc):
                raise MCPTransportError(f"MCP transport failed during connect: {exc}") from exc
            raise

    async def cleanup(self) -> str | None:
        try:
            await self.exit_stack.aclose()
            return None
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

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


    async def process_query(
        self,
        query: str,
        *,
        task_id: str | None = None,
        system_prompt: str | None = None,
        system_prompt_label: str | None = None,
        return_trace: bool = False,
    ) -> str | dict[str, Any]:
        """Process a user query and log query-level and tool-level behavior."""
        max_steps = 30  # safety to prevent infinite loops; adjust as needed
        if not self.session:
            raise RuntimeError("Not connected to MCP server. Call connect_to_server() first.")

        effective_system_prompt = (system_prompt or self.system_prompt).strip()
        effective_prompt_label = (
            system_prompt_label
            or ("override" if system_prompt else "default")
        )

        query_record = self.logger.build_query_run(
            query_id=self.logger.new_query_id(),
            task_id=task_id,
            session_id=self.session_id,
            user_query=query,
            model=self.model,
            toolset_id=getattr(self, "toolset_id", None),
            system_prompt_label=effective_prompt_label,
            system_prompt_overridden=bool(system_prompt),
        )
        query_id = query_record.query_id
        total_steps = 0
        tool_call_count = 0
        schema_valid_true_count = 0
        ccr_applicable_calls = 0
        ccr_true_calls = 0
        final_answer = ""
        tools_used: list[str] = []
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        query_status = "completed"
        query_error_type: str | None = None
        query_error_message: str | None = None
        messages: list[dict[str, Any]] = [{"role": "system", "content": effective_system_prompt}]

        def add_usage(resp: Any) -> None:
            nonlocal input_tokens, output_tokens, total_tokens
            usage = getattr(resp, "usage", None)
            if not usage:
                return
            input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
        
        try:
            messages.append({"role": "user", "content": query})

            # Step loop: model may call tool(s), then we execute, then model may call again, etc.
            for step in range(1, max_steps + 1):
                total_steps = step
                resp = await self._create_chat_completion(
                    **self._build_completion_kwargs(messages, include_tools=True)
                )
                add_usage(resp)

                if not resp or not getattr(resp, "choices", None):
                    raise RuntimeError(f"LLM returned no choices for model={self.model}")


                msg = resp.choices[0].message
                # Save assistant message to history (important for tool_call_id linking)
                messages.append(msg.model_dump())

                # If no tool calls, we're done.
                if not msg.tool_calls:
                    final_answer = msg.content or ""
                    break

                # Execute all tool calls in this message
                for tc in msg.tool_calls:
                    tool_call_count += 1
                    tool_name = tc.function.name
                    if tool_name not in tools_used:
                        tools_used.append(tool_name)
                    raw_args = tc.function.arguments or "{}"
                    tool_record = self.logger.build_tool_call(
                        query_id=query_id,
                        task_id=task_id,
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
                        tool_record.schema_errors = ["Tool arguments must be valid JSON."]
                        if has_explicit_constraints(tool_name):
                            tool_record.constraint_valid = False
                            ccr_applicable_calls += 1
                        else:
                            tool_record.constraint_valid = None
                        tool_record.output_schema_valid = None
                        tool_record.failure_stage = "parse"
                        tool_record.error_type = type(e).__name__
                        tool_record.error_message = str(e)

                        err_text = f"Tool arguments JSON parse failed for {tool_name}: {e}. raw={raw_args}"
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": tool_name,
                                "content": err_text,
                            }
                        )
                        self.logger.log_tool_call(tool_record)
                        continue

                    validation = validate_tool_arguments(
                        tool_name=tool_name,
                        arguments=tool_args,
                        openai_tool_def=self.openai_tool_map.get(tool_name),
                    )
                    tool_record.schema_valid = validation.schema_valid
                    tool_record.schema_errors = validation.schema_errors
                    tool_record.constraint_valid = validation.constraint_valid
                    if tool_record.schema_valid is True:
                        schema_valid_true_count += 1
                    if has_explicit_constraints(tool_name):
                        ccr_applicable_calls += 1
                        if tool_record.constraint_valid is True:
                            ccr_true_calls += 1

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
                        messages.append(
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
                        messages.append(
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
            if not final_answer:
                final = await self._create_chat_completion(
                    **self._build_completion_kwargs(messages, include_tools=False),
                )
                add_usage(final)
                if not final or not getattr(final, "choices", None):
                    raise RuntimeError(
                        f"LLM returned no choices during final summarization for model={self.model}"
                    )
                final_msg = final.choices[0].message
                messages.append(final_msg.model_dump())
                final_answer = final_msg.content or ""

            self.messages = messages
            parameter_schema_valid_rate = (
                schema_valid_true_count / tool_call_count if tool_call_count > 0 else None
            )
            constraint_compliance_rate = (
                ccr_true_calls / ccr_applicable_calls if ccr_applicable_calls > 0 else None
            )
            if return_trace:
                return {
                    "query_id": query_id,
                    "task_set_id": self.logger.task_set_id,
                    "task_id": task_id,
                    "session_id": self.session_id,
                    "user_query": query,
                    "final_answer": final_answer,
                    "total_steps": total_steps,
                    "total_tool_calls": tool_call_count,
                    "parameter_schema_valid_rate": parameter_schema_valid_rate,
                    "ccr_applicable_calls": ccr_applicable_calls,
                    "ccr_true_calls": ccr_true_calls,
                    "constraint_compliance_rate": constraint_compliance_rate,
                    "tools_used": tools_used,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "system_prompt_label": effective_prompt_label,
                    "system_prompt_overridden": bool(system_prompt),
                    "query_runs_path": str(self.logger.query_runs_path),
                    "tool_calls_path": str(
                        self.logger.task_tool_calls_path(task_id=task_id, query_id=query_id)
                    ),
                }
            return final_answer
        except Exception as exc:
            query_status = "failed"
            mapped_exc: Exception = exc
            if is_authentication_error(exc):
                mapped_exc = MCPAuthenticationError(f"MCP authentication failed during query: {exc}")
            elif is_transport_error(exc):
                mapped_exc = MCPTransportError(f"MCP transport failed during query: {exc}")

            query_error_type = type(mapped_exc).__name__
            query_error_message = str(mapped_exc)
            if mapped_exc is exc:
                raise
            raise mapped_exc from exc
        finally:
            query_record.status = query_status
            query_record.error_type = query_error_type
            query_record.error_message = query_error_message
            query_record.end_time = utc_now_iso()
            query_record.final_answer = final_answer
            query_record.total_steps = total_steps
            query_record.total_tool_calls = tool_call_count
            query_record.parameter_schema_valid_rate = (
                schema_valid_true_count / tool_call_count if tool_call_count > 0 else None
            )
            query_record.ccr_applicable_calls = ccr_applicable_calls
            query_record.ccr_true_calls = ccr_true_calls
            query_record.constraint_compliance_rate = (
                ccr_true_calls / ccr_applicable_calls if ccr_applicable_calls > 0 else None
            )
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











