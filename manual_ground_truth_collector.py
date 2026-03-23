import asyncio
import json
import os
from MCPClient import MCPClient
aio_key = os.environ["AIO_AUTH_KEY"]
openrouter_key = os.environ["OPENROUTER_API_KEY"]


def drop_none_values(value):
    """Recursively remove keys with None values from dictionaries."""
    if isinstance(value, dict):
        return {k: drop_none_values(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [drop_none_values(v) for v in value]
    return value

def prompt_user_for_tool_args(tool_name):
    """Prompt the user to input arguments for the specified tool."""
    while True:
        print(f"\nEnter arguments for tool '{tool_name}' as a JSON object (multi-line input supported):")
        print("Paste your JSON object and press Enter twice when done.")

        lines = []
        while True:
            line = input()
            if line.strip() == "":  # Empty line indicates end of input
                break
            lines.append(line)

        user_input = "\n".join(lines)

        try:
            # Attempt to parse the collected input as JSON
            tool_args = json.loads(user_input)

            # Ensure the parsed input is a dictionary
            if isinstance(tool_args, dict):
                print("\nParsed arguments:")
                print(json.dumps(tool_args, indent=2))
                confirm = input("Are these arguments correct? (yes/no): ").strip().lower()
                if confirm == "yes":
                    return tool_args
                print("Please re-enter the arguments.")
            else:
                print("Error: Arguments must be a JSON object. Try again.")
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON. {e}. Try again.")


async def main():
    print("Manual Ground Truth Collector")

    # Initialize MCPClient
    aio_key = os.environ.get("AIO_AUTH_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openrouter_model = "nvidia/nemotron-3-super-120b-a12b:free"

    client = MCPClient(auth_key=aio_key, openrouter_api_key=openrouter_key, openrouter_model=openrouter_model)

    try:
        # Authenticate and load tools
        client.authenticate()
        toolset_path = "toolsets/aio_mcp_toolset_v2.json"
        client.load_tools(toolset_path)
        await client.connect_to_server()

        # Interact with tools
        available_tools = client._get_openai_tool_names()
        print("\nAvailable tools:")
        for tool in available_tools:
            print(f"- {tool}")

        while True:
            tool_name = input("\nEnter the name of the tool to interact with (or 'exit' to quit): ")
            if tool_name.lower() == 'exit':
                break
            if tool_name not in available_tools:
                print("Error: Tool not found. Please choose from the available tools.")
                continue

            tool_args = prompt_user_for_tool_args(tool_name)
            sanitized_tool_args = drop_none_values(tool_args)
            if sanitized_tool_args != tool_args:
                print("\nSanitized arguments (removed null values):")
                print(json.dumps(sanitized_tool_args, indent=2))

            result = await client.session.call_tool(tool_name, sanitized_tool_args)
            print(f"\nResult for tool '{tool_name}':\n{result}")

    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())