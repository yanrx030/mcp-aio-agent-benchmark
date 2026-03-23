import asyncio
from MCPClient import MCPClient
import os


aio_key = os.environ["AIO_AUTH_KEY"]
openrouter_key = os.environ["OPENROUTER_API_KEY"]

async def main():
    print("Hello from aired-eval!")
    client = MCPClient(auth_key=aio_key, openrouter_api_key=openrouter_key,openrouter_model="nvidia/nemotron-3-super-120b-a12b:free")
    try:
        client.authenticate()
        client.load_tools("toolsets/aio_mcp_toolset_v2.json")
        await client.connect_to_server()
        # tool_name = "aggregate_by_time"
        # tool_args = {          "collection": "reddit",
        #   "aggregation_level": "month",
        #   "startdate": "2024-01-01",
        #   "enddate": "2024-12-31",
        #   "sentiment": False}
        # result = await client.session.call_tool(tool_name, tool_args)
        # print(f"Result for {tool_name}: {result}")  

        result = await client.process_query("How many total documents are available on reddit as of February 2026?")
        print("\nResult:\n", result)


    finally:
        await client.cleanup()


    



if __name__ == "__main__":
    asyncio.run(main())
