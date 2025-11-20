import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

async def main():
    client = MultiServerMCPClient(
        {
            "playwright": {
                "transport": "streamable_http",
                "url": "http://localhost:8931/mcp",
            }
        }
    )
    
    tools = await client.get_tools()
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    agent = create_agent(
        llm,
        tools
    )
    
    username = os.getenv("USER_NAME")
    password = os.getenv("PASSWORD") 
    verification_code = os.getenv("VERIFICATION_CODE")
    
    # Execute all steps in a single conversation to maintain browser context
    full_task = f"""Please complete the following login sequence step by step:
    1. Navigate to https://v4staging.scantist.io/login 
    2. Fill in username field with '{username}' (use simple fill method)
    3. Fill in password field with '{password}' (use simple fill method)
    4. Fill in verification code field with '{verification_code}' (use simple fill method)
    5. Click Sign In button 
    6. Wait for the dashboard page to load completely
    7. After successful login to the dashboard, get the page snapshot and identify all navigation menu items. Create a text file called 'menu_items.txt' that contains all the dashboard navigation items."""

    try:
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": full_task}]}
        )
        print("Login sequence completed:", response)
    except Exception as e:
        print(f"Error during login sequence: {e}")
        if "not found in the current page snapshot" in str(e):
            print("Browser reference lost, try restarting the MCP server")
        response = None

if __name__ == "__main__":
    asyncio.run(main())