import asyncio
import os
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def save_text_to_file(filename: str, content: str, directory: str = ".") -> str:
    """
    Save text content to a local file.
    
    Args:
        filename: Name of the file to save (e.g., 'menu_items.txt')
        content: Text content to save
        directory: Directory to save the file in (default: current directory)
    
    Returns:
        Success message with file path
    """
    try:
        # Create directory if it doesn't exist
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create full file path
        file_path = dir_path / filename
        
        # Write content to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"Successfully saved content to {file_path.absolute()}"
    except Exception as e:
        return f"Error saving file: {str(e)}"

async def main():
    client = MultiServerMCPClient(
        {
            "playwright": {
                "transport": "streamable_http",
                "url": "http://localhost:8931/mcp",
            }
        }
    )
    
    playwright_tools = await client.get_tools()
    
    # Combine Playwright tools with our file system tool
    all_tools = playwright_tools + [save_text_to_file]
    
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    agent = create_agent(
        llm,
        all_tools
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
    7. After successful login to the dashboard, get the page snapshot and identify all navigation menu items on the left. Save the menu items to 'menu_items.txt' using the save_text_to_file tool."""

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