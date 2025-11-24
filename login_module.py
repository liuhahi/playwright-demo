import asyncio
import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from neo4j_tools import initialize_neo4j, save_page_to_neo4j, extract_navigation_items, save_navigation_to_neo4j

class LoginModule:
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
    async def get_tools(self):
        """Get MCP tools combined with Neo4j tools"""
        playwright_tools = await self.mcp_client.get_tools()
        return playwright_tools + [
            initialize_neo4j,
            save_page_to_neo4j,
            extract_navigation_items,
            save_navigation_to_neo4j
        ]
        
    async def login_flow(self, username: str, password: str, verification_code: str):
        """Execute login sequence"""
        tools = await self.get_tools()
        agent = create_agent(self.llm, tools)
        
        login_task = f"""Please complete the login sequence:
        
        1. Initialize Neo4j database connection using initialize_neo4j tool
        
        2. Navigate to https://v4staging.scantist.io/login 
        
        3. Fill in username field with '{username}' (use simple fill method)
        
        4. Fill in password field with '{password}' (use simple fill method)
        
        5. Fill in verification code field with '{verification_code}' (use simple fill method)
        
        6. Click Sign In button and wait for the dashboard page to load completely
        
        7. After successful login to the dashboard:
           a. Save the dashboard page to Neo4j
           b. Extract navigation items and save to Neo4j
           
        8. Return success status and current page URL"""
        
        try:
            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": login_task}]}
            )
            return {"success": True, "response": response}
        except Exception as e:
            return {"success": False, "error": str(e)}