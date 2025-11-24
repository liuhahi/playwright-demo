import asyncio
import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from neo4j_tools import save_page_to_neo4j, extract_navigation_items, save_navigation_to_neo4j, save_tab_feature_to_neo4j

class NavigationModule:
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
            save_page_to_neo4j,
            extract_navigation_items,
            save_navigation_to_neo4j,
            save_tab_feature_to_neo4j
        ]
        
    async def find_and_click_item(self, target_text: str, item_type: str = "link"):
        """Find and click on a specific navigation item"""
        tools = await self.get_tools()
        agent = create_agent(self.llm, tools)
        
        navigation_task = f"""Find and navigate to the '{target_text}' {item_type}:
        
        1. Take a screenshot to see current page state
        
        2. Look for and click on "{target_text}" navigation item, sidebar menu, or any clickable element
           - Check sidebar navigation, top menu, dashboard cards, or any visible UI elements
           - Look for text containing "{target_text}" (case insensitive)
           - Try multiple selectors if needed (text, button, link, etc.)
        
        3. Wait for the new page to load completely
        
        4. Save the new page to Neo4j using save_page_to_neo4j
        
        5. Save the navigation action as a feature using save_tab_feature_to_neo4j (type="{item_type}")
        
        6. Extract and save navigation items from the new page
        
        7. Return the new page URL and success status"""
        
        try:
            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": navigation_task}]}
            )
            return {"success": True, "response": response}
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def navigate_to_projects(self):
        """Navigate specifically to the Projects page"""
        return await self.find_and_click_item("Projects", "navigation")
        
    async def click_project_item(self, project_name: str):
        """Click on a specific project in the projects list"""
        tools = await self.get_tools()
        agent = create_agent(self.llm, tools)
        
        project_task = f"""Navigate to the specific project '{project_name}':
        
        1. Take a screenshot to see current projects list
        
        2. Look for project named "{project_name}" in the projects list
           - Look for project cards, table rows, or list items
           - Click on the project name, view button, or project link
        
        3. Wait for the project detail page to load completely
        
        4. Save the project detail page to Neo4j using save_page_to_neo4j
        
        5. Save the project access as a feature using save_tab_feature_to_neo4j (type="project_access")
        
        6. Extract navigation items from project detail page
        
        7. Return the project detail page URL and available tabs"""
        
        try:
            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": project_task}]}
            )
            return {"success": True, "response": response}
        except Exception as e:
            return {"success": False, "error": str(e)}