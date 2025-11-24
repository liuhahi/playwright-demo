import asyncio
import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from neo4j_tools import save_page_to_neo4j, extract_navigation_items, save_navigation_to_neo4j, save_tab_feature_to_neo4j

class PageExplorationModule:
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
        
    async def explore_project_tabs(self, project_url: str):
        """Systematically explore all tabs in a project detail page"""
        tools = await self.get_tools()
        agent = create_agent(self.llm, tools)
        
        exploration_task = f"""Comprehensive exploration of project detail tabs and sections:
        
        Current project URL: {project_url}
        
        1. Take a screenshot to see current project page
        
        2. Identify ALL available tabs (e.g., Overview, Scan Results, Vulnerabilities, Licenses, Components, etc.)
        
        3. For EACH tab systematically:
           a. Click on the tab and wait for content to load
           b. Save each tab page to Neo4j using save_page_to_neo4j with specific URL
           c. Save each tab as a specific feature using save_tab_feature_to_neo4j (with type="tab")
           d. Extract ALL navigation items from each tab page
           e. Look for sub-tabs, filters, buttons, or additional navigation within each section
           f. For any sub-tabs found, save them using save_tab_feature_to_neo4j with parent_page reference
        
        4. For any scan results, data tables, or reports:
           a. Click on individual result items to see detail views
           b. Save each detail view page to Neo4j using save_page_to_neo4j
           c. Save each detail view as a feature using save_tab_feature_to_neo4j (with type="detail_view")
           d. Explore any expandable sections or drill-down options
        
        5. Return summary of all tabs and features explored"""
        
        try:
            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": exploration_task}]}
            )
            return {"success": True, "response": response}
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def explore_interactive_elements(self, page_url: str):
        """Explore buttons, filters, and interactive elements on current page"""
        tools = await self.get_tools()
        agent = create_agent(self.llm, tools)
        
        interactive_task = f"""Explore additional interactive elements on current page:
        
        Current page URL: {page_url}
        
        1. Take a screenshot to see current page state
        
        2. Look for and interact with:
           a. Buttons, filters, or action items - save using save_tab_feature_to_neo4j (type="button" or "filter")
           b. Expandable sections or accordions - save using save_tab_feature_to_neo4j (type="accordion" or "section")
           c. Chart elements or data visualizations - save using save_tab_feature_to_neo4j (type="chart" or "visualization")
           d. Modal dialogs or popup windows - save using save_tab_feature_to_neo4j (type="modal" or "dialog")
        
        3. For each interactive element:
           a. Try to interact with it (click, expand, filter)
           b. Save the resulting state/page to Neo4j
           c. Extract any new navigation items revealed
        
        4. Return summary of interactive elements explored"""
        
        try:
            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": interactive_task}]}
            )
            return {"success": True, "response": response}
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def explore_single_tab(self, tab_name: str, parent_url: str = ""):
        """Explore a specific tab by name"""
        tools = await self.get_tools()
        agent = create_agent(self.llm, tools)
        
        tab_task = f"""Explore the '{tab_name}' tab:
        
        Parent page URL: {parent_url}
        
        1. Look for and click on the "{tab_name}" tab
        
        2. Wait for the tab content to load completely
        
        3. Save the tab page to Neo4j using save_page_to_neo4j
        
        4. Save the tab as a feature using save_tab_feature_to_neo4j (type="tab", parent_page="{parent_url}")
        
        5. Extract and save all navigation items from the tab
        
        6. Look for any sub-sections, filters, or interactive elements within the tab
        
        7. If there are data tables or result lists, try clicking on individual items
        
        8. Return summary of tab content and any sub-features found"""
        
        try:
            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": tab_task}]}
            )
            return {"success": True, "response": response}
        except Exception as e:
            return {"success": False, "error": str(e)}