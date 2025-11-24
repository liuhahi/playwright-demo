import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from login_module import LoginModule
from navigation_module import NavigationModule
from page_exploration_module import PageExplorationModule

load_dotenv()

class WebsiteExplorer:
    def __init__(self):
        self.mcp_client = None
        self.login_module = None
        self.navigation_module = None
        self.exploration_module = None
        self.explored_projects = []
        
    async def initialize(self):
        """Initialize MCP client and modules"""
        self.mcp_client = MultiServerMCPClient(
            {
                "playwright": {
                    "transport": "streamable_http",
                    "url": "http://localhost:8931/mcp",
                }
            }
        )
        
        self.login_module = LoginModule(self.mcp_client)
        self.navigation_module = NavigationModule(self.mcp_client)
        self.exploration_module = PageExplorationModule(self.mcp_client)
        
    async def login(self, username: str, password: str, verification_code: str):
        """Execute login flow"""
        print("Starting login flow...")
        result = await self.login_module.login_flow(username, password, verification_code)
        if result["success"]:
            print("Login completed successfully")
        else:
            print(f"Login failed: {result['error']}")
        return result
        
    async def navigate_to_projects(self):
        """Navigate to projects page"""
        print("Navigating to Projects page...")
        result = await self.navigation_module.navigate_to_projects()
        if result["success"]:
            print("Successfully navigated to Projects page")
        else:
            print(f"Navigation failed: {result['error']}")
        return result
        
    async def explore_project(self, project_name: str):
        """Explore a specific project in detail"""
        print(f"Exploring project: {project_name}")
        
        # Navigate to the project
        nav_result = await self.navigation_module.click_project_item(project_name)
        if not nav_result["success"]:
            print(f"Failed to navigate to project {project_name}: {nav_result['error']}")
            return nav_result
            
        # Explore all tabs in the project
        exploration_result = await self.exploration_module.explore_project_tabs(f"project/{project_name}")
        if exploration_result["success"]:
            print(f"Successfully explored project {project_name}")
            self.explored_projects.append(project_name)
        else:
            print(f"Project exploration failed: {exploration_result['error']}")
            
        return exploration_result
        
    async def explore_specific_tab(self, tab_name: str, parent_url: str = ""):
        """Explore a specific tab by name"""
        print(f"Exploring tab: {tab_name}")
        result = await self.exploration_module.explore_single_tab(tab_name, parent_url)
        if result["success"]:
            print(f"Successfully explored tab: {tab_name}")
        else:
            print(f"Tab exploration failed: {result['error']}")
        return result
        
    async def explore_interactive_elements(self, page_url: str):
        """Explore interactive elements on current page"""
        print(f"Exploring interactive elements on: {page_url}")
        result = await self.exploration_module.explore_interactive_elements(page_url)
        if result["success"]:
            print("Successfully explored interactive elements")
        else:
            print(f"Interactive elements exploration failed: {result['error']}")
        return result
        
    async def full_website_exploration(self, username: str, password: str, verification_code: str, project_names: list = None):
        """Complete website exploration workflow"""
        print("Starting full website exploration...")
        
        # Step 1: Login
        login_result = await self.login(username, password, verification_code)
        if not login_result["success"]:
            return {"success": False, "error": "Login failed", "details": login_result}
            
        # Step 2: Navigate to Projects
        projects_result = await self.navigate_to_projects()
        if not projects_result["success"]:
            return {"success": False, "error": "Projects navigation failed", "details": projects_result}
            
        # Step 3: Explore projects
        if project_names:
            # Explore specific projects
            for project_name in project_names:
                await self.explore_project(project_name)
        else:
            # If no specific projects provided, explore first available project
            print("No specific projects provided, will explore first available project")
            await self.explore_project("first_available")
            
        # Step 4: Generate summary
        summary = {
            "success": True,
            "explored_projects": self.explored_projects,
            "total_projects_explored": len(self.explored_projects),
            "login_status": "completed",
            "projects_navigation": "completed"
        }
        
        print(f"Exploration completed. Summary: {summary}")
        return summary
        
    async def custom_workflow(self, steps: list):
        """Execute custom workflow with specific steps"""
        print(f"Starting custom workflow with {len(steps)} steps...")
        results = []
        
        for i, step in enumerate(steps):
            print(f"Executing step {i+1}: {step['action']}")
            
            if step["action"] == "login":
                result = await self.login(step["username"], step["password"], step["verification_code"])
            elif step["action"] == "navigate_to_projects":
                result = await self.navigate_to_projects()
            elif step["action"] == "explore_project":
                result = await self.explore_project(step["project_name"])
            elif step["action"] == "explore_tab":
                result = await self.explore_specific_tab(step["tab_name"], step.get("parent_url", ""))
            elif step["action"] == "explore_interactive":
                result = await self.explore_interactive_elements(step["page_url"])
            else:
                result = {"success": False, "error": f"Unknown action: {step['action']}"}
                
            results.append({"step": i+1, "action": step["action"], "result": result})
            
            if not result["success"]:
                print(f"Step {i+1} failed: {result['error']}")
                break
                
        return {"success": True, "steps_completed": len(results), "results": results}