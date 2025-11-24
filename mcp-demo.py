import asyncio
import os
from dotenv import load_dotenv
from website_explorer import WebsiteExplorer

load_dotenv()

async def main():
    # Initialize the website explorer
    explorer = WebsiteExplorer()
    await explorer.initialize()
    
    # Get credentials from environment
    username = os.getenv("USER_NAME")
    password = os.getenv("PASSWORD") 
    verification_code = os.getenv("VERIFICATION_CODE")
    
    # Option 1: Full automated exploration
    print("Starting full website exploration...")
    result = await explorer.full_website_exploration(username, password, verification_code)
    print(f"Exploration completed: {result}")
    
    # Option 2: Custom workflow (alternative approach)
    # custom_steps = [
    #     {"action": "login", "username": username, "password": password, "verification_code": verification_code},
    #     {"action": "navigate_to_projects"},
    #     {"action": "explore_project", "project_name": "specific_project_name"},
    #     {"action": "explore_tab", "tab_name": "Vulnerabilities", "parent_url": "project/specific_project_name"},
    #     {"action": "explore_interactive", "page_url": "current_page_url"}
    # ]
    # result = await explorer.custom_workflow(custom_steps)

if __name__ == "__main__":
    asyncio.run(main())