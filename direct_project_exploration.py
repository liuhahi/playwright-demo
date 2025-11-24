#!/usr/bin/env python3
"""
Direct script to force navigation to Projects page and explore project details
"""

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
    
    playwright_tools = await client.get_tools()
    
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    agent = create_agent(llm, playwright_tools)
    
    # Direct and focused task for project exploration
    task = """You MUST complete this exact sequence step by step:

1. Take a screenshot of the current page to see what's available
2. Look for ANY navigation element that could lead to projects:
   - Sidebar navigation items
   - Top menu items 
   - Dashboard cards or widgets
   - Buttons with text like "Projects", "View Projects", "Project List", etc.
   - Icons that might represent projects or lists

3. If you see a sidebar navigation (usually on the left), click on ANY item that could be projects-related

4. If there's no obvious "Projects" link, try clicking on navigation items one by one to find projects:
   - Look for icons that might represent project management
   - Try "Dashboard", "Overview", "Portfolio", or similar terms
   - Check if there are dropdown menus

5. Once you find the projects page:
   - Take a screenshot to confirm you're on projects page
   - Look for a list or grid of projects
   - Click on ANY project in the list to access project details

6. On the project detail page:
   - Take a screenshot
   - Look for tabs across the top or side (Overview, Scan Results, Vulnerabilities, etc.)
   - Click on EACH available tab one by one
   - Take a screenshot after clicking each tab

7. Provide a summary of:
   - How you found the projects page
   - What projects were listed
   - What tabs were available in project details
   - What content was found in each tab

START BY TAKING A SCREENSHOT TO SEE THE CURRENT STATE OF THE PAGE."""

    try:
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": task}]}
        )
        print("Project exploration completed:", response)
    except Exception as e:
        print(f"Error during project exploration: {e}")

if __name__ == "__main__":
    asyncio.run(main())