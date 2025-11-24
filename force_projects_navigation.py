#!/usr/bin/env python3
"""
Force navigation to Projects page and save each discovered tab/page as a feature
"""

import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from dotenv import load_dotenv
from working_memory_importer import WebsiteWMImporter
from datetime import datetime

load_dotenv()

# Global Neo4j importer instance  
neo4j_importer = None

@tool
def save_page_with_feature(url: str, title: str = "", feature_name: str = "", feature_type: str = "page") -> str:
    """Save both a page and its associated feature to Neo4j in one operation."""
    global neo4j_importer
    if not neo4j_importer:
        neo4j_importer = WebsiteWMImporter("bolt://localhost:7687", "neo4j", "12345678")
        neo4j_importer.connect()
        neo4j_importer.ensure_schema()
    
    try:
        with neo4j_importer._driver.session() as session:
            # Save page
            page_data = {
                "url": url,
                "route": url.split('scantist.io')[-1] if 'scantist.io' in url else url,
                "title": title,
                "status": "explored",
                "last_crawled_at": datetime.now().isoformat()
            }
            
            page_query = """
            MERGE (p:Page {url: $url})
            SET p.title = $title, p.route = $route, p.status = $status, p.last_crawled_at = $last_crawled_at
            """
            session.run(page_query, **page_data)
            
            # Save feature
            if feature_name:
                feature_data = {
                    "page_url": url,
                    "name": feature_name,
                    "type": feature_type,
                    "selector": f"{feature_type}={feature_name}",
                    "created_at": datetime.now().isoformat()
                }
                
                feature_query = """
                MERGE (p:Page {url: $page_url})
                CREATE (f:Feature {
                    page_url: $page_url,
                    name: $name,
                    type: $type,
                    selector: $selector,
                    created_at: $created_at
                })
                CREATE (p)-[:CONTAINS]->(f)
                """
                session.run(feature_query, **feature_data)
            
        return f"✅ Saved page: {url} with feature: {feature_name} ({feature_type})"
    except Exception as e:
        return f"❌ Error saving: {str(e)}"

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
    
    # Add our custom tool
    all_tools = playwright_tools + [save_page_with_feature]
    
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    agent = create_agent(llm, all_tools)
    
    # Focused task to navigate to projects and save each page/tab
    task = """EXECUTE THIS EXACT SEQUENCE:

1. Take a screenshot first to see current page state

2. Look for navigation to Projects - check ALL possibilities:
   - Look for sidebar navigation items (usually on left side)
   - Look for top menu items
   - Look for any buttons/links with "Project" or "Projects" text
   - Look for dashboard cards that might lead to projects

3. Click on ANY element that might lead to projects page:
   - If you find a "Projects" link or button, click it
   - If no direct link, try navigating through menus
   - Look for icons that might represent project lists

4. Once you reach a projects page:
   - Take a screenshot to confirm
   - Use save_page_with_feature to save the projects list page (feature_type="projects_page")
   - Look for individual projects in the list
   - Click on the FIRST project you find

5. When you reach a project detail page:
   - Take a screenshot
   - Use save_page_with_feature to save this project detail page (feature_type="project_detail") 
   - Look for ALL tabs (Overview, Scan Results, Vulnerabilities, Components, etc.)
   - For EACH tab you find:
     a. Click on the tab
     b. Wait for content to load
     c. Take a screenshot  
     d. Use save_page_with_feature to save this tab page (feature_type="tab", feature_name="TabName")

6. Continue until you have explored at least 3-4 different tabs

7. Report back what you discovered:
   - How many projects were in the list
   - What tabs were available
   - What content was in each tab

START NOW by taking a screenshot to see the current page state."""

    try:
        print("🚀 Starting forced navigation to Projects...")
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": task}]}
        )
        print("✅ Projects navigation completed!")
        print("📊 Response:", response)
        
        # Check what was saved
        print("\n📈 Checking saved data...")
        if neo4j_importer:
            with neo4j_importer._driver.session() as session:
                result = session.run("MATCH (p:Page) RETURN count(*) as count")
                page_count = list(result)[0]['count']
                result = session.run("MATCH (f:Feature) RETURN count(*) as count")
                feature_count = list(result)[0]['count']
                print(f"📄 Total pages: {page_count}")
                print(f"🔧 Total features: {feature_count}")
                
    except Exception as e:
        print(f"❌ Error during navigation: {e}")

if __name__ == "__main__":
    asyncio.run(main())