import os
from pathlib import Path
from langchain.tools import tool
from working_memory_importer import WebsiteWMImporter
from datetime import datetime
import json
import hashlib
from urllib.parse import urlparse

# Global Neo4j importer instance
neo4j_importer = None

@tool
def initialize_neo4j(neo4j_uri: str = "bolt://localhost:7687", neo4j_user: str = "neo4j", neo4j_password: str = "12345678") -> str:
    """
    Initialize connection to Neo4j database.
    
    Args:
        neo4j_uri: Neo4j connection URI (default: bolt://localhost:7687)
        neo4j_user: Neo4j username (default: neo4j)
        neo4j_password: Neo4j password (default: 12345678)
    
    Returns:
        Success or error message
    """
    global neo4j_importer
    try:
        neo4j_importer = WebsiteWMImporter(neo4j_uri, neo4j_user, neo4j_password)
        if neo4j_importer.connect():
            neo4j_importer.ensure_schema()
            return "Successfully connected to Neo4j and ensured schema"
        else:
            return "Failed to connect to Neo4j"
    except Exception as e:
        return f"Error initializing Neo4j: {str(e)}"

@tool
def save_page_to_neo4j(url: str, title: str = "", route: str = "", status: str = "explored", html_content: str = "") -> str:
    """
    Save page information to Neo4j.
    
    Args:
        url: Page URL
        title: Page title
        route: Page route/path
        status: Page status (default: explored)
        html_content: HTML content for hashing
    
    Returns:
        Success or error message
    """
    global neo4j_importer
    if not neo4j_importer:
        return "Neo4j not initialized. Call initialize_neo4j first."
    
    try:
        # Generate hash from HTML content
        html_hash = hashlib.md5(html_content.encode()).hexdigest() if html_content else ""
        
        page_data = {
            "url": url,
            "route": route or urlparse(url).path,
            "title": title,
            "status": status,
            "html_hash": html_hash,
            "last_crawled_at": datetime.now().isoformat()
        }
        
        if neo4j_importer.upsert_page(page_data):
            return f"Successfully saved page to Neo4j: {url}"
        else:
            return f"Failed to save page to Neo4j: {url}"
    except Exception as e:
        return f"Error saving page to Neo4j: {str(e)}"

@tool
def extract_navigation_items(snapshot_data: str, page_url: str, screenshot_b64: str = "") -> str:
    """
    Extract navigation items and clickable elements using GPT vision and snapshot analysis.
    
    Args:
        snapshot_data: Page snapshot or HTML content
        page_url: URL of the current page
        screenshot_b64: Base64 encoded screenshot for vision analysis (optional)
    
    Returns:
        JSON string of extracted navigation items analyzed by GPT
    """
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Prepare the prompt for GPT to analyze the page
        prompt = f"""
        Analyze this web page content and screenshot to identify all navigatable items, clickable elements, and interactive components.
        
        Page URL: {page_url}
        
        Please identify and extract:
        1. Navigation menu items (sidebar, top nav, breadcrumbs)
        2. Buttons and clickable elements
        3. Links and hyperlinks
        4. Form controls (if interactive)
        5. Tabs and accordion items
        6. Any other interactive UI elements
        
        For each item, provide:
        - text: The visible text or label
        - type: The type of element (nav_link, button, tab, form_element, etc.)
        - description: Brief description of what this element does
        - location: Where on the page this element appears (header, sidebar, main content, etc.)
        
        Return the results as a JSON array of objects with these properties.
        
        Page Content/Snapshot:
        {snapshot_data[:8000]}  # Limit content size
        """
        
        # Prepare messages for the API call
        messages = [
            {
                "role": "system",
                "content": "You are a web page analyst expert at identifying navigatable and interactive elements. Provide detailed, accurate analysis of web page components."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Add screenshot to the analysis if provided
        if screenshot_b64:
            messages[1]["content"] = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{screenshot_b64}"
                    }
                }
            ]
        
        # Call GPT to analyze the page
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=2000,
            temperature=0.1
        )
        
        # Extract the analysis from the response
        analysis_text = response.choices[0].message.content
        
        # Try to parse JSON from the response
        import re
        json_match = re.search(r'```json\n(.*?)\n```', analysis_text, re.DOTALL)
        if json_match:
            navigation_data = json_match.group(1)
        else:
            # Look for JSON array in the response
            json_match = re.search(r'\[\s*{.*?}\s*\]', analysis_text, re.DOTALL)
            if json_match:
                navigation_data = json_match.group(0)
            else:
                # Fallback: return the analysis as structured text
                navigation_items = []
                lines = analysis_text.split('\n')
                current_item = {}
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('- text:'):
                        if current_item:
                            navigation_items.append(current_item)
                        current_item = {"text": line.replace('- text:', '').strip()}
                    elif line.startswith('- type:') and current_item:
                        current_item["type"] = line.replace('- type:', '').strip()
                    elif line.startswith('- description:') and current_item:
                        current_item["description"] = line.replace('- description:', '').strip()
                    elif line.startswith('- location:') and current_item:
                        current_item["location"] = line.replace('- location:', '').strip()
                        current_item["page_url"] = page_url
                
                if current_item:
                    navigation_items.append(current_item)
                
                navigation_data = json.dumps(navigation_items, indent=2)
        
        return navigation_data
        
    except Exception as e:
        return f"Error extracting navigation items with GPT: {str(e)}"

@tool
def save_navigation_to_neo4j(navigation_data: str, page_url: str) -> str:
    """
    Save extracted navigation items to Neo4j as features.
    
    Args:
        navigation_data: JSON string of navigation items
        page_url: URL of the page containing the navigation
    
    Returns:
        Success or error message
    """
    global neo4j_importer
    if not neo4j_importer:
        return "Neo4j not initialized. Call initialize_neo4j first."
    
    try:
        navigation_items = json.loads(navigation_data)
        
        with neo4j_importer._driver.session() as session:
            for item in navigation_items:
                feature_data = {
                    "page_url": page_url,
                    "name": item.get("text", ""),
                    "type": item.get("type", "navigation"),
                    "href": item.get("href", ""),
                    "absolute_url": item.get("absolute_url", ""),
                    "selector": f"text={item.get('text', '')}",
                    "created_at": datetime.now().isoformat()
                }
                
                query = """
                MERGE (p:Page {url: $page_url})
                CREATE (f:Feature {
                    page_url: $page_url,
                    name: $name,
                    type: $type,
                    href: $href,
                    absolute_url: $absolute_url,
                    selector: $selector,
                    created_at: $created_at
                })
                CREATE (p)-[:CONTAINS]->(f)
                """
                
                session.run(query, **feature_data)
        
        return f"Successfully saved {len(navigation_items)} navigation items to Neo4j"
    except Exception as e:
        return f"Error saving navigation to Neo4j: {str(e)}"

@tool
def save_tab_feature_to_neo4j(page_url: str, tab_name: str, tab_type: str = "tab", description: str = "", parent_page: str = "") -> str:
    """
    Save individual tabs and UI features as specific features in Neo4j.
    
    Args:
        page_url: URL of the page containing the tab
        tab_name: Name of the tab or feature
        tab_type: Type of feature (tab, button, section, filter, etc.)
        description: Description of the tab functionality
        parent_page: URL of the parent page if this is a sub-feature
    
    Returns:
        Success or error message
    """
    global neo4j_importer
    if not neo4j_importer:
        return "Neo4j not initialized. Call initialize_neo4j first."
    
    try:
        with neo4j_importer._driver.session() as session:
            feature_data = {
                "page_url": page_url,
                "name": tab_name,
                "type": tab_type,
                "description": description,
                "parent_page": parent_page,
                "selector": f"tab={tab_name}",
                "created_at": datetime.now().isoformat()
            }
            
            # Create the feature and link it to the page
            query = """
            MERGE (p:Page {url: $page_url})
            CREATE (f:Feature {
                page_url: $page_url,
                name: $name,
                type: $type,
                description: $description,
                parent_page: $parent_page,
                selector: $selector,
                created_at: $created_at
            })
            CREATE (p)-[:CONTAINS]->(f)
            """
            
            # If there's a parent page, also create relationship
            if parent_page:
                query += """
                WITH f, p
                MERGE (parent:Page {url: $parent_page})
                CREATE (parent)-[:HAS_SUB_FEATURE]->(f)
                """
            
            session.run(query, **feature_data)
        
        return f"Successfully saved {tab_type} '{tab_name}' to Neo4j for page: {page_url}"
    except Exception as e:
        return f"Error saving tab feature to Neo4j: {str(e)}"

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