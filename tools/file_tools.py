from pathlib import Path
import base64

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

def save_screenshot_to_folder(page_name: str, screenshot_data: str, folder: str = "screenshots") -> str:
    """
    Save a screenshot to a specified folder.
    
    Args:
        page_name: Name of the page (will be used as filename)
        screenshot_data: Base64 encoded screenshot data or file path
        folder: Folder to save screenshots in (default: 'screenshots')
    
    Returns:
        Success message with file path
    """
    try:
        # Create screenshots directory if it doesn't exist
        dir_path = Path(folder)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Clean page name for filename
        clean_name = "".join(c for c in page_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_name = clean_name.replace(' ', '_')
        filename = f"{clean_name}_screenshot.png"
        
        # Create full file path
        file_path = dir_path / filename
        
        # If screenshot_data looks like base64, decode and save
        if screenshot_data.startswith('data:image') or len(screenshot_data) > 100:
            # Handle base64 data
            if screenshot_data.startswith('data:image'):
                screenshot_data = screenshot_data.split(',')[1]
            
            with open(file_path, 'wb') as f:
                f.write(base64.b64decode(screenshot_data))
        else:
            # Assume it's a file path, copy the file
            import shutil
            shutil.copy2(screenshot_data, file_path)
        
        return f"Successfully saved screenshot to {file_path.absolute()}"
    except Exception as e:
        return f"Error saving screenshot: {str(e)}"