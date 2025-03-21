from typing import Dict, Optional
import requests
from autogen_core.code_executor import ImportFromModule
from autogen_core.tools import FunctionTool
 
from bs4 import BeautifulSoup
import html2text
from urllib.parse import urljoin 


async def fetch_webpage(
    url: str,
    include_images: bool = True,
    max_length: Optional[int] = None,
    headers: Optional[Dict[str, str]] = None
) -> str:
    """Fetch a webpage and convert it to markdown format.
    
    Args:
        url: The URL of the webpage to fetch
        include_images: Whether to include image references in the markdown
        max_length: Maximum length of the output markdown (if None, no limit)
        headers: Optional HTTP headers for the request
        
    Returns:
        str: Markdown version of the webpage content
        
    Raises:
        ValueError: If the URL is invalid or the page can't be fetched
    """
    # Use default headers if none provided
    print("----------------------------")
    print("[debug] fetch_webpage() -> ")
    print(f"url : {url}")
    print(f"include image : {include_images}")
    print(f"max_length (by agent) : {max_length}")
    ####### Modified (Prommin) ########
    max_length_manual = -1
    if max_length_manual != -1 : 
      max_length = max_length_manual 
      print(f"max_length (forced) : {max_length}")
    ###################################

    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    try:
        # Fetch the webpage
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Convert relative URLs to absolute
        for tag in soup.find_all(['a', 'img']):
            if tag.get('href'):
                tag['href'] = urljoin(url, tag['href'])
            if tag.get('src'):
                tag['src'] = urljoin(url, tag['src'])
        
        # Configure HTML to Markdown converter
        h2t = html2text.HTML2Text()
        h2t.body_width = 0  # No line wrapping
        h2t.ignore_images = not include_images
        h2t.ignore_emphasis = False
        h2t.ignore_links = False
        h2t.ignore_tables = False
        
        # Convert to markdown
        markdown = h2t.handle(str(soup))
        
        # Trim if max_length is specified
        if max_length and len(markdown) > max_length:
            markdown = markdown[:max_length] + "\n...(truncated)"
        print("----------------------------")
        return markdown.strip()
        
    except requests.RequestException as e:
        print(f"fetch_webpage() -> [debug] Fetch Error {str(e)}")
        raise ValueError(f"Failed to fetch webpage: {str(e)}")
    except Exception as e:
        print(f"fetch_webpage() -> [debug] Fetch Error {str(e)}")
        raise ValueError(f"Error processing webpage: {str(e)}")

# Create the webpage fetching tool
fetch_webpage_tool = FunctionTool(
    func=fetch_webpage,
    description="Fetch a webpage and convert it to markdown format, with options for including images and limiting length",
    global_imports=[
        "os",
        "html2text",
        ImportFromModule("typing", ("Optional", "Dict")),
        "requests",
        ImportFromModule("bs4", ("BeautifulSoup",)),
        ImportFromModule("html2text", ("HTML2Text",)),
        ImportFromModule("urllib.parse", ("urljoin",))
    ]
)