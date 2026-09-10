import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

def search_web(query: str, max_results: int = 3) -> list[dict]:
    """Performs a web search via DuckDuckGo"""
    print(f"Searching web for: {query}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as e:
        return [{"error": f"Search failed: {e}"}]

def read_url_content(url: str) -> str:
    """Fetches and extracts text from a specific webpage"""
    print(f"Reading URL: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract main text
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
            
        text = soup.get_text(separator=' ', strip=True)
        # Truncate if too long (e.g. max 5000 chars for context)
        return text[:5000] + ("..." if len(text) > 5000 else "")
    except Exception as e:
        return f"Error reading URL: {e}"
