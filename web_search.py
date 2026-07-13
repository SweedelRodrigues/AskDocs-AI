import os
import json
import logging
import requests

logger = logging.getLogger(__name__)

# Load config to get API keys if available
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
config_data = {}
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading config.json in web_search: {str(e)}")

def get_api_key(key_name):
    # Try config.json first, then environment variables
    return config_data.get(key_name) or os.environ.get(key_name)

def tavily_search(query: str, api_key: str, max_results: int = 5):
    """
    Performs a web search using the Tavily API.
    """
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "include_answer": False,
        "max_results": max_results
    }
    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    results = []
    for r in data.get("results", []):
        results.append({
            "title": r.get("title", "Tavily Result"),
            "snippet": r.get("content", ""),
            "link": r.get("url", "")
        })
    return results

def serper_search(query: str, api_key: str, max_results: int = 5):
    """
    Performs a web search using the Serper API.
    """
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "q": query,
        "num": max_results
    }
    response = requests.post(url, headers=headers, json=payload, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    results = []
    for r in data.get("organic", []):
        results.append({
            "title": r.get("title", "Serper Result"),
            "snippet": r.get("snippet", ""),
            "link": r.get("link", "")
        })
    return results

def duckduckgo_search(query: str, max_results: int = 5):
    """
    Performs a web search using DuckDuckGo search.
    Tries the duckduckgo-search package, falls back to langchain_community.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                raise ValueError("DuckDuckGo search returned zero results (possibly rate limited).")
            return [
                {
                    "title": r.get("title", "DuckDuckGo Result"),
                    "snippet": r.get("body", ""),
                    "link": r.get("href", "")
                }
                for r in results
            ]
    except Exception as e_ddg:
        logger.warning(f"duckduckgo-search package failed or not installed: {str(e_ddg)}")
        # Try LangChain fallback
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
            wrapper = DuckDuckGoSearchAPIWrapper(max_results=max_results)
            ddg = DuckDuckGoSearchRun(api_wrapper=wrapper)
            res = ddg.run(query)
            return [{
                "title": "DuckDuckGo Web Result",
                "snippet": res,
                "link": f"https://duckduckgo.com/?q={requests.utils.quote(query)}"
            }]
        except Exception as e_lc:
            raise RuntimeError(f"DuckDuckGo search failed. Details: DDG library error: {str(e_ddg)}, Langchain fallback error: {str(e_lc)}")

def perform_web_search(query: str, max_results: int = 5):
    """
    Intelligent search router that uses Tavily, Serper, or DuckDuckGo in priority order.
    Returns: (list of search result dicts, provider_name_string)
    """
    tavily_key = get_api_key("TAVILY_API_KEY")
    if tavily_key:
        try:
            logger.info("Using Tavily Search")
            return tavily_search(query, tavily_key, max_results), "Tavily"
        except Exception as e:
            logger.error(f"Tavily search failed, falling back: {str(e)}")
            
    serper_key = get_api_key("SERPER_API_KEY")
    if serper_key:
        try:
            logger.info("Using Serper Search")
            return serper_search(query, serper_key, max_results), "Serper"
        except Exception as e:
            logger.error(f"Serper search failed, falling back: {str(e)}")
            
    # Default to DuckDuckGo
    logger.info("Using DuckDuckGo Search")
    return duckduckgo_search(query, max_results), "DuckDuckGo"
