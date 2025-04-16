"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated

from react_agent.configuration import Configuration


async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


async def search_internal_docs(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> dict[str, Any]:
    """Search for internal documents in the Memento service.
    
    This function performs a search in the internal document database (Memento service)
    and returns matching documents based on the query.
    
    Args:
        query: The search query for finding relevant internal documents
    
    Returns:
        A dictionary with the search results
    """
    # For now, just log to console - will be replaced with actual API call later
    print(f"[MEMENTO TOOL] Searching internal documents for: {query}")
    
    # Return fallback data to simulate a successful search
    return {
        "status": "success",
        "message": f"Found documents related to '{query}'",
        "results": [
            {
                "title": f"{query} 결과 문서",
                "content": f"이것은 '{query}'에 대한 검색 결과입니다.",
                "relevance": 0.95
            }
        ]
    }


async def add_todo(
    title: str,
    due_date: Optional[str] = None,
    assignee: Optional[str] = None,
    checkpoints: Optional[List[str]] = None,
    *, 
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> dict[str, Any]:
    """Add a new todo item to the database.
    
    This function takes todo information and adds it to the todo database.
    Currently just logs the information to the console for demonstration purposes.
    
    Args:
        title: The title or description of the todo item
        due_date: When the todo item is due (optional)
        assignee: Who is responsible for the todo item (optional)
        checkpoints: List of checkpoints or milestones for the todo (optional)
    
    Returns:
        A dictionary with the created todo information
    """
    todo_item = {
        "title": title,
        "due_date": due_date,
        "assignee": assignee,
        "checkpoints": checkpoints or [],
    }
    
    # For now, just log to console - will be replaced with API call later
    print(f"[TODO TOOL] Adding new todo: {todo_item}")
    
    return {
        "status": "success",
        "message": "Todo added successfully",
        "todo": todo_item
    }


TOOLS: List[Callable[..., Any]] = [search, search_internal_docs, add_todo]

# Note: These base tools will be combined with MCP tools in the ReAct agent.
# The combination happens in the make_graph function in graph.py.
# MCP tools are loaded from the configuration and added to these base tools.
