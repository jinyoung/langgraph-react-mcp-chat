"""Define the configurable parameters for the agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig, ensure_config
from dotenv import load_dotenv

from react_agent import prompts

# Load environment variables from .env file
load_dotenv()


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    mcp_tools: str = field(
        default=os.getenv("MCP_TOOLS", "mcp_config.json"),
        metadata={"description": "The path to the MCP tools configuration file."},
    )

    recursion_limit: int = field(
        default=int(os.getenv("RECURSION_LIMIT", "30")),
        metadata={
            "description": "The maximum number of recursive calls that Agent can make."
        },
    )
    
    model_provider: Optional[str] = field(
        default=os.getenv("MODEL_PROVIDER", "openai"),
        metadata={
            "description": "The model provider to use (anthropic or openai)."
        },
    )
    
    model_name: Optional[str] = field(
        default=os.getenv("MODEL_NAME", None),
        metadata={
            "description": "The specific model name to use. If not provided, defaults will be used (claude-3-7-sonnet-latest for Anthropic, gpt-4o for OpenAI)."
        },
    )
    
    max_search_results: int = field(
        default=int(os.getenv("MAX_SEARCH_RESULTS", "3")),
        metadata={
            "description": "The maximum number of search results to return from search tools."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
