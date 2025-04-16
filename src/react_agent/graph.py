from datetime import datetime, timezone
from typing import Dict, List, Literal, cast, Optional, Callable, Any
import os

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent import utils
from contextlib import asynccontextmanager
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

memory = MemorySaver()


@asynccontextmanager
async def make_graph(mcp_tools: Dict[str, Dict[str, str]], model: BaseChatModel):
    async with MultiServerMCPClient(mcp_tools) as client:
        # Get MCP tools and combine with existing TOOLS
        mcp_tool_objects = client.get_tools()
        all_tools = TOOLS + mcp_tool_objects
        
        # Create agent with combined tools
        agent = create_react_agent(model, all_tools, checkpointer=memory)
        yield agent


def get_model(model_provider: str, model_name: Optional[str] = None) -> BaseChatModel:
    """Get the appropriate model based on provider and name.
    
    Args:
        model_provider (str): The model provider (anthropic or openai)
        model_name (Optional[str]): The specific model name to use
        
    Returns:
        BaseChatModel: A LangChain chat model
    """
    if model_provider.lower() == "anthropic":
        return ChatAnthropic(
            model=model_name or "claude-3-7-sonnet-latest", 
            temperature=0.0, 
            max_tokens=64000,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    elif model_provider.lower() == "openai":
        return ChatOpenAI(
            model=model_name or "gpt-4o", 
            temperature=0.0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")


async def call_model(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_runnable_config(config)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=timezone.utc).isoformat()
    )

    # Make sure state is available to the tools
    config["state"] = state

    mcp_json_path = configuration.mcp_tools

    mcp_tools_config = await utils.load_mcp_config_json(mcp_json_path)

    # Extract the servers configuration from mcpServers key
    mcp_tools = mcp_tools_config.get("mcpServers", {})
    print(mcp_tools)
    
    # Get model provider and name from configuration
    model_provider = configuration.model_provider or "openai"
    model_name = configuration.model_name
    
    # Initialize the model
    model = get_model(model_provider, model_name)

    response = None

    async with make_graph(mcp_tools, model) as my_agent:
        # Create the messages list
        messages = [
            SystemMessage(content=system_message),
            *state.messages,
        ]

        # Pass messages with the correct dictionary structure
        response = cast(
            AIMessage,
            await my_agent.ainvoke(
                {"messages": messages},
                config,
            ),
        )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response["messages"][-1]]}


# Create a function to get combined tools (MCP tools + TOOLS)
async def get_combined_tools(config: RunnableConfig) -> List[Callable[..., Any]]:
    """Get combined tools from both TOOLS and MCP tools.
    
    Args:
        config (RunnableConfig): Configuration for the run.
        
    Returns:
        List[Callable[..., Any]]: Combined list of tools.
    """
    configuration = Configuration.from_runnable_config(config)
    mcp_json_path = configuration.mcp_tools
    mcp_tools_config = await utils.load_mcp_config_json(mcp_json_path)
    mcp_tools = mcp_tools_config.get("mcpServers", {})
    
    async with MultiServerMCPClient(mcp_tools) as client:
        mcp_tool_objects = client.get_tools()
        return TOOLS + mcp_tool_objects


# Define a new graph

builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the two nodes we will cycle between
builder.add_node(call_model)

# We'll use a custom tool node function that combines TOOLS with MCP tools
async def tool_node_with_combined_tools(state, config):
    """A custom tool node that uses combined tools from both TOOLS and MCP tools."""
    # Make sure state is available to the tools
    config["state"] = state
    
    combined_tools = await get_combined_tools(config)
    tool_node = ToolNode(combined_tools)
    return await tool_node.invoke(state, config)

builder.add_node("tools", tool_node_with_combined_tools)

# Set the entrypoint as `call_model`
# This means that this node is the first one called
builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"


# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")

# Compile the builder into an executable graph
# You can customize this by adding interrupt points for state updates
graph = builder.compile(
    interrupt_before=[],  # Add node names here to update state before they're called
    interrupt_after=[],  # Add node names here to update state after they're called
)
graph.name = "ReAct Agent"  # This customizes the name in LangSmith
