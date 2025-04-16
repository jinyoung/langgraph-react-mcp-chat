"""FastAPI service example for the multi-tenant LangGraph agent.

This example creates a simple FastAPI service that manages conversations
for multiple tenants, ensuring that each tenant's data is properly isolated.
"""

import os
import uuid
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from react_agent import graph, create_state_with_tenant, State

app = FastAPI(title="Multi-Tenant LangGraph Agent API")

# In-memory conversation store - in production, use a database
conversations: Dict[str, Dict[str, State]] = {}


class ConversationRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class Message(BaseModel):
    role: str
    content: str


class ConversationResponse(BaseModel):
    conversation_id: str
    messages: List[Message]


def get_tenant_id(x_tenant_id: str = Header(...)):
    """Extract and validate tenant ID from header."""
    if not x_tenant_id:
        raise HTTPException(status_code=400, detail="X-Tenant-ID header is required")
    return x_tenant_id


@app.post("/conversation", response_model=ConversationResponse)
async def create_or_continue_conversation(
    request: ConversationRequest, tenant_id: str = Depends(get_tenant_id)
):
    """Create a new conversation or continue an existing one."""
    # Initialize tenant's conversation store if doesn't exist
    if tenant_id not in conversations:
        conversations[tenant_id] = {}

    # Get or create conversation state
    if request.conversation_id and request.conversation_id in conversations[tenant_id]:
        # Continue existing conversation
        state = conversations[tenant_id][request.conversation_id]
        # Add new message
        state.messages.append(HumanMessage(content=request.message))
    else:
        # Create new conversation
        conversation_id = str(uuid.uuid4())
        state = create_state_with_tenant(
            messages=[HumanMessage(content=request.message)],
            tenant_id=tenant_id
        )
        conversations[tenant_id][conversation_id] = state
        request.conversation_id = conversation_id

    # Invoke the graph
    result = await graph.ainvoke(state)
    
    # Update state in storage
    conversations[tenant_id][request.conversation_id] = result
    
    # Format response
    messages = []
    for msg in result.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        messages.append(Message(role=role, content=msg.content))
    
    return ConversationResponse(
        conversation_id=request.conversation_id,
        messages=messages
    )


@app.get("/conversations", response_model=List[str])
async def list_conversations(tenant_id: str = Depends(get_tenant_id)):
    """List all conversation IDs for a tenant."""
    tenant_conversations = conversations.get(tenant_id, {})
    return list(tenant_conversations.keys())


@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str, tenant_id: str = Depends(get_tenant_id)):
    """Delete a specific conversation."""
    if tenant_id not in conversations or conversation_id not in conversations[tenant_id]:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    del conversations[tenant_id][conversation_id]
    return {"status": "success", "message": "Conversation deleted"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 