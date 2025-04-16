"""Example of initializing the LangGraph agent with a specific tenant ID.

This example demonstrates how to create a new conversation thread
with a specific tenant ID, which ensures that all tool calls within
that conversation are scoped to the specific tenant.
"""

import asyncio
from langchain_core.messages import HumanMessage
from react_agent import create_state_with_tenant, graph

async def main():
    # Create initial messages
    messages = [
        HumanMessage(content="내부 문서를 검색해서 '프로젝트 계획'에 대한 정보를 찾아줘")
    ]
    
    # Initialize state with specific tenant ID
    tenant_id = "tenant123"  # In a real app, this would come from your auth system
    state = create_state_with_tenant(messages=messages, tenant_id=tenant_id)
    
    # Invoke the graph with the initial state
    print(f"Starting conversation for tenant: {tenant_id}")
    result = await graph.ainvoke(state)
    
    # Print result
    print("\nResult:")
    for message in result.messages:
        print(f"{message.type}: {message.content}")
    
    # You can continue the conversation by adding more messages
    state.messages.append(HumanMessage(content="지금 정보를 바탕으로 할일 목록에 '계획 검토' 항목을 추가해줘"))
    result = await graph.ainvoke(state)
    
    print("\nUpdated Result:")
    for message in result.messages:
        print(f"{message.type}: {message.content}")

if __name__ == "__main__":
    asyncio.run(main()) 