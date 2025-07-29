# Copyright (c) Microsoft. All rights reserved.
# flake8: noqa
import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime

from langfuse import Langfuse
from opentelemetry import trace
from opentelemetry.trace import NoOpTracerProvider, Status, StatusCode
from pydantic import Field

from agents import get_agents
from observability import enable_observability
from reasoning_agent import create_reasoning_compatible_agent

# ✅ Import the enhanced memory system
from enhanced_memory_system import (
    SharedMemoryManager, 
    AgentMemoryContext, 
    RAGIntegrationManager,
    get_shared_memory_manager,
    STATE_MANAGEMENT_EVENT,
    AGENT_INTERACTION_EVENT,
    SYSTEM
)

from semantic_kernel.agents import (
    BooleanResult,
    ChatCompletionAgent,
    GroupChatManager,
    GroupChatOrchestration,
    MessageResult,
    StringResult,
)
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
from semantic_kernel.contents import (
    AuthorRole,
    ChatHistory,
    ChatMessageContent,
    FunctionCallContent,
    FunctionResultContent,
    StreamingChatMessageContent,
)
from semantic_kernel.functions.kernel_arguments import KernelArguments

# ✅ Add logger for events
logger = logging.getLogger(__name__)

# langfuse = Langfuse(
#     secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
#     public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
#     host="https://us.cloud.langfuse.com",
# )

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover

# ✅ Global memory components - will be initialized in main()
shared_memory_manager = None
rag_manager = None

# Global conversation context
current_conversation_id = str(uuid.uuid4())
current_session_metadata = {
    "session_start": datetime.now().isoformat(),
    "user_preferences": {},
    "planning_constraints": {}
}

# Flag to indicate if a new message is being received
is_new_message = True


def enhanced_streaming_agent_response_callback(message: StreamingChatMessageContent, is_final: bool) -> None:
    """Enhanced streaming callback with memory tracking via events."""
    global is_new_message

    # Only create span for final messages to reduce duplicate spans
    if is_final and (message.content or message.items):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("streaming_message_final") as stream_span:
            
            # ✅ Create memory context
            memory_context = AgentMemoryContext(
                agent_id=message.name or "unknown_agent",
                conversation_id=current_conversation_id,
                trace_id=stream_span.get_span_context().trace_id.to_bytes(16, 'big').hex(),
                span_id=stream_span.get_span_context().span_id.to_bytes(8, 'big').hex(),
                session_metadata=current_session_metadata
            )
            
            stream_span.set_attributes({
                "gen_ai.operation.name": "memory_operation",
                "gen_ai.memory.operation_type": "write",
                "gen_ai.memory.source_type": "streaming_response",
                "message.agent_name": message.name,
                "message.content_length": len(message.content) if message.content else 0,
                "message.processing_complete": True,
                "conversation.id": memory_context.conversation_id,
            })

            # Store message in agent's local memory
            if message.content:
                asyncio.create_task(
                    _store_agent_message_memory(message, memory_context)
                )

            # Enhanced function call and result tracking  
            function_calls = []
            function_results = []

            for item in message.items:
                if isinstance(item, FunctionCallContent):
                    function_calls.append({
                        "function": item.name,
                        "arguments": str(item.arguments)[:500],
                    })
                elif isinstance(item, FunctionResultContent):
                    function_results.append({
                        "function": item.name,
                        "result": str(item.result)[:500],
                    })

            if function_calls or function_results:
                asyncio.create_task(
                    _store_function_call_memory(function_calls, function_results, memory_context)
                )

            # ENHANCED: Add input/output visibility for Langfuse
            if message.content:
                stream_span.add_event(
                    "gen_ai.content.completion",
                    {
                        "gen_ai.completion": str(message.content)[:1000],
                        "gen_ai.assistant.message": str(message.content)[:1000],
                        "output.agent": message.name,
                        "output.type": "streaming_final",
                        "output.role": str(message.role),
                        "memory.stored": True,
                    },
                )

            if function_calls:
                stream_span.add_event(
                    "gen_ai.tool.calls",
                    {
                        "tool.calls": json.dumps(function_calls),
                        "tool.call_count": len(function_calls),
                        "agent": message.name,
                    },
                )

            if function_results:
                stream_span.add_event(
                    "gen_ai.tool.results",
                    {
                        "tool.results": json.dumps(function_results),
                        "tool.result_count": len(function_results),
                        "agent": message.name,
                    },
                )

    if is_new_message:
        is_new_message = False
    print(message.content, end="", flush=True)

    for item in message.items:
        if isinstance(item, FunctionCallContent):
            print(f"Calling '{item.name}' with arguments '{item.arguments}'", end="", flush=True)
        if isinstance(item, FunctionResultContent):
            print(f"Result from '{item.name}' is '{item.result}'", end="", flush=True)

    if is_final:
        print()
        is_new_message = True


async def _store_agent_message_memory(message: StreamingChatMessageContent, memory_context: AgentMemoryContext):
    """Store agent message in local memory."""
    try:
        agent_store = shared_memory_manager.get_agent_memory(memory_context.agent_id)
        
        from semantic_kernel.memory.memory_record import MemoryRecord
        import numpy as np
        
        message_record = MemoryRecord(
            is_reference=False,
            external_source_name="agent_response",
            id=f"msg_{uuid.uuid4()}",
            description=f"Response from {memory_context.agent_id}",
            text=message.content,
            additional_metadata=json.dumps({
                "agent_id": memory_context.agent_id,
                "conversation_id": memory_context.conversation_id,
                "message_type": "response",
                "role": str(message.role),
                "timestamp": datetime.now().isoformat()
            }),
            embedding=np.zeros(1536)  # Placeholder embedding
        )
        
        await shared_memory_manager.ensure_agent_collections(memory_context.agent_id)
        await agent_store.upsert_with_telemetry(
            f"{memory_context.agent_id}_working_memory",
            message_record,
            memory_context
        )
        
    except Exception as e:
        logging.error(f"Failed to store agent message memory: {e}")


async def _store_function_call_memory(function_calls: list, function_results: list, memory_context: AgentMemoryContext):
    """Store function call information in shared memory."""
    try:
        from semantic_kernel.memory.memory_record import MemoryRecord
        import numpy as np
        
        if function_calls or function_results:
            function_record = MemoryRecord(
                is_reference=False,
                external_source_name="function_execution",
                id=f"func_{uuid.uuid4()}",
                description=f"Function execution by {memory_context.agent_id}",
                text=json.dumps({
                    "function_calls": function_calls,
                    "function_results": function_results
                }),
                additional_metadata=json.dumps({
                    "agent_id": memory_context.agent_id,
                    "conversation_id": memory_context.conversation_id,
                    "execution_type": "tool_usage",
                    "timestamp": datetime.now().isoformat()
                }),
                embedding=np.zeros(1536)
            )
            
            await shared_memory_manager.shared_store.upsert_with_telemetry(
                "agent_interactions",
                function_record,
                memory_context
            )
            
    except Exception as e:
        logging.error(f"Failed to store function call memory: {e}")


def human_response_function(chat_histoy: ChatHistory) -> ChatMessageContent:
    """Observer function to print the messages from the agents."""
    user_input = input("User: ")
    return ChatMessageContent(role=AuthorRole.USER, content=user_input)


class AgentBaseGroupChatManager(GroupChatManager):
    """A group chat managers that uses a ChatCompletionAgent."""

    agent: ChatCompletionAgent
    logger: logging.Logger = Field(default=logging.getLogger("semantic_kernel.AgentBaseGroupChatManager"))

    def __init__(self, **kwargs):
        """Initialize the base group chat manager with a ChatCompletionAgent."""
        agent = create_reasoning_compatible_agent(
            name="Orchestrator",
            description="The manager of the group chat, responsible for coordinating the agents.",
            instructions=(
                "You are the manager of a group chat for travel planning. "
                "Coordinate the conversation between different travel agents "
                "to help users plan their trips effectively."
            ),
        )

        super().__init__(agent=agent, **kwargs)

    @override
    async def should_request_user_input(self, chat_history: ChatHistory) -> BooleanResult:
        """Determine if the manager should request user input based on the chat history."""
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("user_input_received") as input_span:
            # Capture the number of messages and the role of the last message
            input_span.set_attributes({
                "chat.message_count": len(chat_history.messages),
                "chat.last_message_role": str(chat_history.messages[-1].role) if chat_history.messages else "none",
            })

            if len(chat_history.messages) == 0:
                input_span.set_attribute("chat.last_message_content", "No messages yet.")
                return BooleanResult(
                    result=False,
                    reason="No agents have spoken yet.",
                )

            last_message = chat_history.messages[-1]
            input_span.set_attribute("chat.last_message_content", last_message.content)

            if last_message.role == AuthorRole.USER:
                input_span.set_attribute("chat.input_decision", "No input needed. Last message is from user.")
                return BooleanResult(
                    result=False,
                    reason="User input is not needed if the last message is from the user.",
                )

            messages = chat_history.messages[:]
            messages.append(ChatMessageContent(role=AuthorRole.USER, content="Does the group need further user input?"))

            settings = AzureChatPromptExecutionSettings()
            settings.response_format = BooleanResult

            response = await self.agent.get_response(messages, arguments=KernelArguments(settings=settings))
            result = BooleanResult.model_validate_json(response.message.content)

            input_span.set_attributes({
                "chat.input_decision": "input_needed" if result.result else "continue_conversation",
                "chat.input_reasoning": result.reason,
            })

            return result

    @override
    async def should_terminate(self, chat_history: ChatHistory) -> BooleanResult:
        """Provide concrete implementation for should_terminate."""
       
        should_terminate = await super().should_terminate(chat_history)
        if should_terminate.result:
            return should_terminate

        if len(chat_history.messages) == 0:
            return BooleanResult(
                    result=False,
                    reason="No agents have spoken yet.",
            )

        messages = chat_history.messages[:]
        messages.append(
                ChatMessageContent(
                    role=AuthorRole.USER,
                    content="Has the user's request been satisfied?",
                )
        )

        settings = AzureChatPromptExecutionSettings()
        settings.response_format = BooleanResult

        response = await self.agent.get_response(messages, arguments=KernelArguments(settings=settings))
        return BooleanResult.model_validate_json(response.message.content)
  
    @override
    async def select_next_agent(
        self,
        chat_history: ChatHistory,
        participant_descriptions: dict[str, str],
    ) -> StringResult:
        """✅ ENHANCED: Select next agent with memory events and context propagation."""
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("plan_task") as selection_span:
            # ✅ Create memory context for this selection
            memory_context = AgentMemoryContext(
                agent_id=self.agent.name,
                conversation_id=current_conversation_id,
                trace_id=selection_span.get_span_context().trace_id.to_bytes(16, 'big').hex(),
                span_id=selection_span.get_span_context().span_id.to_bytes(8, 'big').hex(),
                session_metadata=current_session_metadata
            )
            
            selection_span.set_attributes({
                "gen_ai.planning.type": "plan_task",
                "gen_ai.planning.complexity": "moderate",
                "gen_ai.planning.stage": "agent_coordination",
                "gen_ai.operation.name": "plan_task",
                "agent.selection.total_participants": len(participant_descriptions),
                "agent.selection.available_agents": list(participant_descriptions.keys()),
                "agent.selection.conversation_length": len(chat_history.messages),
                "conversation.id": memory_context.conversation_id,
            })

            # ✅ Convert chat history to serializable format for events
            chat_history_data = []
            for msg in chat_history.messages[-5:]:  # Last 5 messages
                chat_history_data.append({
                    "role": str(msg.role),
                    "content": msg.content[:200] if msg.content else "",  # Truncate
                    "agent": getattr(msg, 'name', 'unknown')
                })

            messages = chat_history.messages[:]
            messages.append(
                ChatMessageContent(
                    role=AuthorRole.USER,
                    content=(
                        "Who should speak next based on the conversation? Pick ONE agent from the participants:\n"
                        + "\n".join([f"- {k}: {v}" for k, v in participant_descriptions.items()])
                        + "\n\nYou must respond with a JSON object containing exactly the agent name from the list above.\n"
                        + 'Format: {"result": "agent_name", "reason": "explanation"}\n'
                        + f"Valid agent names: {list(participant_descriptions.keys())}\n"
                        + "The 'result' field must contain ONLY the agent name, nothing else."
                    ),
                )
            )

            settings = AzureChatPromptExecutionSettings()
            settings.response_format = StringResult

            # ✅ THIS IS WHERE LLM MAKES THE DECISION - IN LLM SPAN
            response = await self.agent.get_response(messages, arguments=KernelArguments(settings=settings))
            result = StringResult.model_validate_json(response.message.content)

            # Enhanced validation and error handling
            selected_agent = result.result.strip()

            # Check if the result contains extra text and try to extract the agent name
            if selected_agent not in participant_descriptions:
                # Try to find a valid agent name within the response
                for agent_name in participant_descriptions.keys():
                    if agent_name in selected_agent:
                        selection_span.set_attribute(
                            "agent.selection.corrected", f"from '{selected_agent}' to '{agent_name}'"
                        )
                        result.result = agent_name
                        selected_agent = agent_name
                        break

                # If still not found, raise the error with better debugging info
                if selected_agent not in participant_descriptions:
                    selection_span.set_attribute("agent.selection.error", "invalid_agent_selected")
                    selection_span.set_attribute("agent.selection.raw_response", response.message.content[:500])
                    raise ValueError(
                        f"Selected agent '{selected_agent}' is not in the list of participants: "
                        f"{list(participant_descriptions.keys())}. Raw response: {response.message.content[:200]}"
                    )

            # ✅ PROPAGATE CONTEXT AND EMIT EVENTS
            context_data = {
                "selection_reason": result.reason,
                "conversation_state": "agent_transition",
                "last_message_summary": chat_history.messages[-1].content[:200] if chat_history.messages else "",
                "participant_count": len(participant_descriptions),
                "selection_timestamp": datetime.now().isoformat()
            }
            
            # ✅ PROPAGATE CONTEXT - EVENTS EMITTED IN LLM SPAN
            await shared_memory_manager.propagate_context_to_agent(
                source_agent=self.agent.name,
                target_agent=selected_agent,
                context_data=context_data,
                memory_context=memory_context,
                chat_history=chat_history_data  # ✅ Pass chat history for context propagation event
            )

            selection_span.set_attributes({
                "agent.selection.selected": selected_agent,
                "agent.selection.reasoning": result.reason,
                "planning.decision": "agent_selected",
            })

            self.logger.info(
                "Selected agent: %s, Reason: %s",
                selected_agent,
                result.reason,
                extra={
                    "gen_ai.interaction.source_agent_id": self.agent.name,
                    "gen_ai.interaction.target_agent_id": selected_agent,
                    "gen_ai.interaction.message_type": "request",
                    "gen_ai.interaction.pattern": "group_chat",
                },
            )

            return result

    @override
    async def filter_results(
        self,
        chat_history: ChatHistory,
    ) -> MessageResult:
        """Provide concrete implementation for filtering results."""
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("memory_operation_summary") as memory_span:
            memory_span.set_attributes({
                "gen_ai.operation.name": "memory_operation",
                "gen_ai.memory.operation_type": "write",
                "gen_ai.memory.source_type": "conversation_summary",
                "gen_ai.memory.memory_type": "working",
                "gen_ai.memory.size_bytes": len(str(chat_history.messages)) * 2,
                "conversation.message_count": len(chat_history.messages),
            })

            messages = chat_history.messages[:]
            messages.append(ChatMessageContent(role=AuthorRole.USER, content="Please summarize the conversation."))

            settings = AzureChatPromptExecutionSettings()
            settings.response_format = StringResult

            response = await self.agent.get_response(messages, arguments=KernelArguments(settings=settings))
            string_with_reason = StringResult.model_validate_json(response.message.content)

            memory_span.set_attributes({
                "memory.summary_length": len(string_with_reason.result),
                "memory.summary_generated": True,
                "memory.result_processed": True,
            })

            return MessageResult(
                result=ChatMessageContent(
                    role=AuthorRole.ASSISTANT,
                    content=string_with_reason.result,
                ),
                reason=string_with_reason.reason,
            )


@enable_observability
async def main():
    """Main function to run the agents with comprehensive telemetry."""
    global shared_memory_manager, rag_manager
    
    # ✅ Initialize memory components with async
    shared_memory_manager = get_shared_memory_manager()
    rag_manager = RAGIntegrationManager(shared_memory_manager)
    
    tracer = trace.get_tracer(__name__)

    # Create comprehensive session context
    with tracer.start_as_current_span("travel_planning_session") as session_span:
        session_span.set_attributes({
            "user.id": "demo_user_enhanced",
            "conversation.id": current_conversation_id,
            "session.type": "enhanced_multi_agent_demo",
            # Langfuse-specific attributes
            "langfuse.trace.type": "multi_agent_orchestration",
            "langfuse.version": "1.0",
            "langfuse.session_type": "group_chat",
        })

        # 1. Create a Group Chat orchestration with multiple agents
        agents: dict[str, ChatCompletionAgent] = get_agents()

        group_chat_orchestration = GroupChatOrchestration(
            members=[
                agents["planner"],
                agents["flight_agent"],
                agents["hotel_agent"],
            ],
            manager=AgentBaseGroupChatManager(max_rounds=20, human_response_function=human_response_function),
            streaming_agent_response_callback=enhanced_streaming_agent_response_callback,
        )

        # 2. Comprehensive task execution with telemetry
        task_description = (
            "Plan a trip to Bali for 5 days including flights, hotels, and "
            "activities for a vegetarian family of 4 members. The family lives in Seattle, WA, USA. "
            "Their vacation starts on July 30th 2025. They have a strict budget of $5000 for the trip. "
            "Please think through this step-by-step: first assess the budget allocation, then find suitable flights, "
            "select appropriate vegetarian-friendly accommodations, and plan activities. "
            "Show your reasoning process and provide a detailed plan. When the plan is ready, make the flight and hotel bookings."
        )
        session_span.add_event(
            "gen_ai.content.prompt",
            {
                "gen_ai.prompt": task_description,
                "gen_ai.user.message": task_description,
                "input.type": "user_request",
                "input.source": "main_function",
                "input.task_type": "multi_agent_planning",
                "input.constraints": json.dumps({
                    "budget": 5000,
                    "duration_days": 5,
                    "travelers": 4,
                    "dietary": "vegetarian",
                    "destination": "Bali",
                    "origin": "Seattle, WA",
                }),
                "langfuse.input": task_description,  # Langfuse-specific
            },
        )

        with tracer.start_as_current_span("comprehensive_task_execution") as task_span:
            task_span.set_attributes({
                "gen_ai.operation.name": "execute_task",
                "gen_ai.system": "semantic_kernel_multi_agent",
                "gen_ai.task.id": "bali_trip_planning_001",
                "gen_ai.task.description": task_description,
                "gen_ai.task.status": "in_progress",
                "gen_ai.task.expected_output": "Complete travel plan with bookings",
                "gen_ai.task.constraints": ["budget:5000", "duration:5_days", "travelers:4", "dietary:vegetarian"],
                "gen_ai.task.assigned_agents": ["planner", "flight_agent", "hotel_agent"],
                "task.destination": "Bali",
                "task.duration_days": 5,
                "task.travelers": 4,
                "task.budget": 5000,
                "task.expected_tool_usage": ["flight_search", "hotel_search", "planning_tools"],
            })

            # Create a runtime and start it
            runtime = InProcessRuntime(tracer_provider=NoOpTracerProvider())
            runtime.start()

            # Invoke the orchestration with a task and the runtime
            orchestration_result = await group_chat_orchestration.invoke(
                task=task_description,
                runtime=runtime,
            )

            # 3. Wait for the results with memory operation tracking
            value = await orchestration_result.get()
            print(f"\n✅ Final Result:\n{value}")

            task_span.set_attribute("gen_ai.task.status", "completed")
            session_span.add_event(
                "gen_ai.content.completion",
                {
                    "gen_ai.completion": str(value)[:2000],  # Truncate if too long
                    "gen_ai.assistant.message": str(value)[:2000],
                    "output.type": "final_travel_plan",
                    "output.source": "multi_agent_orchestration",
                    "output.success": True,
                    "langfuse.output": str(value)[:2000],  # Langfuse-specific
                },
            )

        # ✅ Print final memory summary
        print("\n" + "="*60)
        print("📊 STATE MANAGEMENT DATA SUMMARY")
        print("="*60)
        
        all_state_data = shared_memory_manager.get_all_state_management_data()
        print(f"Total state management events: {len(all_state_data)}")
        
        for event_type in ["context_propagation", "local_agent_memory", "rag_knowledge"]:
            type_events = [e for e in all_state_data if e.get("type") == event_type]
            print(f"- {event_type}: {len(type_events)} events")
        
        print("="*60)

        # 4. Stop the runtime after the invocation is complete
        await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())