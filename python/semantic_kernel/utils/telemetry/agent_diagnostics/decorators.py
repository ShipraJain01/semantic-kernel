# Copyright (c) Microsoft. All rights reserved.

import functools
import json
import logging
from collections.abc import AsyncIterable, AsyncGenerator, Awaitable, Callable
from typing import ParamSpec, TypeVar, cast, Any
import inspect

from opentelemetry.trace import get_tracer

from semantic_kernel.utils.feature_stage_decorator import experimental
from semantic_kernel.utils.telemetry.agent_diagnostics import gen_ai_attributes
from semantic_kernel.utils.telemetry.model_diagnostics.model_diagnostics_settings import ModelDiagnosticSettings

P = ParamSpec("P")
T = TypeVar("T")

# Creates a tracer from the global tracer provider
tracer = get_tracer(__name__)

logger = logging.getLogger(__name__)

# Reuse settings from model diagnostics
MODEL_DIAGNOSTICS_SETTINGS = ModelDiagnosticSettings()


def are_sensitive_events_enabled() -> bool:
    """Check if sensitive events are enabled."""
    return MODEL_DIAGNOSTICS_SETTINGS.enable_otel_diagnostics_sensitive


@experimental
def trace_agent_invocation(invoke_func: Callable[P, AsyncIterable[T]]) -> Callable[P, AsyncIterable[T]]:
    """Decorator to trace agent invocation."""
    OPERATION_NAME = "invoke_agent"

    @functools.wraps(invoke_func)
    async def wrapper_decorator(*args: P.args, **kwargs: P.kwargs) -> AsyncIterable[T]:
        from semantic_kernel.agents.agent import Agent
        from semantic_kernel.contents.chat_message_content import ChatMessageContent
        from semantic_kernel.contents.function_call_content import FunctionCallContent
        from semantic_kernel.contents.function_result_content import FunctionResultContent

        agent = cast(Agent, args[0])
        
        # Extract messages from args/kwargs
        messages_arg = None
        if len(args) > 1:
            messages_arg = args[1]
        elif "messages" in kwargs:
            messages_arg = kwargs["messages"]
            
        with tracer.start_as_current_span(f"{OPERATION_NAME} {agent.name}") as span:
            # Set basic attributes
            span.set_attributes({
                gen_ai_attributes.OPERATION: OPERATION_NAME,
                gen_ai_attributes.AGENT_ID: agent.id,
                gen_ai_attributes.AGENT_NAME: agent.name,
            })

            if agent.description:
                span.set_attribute(gen_ai_attributes.AGENT_DESCRIPTION, agent.description)

            # Extract registered agents and tools
            registered_agent_ids = _get_registered_agent_ids(agent)
            span.set_attribute(gen_ai_attributes.AGENT_CHILD_AGENTS, json.dumps(registered_agent_ids))

            # Extract tool definitions from kernel if available
            kernel = kwargs.get("kernel")
            tool_definitions = _get_tool_definitions(agent, kernel)
            if tool_definitions:
                span.set_attribute(gen_ai_attributes.TOOL_DEFINITIONS, json.dumps(tool_definitions))

            # Process input messages and set invocation input
            invocation_input = _process_invocation_input(messages_arg, agent)
            if invocation_input:
                span.set_attribute(gen_ai_attributes.AGENT_INVOCATION_INPUT, json.dumps(invocation_input))

            # Collect output messages
            output_messages = []
            tool_call_results = []
            final_response = None
            
            try:
                async for response in invoke_func(*args, **kwargs):
                    # Store the final response
                    final_response = response
                    
                    # Process response if it contains message content
                    if hasattr(response, "message") and isinstance(response.message, ChatMessageContent):
                        message = response.message
                        output_messages.append(message)
                        
                        # Extract tool calls and results from message
                        if hasattr(message, "items") and message.items:
                            for item in message.items:
                                if isinstance(item, FunctionCallContent):
                                    tool_call_results.append({
                                        "type": "tool_call",
                                        "id": getattr(item, "id", ""),
                                        "name": item.function_name,
                                        "arguments": getattr(item, "arguments", {})
                                    })
                                elif isinstance(item, FunctionResultContent):
                                    tool_call_results.append({
                                        "type": "tool_call_response", 
                                        "id": getattr(item, "id", ""),
                                        "result": str(item.result) if hasattr(item, "result") else ""
                                    })
                    
                    yield response
                
                # Process final output message and set invocation output
                invocation_output = _process_invocation_output(final_response, output_messages, tool_call_results)
                if invocation_output:
                    span.set_attribute(gen_ai_attributes.AGENT_INVOCATION_OUTPUT, json.dumps(invocation_output))
                    
            except Exception as e:
                logger.error(f"Error in agent invocation: {e}")
                raise

    # Mark the wrapper decorator as an agent diagnostics decorator
    wrapper_decorator.__agent_diagnostics__ = True  # type: ignore

    return wrapper_decorator


@experimental
def trace_agent_invocation_streaming(invoke_func: Callable[P, AsyncGenerator[T, Any]]) -> Callable[P, AsyncGenerator[T, Any]]:
    """Decorator to trace streaming agent invocation."""
    OPERATION_NAME = "invoke_agent"

    @functools.wraps(invoke_func)
    async def wrapper_decorator(*args: P.args, **kwargs: P.kwargs) -> AsyncGenerator[T, Any]:
        from semantic_kernel.agents.agent import Agent
        from semantic_kernel.contents.chat_message_content import ChatMessageContent
        from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent

        agent = cast(Agent, args[0])
        
        # Extract messages from args/kwargs
        messages_arg = None
        if len(args) > 1:
            messages_arg = args[1]
        elif "messages" in kwargs:
            messages_arg = kwargs["messages"]

        # Collect all streaming responses grouped by choice index
        all_streaming_responses: dict[int, list[T]] = {}
        
        with tracer.start_as_current_span(f"{OPERATION_NAME} {agent.name}") as span:
            try:
                # Set basic attributes
                span.set_attributes({
                    gen_ai_attributes.OPERATION: OPERATION_NAME,
                    gen_ai_attributes.AGENT_ID: agent.id,
                    gen_ai_attributes.AGENT_NAME: agent.name,
                })

                if agent.description:
                    span.set_attribute(gen_ai_attributes.AGENT_DESCRIPTION, agent.description)

                # Extract registered agents and tools
                registered_agent_ids = _get_registered_agent_ids(agent)
                span.set_attribute(gen_ai_attributes.AGENT_CHILD_AGENTS, json.dumps(registered_agent_ids))

                # Extract tool definitions from kernel if available
                kernel = kwargs.get("kernel")
                tool_definitions = _get_tool_definitions(agent, kernel)
                if tool_definitions:
                    span.set_attribute(gen_ai_attributes.TOOL_DEFINITIONS, json.dumps(tool_definitions))

                # Process input messages and set invocation input
                invocation_input = _process_invocation_input(messages_arg, agent)
                if invocation_input:
                    span.set_attribute(gen_ai_attributes.AGENT_INVOCATION_INPUT, json.dumps(invocation_input))

                # Stream and collect responses
                async for streaming_response in invoke_func(*args, **kwargs):
                    # Determine choice index for grouping responses
                    choice_index = 0
                    if hasattr(streaming_response, "choice_index"):
                        choice_index = streaming_response.choice_index
                    elif hasattr(streaming_response, "message") and hasattr(streaming_response.message, "choice_index"):
                        choice_index = streaming_response.message.choice_index
                    
                    # Group streaming responses by choice index
                    if choice_index not in all_streaming_responses:
                        all_streaming_responses[choice_index] = []
                    all_streaming_responses[choice_index].append(streaming_response)
                    
                    yield streaming_response

                # Process collected streaming responses and set invocation output
                _set_agent_invocation_response(span, all_streaming_responses, messages_arg)
                
            except Exception as e:
                logger.error(f"Error in streaming agent invocation: {e}")
                raise

    # Mark the wrapper decorator as an agent diagnostics decorator
    wrapper_decorator.__agent_diagnostics__ = True  # type: ignore

    return wrapper_decorator


def _set_agent_invocation_response(
    current_span,
    streaming_responses: dict[int, list[Any]],
    messages_arg: Any,
) -> None:
    """Set the agent invocation response for a given span."""
    from semantic_kernel.contents.chat_message_content import ChatMessageContent
    from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent
    from semantic_kernel.contents.utils.author_role import AuthorRole
    
    try:
        # Try to get the final assistant message from chat history first (most accurate)
        final_assistant_message = None
        chat_history = None
        
        # Extract chat history from messages_arg
        if hasattr(messages_arg, "messages"):  # ChatHistory object
            chat_history = messages_arg
        elif isinstance(messages_arg, list):
            chat_history = messages_arg
        
        # Find the last assistant message in chat history
        if chat_history:
            if hasattr(chat_history, "messages"):
                # ChatHistory object
                for message in reversed(chat_history.messages):
                    if isinstance(message, ChatMessageContent) and message.role == AuthorRole.ASSISTANT:
                        final_assistant_message = message
                        break
            elif isinstance(chat_history, list):
                # List of messages
                for message in reversed(chat_history):
                    if isinstance(message, ChatMessageContent) and message.role == AuthorRole.ASSISTANT:
                        final_assistant_message = message
                        break
        
        invocation_output = []
        
        if final_assistant_message:
            # Use the final assistant message from chat history (most complete)
            parts = []
            
            # Add text content
            if hasattr(final_assistant_message, "content") and final_assistant_message.content:
                parts.append({"type": "text", "content": final_assistant_message.content})
            
            # Extract tool calls and results from message
            if hasattr(final_assistant_message, "items") and final_assistant_message.items:
                for item in final_assistant_message.items:
                    if hasattr(item, "function_name"):  # FunctionCallContent
                        parts.append({
                            "type": "tool_call",
                            "id": getattr(item, "id", ""),
                            "name": item.function_name,
                            "arguments": getattr(item, "arguments", {})
                        })
                    elif hasattr(item, "result"):  # FunctionResultContent
                        parts.append({
                            "type": "tool_call_response",
                            "id": getattr(item, "id", ""),
                            "result": str(item.result)
                        })
            
            invocation_output.append({
                "role": str(final_assistant_message.role).lower(),
                "body": parts,
                "finish_reason": getattr(final_assistant_message, "finish_reason", "stop")
            })
        else:
            # Fall back to processing streaming responses
            for choice_index, responses in streaming_responses.items():
                if not responses:
                    continue
                
                # Aggregate streaming content
                aggregated_content = ""
                finish_reason = "stop"
                message_parts = []
                
                for response in responses:
                    if hasattr(response, "message") and isinstance(response.message, (ChatMessageContent, StreamingChatMessageContent)):
                        message = response.message
                        
                        # Aggregate text content
                        if hasattr(message, "content") and message.content:
                            aggregated_content += message.content
                        
                        # Extract finish reason from last message with one
                        if hasattr(message, "finish_reason") and message.finish_reason:
                            finish_reason = str(message.finish_reason)
                        
                        # Extract message parts (tool calls, etc.)
                        if hasattr(message, "items") and message.items:
                            for item in message.items:
                                if hasattr(item, "function_name"):  # FunctionCallContent
                                    message_parts.append({
                                        "type": "tool_call",
                                        "id": getattr(item, "id", ""),
                                        "name": item.function_name,
                                        "arguments": getattr(item, "arguments", {})
                                    })
                                elif hasattr(item, "result"):  # FunctionResultContent
                                    message_parts.append({
                                        "type": "tool_call_response",
                                        "id": getattr(item, "id", ""),
                                        "result": str(item.result)
                                    })
                
                # Create parts list
                parts = []
                if aggregated_content:
                    parts.append({"type": "text", "content": aggregated_content})
                parts.extend(message_parts)
                
                if parts:  # Only add if we have content
                    response_data = {
                        "role": "assistant",
                        "body": parts,
                        "finish_reason": finish_reason
                    }
                    if len(streaming_responses) > 1:
                        response_data["choice_index"] = choice_index
                    invocation_output.append(response_data)

        # Set the invocation output attribute
        if invocation_output:
            current_span.set_attribute(gen_ai_attributes.AGENT_INVOCATION_OUTPUT, json.dumps(invocation_output))
            
        # Log completion event if sensitive events are enabled
        if are_sensitive_events_enabled():
            for output in invocation_output:
                logger.info(
                    json.dumps(output),
                    extra={
                        gen_ai_attributes.EVENT_NAME: gen_ai_attributes.CHOICE,
                        gen_ai_attributes.SYSTEM: "semantic-kernel",
                    },
                )
                
    except Exception as e:
        logger.warning(f"Failed to set agent invocation response: {e}")


def _get_registered_agent_ids(agent) -> dict | str:
    """Extract registered sub-agent information if the agent manages other agents."""
    child_agents = {}
    
    # Check for agents collection (orchestrators)
    if hasattr(agent, "agents") and agent.agents:
        for sub_agent in agent.agents:
            agent_id = None
            agent_name = "unknown"
            agent_type = "specialist"
            
            if hasattr(sub_agent, "id"):
                agent_id = sub_agent.id
            elif hasattr(sub_agent, "name"):
                agent_id = sub_agent.name
            
            if agent_id:
                if hasattr(sub_agent, "name"):
                    agent_name = sub_agent.name
                
                # Try to determine agent type from class name or attributes
                if hasattr(sub_agent, "__class__"):
                    class_name = sub_agent.__class__.__name__.lower()
                    if "orchestrator" in class_name or "coordinator" in class_name:
                        agent_type = "coordinator"
                    elif "chat" in class_name or "completion" in class_name:
                        agent_type = "chat_completion"
                    elif "function" in class_name:
                        agent_type = "function_calling"
                
                child_agents[agent_id] = {
                    "name": agent_name,
                    "type": agent_type
                }
    
    # Check for members (group chat)
    if hasattr(agent, "members") and agent.members:
        for member in agent.members:
            agent_id = None
            agent_name = "unknown"
            agent_type = "specialist"
            
            if hasattr(member, "id"):
                agent_id = member.id
            elif hasattr(member, "name"):
                agent_id = member.name
            
            if agent_id:
                if hasattr(member, "name"):
                    agent_name = member.name
                
                # Try to determine agent type
                if hasattr(member, "__class__"):
                    class_name = member.__class__.__name__.lower()
                    if "orchestrator" in class_name or "coordinator" in class_name:
                        agent_type = "coordinator"
                    elif "chat" in class_name or "completion" in class_name:
                        agent_type = "chat_completion"
                    elif "function" in class_name:
                        agent_type = "function_calling"
                
                child_agents[agent_id] = {
                    "name": agent_name,
                    "type": agent_type
                }
    
    # Check for participants (another common attribute)
    if hasattr(agent, "participants") and agent.participants:
        for participant in agent.participants:
            agent_id = None
            agent_name = "unknown"
            agent_type = "specialist"
            
            if hasattr(participant, "id"):
                agent_id = participant.id
            elif hasattr(participant, "name"):
                agent_id = participant.name
            elif isinstance(participant, str):
                agent_id = participant
                agent_name = participant
            
            if agent_id:
                if hasattr(participant, "name"):
                    agent_name = participant.name
                
                # Try to determine agent type
                if hasattr(participant, "__class__"):
                    class_name = participant.__class__.__name__.lower()
                    if "orchestrator" in class_name or "coordinator" in class_name:
                        agent_type = "coordinator"
                    elif "chat" in class_name or "completion" in class_name:
                        agent_type = "chat_completion"
                    elif "function" in class_name:
                        agent_type = "function_calling"
                
                child_agents[agent_id] = {
                    "name": agent_name,
                    "type": agent_type
                }
    
    # If no child agents found, return "NA"
    return child_agents if child_agents else "NA"


def _get_tool_definitions(agent, kernel) -> list[dict]:
    """Extract tool definitions from the agent and kernel in OpenAI function format."""
    tool_definitions = []
    
    try:
        # Check kernel functions if available - this is the primary source
        if kernel and hasattr(kernel, "get_full_list_of_function_metadata"):
            try:
                functions = kernel.get_full_list_of_function_metadata()
                for func in functions:
                    tool_def = _create_openai_function_definition(func)
                    if tool_def:
                        tool_definitions.append(tool_def)
            except Exception as e:
                logger.warning(f"Failed to extract kernel functions: {e}")
        
        # Check agent's kernel if it has one
        if hasattr(agent, "kernel") and agent.kernel:
            try:
                functions = agent.kernel.get_full_list_of_function_metadata()
                for func in functions:
                    tool_def = _create_openai_function_definition(func)
                    if tool_def:
                        # Avoid duplicates by checking if already exists
                        if not any(existing["name"] == tool_def["name"] for existing in tool_definitions):
                            tool_definitions.append(tool_def)
            except Exception as e:
                logger.warning(f"Failed to extract agent kernel functions: {e}")
        
        # Check for direct plugin access on agent
        if hasattr(agent, "plugins") and agent.plugins:
            for plugin_name, plugin in agent.plugins.items():
                if hasattr(plugin, "functions") and plugin.functions:
                    for func_name, func in plugin.functions.items():
                        tool_def = _create_openai_function_definition_from_func(func, func_name, plugin_name)
                        if tool_def:
                            # Avoid duplicates by checking if already exists
                            if not any(existing["name"] == tool_def["name"] for existing in tool_definitions):
                                tool_definitions.append(tool_def)
        
        # Check if agent has a service with plugins
        if hasattr(agent, "service") and hasattr(agent.service, "kernel"):
            try:
                functions = agent.service.kernel.get_full_list_of_function_metadata()
                for func in functions:
                    tool_def = _create_openai_function_definition(func)
                    if tool_def:
                        # Avoid duplicates by checking if already exists
                        if not any(existing["name"] == tool_def["name"] for existing in tool_definitions):
                            tool_definitions.append(tool_def)
            except Exception as e:
                logger.warning(f"Failed to extract service kernel functions: {e}")
                
    except Exception as e:
        logger.warning(f"Error extracting tool definitions: {e}")
    
    return tool_definitions


def _create_openai_function_definition(func_metadata) -> dict | None:
    """Create OpenAI function definition from function metadata."""
    try:
        if not hasattr(func_metadata, "name"):
            return None
            
        function_def = {
            "type": "function",
            "name": getattr(func_metadata, "name", "unknown"),
            "description": getattr(func_metadata, "description", ""),
        }
        
        # Extract parameters schema
        parameters_schema = _extract_parameters_schema_from_metadata(func_metadata)
        if parameters_schema:
            function_def["parameters"] = parameters_schema
            
        return function_def
        
    except Exception as e:
        logger.warning(f"Failed to create function definition from metadata: {e}")
        return None


def _create_openai_function_definition_from_func(func, func_name: str, plugin_name: str = "") -> dict | None:
    """Create OpenAI function definition from a function object."""
    try:
        function_def = {
            "type": "function", 
            "name": func_name,
            "description": getattr(func, "description", "") or _extract_docstring_description(func),
        }
        
        # Extract parameters schema from function
        parameters_schema = _extract_parameters_schema_from_function(func)
        if parameters_schema:
            function_def["parameters"] = parameters_schema
            
        return function_def
        
    except Exception as e:
        logger.warning(f"Failed to create function definition from function object: {e}")
        return None


def _extract_parameters_schema_from_metadata(func_metadata) -> dict | None:
    """Extract parameters schema from function metadata."""
    try:
        if not hasattr(func_metadata, "parameters") or not func_metadata.parameters:
            return None
            
        properties = {}
        required = []
        
        for param in func_metadata.parameters:
            param_name = getattr(param, "name", "")
            if not param_name:
                continue
                
            param_schema = {
                "type": _map_type_to_json_schema(getattr(param, "type_name", "string")),
                "description": getattr(param, "description", "")
            }
            
            # Handle enum values if available
            if hasattr(param, "enum") and param.enum:
                param_schema["enum"] = param.enum
            
            properties[param_name] = param_schema
            
            # Determine if parameter is required
            if getattr(param, "is_required", True) and not getattr(param, "default_value", None):
                required.append(param_name)
        
        if not properties:
            return None
            
        schema = {
            "type": "object",
            "properties": properties
        }
        
        if required:
            schema["required"] = required
            
        return schema
        
    except Exception as e:
        logger.warning(f"Failed to extract parameters schema from metadata: {e}")
        return None


def _extract_parameters_schema_from_function(func) -> dict | None:
    """Extract parameters schema from a function object using inspection."""
    try:
        # Try to get function signature
        if not callable(func):
            return None
            
        sig = inspect.signature(func)
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            # Skip 'self' and 'cls' parameters
            if param_name in ["self", "cls"]:
                continue
                
            param_schema = {
                "type": _map_python_type_to_json_schema(param.annotation),
                "description": ""  # Could extract from docstring if needed
            }
            
            properties[param_name] = param_schema
            
            # Determine if parameter is required (no default value)
            if param.default == param.empty:
                required.append(param_name)
        
        if not properties:
            return None
            
        schema = {
            "type": "object", 
            "properties": properties
        }
        
        if required:
            schema["required"] = required
            
        return schema
        
    except Exception as e:
        logger.warning(f"Failed to extract parameters schema from function: {e}")
        return None


def _map_type_to_json_schema(type_name: str) -> str:
    """Map semantic kernel type names to JSON schema types."""
    type_mapping = {
        "str": "string",
        "string": "string", 
        "int": "integer",
        "integer": "integer",
        "float": "number",
        "number": "number",
        "bool": "boolean",
        "boolean": "boolean",
        "list": "array",
        "array": "array",
        "dict": "object",
        "object": "object",
    }
    
    return type_mapping.get(type_name.lower(), "string")


def _map_python_type_to_json_schema(annotation) -> str:
    """Map Python type annotations to JSON schema types."""
    if annotation == inspect.Parameter.empty:
        return "string"
        
    # Handle basic types
    if annotation == str:
        return "string"
    elif annotation == int:
        return "integer" 
    elif annotation == float:
        return "number"
    elif annotation == bool:
        return "boolean"
    elif annotation == list:
        return "array"
    elif annotation == dict:
        return "object"
    
    # Handle type strings
    if isinstance(annotation, str):
        return _map_type_to_json_schema(annotation)
    
    # Handle typing module types
    if hasattr(annotation, "__origin__"):
        origin = annotation.__origin__
        if origin == list:
            return "array"
        elif origin == dict:
            return "object"
        elif origin == tuple:
            return "array"
        elif origin == set:
            return "array"
    
    # Default to string for unknown types
    return "string"


def _extract_docstring_description(func) -> str:
    """Extract description from function docstring."""
    try:
        if hasattr(func, "__doc__") and func.__doc__:
            # Extract first line of docstring as description
            lines = func.__doc__.strip().split('\n')
            return lines[0].strip() if lines else ""
        return ""
    except Exception:
        return ""


def _extract_function_parameters(func) -> dict:
    """Extract parameters from a function object."""
    try:
        if hasattr(func, "parameters"):
            return func.parameters
        elif hasattr(func, "__annotations__"):
            return {name: str(annotation) for name, annotation in func.__annotations__.items() if name != 'return'}
        elif hasattr(func, "metadata") and hasattr(func.metadata, "parameters"):
            return {param.name: param.type_name for param in func.metadata.parameters}
        return {}
    except Exception:
        return {}


def _extract_function_metadata_parameters(func_metadata) -> dict:
    """Extract parameters from function metadata."""
    try:
        if hasattr(func_metadata, "parameters"):
            return {param.name: param.type_name for param in func_metadata.parameters}
        return {}
    except Exception:
        return {}


def _process_invocation_input(messages_arg, agent) -> list[dict]:
    """Process input messages into structured format."""
    from semantic_kernel.contents.chat_message_content import ChatMessageContent
    from semantic_kernel.contents.utils.author_role import AuthorRole
    
    input_messages = []
    
    # Add system instructions if available
    if hasattr(agent, "instructions") and agent.instructions:
        input_messages.append({
            "role": "system",
            "body": [{"type": "text", "content": agent.instructions}]
        })
    
    # Process input messages
    messages = []
    if isinstance(messages_arg, str):
        messages = [ChatMessageContent(role=AuthorRole.USER, content=messages_arg)]
    elif isinstance(messages_arg, ChatMessageContent):
        messages = [messages_arg]
    elif isinstance(messages_arg, list):
        messages = messages_arg
    elif hasattr(messages_arg, "messages"):  # ChatHistory object
        messages = messages_arg.messages
    
    for message in messages:
        if isinstance(message, ChatMessageContent):
            parts = []
            
            # Add text content
            if hasattr(message, "content") and message.content:
                parts.append({"type": "text", "content": message.content})
            
            # Add function call content
            if hasattr(message, "items") and message.items:
                for item in message.items:
                    if hasattr(item, "function_name"):  # FunctionCallContent
                        parts.append({
                            "type": "tool_call",
                            "id": getattr(item, "id", ""),
                            "name": item.function_name,
                            "arguments": getattr(item, "arguments", {})
                        })
                    elif hasattr(item, "result"):  # FunctionResultContent
                        parts.append({
                            "type": "tool_call_response",
                            "id": getattr(item, "id", ""),
                            "result": str(item.result)
                        })
            
            if parts:
                input_messages.append({
                    "role": str(message.role).lower(),
                    "body": parts
                })
    
    return input_messages


def _process_invocation_output(final_response, output_messages, tool_call_results) -> list[dict]:
    """Process output messages into structured format."""
    output_data = []
    
    # Process the final response first
    if final_response and hasattr(final_response, "message"):
        message = final_response.message
        if hasattr(message, "role") and hasattr(message, "content"):
            parts = []
            
            # Add text content
            if message.content:
                parts.append({"type": "text", "content": message.content})
            
            # Add tool calls and results from the message itself
            if hasattr(message, "items") and message.items:
                for item in message.items:
                    if hasattr(item, "function_name"):  # FunctionCallContent
                        parts.append({
                            "type": "tool_call",
                            "id": getattr(item, "id", ""),
                            "name": item.function_name,
                            "arguments": getattr(item, "arguments", {})
                        })
                    elif hasattr(item, "result"):  # FunctionResultContent
                        parts.append({
                            "type": "tool_call_response",
                            "id": getattr(item, "id", ""),
                            "result": str(item.result)
                        })
            
            # Add collected tool call results
            for tool_result in tool_call_results:
                parts.append(tool_result)
            
            msg_data = {
                "role": str(message.role).lower(),
                "body": parts
            }
            
            # Add finish reason if available
            if hasattr(message, "finish_reason") and message.finish_reason:
                msg_data["finish_reason"] = str(message.finish_reason)
            
            output_data.append(msg_data)
    
    # If no final response, process collected output messages
    elif output_messages:
        for message in output_messages:
            if hasattr(message, "role") and hasattr(message, "content"):
                parts = []
                
                # Add text content
                if message.content:
                    parts.append({"type": "text", "content": message.content})
                
                # Add tool calls and results
                if hasattr(message, "items") and message.items:
                    for item in message.items:
                        if hasattr(item, "function_name"):  # FunctionCallContent
                            parts.append({
                                "type": "tool_call",
                                "id": getattr(item, "id", ""),
                                "name": item.function_name,
                                "arguments": getattr(item, "arguments", {})
                            })
                        elif hasattr(item, "result"):  # FunctionResultContent
                            parts.append({
                                "type": "tool_call_response",
                                "id": getattr(item, "id", ""),
                                "result": str(item.result)
                            })
                
                msg_data = {
                    "role": str(message.role).lower(),
                    "body": parts
                }
                
                # Add finish reason if available
                if hasattr(message, "finish_reason") and message.finish_reason:
                    msg_data["finish_reason"] = str(message.finish_reason)
                
                output_data.append(msg_data)
    
    return output_data


@experimental
def trace_agent_get_response(get_response_func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    """Decorator to trace agent get_response method."""
    OPERATION_NAME = "invoke_agent"

    @functools.wraps(get_response_func)
    async def wrapper_decorator(*args: P.args, **kwargs: P.kwargs) -> T:
        from semantic_kernel.agents.agent import Agent
        from semantic_kernel.contents.chat_message_content import ChatMessageContent
        from semantic_kernel.contents.utils.author_role import AuthorRole

        agent = cast(Agent, args[0])
        
        # Extract messages from args/kwargs
        messages_arg = None
        if len(args) > 1:
            messages_arg = args[1]
        elif "messages" in kwargs:
            messages_arg = kwargs["messages"]

        with tracer.start_as_current_span(f"{OPERATION_NAME} {agent.name}") as span:
            # Set basic attributes
            span.set_attributes({
                gen_ai_attributes.OPERATION: OPERATION_NAME,
                gen_ai_attributes.AGENT_ID: agent.id,
                gen_ai_attributes.AGENT_NAME: agent.name,
            })

            if agent.description:
                span.set_attribute(gen_ai_attributes.AGENT_DESCRIPTION, agent.description)

            # Extract registered agents and tools
            registered_agent_ids = _get_registered_agent_ids(agent)
            span.set_attribute(gen_ai_attributes.AGENT_CHILD_AGENTS, json.dumps(registered_agent_ids))

            # Extract tool definitions from kernel if available
            kernel = kwargs.get("kernel")
            tool_definitions = _get_tool_definitions(agent, kernel)
            if tool_definitions:
                span.set_attribute(gen_ai_attributes.TOOL_DEFINITIONS, json.dumps(tool_definitions))

            # Process input messages and set invocation input
            invocation_input = _process_invocation_input(messages_arg, agent)
            if invocation_input:
                span.set_attribute(gen_ai_attributes.AGENT_INVOCATION_INPUT, json.dumps(invocation_input))

            try:
                result = await get_response_func(*args, **kwargs)
                
                # Extract final assistant response from chat history
                final_assistant_message = None
                
                # Try to get chat history from messages_arg
                chat_history = None
                if hasattr(messages_arg, "messages"):  # ChatHistory object
                    chat_history = messages_arg
                elif isinstance(messages_arg, list):
                    # Create a temporary structure to find assistant messages
                    chat_history = messages_arg
                
                # Find the last assistant message in chat history
                if chat_history:
                    if hasattr(chat_history, "messages"):
                        # ChatHistory object
                        for message in reversed(chat_history.messages):
                            if isinstance(message, ChatMessageContent) and message.role == AuthorRole.ASSISTANT:
                                final_assistant_message = message
                                break
                    elif isinstance(chat_history, list):
                        # List of messages
                        for message in reversed(chat_history):
                            if isinstance(message, ChatMessageContent) and message.role == AuthorRole.ASSISTANT:
                                final_assistant_message = message
                                break
                
                # If no assistant message found in chat history, try to extract from result
                if not final_assistant_message and result:
                    if hasattr(result, "message") and isinstance(result.message, ChatMessageContent):
                        final_assistant_message = result.message
                    elif isinstance(result, ChatMessageContent):
                        final_assistant_message = result
                
                # Process the final assistant message for invocation output
                if final_assistant_message:
                    # Create a mock response structure for processing
                    mock_response = type('MockResponse', (), {'message': final_assistant_message})()
                    invocation_output = _process_invocation_output(mock_response, [], [])
                    if invocation_output:
                        span.set_attribute(gen_ai_attributes.AGENT_INVOCATION_OUTPUT, json.dumps(invocation_output))
                
                return result
                
            except Exception as e:
                logger.error(f"Error in agent get_response: {e}")
                raise

    # Mark the wrapper decorator as an agent diagnostics decorator
    wrapper_decorator.__agent_diagnostics__ = True  # type: ignore

    return wrapper_decorator