# Copyright (c) Microsoft. All rights reserved.

from typing import TYPE_CHECKING, Any
import json

from opentelemetry import trace, context
from opentelemetry.context import attach, detach
import logging

from semantic_kernel.utils.telemetry.model_diagnostics.gen_ai_attributes import (
    OPERATION,
    TOOL_CALL_ID,
    TOOL_DESCRIPTION,
    TOOL_NAME,
    SYSTEM,
    EVENT_NAME,
    CHOICE,
)
from semantic_kernel.utils.telemetry.model_diagnostics.decorators import are_sensitive_events_enabled

if TYPE_CHECKING:
    from semantic_kernel.functions.kernel_function import KernelFunction

# The operation name is defined by OTeL GenAI semantic conventions:
# https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool-span
OPERATION_NAME = "execute_tool"

logger = logging.getLogger(__name__)

# Context key for storing our tracer span
_FUNCTION_TRACER_KEY = "sk_function_tracer"


class FunctionTracerSpan:
    """Custom span wrapper that captures function input/output."""
    
    def __init__(self, span: trace.Span, function: "KernelFunction"):
        self.span = span
        self.function = function
        self._function_context = None
        self._actual_span = None
        self._context_token = None
        self._captured_arguments = {}  # Store arguments for choice event
        
    def __enter__(self):
        # Enter the span context
        self._actual_span = self.span.__enter__()
        
        # Store this wrapper in OpenTelemetry context
        ctx = context.set_value(_FUNCTION_TRACER_KEY, self)
        self._context_token = attach(ctx)
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Capture function result before closing span
        if hasattr(self, '_function_context') and self._function_context:
            self._capture_execution_data()
            
            # Emit gen_ai.choice event on successful execution
            if exc_type is None:
                self._emit_choice_event()
        
        # Record exception if one occurred
        if exc_type is not None:
            if self._actual_span:
                self._actual_span.record_exception(exc_val)
                self._actual_span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc_val)))
            
        # Detach our context
        if self._context_token:
            detach(self._context_token)
            
        # Exit the span
        self.span.__exit__(exc_type, exc_val, exc_tb)
    
    def set_function_context(self, context):
        """Store the function invocation context to extract data later."""
        self._function_context = context
        # Capture input arguments immediately
        if hasattr(context, 'arguments') and context.arguments:
            self._capture_input_arguments(context.arguments)
    
    def _capture_input_arguments(self, arguments):
        """Capture and set input arguments as span attributes."""
        try:
            # Convert KernelArguments to dict
            args_dict = {}
            
            # Handle different argument types
            if hasattr(arguments, 'items'):
                # If arguments is dict-like (KernelArguments has items() method)
                args_dict = dict(arguments.items())
            elif hasattr(arguments, '__dict__'):
                # If it has attributes
                args_dict = {k: v for k, v in arguments.__dict__.items() if not k.startswith('_')}
            elif hasattr(arguments, '__iter__') and not isinstance(arguments, str):
                # If arguments is iterable (but not string)
                try:
                    for key, value in arguments:
                        args_dict[key] = value
                except:
                    # If iteration fails, convert to string
                    args_dict = {"arguments": str(arguments)}
            else:
                # Fallback to string representation
                args_dict = {"arguments": str(arguments)}
            
            # Store captured arguments for choice event
            self._captured_arguments = args_dict
            
            # Use the actual span for setting attributes
            span = self._actual_span or self.span
            
            # Set gen_ai.tool.call.arguments in JSON array format
            args_json = json.dumps(args_dict, default=str)
            if len(args_json) > 5000:
                args_json = args_json[:4997] + "..."
            
            # Format as JSON array containing the arguments JSON string
            arguments_array = [args_json]
            span.set_attribute("gen_ai.tool.call.arguments", json.dumps(arguments_array))
            
            logger.debug(f"Captured input arguments for {self.function.name}: {args_dict}")
            
        except Exception as e:
            logger.warning(f"Failed to capture input arguments: {e}")
    
    def _capture_execution_data(self):
        """Capture function execution result."""
        try:
            if hasattr(self._function_context, 'result') and self._function_context.result is not None:
                result = self._function_context.result
                
                # Extract the actual value from FunctionResult
                if hasattr(result, 'value'):
                    value = result.value
                else:
                    value = result
                
                # Use the actual span for setting attributes
                span = self._actual_span or self.span
                
                # Convert result to JSON string
                if isinstance(value, (dict, list)):
                    # If it's already a dict or list, serialize it directly
                    result_json = json.dumps(value, default=str)
                else:
                    # For other types, try to create a structured result
                    str_value = str(value)
                    if len(str_value) > 5000:
                        str_value = str_value[:4997] + "..."
                    
                    # Try to parse as JSON first, fallback to string wrapper
                    try:
                        # Try to parse the string as JSON
                        parsed_result = json.loads(str_value)
                        result_json = json.dumps(parsed_result)
                    except (json.JSONDecodeError, ValueError):
                        # If not valid JSON, wrap in a simple object
                        result_json = json.dumps({"result": str_value})
                
                # Format as JSON array containing the result JSON string
                result_array = [result_json]
                span.set_attribute("gen_ai.tool.call.result", json.dumps(result_array))
                
                # Check if result has metadata with arguments and update input arguments
                if hasattr(result, 'metadata') and result.metadata:
                    if isinstance(result.metadata, dict) and 'arguments' in result.metadata:
                        metadata_arguments = result.metadata['arguments']
                        
                        # Convert metadata arguments to proper format
                        if isinstance(metadata_arguments, dict):
                            args_json = json.dumps(metadata_arguments, default=str)
                        else:
                            args_json = json.dumps({"arguments": str(metadata_arguments)}, default=str)
                        
                        if len(args_json) > 5000:
                            args_json = args_json[:4997] + "..."
                        
                        # Update gen_ai.tool.call.arguments with metadata arguments
                        arguments_array = [args_json]
                        span.set_attribute("gen_ai.tool.call.arguments", json.dumps(arguments_array))
                        
                        # Update captured arguments for choice event
                        self._captured_arguments = metadata_arguments if isinstance(metadata_arguments, dict) else {"arguments": str(metadata_arguments)}
                        
                        logger.debug(f"Updated input arguments from metadata for {self.function.name}: {metadata_arguments}")
                
                logger.debug(f"Captured output for {self.function.name}: {result_json[:100]}...")
                    
        except Exception as e:
            logger.warning(f"Failed to capture execution result: {e}")
    
    def _emit_choice_event(self):
        """Emit gen_ai.choice event for function execution."""
        if not are_sensitive_events_enabled():
            return
        
        try:
            # Extract the result value
            result_value = None
            metadata_arguments = None
            
            if hasattr(self._function_context, 'result') and self._function_context.result is not None:
                result = self._function_context.result
                if hasattr(result, 'value'):
                    result_value = result.value
                else:
                    result_value = result
                
                # Extract arguments from result metadata if available
                if hasattr(result, 'metadata') and result.metadata:
                    if isinstance(result.metadata, dict) and 'arguments' in result.metadata:
                        metadata_arguments = result.metadata['arguments']
            
            # Use metadata arguments if available, otherwise fallback to captured arguments
            arguments_for_tool_call = metadata_arguments if metadata_arguments is not None else self._captured_arguments
            
            # Generate tool call ID
            tool_call_id = f"call_{self.function.name}_{id(self._function_context)}"
            
            # Build the tool call structure
            tool_call = {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": self.function.fully_qualified_name,
                    "arguments": json.dumps(arguments_for_tool_call, default=str) if isinstance(arguments_for_tool_call, dict) else str(arguments_for_tool_call)
                }
            }
            
            # Build the choice event body
            choice_body = {
                "index": 0,  # Single choice for function execution
                "finish_reason": "stop",  # Function completed successfully
                "message": {
                    "content": str(result_value) if result_value is not None else "",
                    "role": "tool",  # Functions are tools
                    "tool_calls": [tool_call]
                }
            }
            
            # Log the choice event
            logger.info(
                json.dumps(choice_body),
                extra={
                    EVENT_NAME: CHOICE,
                    SYSTEM: "semantic-kernel",
                }
            )
            
            logger.debug(f"Emitted gen_ai.choice event for function: {self.function.name}")
            
        except Exception as e:
            logger.warning(f"Failed to emit gen_ai.choice event: {e}")


def start_as_current_span(
    tracer: trace.Tracer,
    function: "KernelFunction",
    metadata: dict[str, Any] | None = None,
):
    """Starts a span for the given function using the provided tracer.

    Args:
        tracer (trace.Tracer): The OpenTelemetry tracer to use.
        function (KernelFunction): The function for which to start the span.
        metadata (dict[str, Any] | None): Optional metadata to include in the span attributes.

    Returns:
        FunctionTracerSpan: Custom span wrapper that captures input/output.
    """
    attributes = {
        OPERATION: OPERATION_NAME,
        TOOL_NAME: function.fully_qualified_name,
        "gen_ai.tool.type": "function",
        SYSTEM: "semantic-kernel",
    }

    tool_call_id = metadata.get("id", None) if metadata else None
    if tool_call_id:
        attributes[TOOL_CALL_ID] = tool_call_id
    
    # Add tool description if available
    if function.description:
        attributes[TOOL_DESCRIPTION] = function.description
    
    span = tracer.start_as_current_span(
        f"{OPERATION_NAME} {function.fully_qualified_name}", 
        attributes=attributes
    )
    
    return FunctionTracerSpan(span, function)


# Monkey patch to integrate with function execution
def _patch_function_invocation_context():
    """Patch FunctionInvocationContext to automatically capture data in our span."""
    try:
        from semantic_kernel.filters.functions.function_invocation_context import FunctionInvocationContext
        
        # Store original methods
        original_init = FunctionInvocationContext.__init__
        
        def patched_init(self, *args, **kwargs):
            # Call original init
            original_init(self, *args, **kwargs)
            
            # Get the function tracer from context
            function_tracer = context.get_value(_FUNCTION_TRACER_KEY)
            if function_tracer and isinstance(function_tracer, FunctionTracerSpan):
                function_tracer.set_function_context(self)
                logger.debug(f"Connected FunctionTracerSpan to FunctionInvocationContext for function: {self.function.name if hasattr(self, 'function') else 'unknown'}")
        
        # Apply init patch
        FunctionInvocationContext.__init__ = patched_init
        
        # Try to patch result setter if it exists
        try:
            # Check if result exists and what type it is
            if hasattr(FunctionInvocationContext, 'result'):
                result_attr = getattr(FunctionInvocationContext, 'result')
                
                if isinstance(result_attr, property):
                    # It's a property, patch the setter
                    original_result_setter = result_attr.fset
                    
                    def patched_result_setter(self, value):
                        # Set the result first
                        if original_result_setter:
                            original_result_setter(self, value)
                        else:
                            # If no original setter, set the private attribute directly
                            self._result = value
                        
                        # Capture the result in our tracer
                        function_tracer = context.get_value(_FUNCTION_TRACER_KEY)
                        if function_tracer and isinstance(function_tracer, FunctionTracerSpan):
                            # Force capture of execution data immediately
                            function_tracer._function_context = self
                            function_tracer._capture_execution_data()
                            logger.debug(f"Captured result for function: {self.function.name if hasattr(self, 'function') else 'unknown'}")
                    
                    # Create a new property with our patched setter
                    FunctionInvocationContext.result = property(
                        result_attr.fget,
                        patched_result_setter,
                        result_attr.fdel if hasattr(result_attr, 'fdel') else None,
                        result_attr.__doc__ if hasattr(result_attr, '__doc__') else None
                    )
                    logger.debug("Successfully patched result property")
                else:
                    # It's not a property, use __setattr__ hook instead
                    original_setattr = FunctionInvocationContext.__setattr__
                    
                    def patched_setattr(self, name, value):
                        # Call original setattr
                        original_setattr(self, name, value)
                        
                        # If setting result, capture it
                        if name == 'result':
                            function_tracer = context.get_value(_FUNCTION_TRACER_KEY)
                            if function_tracer and isinstance(function_tracer, FunctionTracerSpan):
                                function_tracer._function_context = self
                                function_tracer._capture_execution_data()
                                logger.debug(f"Captured result via setattr for function: {self.function.name if hasattr(self, 'function') else 'unknown'}")
                    
                    FunctionInvocationContext.__setattr__ = patched_setattr
                    logger.debug("Successfully patched via __setattr__")
            else:
                # result doesn't exist yet, patch __setattr__ to catch when it's created
                original_setattr = FunctionInvocationContext.__setattr__
                
                def patched_setattr(self, name, value):
                    # Call original setattr
                    original_setattr(self, name, value)
                    
                    # If setting result, capture it
                    if name == 'result':
                        function_tracer = context.get_value(_FUNCTION_TRACER_KEY)
                        if function_tracer and isinstance(function_tracer, FunctionTracerSpan):
                            function_tracer._function_context = self
                            function_tracer._capture_execution_data()
                            logger.debug(f"Captured result via setattr for function: {self.function.name if hasattr(self, 'function') else 'unknown'}")
                
                FunctionInvocationContext.__setattr__ = patched_setattr
                logger.debug("Successfully patched via __setattr__ (result not found)")
                
        except Exception as e:
            logger.warning(f"Could not patch result setter, will rely on exit capture: {e}")
        
        logger.info("Successfully patched FunctionInvocationContext for input/output capture")
        
    except ImportError:
        logger.warning("Could not import FunctionInvocationContext for patching")
    except Exception as e:
        logger.error(f"Failed to patch FunctionInvocationContext: {e}", exc_info=True)


# Apply the patch when module is imported
_patch_function_invocation_context()