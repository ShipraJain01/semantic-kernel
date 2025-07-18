# Copyright (c) Microsoft. All rights reserved.

# Constants for tracing agent activities with semantic conventions.
# Ideally, we should use the attributes from the semcov package.
# However, many of the attributes are not yet available in the package,
# so we define them here for now.

# Activity tags
OPERATION = "gen_ai.operation.name"
AGENT_ID = "gen_ai.agent.id"
AGENT_NAME = "gen_ai.agent.name"
AGENT_DESCRIPTION = "gen_ai.agent.description"

# New agent attributes - shiprajain01
AGENT_VERSION = "gen_ai.agent.version"
AGENT_REGISTERED_AGENT_IDS = "gen_ai.agent.registered_agent_ids"
AGENT_REGISTERED_TOOL_IDS = "gen_ai.agent.registered_tool_ids"
AGENT_ROLE = "gen_ai.agent.role"
AGENT_CALLED_TOOL_IDS = "gen_ai.agent.called_tool_ids"
AGENT_INPUT = "gen_ai.agent.input"
AGENT_OUTPUT = "gen_ai.agent.output"

# New agent attributes
AGENT_VERSION = "gen_ai.agent.version"
AGENT_CHILD_AGENTS = "gen_ai.agent.child_agents"
AGENT_ROLE = "gen_ai.agent.role"
AGENT_INVOCATION_INPUT = "gen_ai.agent.invocation_input"
AGENT_INVOCATION_OUTPUT = "gen_ai.agent.invocation_output"
TOOL_DEFINITIONS = "gen_ai.tool.definitions"


# Event names (reuse from model diagnostics) - shiprajain01
EVENT_NAME = "event.name"
SYSTEM = "gen_ai.system"
SYSTEM_MESSAGE = "gen_ai.system.message"
USER_MESSAGE = "gen_ai.user.message"
ASSISTANT_MESSAGE = "gen_ai.assistant.message"
CHOICE = "gen_ai.choice"
