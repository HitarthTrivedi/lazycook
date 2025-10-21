import lazycook5
import asyncio

# Create configured assistant factory
config = lazycook5.create_assistant("GEMINI_API_KEY", conversation_limit=90)

# Run CLI - config is a MultiAgentAssistantConfig, not AutonomousMultiAgentAssistant
asyncio.run(config.run_cli())
