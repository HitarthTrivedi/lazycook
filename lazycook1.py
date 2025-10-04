''' import LAZYCOOK
import asyncio
import os
from LAZYCOOK import RichMultiAgentCLI  # or MultiAgentAssistantConfig

config=LAZYCOOK.create_assistant(api_key="AIzaSyD7ebIM0FrCgiIgseVF4lwcWjTaPyz4r4M",conversation_limit=70)
# Run CLI
asyncio.run(config.run_cli())


# Or create assistant instance directly
assistant=config.create_assistant()
'''


import asyncio
from Baby14 import MultiAgentAssistantConfig
import os

api_key = os.getenv("AIzaSyD7ebIM0FrCgiIgseVF4lwcWjTaPyz4r4M")
config = MultiAgentAssistantConfig(api_key, conversation_limit=70)
asyncio.run(config.run_cli())


