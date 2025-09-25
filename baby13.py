import lazycook
import asyncio
# Create configured assistant
config=lazycook.create_assistant(api_key="Your_gemini_api_key",conversation_limit=70)


# Run CLI
asyncio.run(config.run_cli())


# Or create assistant instance directly
assistant=config.create_assistant()
