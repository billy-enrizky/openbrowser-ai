"""OpenBrowser custom-tools template.

Register additional actions on `tools` before constructing the agent.
"""

import asyncio

from dotenv import load_dotenv

from openbrowser import Agent, ChatOpenAI, Tools

load_dotenv()


async def main() -> None:
	tools = Tools()
	# Example: add a project-specific action to tools.registry.action(...)
	# before creating the agent.
	agent = Agent(
		task='Use the available tools to complete the requested web task.',
		llm=ChatOpenAI(model='gpt-5-mini'),
		tools=tools,
	)
	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())
