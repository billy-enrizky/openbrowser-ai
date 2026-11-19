"""OpenBrowser default template.

Set OPENAI_API_KEY in your environment or .env file before running.
"""

import asyncio

from dotenv import load_dotenv

from openbrowser import Agent, ChatOpenAI

load_dotenv()


async def main() -> None:
	agent = Agent(
		task='Search the web and summarize the most relevant results.',
		llm=ChatOpenAI(model='gpt-5-mini'),
	)
	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())
