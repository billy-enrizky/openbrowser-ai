"""OpenBrowser advanced template with explicit browser and agent settings."""

import asyncio

from dotenv import load_dotenv

from openbrowser import Agent, BrowserProfile, ChatOpenAI

load_dotenv()


async def main() -> None:
	browser_profile = BrowserProfile(
		headless=True,
		keep_alive=False,
		wait_between_actions=0.5,
		viewport={'width': 1440, 'height': 900},
	)
	agent = Agent(
		task='Open example.com, inspect the page, and print a concise summary.',
		llm=ChatOpenAI(model='gpt-5-mini'),
		browser_profile=browser_profile,
		max_actions_per_step=3,
	)
	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())
