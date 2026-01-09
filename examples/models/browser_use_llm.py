"""
Example using ChatBrowserUse - an external LLM service optimized for browser automation.

Setup:
1. Get your API key from https://cloud.browser-use.com/new-api-key
2. Set environment variable: export BROWSER_USE_API_KEY="your-key"

Note: ChatBrowserUse is an external third-party service, not part of OpenBrowser.
"""

import asyncio
import os

from dotenv import load_dotenv

from openbrowser import Agent, ChatBrowserUse

load_dotenv()

if not os.getenv('BROWSER_USE_API_KEY'):
	raise ValueError('BROWSER_USE_API_KEY is not set. Get your key at https://cloud.browser-use.com/new-api-key')


async def main():
	agent = Agent(
		task='Find the number of stars of the openbrowser repo',
		llm=ChatBrowserUse(),
	)

	# Run the agent
	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())
