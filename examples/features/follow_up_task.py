from dotenv import load_dotenv

from openbrowser import Agent, Browser

load_dotenv()

import asyncio


async def main():
	browser = Browser(keep_alive=True)

	await browser.start()

	agent = Agent(task='search for openbrowser.', browser_session=browser)
	await agent.run(max_steps=2)
	agent.add_new_task('return the title of first result')
	await agent.run()

	await browser.kill()


if __name__ == '__main__':
	asyncio.run(main())
