import asyncio
import os

from dotenv import load_dotenv

# test if traceloop is installed
try:
	from traceloop.sdk import Traceloop  # type: ignore
except ImportError:
	print('Traceloop is not installed')
	exit(1)

from openbrowser import Agent

load_dotenv()
api_key = os.getenv('TRACELOOP_API_KEY')
Traceloop.init(api_key=api_key, disable_batch=True)


async def main():
	await Agent('Find the founders of openbrowser').run()


if __name__ == '__main__':
	asyncio.run(main())
