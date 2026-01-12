import asyncio
import os
import pathlib
import shutil

from dotenv import load_dotenv

from openbrowser import Agent, ChatOpenAI

load_dotenv()

SCRIPT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
agent_dir = SCRIPT_DIR / 'alphabet_earnings'
agent_dir.mkdir(exist_ok=True)

task = """
Go to https://abc.xyz/investor/ and find the latest quarterly earnings report PDF.
Read the PDF and save 3 interesting data points in "alphabet_earnings.pdf" and share it with me!
""".strip('\n')

agent = Agent(
	task=task,
	llm=ChatOpenAI(model='o4-mini'),
	file_system_path=str(agent_dir / 'fs'),
	flash_mode=True,
)


async def main():
	await agent.run()
	input(f'Press Enter to clean the file system at {agent_dir}...')
	# clean the file system
	shutil.rmtree(str(agent_dir / 'fs'))


if __name__ == '__main__':
	asyncio.run(main())
