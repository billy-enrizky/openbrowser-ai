"""
Setup:
1. Get your API key from your LLM provider
2. Set environment variable: export GOOGLE_API_KEY="your-key"
"""

from dotenv import load_dotenv

from openbrowser import Agent, ChatGoogle

load_dotenv()

agent = Agent(
	task='Find the number of stars of the following repos: openbrowser, playwright, stagehand, react, nextjs',
	llm=ChatGoogle(),
)
agent.run_sync()
