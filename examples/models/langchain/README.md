# Langchain Models (legacy)

<!-- toc:start -->
## Contents

- [1. How to use](#1-how-to-use)

<!-- toc:end -->

This directory contains an example of how to use Langchain models with the OpenBrowser chat model interface.

## 1. How to use

Run the working example from the repository root:

```bash
uv run python -m examples.models.langchain.example
```

The example wraps a Langchain chat model with `ChatLangchain`:

```python
import asyncio

from langchain_openai import ChatOpenAI

from openbrowser import Agent
from examples.models.langchain.chat import ChatLangchain


async def main():
    langchain_model = ChatOpenAI(model='gpt-4.1-mini', temperature=0.1)
    llm = ChatLangchain(chat=langchain_model)

    agent = Agent(
        task="Go to google.com and search for 'browser automation with Python'",
        llm=llm,
    )
    history = await agent.run()
    print(history.final_result())


if __name__ == '__main__':
    asyncio.run(main())
```
