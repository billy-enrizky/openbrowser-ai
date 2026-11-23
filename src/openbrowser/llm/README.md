# OpenBrowser LLMs

We officially support the following LLMs:

- OpenAI (`ChatOpenAI`)
- Anthropic (`ChatAnthropic`)
- Google (`ChatGoogle`)
- Groq (`ChatGroq`)
- Ollama (`ChatOllama`)
- DeepSeek (`ChatDeepSeek`)
- Cerebras (`ChatCerebras`)
- OpenRouter (`ChatOpenRouter`)
- Azure OpenAI (`ChatAzureOpenAI`)
- AWS Bedrock (`ChatAWSBedrock`, `ChatAnthropicBedrock`)
- Oracle Cloud Infrastructure (`ChatOCIRaw`)
- Browser-Use (`ChatBrowserUse`)

## Migrating from LangChain

Because of how we implemented the LLMs, we can technically support anything. If you want to use a LangChain model, you can use the `ChatLangchain` (NOT OFFICIALLY SUPPORTED) class.

You can find all the details in the [LangChain example](examples/models/langchain/example.py). We suggest you grab that code and use it as a reference.
