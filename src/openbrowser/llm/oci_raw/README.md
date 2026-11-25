# OCI Raw API Integration

<!-- toc:start -->
## Table of Contents

- [1. Features](#1-features)
- [2. Installation](#2-installation)
- [3. Usage](#3-usage)
  - [3.1. Basic Usage](#31-basic-usage)
  - [3.2. Provider-Specific Configuration Examples](#32-provider-specific-configuration-examples)
    - [3.2.1. Meta Llama Model](#321-meta-llama-model)
    - [3.2.2. Cohere Model](#322-cohere-model)
    - [3.2.3. xAI Model](#323-xai-model)
  - [3.3. Structured Output](#33-structured-output)
- [4. Available Models](#4-available-models)
  - [4.1. Tool Calling Support](#41-tool-calling-support)
  - [4.2. Image Support Models](#42-image-support-models)
    - [4.2.1. Vision-Enabled Chat Models](#421-vision-enabled-chat-models)
    - [4.2.2. Image Embedding Models](#422-image-embedding-models)
- [5. Configuration](#5-configuration)
  - [5.1. Provider-Specific API Formats](#51-provider-specific-api-formats)
    - [5.1.1. Meta and xAI Models](#511-meta-and-xai-models)
    - [5.1.2. Cohere Models](#512-cohere-models)
  - [5.2. Authentication Types](#52-authentication-types)
  - [5.3. Model Parameters](#53-model-parameters)
- [6. Error Handling](#6-error-handling)
- [7. Comparison with Langchain Integration](#7-comparison-with-langchain-integration)
- [8. Example Response Format](#8-example-response-format)
- [9. Troubleshooting](#9-troubleshooting)
  - [9.1. Common Issues](#91-common-issues)
  - [9.2. Debug Mode](#92-debug-mode)
- [10. Contributing](#10-contributing)

<!-- toc:end -->

This module provides direct integration with Oracle Cloud Infrastructure's Generative AI service using raw API calls, without Langchain dependencies.

## 1. Features

- **Direct API Integration**: Uses OCI's native Python SDK for direct API calls
- **Async Support**: Full async/await support for non-blocking operations
- **Structured Output**: Support for Pydantic model validation of responses
- **Error Handling**: Comprehensive error handling with proper exception types
- **Authentication**: Support for multiple OCI authentication methods

## 2. Installation

Make sure you have the required OCI dependencies installed:

```bash
pip install oci
```

## 3. Usage

### 3.1. Basic Usage

```python
from openbrowser import Agent
from openbrowser.llm import ChatOCIRaw

# Configure the model
model = ChatOCIRaw(
    model_id="ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceya...",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaayeiis5uk2nuubznrekd...",
    provider="meta",  # or "cohere"
    temperature=1.0,
    max_tokens=600,
    top_p=0.75,
    auth_type="API_KEY",
    auth_profile="DEFAULT"
)

# Use with openbrowser Agent
agent = Agent(
    task="Search for Python tutorials and summarize them",
    llm=model
)

# Run with asyncio
import asyncio
history = asyncio.run(agent.run())
```

### 3.2. Provider-Specific Configuration Examples

#### 3.2.1. Meta Llama Model
```python
meta_model = ChatOCIRaw(
    model_id="ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceya...",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaayeiis5uk2nuubznrekd...",
    provider="meta",  # Uses GenericChatRequest
    temperature=0.7,
    max_tokens=800,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    top_p=0.9
)
```

#### 3.2.2. Cohere Model
```python
cohere_model = ChatOCIRaw(
    model_id="ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceya...",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaayeiis5uk2nuubznrekd...",
    provider="cohere",  # Uses CohereChatRequest
    temperature=1.0,
    max_tokens=600,
    frequency_penalty=0.0,
    top_p=0.75,
    top_k=0  # Cohere-specific parameter
)
```

#### 3.2.3. xAI Model
```python
xai_model = ChatOCIRaw(
    model_id="ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceya...",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaayeiis5uk2nuubznrekd...",
    provider="xai",  # Uses GenericChatRequest
    temperature=1.0,
    max_tokens=20000,
    top_p=1.0,
    top_k=0
)
```

### 3.3. Structured Output

```python
from pydantic import BaseModel

class SearchResult(BaseModel):
    title: str
    summary: str
    relevance_score: float

# Use structured output
response = await model.ainvoke(messages, output_format=SearchResult)
result = response.completion  # This is a SearchResult instance
```

## 4. Available Models

For the complete list of available models in Oracle Cloud Infrastructure Generative AI, refer to the official documentation: [OCI Generative AI Pretrained Models](https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm)

### 4.1. Tool Calling Support

**Important**: Only models that support tool calling/function calling are compatible with openbrowser. Tool calling is essential for openbrowser as the agent needs to call browser automation functions (click, type, scroll, etc.) to interact with web pages.

According to Oracle's documentation, tool calling functionality is available exclusively through the API and is not supported for browser-based use. However, when using openbrowser with OCI models through this integration, the tool calling happens at the application level (not browser-based), making it compatible.

### 4.2. Image Support Models

Several OCI models support image processing capabilities, which are useful when openbrowser needs to analyze webpage screenshots:

#### 4.2.1. Vision-Enabled Chat Models
- **Meta Llama 3.2 90B Vision**: Supports both text and image inputs
- **Meta Llama 3.2 11B Vision**: Supports both text and image inputs

#### 4.2.2. Image Embedding Models
- **Cohere Embed English Image 3**: Supports image inputs for semantic searches
- **Cohere Embed Multilingual Image 3**: Supports multilingual image processing
- **Cohere Embed English Light Image 3**: Lightweight version with image support
- **Cohere Embed Multilingual Light Image 3**: Lightweight multilingual version with image support

These vision-enabled models are particularly useful for openbrowser tasks that require understanding webpage content through screenshots, such as:
- Identifying UI elements and buttons
- Reading text from images
- Understanding page layouts and visual context
- Processing charts, graphs, and visual data

## 5. Configuration

### 5.1. Provider-Specific API Formats

Different model providers in OCI use different API request formats:

#### 5.1.1. Meta and xAI Models
- Use `GenericChatRequest` with messages array
- Support structured conversations with multiple message types
- Parameters: `temperature`, `max_tokens`, `frequency_penalty`, `presence_penalty`, `top_p`

#### 5.1.2. Cohere Models
- Use `CohereChatRequest` with single message string
- Convert conversation history to a single formatted string
- Parameters: `temperature`, `max_tokens`, `frequency_penalty`, `top_p`, `top_k`

The integration automatically detects the correct format based on the `provider` parameter and handles the conversion transparently.

### 5.2. Authentication Types

The integration supports multiple OCI authentication methods:

- `API_KEY`: Uses API key authentication (default)
- `INSTANCE_PRINCIPAL`: Uses instance principal authentication
- `RESOURCE_PRINCIPAL`: Uses resource principal authentication

### 5.3. Model Parameters

- `model_id`: The OCID of your OCI GenAI model
- `service_endpoint`: The OCI service endpoint URL
- `compartment_id`: The OCID of your OCI compartment
- `provider`: Model provider ("meta", "cohere", or "xai")
- `temperature`: Response randomness (0.0-2.0)
- `max_tokens`: Maximum tokens in response
- `top_p`: Top-p sampling parameter
- `frequency_penalty`: Frequency penalty for repetition
- `presence_penalty`: Presence penalty for repetition
- `top_k`: Top-k sampling parameter (used by Cohere models)

## 6. Error Handling

The integration provides proper error handling with specific exception types:

- `ModelRateLimitError`: For rate limiting (429 errors)
- `ModelProviderError`: For other API errors (4xx, 5xx)

## 7. Comparison with Langchain Integration

| Feature | OCI Raw API | Langchain Integration |
|---------|-------------|----------------------|
| Dependencies | OCI SDK only | Langchain + OCI SDK |
| Performance | Direct API calls | Additional abstraction layer |
| Control | Full control over requests | Limited by Langchain interface |
| Updates | Direct OCI SDK updates | Dependent on Langchain updates |
| Complexity | Lower complexity | Higher complexity |

## 8. Example Response Format

The OCI GenAI API returns responses in this format:

```json
{
  "chat_response": {
    "api_format": "GENERIC",
    "choices": [
      {
        "finish_reason": "stop",
        "index": 0,
        "message": {
          "content": [
            {
              "text": "Response text here",
              "type": "TEXT"
            }
          ],
          "role": "ASSISTANT"
        }
      }
    ],
    "usage": {
      "completion_tokens": 18,
      "prompt_tokens": 38,
      "total_tokens": 56
    }
  }
}
```

## 9. Troubleshooting

### 9.1. Common Issues

1. **Authentication Errors**: Ensure your OCI configuration is correct and you have the necessary permissions
2. **Model Not Found**: Verify your model OCID and ensure it's available in your compartment
3. **Rate Limiting**: The integration handles rate limits automatically with proper error types
4. **API Format Mismatch**: If you get "Chat request's apiFormat must match serving model's apiFormat" error, ensure you're using the correct `provider` parameter:
   - Use `provider="meta"` for Meta Llama models
   - Use `provider="cohere"` for Cohere models  
   - Use `provider="xai"` for xAI models

### 9.2. Debug Mode

Enable verbose logging by setting the `verbose` parameter to `True` (not implemented in this version but can be added).

## 10. Contributing

When contributing to this module:

1. Follow the existing code style
2. Add proper type hints
3. Include comprehensive error handling
4. Add tests for new features
5. Update documentation
