# SimpleAI

A standalone AI wrapper library for Go. Part of the Meda ecosystem.

## Installation

```bash
go get github.com/medatechnology/simpleai
```

## Features

- **Multi-Provider Support**: Anthropic, OpenAI, Gemini, Groq, Ollama, **Mistral**
- **Streaming**: Real-time token streaming for all providers
- **Chat Sessions**: Conversation history with automatic management
- **Autocompact**: Automatic context summarization for long conversations
- **Middleware**: Retry with backoff, provider fallback, logging
- **HTTP Handlers**: Ready-to-use handlers for REST API with SSE streaming
- **Prompt Templates**: Go templates with helper functions
- **Memory Management**: Token-based limits, auto-summarization
- **Embeddings**: OpenAI and Ollama vector embeddings
- **RAG**: Retrieval-augmented generation with vector store
- **Docker Support**: Ready-to-deploy container configuration

## Quick Start

```go
package main

import (
    "context"
    "fmt"

    "github.com/medatechnology/simpleai"
    "github.com/medatechnology/simpleai/provider"
)

func main() {
    // Create provider from environment variables
    mistral := provider.NewMistralFromEnv()

    // Create client
    client := simpleai.NewClient(mistral)

    // Create chat session with autocompact
    chat := client.NewChat(
        simpleai.WithSystem("You are a helpful assistant."),
        simpleai.WithAutocompact(simpleai.DefaultAutocompactConfig()),
    )

    // Send message
    resp, err := chat.Send(context.Background(), "Hello!")
    if err != nil {
        panic(err)
    }

    fmt.Println(resp.Content)
}
```

## Providers

All providers support `NewXFromEnv()` for easy configuration from environment variables.

### Mistral AI

```go
// From environment: MISTRAL_API_KEY, MISTRAL_MODEL (optional)
mistral := provider.NewMistralFromEnv()

// Or with config
mistral := provider.NewMistral(provider.MistralConfig{
    APIKey: os.Getenv("MISTRAL_API_KEY"),
    Model:  "mistral-large-latest", // default
})
```

### Anthropic (Claude)

```go
// From environment: ANTHROPIC_API_KEY, ANTHROPIC_MODEL (optional)
anthropic := provider.NewAnthropicFromEnv()

// Or with config
anthropic := provider.NewAnthropic(provider.AnthropicConfig{
    APIKey: os.Getenv("ANTHROPIC_API_KEY"),
    Model:  "claude-3-5-sonnet-20241022", // default
})
```

### OpenAI

```go
// From environment: OPENAI_API_KEY, OPENAI_MODEL, OPENAI_ORGANIZATION (optional)
openai := provider.NewOpenAIFromEnv()

// Or with config
openai := provider.NewOpenAI(provider.OpenAIConfig{
    APIKey: os.Getenv("OPENAI_API_KEY"),
    Model:  "gpt-4o", // default
})
```

### Google Gemini

```go
// From environment: GEMINI_API_KEY, GEMINI_MODEL (optional)
gemini := provider.NewGeminiFromEnv()

// Or with config
gemini := provider.NewGemini(provider.GeminiConfig{
    APIKey: os.Getenv("GEMINI_API_KEY"),
    Model:  "gemini-1.5-pro", // default
})
```

### Groq

```go
// From environment: GROQ_API_KEY, GROQ_MODEL (optional)
groq := provider.NewGroqFromEnv()

// Or with config
groq := provider.NewGroq(provider.GroqConfig{
    APIKey: os.Getenv("GROQ_API_KEY"),
    Model:  "llama-3.3-70b-versatile", // default
})
```

### Ollama (Local)

```go
// From environment: OLLAMA_BASE_URL, OLLAMA_MODEL (optional)
ollama := provider.NewOllamaFromEnv()

// Or with config
ollama := provider.NewOllama(provider.OllamaConfig{
    BaseURL: "http://localhost:11434", // default
    Model:   "llama3.2",               // default
})
```

## Streaming

```go
stream, err := chat.Stream(ctx, "Tell me a story")
if err != nil {
    panic(err)
}

for event := range stream {
    if event.Error != nil {
        panic(event.Error)
    }
    fmt.Print(event.Content)
    if event.Done {
        fmt.Println()
    }
}
```

## Autocompact (Context Summarization)

Automatically summarize old messages when conversation gets too long:

```go
chat := client.NewChat(
    simpleai.WithSystem("You are a helpful assistant."),
    simpleai.WithAutocompact(simpleai.AutocompactConfig{
        Threshold:  20,  // Trigger when history reaches 20 messages
        KeepRecent: 4,   // Keep last 4 messages, summarize the rest
    }),
)

// After many messages, older ones are summarized automatically
// The summary is included in the system prompt for context
```

## HTTP API Server

SimpleAI includes ready-to-use HTTP handlers for building REST APIs with SSE streaming.

### Quick Setup

```go
import (
    "github.com/medatechnology/simpleai"
    shttp "github.com/medatechnology/simpleai/http"
    "github.com/medatechnology/simplehttp/framework/fiber"
)

func main() {
    client := simpleai.NewClient(provider.NewMistralFromEnv())
    chat := client.NewChat(simpleai.WithSystem("You are helpful."))

    server := fiber.NewServer(nil)

    // Non-streaming completion
    server.POST("/api/v1/chat/complete", shttp.CompleteHandler(client))

    // SSE streaming completion
    server.POST("/api/v1/chat/stream", shttp.StreamHandler(client))

    // Chat session with history
    server.POST("/api/v1/doctor/chat", shttp.ChatStreamHandler(chat))

    server.Start(":8080")
}
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/chat` | Simple chat (non-streaming) |
| POST | `/api/v1/chat/complete` | OpenAI-compatible completion |
| POST | `/api/v1/chat/stream` | SSE streaming completion |
| POST | `/api/v1/doctor/chat` | Doctor AI chat with history |

### Request/Response Examples

#### Simple Chat
```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

Response:
```json
{"response": "Hello! How can I help?", "model": "mistral-large-latest", "usage": {...}}
```

#### Streaming Chat (SSE)
```bash
curl -X POST http://localhost:8080/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Tell me a joke"}]}'
```

Response (SSE):
```
data: {"content":"Why","done":false}
data: {"content":" did","done":false}
data: {"content":" the programmer...","done":false}
data: {"done":true,"finish_reason":"stop"}
```

#### Chat with History
```bash
curl -X POST http://localhost:8080/api/v1/doctor/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I have a headache"}'
```

## Docker

### Run with Docker Compose

```bash
# Create .env file
cp .env.example .env
# Edit .env with your MISTRAL_API_KEY

# Start server
docker compose up -d --build

# View logs
docker compose logs -f

# Stop
docker compose down
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MISTRAL_API_KEY` | Yes | - | Mistral AI API key |
| `MISTRAL_MODEL` | No | `mistral-large-latest` | Model to use |
| `OPENAI_API_KEY` | No | - | OpenAI API key (fallback) |
| `SIMPLEHTTP_PORT` | No | `8080` | Server port |

## Middleware

### Retry with Backoff

```go
import "github.com/medatechnology/simpleai/middleware"

client := simpleai.NewClient(provider,
    simpleai.WithMiddleware(middleware.RetrySimple(3)),
)
```

### Provider Fallback

```go
client := simpleai.NewClient(primaryProvider,
    simpleai.WithMiddleware(middleware.FallbackSimple(backupProvider)),
)
```

### Logging

```go
client := simpleai.NewClient(provider,
    simpleai.WithMiddleware(middleware.SimpleLogger(func(msg string) {
        log.Println(msg)
    })),
)
```

## Prompt Templates

```go
import "github.com/medatechnology/simpleai/template"

engine := template.NewEngine()
engine.Load("doctor", `You are Dr. AI.
Patient: {{.Name}}, Age: {{.Age}}
Conditions: {{join .Conditions ", "}}`)

prompt, _ := engine.Execute("doctor", map[string]any{
    "Name":       "John",
    "Age":        35,
    "Conditions": []string{"headache", "fatigue"},
})
```

## Memory Management

### Token-Based History

```go
chat := client.NewChat(
    simpleai.WithMaxTokens(4000),
    simpleai.WithTokenCounter(func(s string) int {
        return len(s) / 4  // ~4 chars per token
    }),
)
```

### Custom Summarizer

```go
import "github.com/medatechnology/simpleai/memory"

summarizer := memory.NewAISummarizer(provider)
chat := client.NewChat(
    simpleai.WithAutocompact(simpleai.AutocompactConfig{
        Threshold:  20,
        KeepRecent: 4,
        Summarizer: summarizer,  // Use custom summarizer
    }),
)
```

## Embeddings

```go
import "github.com/medatechnology/simpleai/embedding"

// OpenAI embeddings
embedder := embedding.NewOpenAI(embedding.OpenAIConfig{
    APIKey: os.Getenv("OPENAI_API_KEY"),
})

// Ollama embeddings (local)
embedder := embedding.NewOllama(embedding.OllamaConfig{
    Model: "nomic-embed-text",
})

// Generate embedding
vector, _ := embedder.Embed(ctx, "Hello world")
```

## RAG (Retrieval-Augmented Generation)

```go
import (
    "github.com/medatechnology/simpleai/embedding"
    "github.com/medatechnology/simpleai/rag"
)

embedder := embedding.NewOpenAI(...)
store := rag.NewMemoryStore()

r := rag.New(embedder, store, rag.Config{
    TopK:          5,
    MinSimilarity: 0.7,
})

r.AddMessage(ctx, msg, "msg_1")
context, _ := r.BuildContext(ctx, "What did we discuss about headaches?")
```

## License

MIT
