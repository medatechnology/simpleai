# SimpleAI

A standalone AI wrapper library for Go. Part of the Meda ecosystem.

## Installation

```bash
go get github.com/medatechnology/simpleai
```

## Features

- **Multi-Provider Support**: Anthropic, OpenAI, Gemini, Groq, Ollama
- **Streaming**: Real-time token streaming for all providers
- **Chat Sessions**: Conversation history with automatic management
- **Middleware**: Retry with backoff, provider fallback, logging
- **Prompt Templates**: Go templates with helper functions
- **Memory Management**: Token-based limits, auto-summarization
- **Embeddings**: OpenAI and Ollama vector embeddings
- **RAG**: Retrieval-augmented generation with vector store

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "os"

    "github.com/medatechnology/simpleai"
    "github.com/medatechnology/simpleai/provider"
)

func main() {
    // Create provider
    anthropic := provider.NewAnthropic(provider.AnthropicConfig{
        APIKey: os.Getenv("ANTHROPIC_API_KEY"),
    })

    // Create client
    client := simpleai.NewClient(anthropic)

    // Create chat session
    chat := client.NewChat(simpleai.WithSystem("You are a helpful assistant."))

    // Send message
    resp, err := chat.Send(context.Background(), "Hello!")
    if err != nil {
        panic(err)
    }

    fmt.Println(resp.Content)
}
```

## Providers

### Anthropic (Claude)

```go
provider := provider.NewAnthropic(provider.AnthropicConfig{
    APIKey: os.Getenv("ANTHROPIC_API_KEY"),
    Model:  "claude-3-5-sonnet-20241022", // default
})
```

### OpenAI

```go
provider := provider.NewOpenAI(provider.OpenAIConfig{
    APIKey: os.Getenv("OPENAI_API_KEY"),
    Model:  "gpt-4o", // default
})
```

### Google Gemini

```go
provider := provider.NewGemini(provider.GeminiConfig{
    APIKey: os.Getenv("GEMINI_API_KEY"),
    Model:  "gemini-1.5-pro", // default
})
```

### Groq

```go
provider := provider.NewGroq(provider.GroqConfig{
    APIKey: os.Getenv("GROQ_API_KEY"),
    Model:  "llama-3.3-70b-versatile", // default
})
```

### Ollama (Local)

```go
provider := provider.NewOllama(provider.OllamaConfig{
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

## Integration with Meda Ecosystem

```go
// Use with simplehttp for REST API
// Use with simpleorm for persistence
// Use with goutil for utilities

import (
    "github.com/medatechnology/simpleai"
    "github.com/medatechnology/simplehttp"
    "github.com/medatechnology/simpleorm"
    "github.com/medatechnology/goutil/utils"
)
```

## License

MIT

## Memory Management

### Token-Based History

```go
chat := client.NewChat(
    simpleai.WithMaxTokens(4000),  // Limit history to 4000 tokens
    simpleai.WithTokenCounter(func(s string) int {
        return len(s) / 4  // ~4 chars per token
    }),
)
```

### Auto-Summarization

```go
import "github.com/medatechnology/simpleai/memory"

summarizer := memory.NewAISummarizer(provider)
mem := memory.NewSimpleWithSummarizer(memory.MemoryConfig{
    MaxTokens:      4000,
    SummarizeAfter: 20,  // Summarize after 20 messages
}, summarizer)
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

// Create embedder and vector store
embedder := embedding.NewOpenAI(...)
store := rag.NewMemoryStore()

// Create RAG instance
r := rag.New(embedder, store, rag.Config{
    TopK:          5,
    MinSimilarity: 0.7,
})

// Add messages to RAG
r.AddMessage(ctx, msg, "msg_1")

// Retrieve relevant context
context, _ := r.BuildContext(ctx, "What did we discuss about headaches?")
```
