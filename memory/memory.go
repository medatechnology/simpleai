package memory

import (
	"context"

	"github.com/medatechnology/simpleai"
)

// Memory defines the interface for conversation memory management
type Memory interface {
	// Add adds a message to memory
	Add(ctx context.Context, msg simpleai.Message) error

	// GetMessages retrieves messages for context, respecting token limits
	GetMessages(ctx context.Context, maxTokens int) ([]simpleai.Message, error)

	// GetRelevant retrieves messages relevant to the query (for RAG)
	GetRelevant(ctx context.Context, query string, topK int) ([]simpleai.Message, error)

	// Clear clears all messages from memory
	Clear(ctx context.Context) error

	// Count returns the number of messages in memory
	Count() int

	// TokenCount returns the total token count of all messages
	TokenCount() int
}

// Summarizer can summarize conversation history
type Summarizer interface {
	// Summarize compresses messages into a summary
	Summarize(ctx context.Context, messages []simpleai.Message) (string, error)
}

// TokenCounter counts tokens in text
type TokenCounter interface {
	// Count returns the token count for text
	Count(text string) int
}

// DefaultTokenCounter provides a simple character-based estimation
type DefaultTokenCounter struct{}

// Count estimates tokens as ~4 characters per token
func (d *DefaultTokenCounter) Count(text string) int {
	return len(text) / 4
}

// MemoryConfig holds configuration for memory implementations
type MemoryConfig struct {
	// MaxTokens is the maximum tokens to keep in history
	MaxTokens int

	// MaxMessages is the maximum messages to keep (0 = unlimited)
	MaxMessages int

	// SummarizeAfter triggers summarization after this many messages
	SummarizeAfter int

	// TokenCounter for counting tokens
	TokenCounter TokenCounter
}

// DefaultMemoryConfig returns sensible defaults
func DefaultMemoryConfig() MemoryConfig {
	return MemoryConfig{
		MaxTokens:      4000,
		MaxMessages:    100,
		SummarizeAfter: 0, // disabled by default
		TokenCounter:   &DefaultTokenCounter{},
	}
}
