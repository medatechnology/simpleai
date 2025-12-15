package simpleai

import (
	"context"
)

// Role represents the role of a message sender
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Message represents a single message in a conversation
type Message struct {
	Role    Role   `json:"role"`
	Content string `json:"content"`
}

// Request represents a completion request to an AI provider
type Request struct {
	Messages     []Message `json:"messages"`
	Model        string    `json:"model,omitempty"`
	MaxTokens    int       `json:"max_tokens,omitempty"`
	Temperature  float64   `json:"temperature,omitempty"`
	TopP         float64   `json:"top_p,omitempty"`
	Stop         []string  `json:"stop,omitempty"`
	Stream       bool      `json:"stream,omitempty"`
	SystemPrompt string    `json:"system_prompt,omitempty"`
}

// Response represents a completion response from an AI provider
type Response struct {
	Content      string `json:"content"`
	Model        string `json:"model"`
	FinishReason string `json:"finish_reason"`
	Usage        Usage  `json:"usage"`
}

// Usage represents token usage statistics
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// StreamEvent represents a streaming response event
type StreamEvent struct {
	Content      string `json:"content"`
	Done         bool   `json:"done"`
	FinishReason string `json:"finish_reason,omitempty"`
	Error        error  `json:"error,omitempty"`
}

// Provider defines the interface for AI providers
type Provider interface {
	// Complete sends a completion request and returns the response
	Complete(ctx context.Context, req *Request) (*Response, error)

	// Stream sends a streaming completion request
	Stream(ctx context.Context, req *Request) (<-chan StreamEvent, error)

	// CountTokens estimates the token count for the given text
	CountTokens(text string) int

	// Name returns the provider name
	Name() string
}

// ProviderConfig holds common configuration for providers
type ProviderConfig struct {
	APIKey      string  `json:"api_key" yaml:"api_key"`
	BaseURL     string  `json:"base_url" yaml:"base_url"`
	Model       string  `json:"model" yaml:"model"`
	MaxTokens   int     `json:"max_tokens" yaml:"max_tokens"`
	Temperature float64 `json:"temperature" yaml:"temperature"`
	TopP        float64 `json:"top_p" yaml:"top_p"`
	Timeout     int     `json:"timeout" yaml:"timeout"` // in seconds
}

// DefaultProviderConfig returns sensible defaults
func DefaultProviderConfig() ProviderConfig {
	return ProviderConfig{
		MaxTokens:   4096,
		Temperature: 0.7,
		TopP:        1.0,
		Timeout:     60,
	}
}
