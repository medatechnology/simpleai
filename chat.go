package simpleai

import (
	"context"
	"sync"
)

// Chat represents a conversation session with an AI provider
type Chat struct {
	client       *Client
	system       string
	history      []Message
	historyLimit int
	maxTokens    int
	tokenCounter func(string) int
	mu           sync.RWMutex
}

// NewChat creates a new chat session
func NewChat(client *Client, opts ...ChatOption) *Chat {
	c := &Chat{
		client:       client,
		history:      []Message{},
		historyLimit: 100, // default limit
	}

	for _, opt := range opts {
		opt(c)
	}

	return c
}

// Send sends a user message and returns the assistant's response
func (c *Chat) Send(ctx context.Context, message string) (*Response, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Add user message to history
	c.history = append(c.history, Message{
		Role:    RoleUser,
		Content: message,
	})

	// Build request with full history
	req := &Request{
		Messages:     c.buildMessages(),
		SystemPrompt: c.system,
	}

	// Send to provider
	resp, err := c.client.Complete(ctx, req)
	if err != nil {
		// Remove the user message on error
		c.history = c.history[:len(c.history)-1]
		return nil, err
	}

	// Add assistant response to history
	c.history = append(c.history, Message{
		Role:    RoleAssistant,
		Content: resp.Content,
	})

	// Trim history if needed
	c.trimHistory()

	return resp, nil
}

// Stream sends a user message and streams the response
func (c *Chat) Stream(ctx context.Context, message string) (<-chan StreamEvent, error) {
	c.mu.Lock()

	// Add user message to history
	c.history = append(c.history, Message{
		Role:    RoleUser,
		Content: message,
	})

	// Build request
	req := &Request{
		Messages:     c.buildMessages(),
		SystemPrompt: c.system,
		Stream:       true,
	}

	c.mu.Unlock()

	// Get stream from provider
	stream, err := c.client.Stream(ctx, req)
	if err != nil {
		c.mu.Lock()
		c.history = c.history[:len(c.history)-1]
		c.mu.Unlock()
		return nil, err
	}

	// Create output channel that accumulates the response
	out := make(chan StreamEvent)
	go func() {
		defer close(out)
		var fullContent string

		for event := range stream {
			fullContent += event.Content
			out <- event

			if event.Done {
				// Add complete response to history
				c.mu.Lock()
				c.history = append(c.history, Message{
					Role:    RoleAssistant,
					Content: fullContent,
				})
				c.trimHistory()
				c.mu.Unlock()
			}
		}
	}()

	return out, nil
}

// History returns a copy of the conversation history
func (c *Chat) History() []Message {
	c.mu.RLock()
	defer c.mu.RUnlock()

	result := make([]Message, len(c.history))
	copy(result, c.history)
	return result
}

// Clear clears the conversation history
func (c *Chat) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.history = []Message{}
}

// SetSystem updates the system prompt
func (c *Chat) SetSystem(prompt string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.system = prompt
}

// System returns the current system prompt
func (c *Chat) System() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.system
}

// buildMessages constructs the message list for the request
func (c *Chat) buildMessages() []Message {
	messages := make([]Message, 0, len(c.history)+1)

	// Add system message if present (for providers that need it in messages)
	if c.system != "" {
		messages = append(messages, Message{
			Role:    RoleSystem,
			Content: c.system,
		})
	}

	// Add conversation history
	messages = append(messages, c.history...)

	return messages
}

// trimHistory removes old messages if over the limit
func (c *Chat) trimHistory() {
	// Trim by message count
	if c.historyLimit > 0 && len(c.history) > c.historyLimit {
		excess := len(c.history) - c.historyLimit
		c.history = c.history[excess:]
	}

	// Trim by token count
	if c.maxTokens > 0 && c.tokenCounter != nil {
		for c.countHistoryTokens() > c.maxTokens && len(c.history) > 1 {
			c.history = c.history[1:]
		}
	}
}

// countHistoryTokens returns the total tokens in history
func (c *Chat) countHistoryTokens() int {
	if c.tokenCounter == nil {
		return 0
	}
	total := 0
	for _, msg := range c.history {
		total += c.tokenCounter(msg.Content)
	}
	return total
}
