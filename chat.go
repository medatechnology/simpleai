package simpleai

import (
	"context"
	"sync"
)

// AutocompactConfig configures automatic conversation compaction
type AutocompactConfig struct {
	// Threshold is the message count that triggers compaction
	Threshold int
	// KeepRecent is how many recent messages to preserve (not summarized)
	KeepRecent int
	// Summarizer is an optional custom summarizer (uses memory.AISummarizer by default)
	// If nil, uses the chat client's provider for summarization
	Summarizer Summarizer
}

// Summarizer can summarize conversation history (mirrors memory.Summarizer)
type Summarizer interface {
	Summarize(ctx context.Context, messages []Message) (string, error)
}

// DefaultAutocompactConfig returns sensible defaults for autocompact
func DefaultAutocompactConfig() AutocompactConfig {
	return AutocompactConfig{
		Threshold:  20,
		KeepRecent: 4,
	}
}

// Chat represents a conversation session with an AI provider
type Chat struct {
	client       *Client
	system       string
	history      []Message
	historyLimit int
	maxTokens    int
	tokenCounter func(string) int
	mu           sync.RWMutex

	// Autocompact fields
	autocompact       *AutocompactConfig
	conversationSummary string // Accumulated summary from compacted messages
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
	messages := make([]Message, 0, len(c.history)+2)

	// Add system message if present (for providers that need it in messages)
	if c.system != "" {
		systemContent := c.system
		// Append conversation summary to system prompt if available
		if c.conversationSummary != "" {
			systemContent += "\n\n[Previous conversation summary: " + c.conversationSummary + "]"
		}
		messages = append(messages, Message{
			Role:    RoleSystem,
			Content: systemContent,
		})
	} else if c.conversationSummary != "" {
		// If no system prompt but we have a summary, add it as a system message
		messages = append(messages, Message{
			Role:    RoleSystem,
			Content: "[Previous conversation summary: " + c.conversationSummary + "]",
		})
	}

	// Add conversation history
	messages = append(messages, c.history...)

	return messages
}

// trimHistory removes old messages if over the limit
func (c *Chat) trimHistory() {
	// Check if autocompact should be triggered
	if c.autocompact != nil && len(c.history) >= c.autocompact.Threshold {
		c.compactHistory()
		return
	}

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

// compactHistory summarizes old messages and keeps only recent ones
func (c *Chat) compactHistory() {
	if c.autocompact == nil || len(c.history) < c.autocompact.Threshold {
		return
	}

	keepRecent := c.autocompact.KeepRecent
	if keepRecent >= len(c.history) {
		return // Nothing to compact
	}

	// Split history into old (to summarize) and recent (to keep)
	oldMessages := c.history[:len(c.history)-keepRecent]
	recentMessages := c.history[len(c.history)-keepRecent:]

	var summaryContent string
	var err error

	// Unlock before making AI call to avoid deadlock
	c.mu.Unlock()

	// Use custom summarizer if provided, otherwise use default AI summarization
	if c.autocompact.Summarizer != nil {
		summaryContent, err = c.autocompact.Summarizer.Summarize(context.Background(), oldMessages)
	} else {
		// Default: use chat client's provider for summarization
		var conversationText string
		for _, msg := range oldMessages {
			conversationText += string(msg.Role) + ": " + msg.Content + "\n\n"
		}

		summaryReq := &Request{
			Messages: []Message{
				{
					Role:    RoleUser,
					Content: "Summarize this conversation concisely, preserving key information:\n\n" + conversationText,
				},
			},
			MaxTokens:   500,
			Temperature: 0.3,
		}

		summaryResp, reqErr := c.client.Complete(context.Background(), summaryReq)
		if reqErr != nil {
			err = reqErr
		} else {
			summaryContent = summaryResp.Content
		}
	}

	// Relock after AI call
	c.mu.Lock()

	if err != nil {
		// If summarization fails, just trim normally
		c.history = recentMessages
		return
	}

	// Append new summary to existing summary
	if c.conversationSummary != "" {
		c.conversationSummary = c.conversationSummary + "\n\n" + summaryContent
	} else {
		c.conversationSummary = summaryContent
	}

	// Keep only recent messages
	c.history = recentMessages
}

// Summary returns the current conversation summary
func (c *Chat) Summary() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.conversationSummary
}

