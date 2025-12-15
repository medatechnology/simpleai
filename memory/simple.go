package memory

import (
	"context"
	"sync"

	"github.com/medatechnology/simpleai"
)

// Simple is an in-memory implementation of Memory with token-based limits
type Simple struct {
	messages     []simpleai.Message
	tokenCounts  []int
	totalTokens  int
	config       MemoryConfig
	summarizer   Summarizer
	summary      string
	mu           sync.RWMutex
}

// NewSimple creates a new simple in-memory store
func NewSimple(config MemoryConfig) *Simple {
	if config.TokenCounter == nil {
		config.TokenCounter = &DefaultTokenCounter{}
	}
	return &Simple{
		messages:    []simpleai.Message{},
		tokenCounts: []int{},
		config:      config,
	}
}

// NewSimpleWithSummarizer creates a simple store with auto-summarization
func NewSimpleWithSummarizer(config MemoryConfig, summarizer Summarizer) *Simple {
	s := NewSimple(config)
	s.summarizer = summarizer
	return s
}

// Add adds a message to memory
func (s *Simple) Add(ctx context.Context, msg simpleai.Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Count tokens for this message
	tokenCount := s.config.TokenCounter.Count(msg.Content)

	s.messages = append(s.messages, msg)
	s.tokenCounts = append(s.tokenCounts, tokenCount)
	s.totalTokens += tokenCount

	// Check if we need to summarize
	if s.summarizer != nil && s.config.SummarizeAfter > 0 {
		if len(s.messages) > s.config.SummarizeAfter {
			if err := s.summarizeOldMessages(ctx); err != nil {
				// Log but don't fail
			}
		}
	}

	// Trim if over limits
	s.trimToLimits()

	return nil
}

// GetMessages retrieves messages respecting token limit
func (s *Simple) GetMessages(ctx context.Context, maxTokens int) ([]simpleai.Message, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if maxTokens <= 0 {
		maxTokens = s.config.MaxTokens
	}

	// Build result from most recent, respecting token limit
	var result []simpleai.Message
	tokenCount := 0

	// Include summary if exists
	if s.summary != "" {
		summaryTokens := s.config.TokenCounter.Count(s.summary)
		if summaryTokens < maxTokens {
			result = append(result, simpleai.Message{
				Role:    simpleai.RoleSystem,
				Content: "[Previous conversation summary]\n" + s.summary,
			})
			tokenCount += summaryTokens
		}
	}

	// Add messages from most recent, going backwards
	for i := len(s.messages) - 1; i >= 0; i-- {
		msgTokens := s.tokenCounts[i]
		if tokenCount+msgTokens > maxTokens {
			break
		}
		result = append([]simpleai.Message{s.messages[i]}, result...)
		tokenCount += msgTokens
	}

	return result, nil
}

// GetRelevant is not supported in simple memory (returns all messages)
func (s *Simple) GetRelevant(ctx context.Context, query string, topK int) ([]simpleai.Message, error) {
	return s.GetMessages(ctx, s.config.MaxTokens)
}

// Clear clears all messages
func (s *Simple) Clear(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.messages = []simpleai.Message{}
	s.tokenCounts = []int{}
	s.totalTokens = 0
	s.summary = ""

	return nil
}

// Count returns message count
func (s *Simple) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.messages)
}

// TokenCount returns total tokens
func (s *Simple) TokenCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.totalTokens
}

// trimToLimits removes old messages to stay within limits
func (s *Simple) trimToLimits() {
	// Trim by message count
	if s.config.MaxMessages > 0 && len(s.messages) > s.config.MaxMessages {
		excess := len(s.messages) - s.config.MaxMessages
		for i := 0; i < excess; i++ {
			s.totalTokens -= s.tokenCounts[i]
		}
		s.messages = s.messages[excess:]
		s.tokenCounts = s.tokenCounts[excess:]
	}

	// Trim by token count
	for s.totalTokens > s.config.MaxTokens && len(s.messages) > 0 {
		s.totalTokens -= s.tokenCounts[0]
		s.messages = s.messages[1:]
		s.tokenCounts = s.tokenCounts[1:]
	}
}

// summarizeOldMessages compresses older messages into a summary
func (s *Simple) summarizeOldMessages(ctx context.Context) error {
	if s.summarizer == nil || len(s.messages) <= s.config.SummarizeAfter/2 {
		return nil
	}

	// Take the first half of messages to summarize
	splitPoint := len(s.messages) / 2
	toSummarize := s.messages[:splitPoint]

	summary, err := s.summarizer.Summarize(ctx, toSummarize)
	if err != nil {
		return err
	}

	// Update summary
	if s.summary != "" {
		s.summary = s.summary + "\n\n" + summary
	} else {
		s.summary = summary
	}

	// Remove summarized messages
	for i := 0; i < splitPoint; i++ {
		s.totalTokens -= s.tokenCounts[i]
	}
	s.messages = s.messages[splitPoint:]
	s.tokenCounts = s.tokenCounts[splitPoint:]

	return nil
}

// Summary returns the current summary
func (s *Simple) Summary() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.summary
}
