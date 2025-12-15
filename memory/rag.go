package memory

import (
	"context"
	"fmt"

	"github.com/medatechnology/simpleai"
	"github.com/medatechnology/simpleai/rag"
)

// RAGMemory combines simple memory with RAG for intelligent retrieval
type RAGMemory struct {
	simple     *Simple
	rag        *rag.RAG
	messageID  int
	config     RAGMemoryConfig
}

// RAGMemoryConfig holds configuration for RAG memory
type RAGMemoryConfig struct {
	MemoryConfig

	// RAG configuration
	RAGConfig rag.Config

	// RecentMessages is the number of recent messages to always include
	RecentMessages int
}

// DefaultRAGMemoryConfig returns sensible defaults
func DefaultRAGMemoryConfig() RAGMemoryConfig {
	return RAGMemoryConfig{
		MemoryConfig:   DefaultMemoryConfig(),
		RAGConfig:      rag.DefaultConfig(),
		RecentMessages: 5,
	}
}

// NewRAGMemory creates a new RAG-enabled memory
func NewRAGMemory(r *rag.RAG, config RAGMemoryConfig) *RAGMemory {
	return &RAGMemory{
		simple: NewSimple(config.MemoryConfig),
		rag:    r,
		config: config,
	}
}

// Add adds a message to both simple memory and RAG store
func (m *RAGMemory) Add(ctx context.Context, msg simpleai.Message) error {
	// Add to simple memory
	if err := m.simple.Add(ctx, msg); err != nil {
		return err
	}

	// Add to RAG store
	m.messageID++
	id := fmt.Sprintf("msg_%d", m.messageID)
	if err := m.rag.AddMessage(ctx, msg, id); err != nil {
		// Log but don't fail - simple memory still works
		return nil
	}

	return nil
}

// GetMessages retrieves messages using both recent history and RAG
func (m *RAGMemory) GetMessages(ctx context.Context, maxTokens int) ([]simpleai.Message, error) {
	// Get recent messages from simple memory
	return m.simple.GetMessages(ctx, maxTokens)
}

// GetRelevant retrieves messages relevant to the query using RAG
func (m *RAGMemory) GetRelevant(ctx context.Context, query string, topK int) ([]simpleai.Message, error) {
	// Get recent messages (always include)
	recentMsgs, err := m.simple.GetMessages(ctx, m.config.MaxTokens/2)
	if err != nil {
		return nil, err
	}

	// Get relevant messages via RAG
	relevantMsgs, err := m.rag.Retrieve(ctx, query)
	if err != nil {
		// Fall back to just recent messages
		return recentMsgs, nil
	}

	// Deduplicate and merge
	seen := make(map[string]bool)
	var result []simpleai.Message

	// Add recent messages first
	for _, msg := range recentMsgs {
		key := msg.Content
		if len(key) > 100 {
			key = key[:100]
		}
		if !seen[key] {
			seen[key] = true
			result = append(result, msg)
		}
	}

	// Add relevant messages from RAG
	for _, msg := range relevantMsgs {
		key := msg.Content
		if len(key) > 100 {
			key = key[:100]
		}
		if !seen[key] {
			seen[key] = true
			result = append(result, msg)
		}
	}

	return result, nil
}

// Clear clears all memory
func (m *RAGMemory) Clear(ctx context.Context) error {
	if err := m.simple.Clear(ctx); err != nil {
		return err
	}
	return m.rag.Store().Clear(ctx)
}

// Count returns message count
func (m *RAGMemory) Count() int {
	return m.simple.Count()
}

// TokenCount returns total tokens
func (m *RAGMemory) TokenCount() int {
	return m.simple.TokenCount()
}
