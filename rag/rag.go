package rag

import (
	"context"

	"github.com/medatechnology/simpleai"
	"github.com/medatechnology/simpleai/embedding"
)

// VectorStore stores and retrieves documents by vector similarity
type VectorStore interface {
	// Add adds a document with its embedding
	Add(ctx context.Context, doc embedding.Document) error

	// AddBatch adds multiple documents
	AddBatch(ctx context.Context, docs []embedding.Document) error

	// Search finds the top-k most similar documents
	Search(ctx context.Context, queryEmbedding []float64, topK int) ([]SearchResult, error)

	// Delete removes a document by ID
	Delete(ctx context.Context, id string) error

	// Clear removes all documents
	Clear(ctx context.Context) error

	// Count returns the number of documents
	Count() int
}

// SearchResult represents a search result with similarity score
type SearchResult struct {
	Document   embedding.Document
	Similarity float64
}

// RAG provides retrieval-augmented generation capabilities
type RAG struct {
	embedder embedding.Embedder
	store    VectorStore
	config   Config
}

// Config holds RAG configuration
type Config struct {
	// TopK is the number of documents to retrieve
	TopK int

	// MinSimilarity is the minimum similarity threshold
	MinSimilarity float64

	// MaxTokens is the maximum tokens for retrieved context
	MaxTokens int

	// IncludeMetadata includes document metadata in context
	IncludeMetadata bool
}

// DefaultConfig returns sensible defaults
func DefaultConfig() Config {
	return Config{
		TopK:            5,
		MinSimilarity:   0.7,
		MaxTokens:       2000,
		IncludeMetadata: false,
	}
}

// New creates a new RAG instance
func New(embedder embedding.Embedder, store VectorStore, config Config) *RAG {
	if config.TopK == 0 {
		config.TopK = 5
	}
	return &RAG{
		embedder: embedder,
		store:    store,
		config:   config,
	}
}

// AddMessage adds a message to the RAG store
func (r *RAG) AddMessage(ctx context.Context, msg simpleai.Message, id string) error {
	emb, err := r.embedder.Embed(ctx, msg.Content)
	if err != nil {
		return err
	}

	doc := embedding.Document{
		ID:        id,
		Content:   msg.Content,
		Embedding: emb,
		Metadata: map[string]any{
			"role": string(msg.Role),
		},
	}

	return r.store.Add(ctx, doc)
}

// Retrieve finds relevant messages for a query
func (r *RAG) Retrieve(ctx context.Context, query string) ([]simpleai.Message, error) {
	// Generate query embedding
	queryEmb, err := r.embedder.Embed(ctx, query)
	if err != nil {
		return nil, err
	}

	// Search for similar documents
	results, err := r.store.Search(ctx, queryEmb, r.config.TopK)
	if err != nil {
		return nil, err
	}

	// Convert to messages
	var messages []simpleai.Message
	for _, result := range results {
		if result.Similarity < r.config.MinSimilarity {
			continue
		}

		role := simpleai.RoleUser
		if roleStr, ok := result.Document.Metadata["role"].(string); ok {
			role = simpleai.Role(roleStr)
		}

		messages = append(messages, simpleai.Message{
			Role:    role,
			Content: result.Document.Content,
		})
	}

	return messages, nil
}

// BuildContext builds context from retrieved messages
func (r *RAG) BuildContext(ctx context.Context, query string) (string, error) {
	messages, err := r.Retrieve(ctx, query)
	if err != nil {
		return "", err
	}

	if len(messages) == 0 {
		return "", nil
	}

	var context string
	context = "[Relevant context from previous conversations]\n"
	for _, msg := range messages {
		context += msg.Content + "\n---\n"
	}

	return context, nil
}

// Store returns the underlying vector store
func (r *RAG) Store() VectorStore {
	return r.store
}

// Embedder returns the underlying embedder
func (r *RAG) Embedder() embedding.Embedder {
	return r.embedder
}
