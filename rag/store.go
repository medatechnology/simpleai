package rag

import (
	"context"
	"sort"
	"sync"

	"github.com/medatechnology/simpleai/embedding"
)

// MemoryStore is an in-memory vector store implementation
type MemoryStore struct {
	documents []embedding.Document
	mu        sync.RWMutex
}

// NewMemoryStore creates a new in-memory vector store
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		documents: []embedding.Document{},
	}
}

// Add adds a document to the store
func (m *MemoryStore) Add(ctx context.Context, doc embedding.Document) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Check for duplicate ID and update if exists
	for i, d := range m.documents {
		if d.ID == doc.ID {
			m.documents[i] = doc
			return nil
		}
	}

	m.documents = append(m.documents, doc)
	return nil
}

// AddBatch adds multiple documents
func (m *MemoryStore) AddBatch(ctx context.Context, docs []embedding.Document) error {
	for _, doc := range docs {
		if err := m.Add(ctx, doc); err != nil {
			return err
		}
	}
	return nil
}

// Search finds the top-k most similar documents
func (m *MemoryStore) Search(ctx context.Context, queryEmbedding []float64, topK int) ([]SearchResult, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(m.documents) == 0 {
		return nil, nil
	}

	// Calculate similarities
	results := make([]SearchResult, 0, len(m.documents))
	for _, doc := range m.documents {
		similarity := embedding.CosineSimilarity(queryEmbedding, doc.Embedding)
		results = append(results, SearchResult{
			Document:   doc,
			Similarity: similarity,
		})
	}

	// Sort by similarity (descending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	// Return top-k
	if topK > len(results) {
		topK = len(results)
	}

	return results[:topK], nil
}

// Delete removes a document by ID
func (m *MemoryStore) Delete(ctx context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	for i, doc := range m.documents {
		if doc.ID == id {
			m.documents = append(m.documents[:i], m.documents[i+1:]...)
			return nil
		}
	}
	return nil
}

// Clear removes all documents
func (m *MemoryStore) Clear(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.documents = []embedding.Document{}
	return nil
}

// Count returns the number of documents
func (m *MemoryStore) Count() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.documents)
}
