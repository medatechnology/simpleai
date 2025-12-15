package embedding

import (
	"context"
)

// Embedder generates vector embeddings from text
type Embedder interface {
	// Embed generates an embedding for a single text
	Embed(ctx context.Context, text string) ([]float64, error)

	// EmbedBatch generates embeddings for multiple texts
	EmbedBatch(ctx context.Context, texts []string) ([][]float64, error)

	// Dimensions returns the embedding vector size
	Dimensions() int

	// Name returns the embedder name
	Name() string
}

// Document represents a text with its embedding
type Document struct {
	ID        string
	Content   string
	Embedding []float64
	Metadata  map[string]any
}

// CosineSimilarity calculates the cosine similarity between two vectors
func CosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (sqrt(normA) * sqrt(normB))
}

// sqrt is a simple square root implementation
func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x / 2
	for i := 0; i < 10; i++ {
		z = z - (z*z-x)/(2*z)
	}
	return z
}
