package embedding

import (
	"context"
	"fmt"
	"net/http"

	medahttp "github.com/medatechnology/goutil/http"
)

const (
	OllamaDefaultURL   = "http://localhost:11434"
	OllamaDefaultModel = "nomic-embed-text"
)

// OllamaConfig holds configuration for Ollama embeddings
type OllamaConfig struct {
	BaseURL string
	Model   string
}

// Ollama implements Embedder using Ollama's local embedding API
type Ollama struct {
	config     OllamaConfig
	client     medahttp.HttpClient
	dimensions int
}

// NewOllama creates a new Ollama embedder
func NewOllama(config OllamaConfig) *Ollama {
	if config.BaseURL == "" {
		config.BaseURL = OllamaDefaultURL
	}
	if config.Model == "" {
		config.Model = OllamaDefaultModel
	}

	client := medahttp.NewHttp()
	client.SetHeader(map[string][]string{
		"Content-Type": {"application/json"},
	})

	return &Ollama{
		config:     config,
		client:     client,
		dimensions: 768, // nomic-embed-text default
	}
}

// Embed generates an embedding for a single text
func (o *Ollama) Embed(ctx context.Context, text string) ([]float64, error) {
	req := ollamaEmbeddingRequest{
		Model:  o.config.Model,
		Prompt: text,
	}

	var result ollamaEmbeddingResponse
	statusCode, err := o.client.Post(o.config.BaseURL+"/api/embeddings", req, &result, nil)
	if err != nil {
		return nil, fmt.Errorf("embedding request failed: %w", err)
	}

	if statusCode != http.StatusOK {
		return nil, fmt.Errorf("embedding request failed with status %d", statusCode)
	}

	// Update dimensions based on actual response
	if len(result.Embedding) > 0 {
		o.dimensions = len(result.Embedding)
	}

	return result.Embedding, nil
}

// EmbedBatch generates embeddings for multiple texts
func (o *Ollama) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	embeddings := make([][]float64, len(texts))
	for i, text := range texts {
		emb, err := o.Embed(ctx, text)
		if err != nil {
			return nil, fmt.Errorf("failed to embed text %d: %w", i, err)
		}
		embeddings[i] = emb
	}
	return embeddings, nil
}

// Dimensions returns the embedding vector size
func (o *Ollama) Dimensions() int {
	return o.dimensions
}

// Name returns the embedder name
func (o *Ollama) Name() string {
	return "ollama"
}

type ollamaEmbeddingRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type ollamaEmbeddingResponse struct {
	Embedding []float64 `json:"embedding"`
}
