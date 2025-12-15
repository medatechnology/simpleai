package embedding

import (
	"context"
	"fmt"
	"net/http"

	medahttp "github.com/medatechnology/goutil/http"
)

const (
	OpenAIEmbeddingURL   = "https://api.openai.com/v1/embeddings"
	OpenAIDefaultModel   = "text-embedding-3-small"
	OpenAISmallDimension = 1536
)

// OpenAIConfig holds configuration for OpenAI embeddings
type OpenAIConfig struct {
	APIKey string
	Model  string
}

// OpenAI implements Embedder using OpenAI's embedding API
type OpenAI struct {
	config OpenAIConfig
	client medahttp.HttpClient
}

// NewOpenAI creates a new OpenAI embedder
func NewOpenAI(config OpenAIConfig) *OpenAI {
	if config.Model == "" {
		config.Model = OpenAIDefaultModel
	}

	client := medahttp.NewHttp()
	client.SetHeader(map[string][]string{
		"Content-Type":  {"application/json"},
		"Authorization": {"Bearer " + config.APIKey},
	})

	return &OpenAI{
		config: config,
		client: client,
	}
}

// Embed generates an embedding for a single text
func (o *OpenAI) Embed(ctx context.Context, text string) ([]float64, error) {
	embeddings, err := o.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}
	return embeddings[0], nil
}

// EmbedBatch generates embeddings for multiple texts
func (o *OpenAI) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	req := openaiEmbeddingRequest{
		Model: o.config.Model,
		Input: texts,
	}

	var result openaiEmbeddingResponse
	statusCode, err := o.client.Post(OpenAIEmbeddingURL, req, &result, nil)
	if err != nil {
		return nil, fmt.Errorf("embedding request failed: %w", err)
	}

	if statusCode != http.StatusOK {
		return nil, fmt.Errorf("embedding request failed with status %d", statusCode)
	}

	embeddings := make([][]float64, len(result.Data))
	for _, d := range result.Data {
		embeddings[d.Index] = d.Embedding
	}

	return embeddings, nil
}

// Dimensions returns the embedding vector size
func (o *OpenAI) Dimensions() int {
	return OpenAISmallDimension
}

// Name returns the embedder name
func (o *OpenAI) Name() string {
	return "openai"
}

type openaiEmbeddingRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type openaiEmbeddingResponse struct {
	Data  []openaiEmbeddingData `json:"data"`
	Model string                `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

type openaiEmbeddingData struct {
	Index     int       `json:"index"`
	Embedding []float64 `json:"embedding"`
}
