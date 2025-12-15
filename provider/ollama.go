package provider

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	medahttp "github.com/medatechnology/goutil/http"
	"github.com/medatechnology/goutil/utils"
	"github.com/medatechnology/simpleai"
)

const (
	OllamaDefaultBaseURL = "http://localhost:11434"
	OllamaDefaultModel   = "llama3.2"
)

// OllamaConfig holds configuration for the Ollama provider
type OllamaConfig struct {
	BaseURL     string
	Model       string
	MaxTokens   int
	Temperature float64
	TopP        float64
}

// Ollama implements the Provider interface for local Ollama models
type Ollama struct {
	config OllamaConfig
	client medahttp.HttpClient
}

// NewOllama creates a new Ollama provider
func NewOllama(config OllamaConfig) *Ollama {
	if config.BaseURL == "" {
		config.BaseURL = OllamaDefaultBaseURL
	}
	if config.Model == "" {
		config.Model = OllamaDefaultModel
	}
	if config.MaxTokens == 0 {
		config.MaxTokens = 4096
	}
	if config.Temperature == 0 {
		config.Temperature = 0.7
	}

	client := medahttp.NewHttp()
	client.SetHeader(map[string][]string{
		"Content-Type": {"application/json"},
	})

	return &Ollama{
		config: config,
		client: client,
	}
}

// NewOllamaFromEnv creates an Ollama provider from environment variables
// Environment variables: OLLAMA_BASE_URL (optional), OLLAMA_MODEL (optional)
func NewOllamaFromEnv() *Ollama {
	return NewOllama(OllamaConfig{
		BaseURL: utils.GetEnvString("OLLAMA_BASE_URL", OllamaDefaultBaseURL),
		Model:   utils.GetEnvString("OLLAMA_MODEL", OllamaDefaultModel),
	})
}

// Name returns the provider name
func (o *Ollama) Name() string {
	return "ollama"
}

// Complete sends a completion request to Ollama
func (o *Ollama) Complete(ctx context.Context, req *simpleai.Request) (*simpleai.Response, error) {
	ollamaReq := o.buildRequest(req, false)

	var ollamaResp ollamaResponse
	statusCode, err := o.client.Post(
		o.config.BaseURL+"/api/chat",
		ollamaReq,
		&ollamaResp,
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if statusCode != 200 {
		return nil, simpleai.NewProviderError(
			"ollama",
			int(statusCode),
			"request failed",
			"http_error",
		)
	}

	return o.parseResponse(&ollamaResp), nil
}

// Stream sends a streaming completion request
func (o *Ollama) Stream(ctx context.Context, req *simpleai.Request) (<-chan simpleai.StreamEvent, error) {
	ollamaReq := o.buildRequest(req, true)

	// Use goutil PostStream for raw response access
	resp, err := o.client.PostStream(o.config.BaseURL+"/api/chat", ollamaReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, o.handleError(resp)
	}

	out := make(chan simpleai.StreamEvent)
	go o.streamResponse(ctx, resp.Body, out)

	return out, nil
}

// CountTokens estimates token count
func (o *Ollama) CountTokens(text string) int {
	return len(text) / 4
}

// Internal types for Ollama API
type ollamaRequest struct {
	Model    string          `json:"model"`
	Messages []ollamaMessage `json:"messages"`
	Stream   bool            `json:"stream"`
	Options  ollamaOptions   `json:"options,omitempty"`
}

type ollamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ollamaOptions struct {
	NumPredict  int      `json:"num_predict,omitempty"`
	Temperature float64  `json:"temperature,omitempty"`
	TopP        float64  `json:"top_p,omitempty"`
	Stop        []string `json:"stop,omitempty"`
}

type ollamaResponse struct {
	Model              string        `json:"model"`
	CreatedAt          string        `json:"created_at"`
	Message            ollamaMessage `json:"message"`
	Done               bool          `json:"done"`
	DoneReason         string        `json:"done_reason"`
	TotalDuration      int64         `json:"total_duration"`
	LoadDuration       int64         `json:"load_duration"`
	PromptEvalCount    int           `json:"prompt_eval_count"`
	PromptEvalDuration int64         `json:"prompt_eval_duration"`
	EvalCount          int           `json:"eval_count"`
	EvalDuration       int64         `json:"eval_duration"`
}

type ollamaErrorResponse struct {
	Error string `json:"error"`
}

func (o *Ollama) buildRequest(req *simpleai.Request, stream bool) *ollamaRequest {
	messages := make([]ollamaMessage, 0, len(req.Messages)+1)

	if req.SystemPrompt != "" {
		messages = append(messages, ollamaMessage{
			Role:    "system",
			Content: req.SystemPrompt,
		})
	}

	for _, msg := range req.Messages {
		messages = append(messages, ollamaMessage{
			Role:    string(msg.Role),
			Content: msg.Content,
		})
	}

	model := req.Model
	if model == "" {
		model = o.config.Model
	}

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = o.config.MaxTokens
	}

	temp := req.Temperature
	if temp == 0 {
		temp = o.config.Temperature
	}

	return &ollamaRequest{
		Model:    model,
		Messages: messages,
		Stream:   stream,
		Options: ollamaOptions{
			NumPredict:  maxTokens,
			Temperature: temp,
			TopP:        req.TopP,
			Stop:        req.Stop,
		},
	}
}

func (o *Ollama) handleError(resp *http.Response) error {
	body, _ := io.ReadAll(resp.Body)

	var errResp ollamaErrorResponse
	if err := json.Unmarshal(body, &errResp); err == nil && errResp.Error != "" {
		return simpleai.NewProviderError(
			"ollama",
			resp.StatusCode,
			errResp.Error,
			"error",
		)
	}

	return simpleai.NewProviderError(
		"ollama",
		resp.StatusCode,
		string(body),
		"unknown",
	)
}

func (o *Ollama) parseResponse(resp *ollamaResponse) *simpleai.Response {
	return &simpleai.Response{
		Content:      resp.Message.Content,
		Model:        resp.Model,
		FinishReason: resp.DoneReason,
		Usage: simpleai.Usage{
			PromptTokens:     resp.PromptEvalCount,
			CompletionTokens: resp.EvalCount,
			TotalTokens:      resp.PromptEvalCount + resp.EvalCount,
		},
	}
}

func (o *Ollama) streamResponse(ctx context.Context, body io.ReadCloser, out chan<- simpleai.StreamEvent) {
	defer close(out)
	defer body.Close()

	scanner := bufio.NewScanner(body)
	for scanner.Scan() {
		select {
		case <-ctx.Done():
			out <- simpleai.StreamEvent{Error: ctx.Err(), Done: true}
			return
		default:
		}

		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var resp ollamaResponse
		if err := json.Unmarshal([]byte(line), &resp); err != nil {
			continue
		}

		if resp.Message.Content != "" {
			out <- simpleai.StreamEvent{Content: resp.Message.Content}
		}

		if resp.Done {
			out <- simpleai.StreamEvent{
				Done:         true,
				FinishReason: resp.DoneReason,
			}
			return
		}
	}

	if err := scanner.Err(); err != nil {
		out <- simpleai.StreamEvent{Error: err, Done: true}
	}
}
