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
	MistralDefaultBaseURL = "https://api.mistral.ai"
	MistralDefaultModel   = "mistral-large-latest"
)

// MistralConfig holds configuration for the Mistral provider
type MistralConfig struct {
	APIKey      string
	BaseURL     string
	Model       string
	MaxTokens   int
	Temperature float64
	TopP        float64
	SafePrompt  bool // Enable Mistral's safety prompt
}

// Mistral implements the Provider interface for Mistral AI models
type Mistral struct {
	config MistralConfig
	client medahttp.HttpClient
}

// NewMistral creates a new Mistral provider
func NewMistral(config MistralConfig) *Mistral {
	if config.BaseURL == "" {
		config.BaseURL = MistralDefaultBaseURL
	}
	if config.Model == "" {
		config.Model = MistralDefaultModel
	}
	if config.MaxTokens == 0 {
		config.MaxTokens = 4096
	}
	if config.Temperature == 0 {
		config.Temperature = 0.7
	}

	headers := map[string][]string{
		"Content-Type":  {"application/json"},
		"Authorization": {"Bearer " + config.APIKey},
	}

	client := medahttp.NewHttp()
	client.SetHeader(headers)

	return &Mistral{
		config: config,
		client: client,
	}
}

// NewMistralFromEnv creates a Mistral provider from environment variables
// Environment variables: MISTRAL_API_KEY, MISTRAL_MODEL (optional)
func NewMistralFromEnv() *Mistral {
	return NewMistral(MistralConfig{
		APIKey: utils.GetEnvString("MISTRAL_API_KEY", ""),
		Model:  utils.GetEnvString("MISTRAL_MODEL", MistralDefaultModel),
	})
}

// Name returns the provider name
func (m *Mistral) Name() string {
	return "mistral"
}

// Complete sends a completion request to Mistral
func (m *Mistral) Complete(ctx context.Context, req *simpleai.Request) (*simpleai.Response, error) {
	mistralReq := m.buildRequest(req)

	var mistralResp mistralResponse
	statusCode, err := m.client.Post(
		m.config.BaseURL+"/v1/chat/completions",
		mistralReq,
		&mistralResp,
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if statusCode != 200 {
		return nil, simpleai.NewProviderError(
			"mistral",
			int(statusCode),
			"request failed",
			"http_error",
		)
	}

	return m.parseResponse(&mistralResp), nil
}

// Stream sends a streaming completion request
func (m *Mistral) Stream(ctx context.Context, req *simpleai.Request) (<-chan simpleai.StreamEvent, error) {
	mistralReq := m.buildRequest(req)
	mistralReq.Stream = true

	// Use goutil PostStream for raw response access
	resp, err := m.client.PostStream(m.config.BaseURL+"/v1/chat/completions", mistralReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, m.handleError(resp)
	}

	out := make(chan simpleai.StreamEvent)
	go m.streamResponse(ctx, resp.Body, out)

	return out, nil
}

// CountTokens estimates token count
func (m *Mistral) CountTokens(text string) int {
	return len(text) / 4
}

// Internal types for Mistral API (OpenAI-compatible format)
type mistralRequest struct {
	Model       string           `json:"model"`
	Messages    []mistralMessage `json:"messages"`
	MaxTokens   int              `json:"max_tokens,omitempty"`
	Temperature float64          `json:"temperature,omitempty"`
	TopP        float64          `json:"top_p,omitempty"`
	Stream      bool             `json:"stream,omitempty"`
	SafePrompt  bool             `json:"safe_prompt,omitempty"`
	RandomSeed  int              `json:"random_seed,omitempty"`
}

type mistralMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type mistralResponse struct {
	ID      string          `json:"id"`
	Object  string          `json:"object"`
	Created int64           `json:"created"`
	Model   string          `json:"model"`
	Choices []mistralChoice `json:"choices"`
	Usage   mistralUsage    `json:"usage"`
}

type mistralChoice struct {
	Index        int            `json:"index"`
	Message      mistralMessage `json:"message"`
	Delta        mistralMessage `json:"delta"`
	FinishReason string         `json:"finish_reason"`
}

type mistralUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type mistralErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

func (m *Mistral) buildRequest(req *simpleai.Request) *mistralRequest {
	messages := make([]mistralMessage, 0, len(req.Messages)+1)

	if req.SystemPrompt != "" {
		messages = append(messages, mistralMessage{
			Role:    "system",
			Content: req.SystemPrompt,
		})
	}

	for _, msg := range req.Messages {
		messages = append(messages, mistralMessage{
			Role:    string(msg.Role),
			Content: msg.Content,
		})
	}

	model := req.Model
	if model == "" {
		model = m.config.Model
	}

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = m.config.MaxTokens
	}

	temp := req.Temperature
	if temp == 0 {
		temp = m.config.Temperature
	}

	return &mistralRequest{
		Model:       model,
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: temp,
		TopP:        req.TopP,
		SafePrompt:  m.config.SafePrompt,
	}
}

func (m *Mistral) handleError(resp *http.Response) error {
	body, _ := io.ReadAll(resp.Body)

	var errResp mistralErrorResponse
	if err := json.Unmarshal(body, &errResp); err == nil && errResp.Error.Message != "" {
		return simpleai.NewProviderError(
			"mistral",
			resp.StatusCode,
			errResp.Error.Message,
			errResp.Error.Type,
		)
	}

	return simpleai.NewProviderError(
		"mistral",
		resp.StatusCode,
		string(body),
		"unknown",
	)
}

func (m *Mistral) parseResponse(resp *mistralResponse) *simpleai.Response {
	var content string
	var finishReason string

	if len(resp.Choices) > 0 {
		content = resp.Choices[0].Message.Content
		finishReason = resp.Choices[0].FinishReason
	}

	return &simpleai.Response{
		Content:      content,
		Model:        resp.Model,
		FinishReason: finishReason,
		Usage: simpleai.Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}
}

func (m *Mistral) streamResponse(ctx context.Context, body io.ReadCloser, out chan<- simpleai.StreamEvent) {
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

		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			out <- simpleai.StreamEvent{Done: true}
			return
		}

		var resp mistralResponse
		if err := json.Unmarshal([]byte(data), &resp); err != nil {
			continue
		}

		if len(resp.Choices) > 0 {
			choice := resp.Choices[0]
			if choice.Delta.Content != "" {
				out <- simpleai.StreamEvent{Content: choice.Delta.Content}
			}
			if choice.FinishReason != "" {
				out <- simpleai.StreamEvent{
					Done:         true,
					FinishReason: choice.FinishReason,
				}
				return
			}
		}
	}

	if err := scanner.Err(); err != nil {
		out <- simpleai.StreamEvent{Error: err, Done: true}
	}
}
