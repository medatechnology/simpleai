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
	"github.com/medatechnology/simpleai"
)

const (
	AnthropicDefaultBaseURL = "https://api.anthropic.com"
	AnthropicDefaultModel   = "claude-3-5-sonnet-20241022"
	AnthropicAPIVersion     = "2023-06-01"
)

// AnthropicConfig holds configuration for the Anthropic provider
type AnthropicConfig struct {
	APIKey      string
	BaseURL     string
	Model       string
	MaxTokens   int
	Temperature float64
	TopP        float64
}

// Anthropic implements the Provider interface for Anthropic's Claude
type Anthropic struct {
	config AnthropicConfig
	client medahttp.HttpClient
}

// NewAnthropic creates a new Anthropic provider
func NewAnthropic(config AnthropicConfig) *Anthropic {
	if config.BaseURL == "" {
		config.BaseURL = AnthropicDefaultBaseURL
	}
	if config.Model == "" {
		config.Model = AnthropicDefaultModel
	}
	if config.MaxTokens == 0 {
		config.MaxTokens = 4096
	}
	if config.Temperature == 0 {
		config.Temperature = 0.7
	}

	client := medahttp.NewHttp()
	client.SetHeader(map[string][]string{
		"Content-Type":      {"application/json"},
		"x-api-key":         {config.APIKey},
		"anthropic-version": {AnthropicAPIVersion},
	})

	return &Anthropic{
		config: config,
		client: client,
	}
}

// Name returns the provider name
func (a *Anthropic) Name() string {
	return "anthropic"
}

// Complete sends a completion request to Anthropic
func (a *Anthropic) Complete(ctx context.Context, req *simpleai.Request) (*simpleai.Response, error) {
	anthropicReq := a.buildRequest(req)

	var anthropicResp anthropicResponse
	statusCode, err := a.client.Post(
		a.config.BaseURL+"/v1/messages",
		anthropicReq,
		&anthropicResp,
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if statusCode != 200 {
		return nil, simpleai.NewProviderError(
			"anthropic",
			int(statusCode),
			"request failed",
			"http_error",
		)
	}

	return a.parseResponse(&anthropicResp), nil
}

// Stream sends a streaming completion request
func (a *Anthropic) Stream(ctx context.Context, req *simpleai.Request) (<-chan simpleai.StreamEvent, error) {
	anthropicReq := a.buildRequest(req)
	anthropicReq.Stream = true

	// Use goutil PostStream for raw response access
	resp, err := a.client.PostStream(a.config.BaseURL+"/v1/messages", anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, a.handleError(resp)
	}

	out := make(chan simpleai.StreamEvent)
	go a.streamResponse(ctx, resp.Body, out)

	return out, nil
}

// CountTokens estimates token count (approximate)
func (a *Anthropic) CountTokens(text string) int {
	// Rough estimate: ~4 chars per token for English
	return len(text) / 4
}

// Internal types for Anthropic API
type anthropicRequest struct {
	Model       string             `json:"model"`
	Messages    []anthropicMessage `json:"messages"`
	System      string             `json:"system,omitempty"`
	MaxTokens   int                `json:"max_tokens"`
	Temperature float64            `json:"temperature,omitempty"`
	TopP        float64            `json:"top_p,omitempty"`
	Stream      bool               `json:"stream,omitempty"`
	Stop        []string           `json:"stop_sequences,omitempty"`
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type anthropicResponse struct {
	ID           string                  `json:"id"`
	Type         string                  `json:"type"`
	Role         string                  `json:"role"`
	Content      []anthropicContentBlock `json:"content"`
	Model        string                  `json:"model"`
	StopReason   string                  `json:"stop_reason"`
	StopSequence string                  `json:"stop_sequence"`
	Usage        anthropicUsage          `json:"usage"`
}

type anthropicContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type anthropicErrorResponse struct {
	Type  string `json:"type"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// Streaming event types
type anthropicStreamEvent struct {
	Type         string                 `json:"type"`
	Index        int                    `json:"index,omitempty"`
	ContentBlock *anthropicContentBlock `json:"content_block,omitempty"`
	Delta        *anthropicDelta        `json:"delta,omitempty"`
	Message      *anthropicResponse     `json:"message,omitempty"`
	Usage        *anthropicUsage        `json:"usage,omitempty"`
}

type anthropicDelta struct {
	Type       string `json:"type"`
	Text       string `json:"text"`
	StopReason string `json:"stop_reason,omitempty"`
}

func (a *Anthropic) buildRequest(req *simpleai.Request) *anthropicRequest {
	messages := make([]anthropicMessage, 0, len(req.Messages))
	var systemPrompt string

	for _, msg := range req.Messages {
		if msg.Role == simpleai.RoleSystem {
			systemPrompt = msg.Content
			continue
		}
		messages = append(messages, anthropicMessage{
			Role:    string(msg.Role),
			Content: msg.Content,
		})
	}

	// Use request system prompt if provided, otherwise use extracted
	if req.SystemPrompt != "" {
		systemPrompt = req.SystemPrompt
	}

	model := req.Model
	if model == "" {
		model = a.config.Model
	}

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = a.config.MaxTokens
	}

	temp := req.Temperature
	if temp == 0 {
		temp = a.config.Temperature
	}

	return &anthropicRequest{
		Model:       model,
		Messages:    messages,
		System:      systemPrompt,
		MaxTokens:   maxTokens,
		Temperature: temp,
		TopP:        req.TopP,
		Stop:        req.Stop,
	}
}

func (a *Anthropic) handleError(resp *http.Response) error {
	body, _ := io.ReadAll(resp.Body)

	var errResp anthropicErrorResponse
	if err := json.Unmarshal(body, &errResp); err == nil && errResp.Error.Message != "" {
		return simpleai.NewProviderError(
			"anthropic",
			resp.StatusCode,
			errResp.Error.Message,
			errResp.Error.Type,
		)
	}

	return simpleai.NewProviderError(
		"anthropic",
		resp.StatusCode,
		string(body),
		"unknown",
	)
}

func (a *Anthropic) parseResponse(resp *anthropicResponse) *simpleai.Response {
	var content string
	for _, block := range resp.Content {
		if block.Type == "text" {
			content += block.Text
		}
	}

	return &simpleai.Response{
		Content:      content,
		Model:        resp.Model,
		FinishReason: resp.StopReason,
		Usage: simpleai.Usage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
	}
}

func (a *Anthropic) streamResponse(ctx context.Context, body io.ReadCloser, out chan<- simpleai.StreamEvent) {
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

		var event anthropicStreamEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		switch event.Type {
		case "content_block_delta":
			if event.Delta != nil && event.Delta.Text != "" {
				out <- simpleai.StreamEvent{Content: event.Delta.Text}
			}
		case "message_delta":
			if event.Delta != nil && event.Delta.StopReason != "" {
				out <- simpleai.StreamEvent{
					Done:         true,
					FinishReason: event.Delta.StopReason,
				}
				return
			}
		case "message_stop":
			out <- simpleai.StreamEvent{Done: true}
			return
		}
	}

	if err := scanner.Err(); err != nil {
		out <- simpleai.StreamEvent{Error: err, Done: true}
	}
}
