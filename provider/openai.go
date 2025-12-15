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
	OpenAIDefaultBaseURL = "https://api.openai.com"
	OpenAIDefaultModel   = "gpt-4o"
)

// OpenAIConfig holds configuration for the OpenAI provider
type OpenAIConfig struct {
	APIKey       string
	BaseURL      string
	Model        string
	MaxTokens    int
	Temperature  float64
	TopP         float64
	Organization string
}

// OpenAI implements the Provider interface for OpenAI's GPT models
type OpenAI struct {
	config OpenAIConfig
	client medahttp.HttpClient
}

// NewOpenAI creates a new OpenAI provider
func NewOpenAI(config OpenAIConfig) *OpenAI {
	if config.BaseURL == "" {
		config.BaseURL = OpenAIDefaultBaseURL
	}
	if config.Model == "" {
		config.Model = OpenAIDefaultModel
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
	if config.Organization != "" {
		headers["OpenAI-Organization"] = []string{config.Organization}
	}

	client := medahttp.NewHttp()
	client.SetHeader(headers)

	return &OpenAI{
		config: config,
		client: client,
	}
}

// Name returns the provider name
func (o *OpenAI) Name() string {
	return "openai"
}

// Complete sends a completion request to OpenAI
func (o *OpenAI) Complete(ctx context.Context, req *simpleai.Request) (*simpleai.Response, error) {
	openaiReq := o.buildRequest(req)

	var openaiResp openaiResponse
	statusCode, err := o.client.Post(
		o.config.BaseURL+"/v1/chat/completions",
		openaiReq,
		&openaiResp,
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if statusCode != 200 {
		return nil, simpleai.NewProviderError(
			"openai",
			int(statusCode),
			"request failed",
			"http_error",
		)
	}

	return o.parseResponse(&openaiResp), nil
}

// Stream sends a streaming completion request
func (o *OpenAI) Stream(ctx context.Context, req *simpleai.Request) (<-chan simpleai.StreamEvent, error) {
	openaiReq := o.buildRequest(req)
	openaiReq.Stream = true

	// Use goutil PostStream for raw response access
	resp, err := o.client.PostStream(o.config.BaseURL+"/v1/chat/completions", openaiReq)
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
func (o *OpenAI) CountTokens(text string) int {
	return len(text) / 4
}

// Internal types for OpenAI API
type openaiRequest struct {
	Model       string          `json:"model"`
	Messages    []openaiMessage `json:"messages"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Temperature float64         `json:"temperature,omitempty"`
	TopP        float64         `json:"top_p,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
	Stop        []string        `json:"stop,omitempty"`
}

type openaiMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openaiResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []openaiChoice `json:"choices"`
	Usage   openaiUsage    `json:"usage"`
}

type openaiChoice struct {
	Index        int           `json:"index"`
	Message      openaiMessage `json:"message"`
	Delta        openaiMessage `json:"delta"`
	FinishReason string        `json:"finish_reason"`
}

type openaiUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type openaiErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

func (o *OpenAI) buildRequest(req *simpleai.Request) *openaiRequest {
	messages := make([]openaiMessage, 0, len(req.Messages)+1)

	if req.SystemPrompt != "" {
		messages = append(messages, openaiMessage{
			Role:    "system",
			Content: req.SystemPrompt,
		})
	}

	for _, msg := range req.Messages {
		messages = append(messages, openaiMessage{
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

	return &openaiRequest{
		Model:       model,
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: temp,
		TopP:        req.TopP,
		Stop:        req.Stop,
	}
}

func (o *OpenAI) handleError(resp *http.Response) error {
	body, _ := io.ReadAll(resp.Body)

	var errResp openaiErrorResponse
	if err := json.Unmarshal(body, &errResp); err == nil && errResp.Error.Message != "" {
		return simpleai.NewProviderError(
			"openai",
			resp.StatusCode,
			errResp.Error.Message,
			errResp.Error.Type,
		)
	}

	return simpleai.NewProviderError(
		"openai",
		resp.StatusCode,
		string(body),
		"unknown",
	)
}

func (o *OpenAI) parseResponse(resp *openaiResponse) *simpleai.Response {
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

func (o *OpenAI) streamResponse(ctx context.Context, body io.ReadCloser, out chan<- simpleai.StreamEvent) {
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

		var resp openaiResponse
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
