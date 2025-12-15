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
	GroqDefaultBaseURL = "https://api.groq.com/openai"
	GroqDefaultModel   = "llama-3.3-70b-versatile"
)

// GroqConfig holds configuration for the Groq provider
type GroqConfig struct {
	APIKey      string
	BaseURL     string
	Model       string
	MaxTokens   int
	Temperature float64
	TopP        float64
}

// Groq implements the Provider interface for Groq's fast inference
type Groq struct {
	config GroqConfig
	client medahttp.HttpClient
}

// NewGroq creates a new Groq provider
func NewGroq(config GroqConfig) *Groq {
	if config.BaseURL == "" {
		config.BaseURL = GroqDefaultBaseURL
	}
	if config.Model == "" {
		config.Model = GroqDefaultModel
	}
	if config.MaxTokens == 0 {
		config.MaxTokens = 4096
	}
	if config.Temperature == 0 {
		config.Temperature = 0.7
	}

	client := medahttp.NewHttp()
	client.SetHeader(map[string][]string{
		"Content-Type":  {"application/json"},
		"Authorization": {"Bearer " + config.APIKey},
	})

	return &Groq{
		config: config,
		client: client,
	}
}

// Name returns the provider name
func (g *Groq) Name() string {
	return "groq"
}

// Complete sends a completion request to Groq
func (g *Groq) Complete(ctx context.Context, req *simpleai.Request) (*simpleai.Response, error) {
	groqReq := g.buildRequest(req)

	var groqResp groqResponse
	statusCode, err := g.client.Post(
		g.config.BaseURL+"/v1/chat/completions",
		groqReq,
		&groqResp,
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if statusCode != 200 {
		return nil, simpleai.NewProviderError(
			"groq",
			int(statusCode),
			"request failed",
			"http_error",
		)
	}

	return g.parseResponse(&groqResp), nil
}

// Stream sends a streaming completion request
func (g *Groq) Stream(ctx context.Context, req *simpleai.Request) (<-chan simpleai.StreamEvent, error) {
	groqReq := g.buildRequest(req)
	groqReq.Stream = true

	// Use goutil PostStream for raw response access
	resp, err := g.client.PostStream(g.config.BaseURL+"/v1/chat/completions", groqReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, g.handleError(resp)
	}

	out := make(chan simpleai.StreamEvent)
	go g.streamResponse(ctx, resp.Body, out)

	return out, nil
}

// CountTokens estimates token count
func (g *Groq) CountTokens(text string) int {
	return len(text) / 4
}

// Groq uses OpenAI-compatible request/response formats
type groqRequest struct {
	Model       string        `json:"model"`
	Messages    []groqMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float64       `json:"temperature,omitempty"`
	TopP        float64       `json:"top_p,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
	Stop        []string      `json:"stop,omitempty"`
}

type groqMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type groqResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []groqChoice `json:"choices"`
	Usage   groqUsage    `json:"usage"`
}

type groqChoice struct {
	Index        int         `json:"index"`
	Message      groqMessage `json:"message"`
	Delta        groqMessage `json:"delta"`
	FinishReason string      `json:"finish_reason"`
}

type groqUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type groqErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

func (g *Groq) buildRequest(req *simpleai.Request) *groqRequest {
	messages := make([]groqMessage, 0, len(req.Messages)+1)

	if req.SystemPrompt != "" {
		messages = append(messages, groqMessage{
			Role:    "system",
			Content: req.SystemPrompt,
		})
	}

	for _, msg := range req.Messages {
		messages = append(messages, groqMessage{
			Role:    string(msg.Role),
			Content: msg.Content,
		})
	}

	model := req.Model
	if model == "" {
		model = g.config.Model
	}

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = g.config.MaxTokens
	}

	temp := req.Temperature
	if temp == 0 {
		temp = g.config.Temperature
	}

	return &groqRequest{
		Model:       model,
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: temp,
		TopP:        req.TopP,
		Stop:        req.Stop,
	}
}

func (g *Groq) handleError(resp *http.Response) error {
	body, _ := io.ReadAll(resp.Body)

	var errResp groqErrorResponse
	if err := json.Unmarshal(body, &errResp); err == nil && errResp.Error.Message != "" {
		return simpleai.NewProviderError(
			"groq",
			resp.StatusCode,
			errResp.Error.Message,
			errResp.Error.Type,
		)
	}

	return simpleai.NewProviderError(
		"groq",
		resp.StatusCode,
		string(body),
		"unknown",
	)
}

func (g *Groq) parseResponse(resp *groqResponse) *simpleai.Response {
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

func (g *Groq) streamResponse(ctx context.Context, body io.ReadCloser, out chan<- simpleai.StreamEvent) {
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

		var resp groqResponse
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
