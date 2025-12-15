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
	GeminiDefaultBaseURL = "https://generativelanguage.googleapis.com"
	GeminiDefaultModel   = "gemini-1.5-pro"
)

// GeminiConfig holds configuration for the Gemini provider
type GeminiConfig struct {
	APIKey      string
	BaseURL     string
	Model       string
	MaxTokens   int
	Temperature float64
	TopP        float64
}

// Gemini implements the Provider interface for Google's Gemini
type Gemini struct {
	config GeminiConfig
	client medahttp.HttpClient
}

// NewGemini creates a new Gemini provider
func NewGemini(config GeminiConfig) *Gemini {
	if config.BaseURL == "" {
		config.BaseURL = GeminiDefaultBaseURL
	}
	if config.Model == "" {
		config.Model = GeminiDefaultModel
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

	return &Gemini{
		config: config,
		client: client,
	}
}

// NewGeminiFromEnv creates a Gemini provider from environment variables
// Environment variables: GEMINI_API_KEY, GEMINI_MODEL (optional)
func NewGeminiFromEnv() *Gemini {
	return NewGemini(GeminiConfig{
		APIKey: utils.GetEnvString("GEMINI_API_KEY", ""),
		Model:  utils.GetEnvString("GEMINI_MODEL", GeminiDefaultModel),
	})
}

// Name returns the provider name
func (g *Gemini) Name() string {
	return "gemini"
}

// Complete sends a completion request to Gemini
func (g *Gemini) Complete(ctx context.Context, req *simpleai.Request) (*simpleai.Response, error) {
	geminiReq := g.buildRequest(req)

	model := req.Model
	if model == "" {
		model = g.config.Model
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:generateContent?key=%s",
		g.config.BaseURL, model, g.config.APIKey)

	var geminiResp geminiResponse
	statusCode, err := g.client.Post(url, geminiReq, &geminiResp, nil)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if statusCode != 200 {
		return nil, simpleai.NewProviderError(
			"gemini",
			int(statusCode),
			"request failed",
			"http_error",
		)
	}

	return g.parseResponse(&geminiResp, model), nil
}

// Stream sends a streaming completion request
func (g *Gemini) Stream(ctx context.Context, req *simpleai.Request) (<-chan simpleai.StreamEvent, error) {
	geminiReq := g.buildRequest(req)

	model := req.Model
	if model == "" {
		model = g.config.Model
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:streamGenerateContent?alt=sse&key=%s",
		g.config.BaseURL, model, g.config.APIKey)

	// Use goutil PostStream for raw response access
	resp, err := g.client.PostStream(url, geminiReq)
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
func (g *Gemini) CountTokens(text string) int {
	return len(text) / 4
}

// Internal types for Gemini API
type geminiRequest struct {
	Contents          []geminiContent  `json:"contents"`
	SystemInstruction *geminiContent   `json:"systemInstruction,omitempty"`
	GenerationConfig  geminiGenConfig  `json:"generationConfig,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text string `json:"text"`
}

type geminiGenConfig struct {
	MaxOutputTokens int      `json:"maxOutputTokens,omitempty"`
	Temperature     float64  `json:"temperature,omitempty"`
	TopP            float64  `json:"topP,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
}

type geminiResponse struct {
	Candidates    []geminiCandidate `json:"candidates"`
	UsageMetadata geminiUsage       `json:"usageMetadata"`
}

type geminiCandidate struct {
	Content       geminiContent `json:"content"`
	FinishReason  string        `json:"finishReason"`
	SafetyRatings []interface{} `json:"safetyRatings"`
}

type geminiUsage struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

type geminiErrorResponse struct {
	Error struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error"`
}

func (g *Gemini) buildRequest(req *simpleai.Request) *geminiRequest {
	contents := make([]geminiContent, 0, len(req.Messages))
	var systemContent *geminiContent

	for _, msg := range req.Messages {
		if msg.Role == simpleai.RoleSystem {
			systemContent = &geminiContent{
				Parts: []geminiPart{{Text: msg.Content}},
			}
			continue
		}

		role := "user"
		if msg.Role == simpleai.RoleAssistant {
			role = "model"
		}

		contents = append(contents, geminiContent{
			Role:  role,
			Parts: []geminiPart{{Text: msg.Content}},
		})
	}

	if req.SystemPrompt != "" {
		systemContent = &geminiContent{
			Parts: []geminiPart{{Text: req.SystemPrompt}},
		}
	}

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = g.config.MaxTokens
	}

	temp := req.Temperature
	if temp == 0 {
		temp = g.config.Temperature
	}

	return &geminiRequest{
		Contents:          contents,
		SystemInstruction: systemContent,
		GenerationConfig: geminiGenConfig{
			MaxOutputTokens: maxTokens,
			Temperature:     temp,
			TopP:            req.TopP,
			StopSequences:   req.Stop,
		},
	}
}

func (g *Gemini) handleError(resp *http.Response) error {
	body, _ := io.ReadAll(resp.Body)

	var errResp geminiErrorResponse
	if err := json.Unmarshal(body, &errResp); err == nil && errResp.Error.Message != "" {
		return simpleai.NewProviderError(
			"gemini",
			resp.StatusCode,
			errResp.Error.Message,
			errResp.Error.Status,
		)
	}

	return simpleai.NewProviderError(
		"gemini",
		resp.StatusCode,
		string(body),
		"unknown",
	)
}

func (g *Gemini) parseResponse(resp *geminiResponse, model string) *simpleai.Response {
	var content string
	var finishReason string

	if len(resp.Candidates) > 0 {
		candidate := resp.Candidates[0]
		finishReason = candidate.FinishReason
		if len(candidate.Content.Parts) > 0 {
			content = candidate.Content.Parts[0].Text
		}
	}

	return &simpleai.Response{
		Content:      content,
		Model:        model,
		FinishReason: finishReason,
		Usage: simpleai.Usage{
			PromptTokens:     resp.UsageMetadata.PromptTokenCount,
			CompletionTokens: resp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      resp.UsageMetadata.TotalTokenCount,
		},
	}
}

func (g *Gemini) streamResponse(ctx context.Context, body io.ReadCloser, out chan<- simpleai.StreamEvent) {
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

		var resp geminiResponse
		if err := json.Unmarshal([]byte(data), &resp); err != nil {
			continue
		}

		if len(resp.Candidates) > 0 {
			candidate := resp.Candidates[0]
			if len(candidate.Content.Parts) > 0 {
				out <- simpleai.StreamEvent{Content: candidate.Content.Parts[0].Text}
			}
			if candidate.FinishReason != "" && candidate.FinishReason != "STOP" {
				out <- simpleai.StreamEvent{
					Done:         true,
					FinishReason: candidate.FinishReason,
				}
				return
			}
		}
	}

	out <- simpleai.StreamEvent{Done: true}

	if err := scanner.Err(); err != nil {
		out <- simpleai.StreamEvent{Error: err, Done: true}
	}
}
