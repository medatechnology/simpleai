package http

import (
	"encoding/json"
	"net/http"

	"github.com/medatechnology/simpleai"
	"github.com/medatechnology/simplehttp"
)

// ChatRequest represents an incoming chat request
type ChatRequest struct {
	Messages    []simpleai.Message `json:"messages"`
	Model       string             `json:"model,omitempty"`
	MaxTokens   int                `json:"max_tokens,omitempty"`
	Temperature float64            `json:"temperature,omitempty"`
	Stream      bool               `json:"stream,omitempty"`
}

// ChatResponse represents a non-streaming chat response
type ChatResponse struct {
	Content      string         `json:"content"`
	Model        string         `json:"model"`
	FinishReason string         `json:"finish_reason"`
	Usage        simpleai.Usage `json:"usage"`
}

// StreamChunk represents a streaming response chunk (SSE data)
type StreamChunk struct {
	Content      string `json:"content,omitempty"`
	Done         bool   `json:"done"`
	FinishReason string `json:"finish_reason,omitempty"`
}

// StreamHandler creates an HTTP handler for streaming AI completions via SSE
func StreamHandler(client *simpleai.Client) simplehttp.HandlerFunc {
	return func(c simplehttp.Context) error {
		var req ChatRequest
		if err := c.BindJSON(&req); err != nil {
			return c.JSON(http.StatusBadRequest, map[string]string{
				"error": "invalid request: " + err.Error(),
			})
		}

		// Convert to simpleai request
		aiReq := &simpleai.Request{
			Messages:    req.Messages,
			Model:       req.Model,
			MaxTokens:   req.MaxTokens,
			Temperature: req.Temperature,
			Stream:      true,
		}

		// Start streaming
		events, err := client.Stream(c.Context(), aiReq)
		if err != nil {
			return c.JSON(http.StatusInternalServerError, map[string]string{
				"error": err.Error(),
			})
		}

		// Stream via SSE
		return c.SSE(func(w simplehttp.SSEWriter) error {
			for event := range events {
				if event.Error != nil {
					// Send error event
					errData, _ := json.Marshal(map[string]string{"error": event.Error.Error()})
					w.SendEvent(simplehttp.SSEEvent{Event: "error", Data: string(errData)})
					return event.Error
				}

				chunk := StreamChunk{
					Content:      event.Content,
					Done:         event.Done,
					FinishReason: event.FinishReason,
				}
				data, _ := json.Marshal(chunk)
				w.Send(string(data))

				if event.Done {
					break
				}
			}
			return nil
		})
	}
}

// CompleteHandler creates an HTTP handler for non-streaming AI completions
func CompleteHandler(client *simpleai.Client) simplehttp.HandlerFunc {
	return func(c simplehttp.Context) error {
		var req ChatRequest
		if err := c.BindJSON(&req); err != nil {
			return c.JSON(http.StatusBadRequest, map[string]string{
				"error": "invalid request: " + err.Error(),
			})
		}

		// Convert to simpleai request
		aiReq := &simpleai.Request{
			Messages:    req.Messages,
			Model:       req.Model,
			MaxTokens:   req.MaxTokens,
			Temperature: req.Temperature,
		}

		// Complete request
		resp, err := client.Complete(c.Context(), aiReq)
		if err != nil {
			return c.JSON(http.StatusInternalServerError, map[string]string{
				"error": err.Error(),
			})
		}

		return c.JSON(http.StatusOK, ChatResponse{
			Content:      resp.Content,
			Model:        resp.Model,
			FinishReason: resp.FinishReason,
			Usage:        resp.Usage,
		})
	}
}

// ChatStreamHandler creates an HTTP handler for streaming chat sessions
func ChatStreamHandler(chat *simpleai.Chat) simplehttp.HandlerFunc {
	return func(c simplehttp.Context) error {
		var req struct {
			Message string `json:"message"`
		}
		if err := c.BindJSON(&req); err != nil {
			return c.JSON(http.StatusBadRequest, map[string]string{
				"error": "invalid request: " + err.Error(),
			})
		}

		// Start streaming
		events, err := chat.Stream(c.Context(), req.Message)
		if err != nil {
			return c.JSON(http.StatusInternalServerError, map[string]string{
				"error": err.Error(),
			})
		}

		// Stream via SSE
		return c.SSE(func(w simplehttp.SSEWriter) error {
			for event := range events {
				if event.Error != nil {
					errData, _ := json.Marshal(map[string]string{"error": event.Error.Error()})
					w.SendEvent(simplehttp.SSEEvent{Event: "error", Data: string(errData)})
					return event.Error
				}

				chunk := StreamChunk{
					Content:      event.Content,
					Done:         event.Done,
					FinishReason: event.FinishReason,
				}
				data, _ := json.Marshal(chunk)
				w.Send(string(data))

				if event.Done {
					break
				}
			}
			return nil
		})
	}
}
