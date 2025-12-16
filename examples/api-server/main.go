package main

import (
	"context"
	"log"
	"net/http"

	"github.com/medatechnology/simpleai"
	shttp "github.com/medatechnology/simpleai/http"
	"github.com/medatechnology/simpleai/middleware"
	"github.com/medatechnology/simpleai/provider"
	"github.com/medatechnology/simplehttp"
	"github.com/medatechnology/simplehttp/framework/fiber"
)

func main() {
	// Create Mistral provider from environment
	mistral := provider.NewMistralFromEnv()

	// Create fallback provider (OpenAI)
	openai := provider.NewOpenAIFromEnv()

	// Create client with middleware
	client := simpleai.NewClient(mistral,
		simpleai.WithMiddleware(middleware.RetrySimple(3)),
		simpleai.WithMiddleware(middleware.FallbackSimple(openai)),
		simpleai.WithMiddleware(middleware.SimpleLogger(func(msg string) {
			log.Println("[AI]", msg)
		})),
	)

	// Create chat session with Doctor AI system prompt
	chat := client.NewChat(
		simpleai.WithSystem(`You are Dr. AI, a knowledgeable medical assistant.
Your responsibilities:
- Provide general health information and guidance
- Analyze symptoms (not diagnose)
- Offer wellness advice and preventive care tips

IMPORTANT: Always remind users to consult real healthcare professionals for medical decisions.`),
		simpleai.WithHistoryLimit(50),
	)

	// Create HTTP server
	config := simplehttp.LoadConfig()
	if config.Port == "" {
		config.Port = "8080"
	}
	server := fiber.NewServer(config)

	// Add logging middleware
	server.Use(simplehttp.MiddlewareRequestID())
	server.Use(simplehttp.MiddlewareLogger(simplehttp.NewDefaultLogger()))

	// Health check endpoint
	server.GET("/health", func(c simplehttp.Context) error {
		return c.JSON(http.StatusOK, map[string]string{
			"status":   "healthy",
			"provider": "mistral",
		})
	})

	// API routes
	api := server.Group("/api/v1")

	// Non-streaming completion
	api.POST("/chat/complete", shttp.CompleteHandler(client))

	// Streaming completion via SSE
	api.POST("/chat/stream", shttp.StreamHandler(client))

	// Chat with history (Doctor AI)
	api.POST("/doctor/chat", shttp.ChatStreamHandler(chat))

	// Simple chat endpoint for testing
	api.POST("/chat", func(c simplehttp.Context) error {
		var req struct {
			Message string `json:"message"`
		}
		if err := c.BindJSON(&req); err != nil {
			return c.JSON(http.StatusBadRequest, map[string]string{"error": err.Error()})
		}

		resp, err := client.Complete(context.Background(), &simpleai.Request{
			Messages: []simpleai.Message{
				{Role: simpleai.RoleUser, Content: req.Message},
			},
		})
		if err != nil {
			return c.JSON(http.StatusInternalServerError, map[string]string{"error": err.Error()})
		}

		return c.JSON(http.StatusOK, map[string]interface{}{
			"response": resp.Content,
			"model":    resp.Model,
			"usage":    resp.Usage,
		})
	})

	// Start server
	log.Printf("🚀 SimpleAI API Server starting on port %s", config.Port)
	log.Println("Endpoints:")
	log.Println("  GET  /health           - Health check")
	log.Println("  POST /api/v1/chat      - Simple chat")
	log.Println("  POST /api/v1/chat/complete - OpenAI-compatible completion")
	log.Println("  POST /api/v1/chat/stream   - SSE streaming")
	log.Println("  POST /api/v1/doctor/chat   - Doctor AI chat with history")

	if err := server.Start(":" + config.Port); err != nil {
		log.Fatal(err)
	}
}
