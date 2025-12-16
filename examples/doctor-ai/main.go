package main

import (
	"context"
	"fmt"

	"github.com/medatechnology/simpleai"
	"github.com/medatechnology/simpleai/middleware"
	"github.com/medatechnology/simpleai/provider"
	"github.com/medatechnology/simpleai/template"
)

func main() {
	// Create primary provider (Mistral) - reads MISTRAL_API_KEY from env
	mistral := provider.NewMistralFromEnv()

	// Create fallback provider (OpenAI) - reads OPENAI_API_KEY from env
	openai := provider.NewOpenAIFromEnv()

	// Create client with middleware
	client := simpleai.NewClient(mistral,
		simpleai.WithMiddleware(middleware.RetrySimple(3)),
		simpleai.WithMiddleware(middleware.FallbackSimple(openai)),
		simpleai.WithMiddleware(middleware.SimpleLogger(func(msg string) {
			fmt.Println("[LOG]", msg)
		})),
	)

	// Create prompt template for Doctor AI
	tmpl := template.NewEngine()
	err := tmpl.Load("doctor_system", `You are Dr. AI, a knowledgeable medical assistant.

Your responsibilities:
- Provide general health information and guidance
- Analyze symptoms (not diagnose)
- Offer wellness advice and preventive care tips

Patient Context:
- Name: {{.Name}}
- Age: {{.Age}}
- Known Conditions: {{if .Conditions}}{{join .Conditions ", "}}{{else}}None{{end}}

IMPORTANT: Always remind users to consult real healthcare professionals for medical decisions.`)
	if err != nil {
		panic(err)
	}

	// Generate system prompt from template
	systemPrompt, err := tmpl.Execute("doctor_system", map[string]any{
		"Name":       "Patient",
		"Age":        35,
		"Conditions": []string{},
	})
	if err != nil {
		panic(err)
	}

	// Create chat session
	chat := client.NewChat(
		simpleai.WithSystem(systemPrompt),
		simpleai.WithHistoryLimit(50),
	)

	// Example conversation
	ctx := context.Background()

	fmt.Println("=== Doctor AI Demo ===")
	fmt.Println()

	// First message
	fmt.Println("Patient: I've been having headaches for the past 3 days, especially in the morning.")
	resp, err := chat.Send(ctx, "I've been having headaches for the past 3 days, especially in the morning.")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Println()
	fmt.Println("Dr. AI:", resp.Content)
	fmt.Println()

	// Follow-up question
	fmt.Println("Patient: Should I be worried? What could be causing this?")
	resp, err = chat.Send(ctx, "Should I be worried? What could be causing this?")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Println()
	fmt.Println("Dr. AI:", resp.Content)
	fmt.Println()

	// Show usage stats
	fmt.Printf("Token usage - Prompt: %d, Completion: %d, Total: %d\n",
		resp.Usage.PromptTokens, resp.Usage.CompletionTokens, resp.Usage.TotalTokens)
}
