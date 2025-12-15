package memory

import (
	"context"
	"fmt"
	"strings"

	"github.com/medatechnology/simpleai"
)

// AISummarizer uses an AI provider to summarize conversations
type AISummarizer struct {
	provider simpleai.Provider
	model    string
}

// NewAISummarizer creates a summarizer using the given AI provider
func NewAISummarizer(provider simpleai.Provider) *AISummarizer {
	return &AISummarizer{
		provider: provider,
	}
}

// NewAISummarizerWithModel creates a summarizer with a specific model
func NewAISummarizerWithModel(provider simpleai.Provider, model string) *AISummarizer {
	return &AISummarizer{
		provider: provider,
		model:    model,
	}
}

// Summarize compresses messages into a concise summary
func (s *AISummarizer) Summarize(ctx context.Context, messages []simpleai.Message) (string, error) {
	if len(messages) == 0 {
		return "", nil
	}

	// Build conversation text
	var sb strings.Builder
	for _, msg := range messages {
		sb.WriteString(fmt.Sprintf("%s: %s\n", msg.Role, msg.Content))
	}

	req := &simpleai.Request{
		Messages: []simpleai.Message{
			{
				Role:    simpleai.RoleUser,
				Content: sb.String(),
			},
		},
		SystemPrompt: `Summarize the following conversation concisely, preserving:
- Key facts and information shared
- Important decisions or conclusions
- Relevant context for future messages
Keep the summary brief (2-4 sentences). Do not include meta-commentary.`,
		Model:       s.model,
		MaxTokens:   500,
		Temperature: 0.3, // Low temperature for consistent summaries
	}

	resp, err := s.provider.Complete(ctx, req)
	if err != nil {
		return "", fmt.Errorf("summarization failed: %w", err)
	}

	return resp.Content, nil
}

// SimpleSummarizer provides a basic non-AI summarization (just truncation)
type SimpleSummarizer struct {
	maxLength int
}

// NewSimpleSummarizer creates a simple truncating summarizer
func NewSimpleSummarizer(maxLength int) *SimpleSummarizer {
	if maxLength <= 0 {
		maxLength = 500
	}
	return &SimpleSummarizer{maxLength: maxLength}
}

// Summarize truncates messages to create a simple summary
func (s *SimpleSummarizer) Summarize(ctx context.Context, messages []simpleai.Message) (string, error) {
	if len(messages) == 0 {
		return "", nil
	}

	var sb strings.Builder
	sb.WriteString("[Conversation excerpt] ")

	for _, msg := range messages {
		if sb.Len() >= s.maxLength {
			break
		}
		excerpt := msg.Content
		if len(excerpt) > 100 {
			excerpt = excerpt[:100] + "..."
		}
		sb.WriteString(fmt.Sprintf("%s: %s | ", msg.Role, excerpt))
	}

	result := sb.String()
	if len(result) > s.maxLength {
		result = result[:s.maxLength] + "..."
	}

	return result, nil
}
