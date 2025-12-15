package simpleai

import "errors"

// Common errors
var (
	ErrNoProvider       = errors.New("simpleai: no provider configured")
	ErrEmptyAPIKey      = errors.New("simpleai: API key is required")
	ErrEmptyMessage     = errors.New("simpleai: message cannot be empty")
	ErrProviderError    = errors.New("simpleai: provider returned an error")
	ErrRateLimited      = errors.New("simpleai: rate limited by provider")
	ErrContextCanceled  = errors.New("simpleai: context canceled")
	ErrStreamClosed     = errors.New("simpleai: stream closed")
	ErrInvalidResponse  = errors.New("simpleai: invalid response from provider")
	ErrMaxTokensReached = errors.New("simpleai: max tokens reached")
)

// ProviderError represents an error from an AI provider
type ProviderError struct {
	Provider   string
	StatusCode int
	Message    string
	Type       string
	Err        error
}

func (e *ProviderError) Error() string {
	if e.Err != nil {
		return e.Provider + ": " + e.Message + ": " + e.Err.Error()
	}
	return e.Provider + ": " + e.Message
}

func (e *ProviderError) Unwrap() error {
	return e.Err
}

// NewProviderError creates a new provider error
func NewProviderError(provider string, statusCode int, message, errType string) *ProviderError {
	return &ProviderError{
		Provider:   provider,
		StatusCode: statusCode,
		Message:    message,
		Type:       errType,
	}
}

// IsRetryable returns true if the error is retryable
func (e *ProviderError) IsRetryable() bool {
	// Rate limited or server errors are retryable
	return e.StatusCode == 429 || e.StatusCode >= 500
}
