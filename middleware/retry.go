package middleware

import (
	"context"
	"math"
	"math/rand"
	"time"

	"github.com/medatechnology/simpleai"
)

// RetryConfig holds configuration for retry middleware
type RetryConfig struct {
	MaxAttempts  int           // Maximum number of attempts (including first)
	InitialDelay time.Duration // Initial delay between retries
	MaxDelay     time.Duration // Maximum delay between retries
	Multiplier   float64       // Backoff multiplier
	Jitter       bool          // Add random jitter to delays
}

// DefaultRetryConfig returns sensible defaults
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts:  3,
		InitialDelay: 1 * time.Second,
		MaxDelay:     30 * time.Second,
		Multiplier:   2.0,
		Jitter:       true,
	}
}

// Retry creates a retry middleware with the given config
func Retry(config RetryConfig) simpleai.Middleware {
	return simpleai.MiddlewareFunc(func(next simpleai.Handler) simpleai.Handler {
		return func(ctx context.Context, req *simpleai.Request) (*simpleai.Response, error) {
			var lastErr error
			delay := config.InitialDelay

			for attempt := 1; attempt <= config.MaxAttempts; attempt++ {
				resp, err := next(ctx, req)
				if err == nil {
					return resp, nil
				}

				lastErr = err

				// Check if error is retryable
				if !isRetryable(err) {
					return nil, err
				}

				// Check if we've exhausted attempts
				if attempt >= config.MaxAttempts {
					break
				}

				// Calculate delay with jitter
				waitTime := delay
				if config.Jitter {
					jitter := time.Duration(rand.Float64() * float64(delay) * 0.3)
					waitTime = delay + jitter
				}

				// Wait before retry
				select {
				case <-ctx.Done():
					return nil, ctx.Err()
				case <-time.After(waitTime):
				}

				// Increase delay for next attempt
				delay = time.Duration(float64(delay) * config.Multiplier)
				if delay > config.MaxDelay {
					delay = config.MaxDelay
				}
			}

			return nil, lastErr
		}
	})
}

// RetrySimple creates a retry middleware with default config and specified max attempts
func RetrySimple(maxAttempts int) simpleai.Middleware {
	config := DefaultRetryConfig()
	config.MaxAttempts = maxAttempts
	return Retry(config)
}

// isRetryable checks if an error is retryable
func isRetryable(err error) bool {
	if providerErr, ok := err.(*simpleai.ProviderError); ok {
		return providerErr.IsRetryable()
	}
	return false
}

// ExponentialBackoff calculates backoff delay
func ExponentialBackoff(attempt int, initialDelay time.Duration, maxDelay time.Duration) time.Duration {
	delay := time.Duration(float64(initialDelay) * math.Pow(2, float64(attempt-1)))
	if delay > maxDelay {
		delay = maxDelay
	}
	return delay
}
