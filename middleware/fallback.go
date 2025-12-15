package middleware

import (
	"context"

	"github.com/medatechnology/simpleai"
)

// FallbackConfig holds configuration for fallback middleware
type FallbackConfig struct {
	Providers []simpleai.Provider // Fallback providers in order
	OnError   func(err error, provider string) // Optional callback on error
}

// Fallback creates a fallback middleware that tries alternative providers
func Fallback(config FallbackConfig) simpleai.Middleware {
	return simpleai.MiddlewareFunc(func(next simpleai.Handler) simpleai.Handler {
		return func(ctx context.Context, req *simpleai.Request) (*simpleai.Response, error) {
			// Try primary provider first
			resp, err := next(ctx, req)
			if err == nil {
				return resp, nil
			}

			// Report error if callback provided
			if config.OnError != nil {
				config.OnError(err, "primary")
			}

			// Try fallback providers
			for _, provider := range config.Providers {
				select {
				case <-ctx.Done():
					return nil, ctx.Err()
				default:
				}

				resp, err = provider.Complete(ctx, req)
				if err == nil {
					return resp, nil
				}

				if config.OnError != nil {
					config.OnError(err, provider.Name())
				}
			}

			// All providers failed
			return nil, err
		}
	})
}

// FallbackSimple creates a fallback middleware with just providers
func FallbackSimple(providers ...simpleai.Provider) simpleai.Middleware {
	return Fallback(FallbackConfig{
		Providers: providers,
	})
}

// FallbackWithLogging creates a fallback middleware that logs errors
func FallbackWithLogging(logger func(msg string), providers ...simpleai.Provider) simpleai.Middleware {
	return Fallback(FallbackConfig{
		Providers: providers,
		OnError: func(err error, provider string) {
			logger("provider " + provider + " failed: " + err.Error())
		},
	})
}
