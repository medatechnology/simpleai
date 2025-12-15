package middleware

import (
	"context"
	"time"

	"github.com/medatechnology/goutil/simplelog"
	"github.com/medatechnology/simpleai"
)

// LogEntry represents a log entry for AI requests
type LogEntry struct {
	Timestamp    time.Time
	Provider     string
	Model        string
	Duration     time.Duration
	InputTokens  int
	OutputTokens int
	Error        error
}

// Logger is a function that receives log entries
type Logger func(entry LogEntry)

// LoggingConfig holds configuration for logging middleware
type LoggingConfig struct {
	Logger     Logger
	LogRequest bool // Log request details (can be verbose)
}

// Logging creates a logging middleware
func Logging(config LoggingConfig) simpleai.Middleware {
	return simpleai.MiddlewareFunc(func(next simpleai.Handler) simpleai.Handler {
		return func(ctx context.Context, req *simpleai.Request) (*simpleai.Response, error) {
			start := time.Now()

			resp, err := next(ctx, req)

			entry := LogEntry{
				Timestamp: start,
				Model:     req.Model,
				Duration:  time.Since(start),
				Error:     err,
			}

			if resp != nil {
				entry.InputTokens = resp.Usage.PromptTokens
				entry.OutputTokens = resp.Usage.CompletionTokens
			}

			if config.Logger != nil {
				config.Logger(entry)
			}

			return resp, err
		}
	})
}

// SimpleLogger creates a logging middleware with a simple log function
func SimpleLogger(logFn func(msg string)) simpleai.Middleware {
	return Logging(LoggingConfig{
		Logger: func(entry LogEntry) {
			if entry.Error != nil {
				logFn("AI request failed: " + entry.Error.Error())
			} else {
				logFn("AI request completed in " + entry.Duration.String())
			}
		},
	})
}

// GoutilLogger creates a logging middleware using goutil/simplelog
func GoutilLogger(debugLevel int) simpleai.Middleware {
	return Logging(LoggingConfig{
		Logger: func(entry LogEntry) {
			if entry.Error != nil {
				simplelog.LogErr(entry.Error, "AI request failed")
			} else {
				simplelog.LogInfoStr("SimpleAI", debugLevel,
					"Request completed in "+entry.Duration.String(),
				)
			}
		},
	})
}
