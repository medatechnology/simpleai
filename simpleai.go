package simpleai

import (
	"context"
	"fmt"
)

// Client is the main entry point for the simpleai library
type Client struct {
	provider   Provider
	middleware []Middleware
	config     *ClientConfig
}

// ClientConfig holds client configuration
type ClientConfig struct {
	DefaultModel       string
	DefaultMaxTokens   int
	DefaultTemperature float64
}

// NewClient creates a new simpleai client with the given provider
func NewClient(provider Provider, opts ...Option) *Client {
	c := &Client{
		provider:   provider,
		middleware: []Middleware{},
		config: &ClientConfig{
			DefaultMaxTokens:   4096,
			DefaultTemperature: 0.7,
		},
	}

	for _, opt := range opts {
		opt(c)
	}

	return c
}

// Complete sends a completion request through the middleware chain
func (c *Client) Complete(ctx context.Context, req *Request) (*Response, error) {
	if c.provider == nil {
		return nil, fmt.Errorf("no provider configured")
	}

	// Apply defaults if not set
	if req.MaxTokens == 0 {
		req.MaxTokens = c.config.DefaultMaxTokens
	}
	if req.Temperature == 0 {
		req.Temperature = c.config.DefaultTemperature
	}

	// Build middleware chain
	handler := func(ctx context.Context, req *Request) (*Response, error) {
		return c.provider.Complete(ctx, req)
	}

	// Apply middleware in reverse order
	for i := len(c.middleware) - 1; i >= 0; i-- {
		handler = c.middleware[i].Wrap(handler)
	}

	return handler(ctx, req)
}

// Stream sends a streaming completion request
func (c *Client) Stream(ctx context.Context, req *Request) (<-chan StreamEvent, error) {
	if c.provider == nil {
		return nil, fmt.Errorf("no provider configured")
	}

	// Apply defaults
	if req.MaxTokens == 0 {
		req.MaxTokens = c.config.DefaultMaxTokens
	}
	if req.Temperature == 0 {
		req.Temperature = c.config.DefaultTemperature
	}
	req.Stream = true

	return c.provider.Stream(ctx, req)
}

// NewChat creates a new chat session with the client's provider
func (c *Client) NewChat(opts ...ChatOption) *Chat {
	return NewChat(c, opts...)
}

// CountTokens estimates token count for the given text
func (c *Client) CountTokens(text string) int {
	if c.provider == nil {
		return 0
	}
	return c.provider.CountTokens(text)
}

// Provider returns the underlying provider
func (c *Client) Provider() Provider {
	return c.provider
}

// SetProvider changes the provider
func (c *Client) SetProvider(p Provider) {
	c.provider = p
}
