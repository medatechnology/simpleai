package simpleai

import "context"

// Handler is a function that processes a request and returns a response
type Handler func(ctx context.Context, req *Request) (*Response, error)

// Middleware wraps a handler to add functionality
type Middleware interface {
	Wrap(next Handler) Handler
}

// MiddlewareFunc is a function that implements Middleware
type MiddlewareFunc func(next Handler) Handler

// Wrap implements the Middleware interface
func (f MiddlewareFunc) Wrap(next Handler) Handler {
	return f(next)
}
