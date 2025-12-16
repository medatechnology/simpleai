package simpleai

// Option is a functional option for configuring the Client
type Option func(*Client)

// WithMiddleware adds middleware to the client
func WithMiddleware(m Middleware) Option {
	return func(c *Client) {
		c.middleware = append(c.middleware, m)
	}
}

// WithDefaultMaxTokens sets the default max tokens
func WithDefaultMaxTokens(n int) Option {
	return func(c *Client) {
		c.config.DefaultMaxTokens = n
	}
}

// WithDefaultTemperature sets the default temperature
func WithDefaultTemperature(t float64) Option {
	return func(c *Client) {
		c.config.DefaultTemperature = t
	}
}

// WithDefaultModel sets the default model
func WithDefaultModel(model string) Option {
	return func(c *Client) {
		c.config.DefaultModel = model
	}
}

// ChatOption is a functional option for configuring a Chat session
type ChatOption func(*Chat)

// WithSystem sets the system prompt for the chat
func WithSystem(prompt string) ChatOption {
	return func(chat *Chat) {
		chat.system = prompt
	}
}

// WithHistoryLimit sets the maximum number of messages to keep in history
func WithHistoryLimit(limit int) ChatOption {
	return func(chat *Chat) {
		chat.historyLimit = limit
	}
}

// WithMessages initializes the chat with existing messages
func WithMessages(messages []Message) ChatOption {
	return func(chat *Chat) {
		chat.history = append(chat.history, messages...)
	}
}

// WithMaxTokens sets the maximum tokens for history (token-based truncation)
func WithMaxTokens(maxTokens int) ChatOption {
	return func(chat *Chat) {
		chat.maxTokens = maxTokens
	}
}

// WithTokenCounter sets a custom token counter
func WithTokenCounter(counter func(string) int) ChatOption {
	return func(chat *Chat) {
		chat.tokenCounter = counter
	}
}

// WithAutocompact enables automatic conversation compaction
// When the conversation exceeds the threshold, older messages are summarized
func WithAutocompact(config AutocompactConfig) ChatOption {
	return func(chat *Chat) {
		chat.autocompact = &config
	}
}
