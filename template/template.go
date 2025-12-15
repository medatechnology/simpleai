package template

import (
	"bytes"
	"fmt"
	"strings"
	"sync"
	"text/template"
)

// Engine manages prompt templates
type Engine struct {
	templates map[string]*template.Template
	mu        sync.RWMutex
	funcs     template.FuncMap
}

// NewEngine creates a new template engine
func NewEngine() *Engine {
	return &Engine{
		templates: make(map[string]*template.Template),
		funcs:     defaultFuncs(),
	}
}

// defaultFuncs returns default template functions
func defaultFuncs() template.FuncMap {
	return template.FuncMap{
		"upper":    strings.ToUpper,
		"lower":    strings.ToLower,
		"title":    strings.Title,
		"trim":     strings.TrimSpace,
		"join":     strings.Join,
		"split":    strings.Split,
		"contains": strings.Contains,
		"replace":  strings.ReplaceAll,
		"default": func(def, val interface{}) interface{} {
			if val == nil || val == "" {
				return def
			}
			return val
		},
		"list": func(items ...interface{}) []interface{} {
			return items
		},
	}
}

// AddFunc adds a custom template function
func (e *Engine) AddFunc(name string, fn interface{}) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.funcs[name] = fn
}

// Load loads a template from a string
func (e *Engine) Load(name, content string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	tmpl, err := template.New(name).Funcs(e.funcs).Parse(content)
	if err != nil {
		return fmt.Errorf("failed to parse template %s: %w", name, err)
	}

	e.templates[name] = tmpl
	return nil
}

// LoadFile loads a template from a file
func (e *Engine) LoadFile(name, path string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	tmpl, err := template.New(name).Funcs(e.funcs).ParseFiles(path)
	if err != nil {
		return fmt.Errorf("failed to load template file %s: %w", path, err)
	}

	e.templates[name] = tmpl
	return nil
}

// Execute executes a template with the given data
func (e *Engine) Execute(name string, data interface{}) (string, error) {
	e.mu.RLock()
	tmpl, ok := e.templates[name]
	e.mu.RUnlock()

	if !ok {
		return "", fmt.Errorf("template %s not found", name)
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return "", fmt.Errorf("failed to execute template %s: %w", name, err)
	}

	return buf.String(), nil
}

// ExecuteString executes a template string directly (without registration)
func (e *Engine) ExecuteString(content string, data interface{}) (string, error) {
	tmpl, err := template.New("inline").Funcs(e.funcs).Parse(content)
	if err != nil {
		return "", fmt.Errorf("failed to parse inline template: %w", err)
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return "", fmt.Errorf("failed to execute inline template: %w", err)
	}

	return buf.String(), nil
}

// Has checks if a template exists
func (e *Engine) Has(name string) bool {
	e.mu.RLock()
	defer e.mu.RUnlock()
	_, ok := e.templates[name]
	return ok
}

// Names returns all registered template names
func (e *Engine) Names() []string {
	e.mu.RLock()
	defer e.mu.RUnlock()

	names := make([]string, 0, len(e.templates))
	for name := range e.templates {
		names = append(names, name)
	}
	return names
}

// Delete removes a template
func (e *Engine) Delete(name string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	delete(e.templates, name)
}

// Clear removes all templates
func (e *Engine) Clear() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.templates = make(map[string]*template.Template)
}

// Prompt is a convenience function to quickly execute a template string
func Prompt(content string, data interface{}) (string, error) {
	engine := NewEngine()
	return engine.ExecuteString(content, data)
}
