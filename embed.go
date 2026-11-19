package braingate

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const (
	// DefaultEmbedBaseURL is the default OpenAI-compatible embeddings server address.
	DefaultEmbedBaseURL = "http://localhost:1234/v1"
	// DefaultEmbedModel is the default primary embedding model identifier.
	DefaultEmbedModel = "text-embedding-embeddinggemma-2"
	// DefaultSecondaryEmbedModel is the default secondary embedding model identifier used for dual-embedding calibration.
	DefaultSecondaryEmbedModel = "text-embedding-qwen3-embedding-0.6b"

	maxEmbedResponseBytes = 16 << 20
)

// Embedder computes dense vector embeddings for a batch of input texts.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float64, error)
}

// EmbedConfig configures an OpenAI-compatible embeddings client.
type EmbedConfig struct {
	BaseURL    string
	Model      string
	APIKey     string
	HTTPClient *http.Client
}

// OpenAIEmbedder calls an OpenAI-compatible /v1/embeddings endpoint.
type OpenAIEmbedder struct {
	endpoint string
	model    string
	apiKey   string
	client   *http.Client
}

// NewEmbedder validates configuration and constructs an OpenAI-compatible embeddings client.
func NewEmbedder(cfg EmbedConfig) (*OpenAIEmbedder, error) {
	rawBase := strings.TrimSpace(cfg.BaseURL)
	if rawBase == "" {
		rawBase = DefaultEmbedBaseURL
	}
	parsed, err := url.Parse(rawBase)
	if err != nil {
		return nil, fmt.Errorf("braingate: invalid embed base URL: %w", err)
	}
	if (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return nil, fmt.Errorf("braingate: embed base URL %q must use http or https with a host", rawBase)
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/")
	if parsed.Path == "" {
		parsed.Path = "/v1/embeddings"
	} else {
		parsed.Path += "/embeddings"
	}

	model := strings.TrimSpace(cfg.Model)
	if model == "" {
		model = DefaultEmbedModel
	}

	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{
			Timeout: 60 * time.Second,
			CheckRedirect: func(*http.Request, []*http.Request) error {
				return http.ErrUseLastResponse
			},
		}
	}

	return &OpenAIEmbedder{
		endpoint: parsed.String(),
		model:    model,
		apiKey:   strings.TrimSpace(cfg.APIKey),
		client:   httpClient,
	}, nil
}

// Embed computes embeddings for texts in order using a single batch request.
func (e *OpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float64, error) {
	if e == nil {
		return nil, errors.New("braingate: embedder is nil")
	}
	if len(texts) == 0 {
		return nil, nil
	}

	payload, err := json.Marshal(struct {
		Model string   `json:"model"`
		Input []string `json:"input"`
	}{
		Model: e.model,
		Input: texts,
	})
	if err != nil {
		return nil, fmt.Errorf("braingate: marshal embeddings request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("braingate: build embeddings request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if e.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.apiKey)
	}

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("braingate: embeddings request failed: %w", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxEmbedResponseBytes+1))
	if err != nil {
		return nil, fmt.Errorf("braingate: read embeddings response: %w", err)
	}
	if len(body) > maxEmbedResponseBytes {
		return nil, fmt.Errorf("braingate: embeddings response exceeded %d bytes", maxEmbedResponseBytes)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("braingate: embeddings HTTP %d: %s", resp.StatusCode, extractEmbedError(body, resp.Status))
	}

	var decoded struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		return nil, fmt.Errorf("braingate: decode embeddings response: %w", err)
	}
	if len(decoded.Data) != len(texts) {
		return nil, fmt.Errorf("braingate: embeddings count mismatch: got %d, want %d", len(decoded.Data), len(texts))
	}

	vectors := make([][]float64, len(texts))
	dim := 0
	for _, item := range decoded.Data {
		if item.Index < 0 || item.Index >= len(texts) {
			return nil, fmt.Errorf("braingate: embedding index %d out of range [0, %d)", item.Index, len(texts))
		}
		if vectors[item.Index] != nil {
			return nil, fmt.Errorf("braingate: duplicate embedding index %d", item.Index)
		}
		if len(item.Embedding) == 0 {
			return nil, fmt.Errorf("braingate: empty embedding vector at index %d", item.Index)
		}
		if dim == 0 {
			dim = len(item.Embedding)
		} else if len(item.Embedding) != dim {
			return nil, fmt.Errorf("braingate: embedding dimension mismatch at index %d: got %d, want %d", item.Index, len(item.Embedding), dim)
		}
		for _, v := range item.Embedding {
			if !finite(v) {
				return nil, fmt.Errorf("braingate: non-finite embedding value at index %d", item.Index)
			}
		}
		vectors[item.Index] = item.Embedding
	}
	return vectors, nil
}

func extractEmbedError(body []byte, status string) string {
	var errPayload struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &errPayload); err == nil && strings.TrimSpace(errPayload.Error.Message) != "" {
		return strings.TrimSpace(errPayload.Error.Message)
	}
	text := strings.TrimSpace(string(body))
	if text == "" {
		return status
	}
	const maxSnippet = 512
	if len(text) > maxSnippet {
		return text[:maxSnippet]
	}
	return text
}
