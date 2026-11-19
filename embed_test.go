package braingate_test

import (
	"encoding/json"
	"io"
	"math"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/snowmerak/braingate"
)

func TestNewEmbedderURLNormalization(t *testing.T) {
	cases := []struct {
		name    string
		baseURL string
		wantURL string
	}{
		{
			name:    "default host without path",
			baseURL: "http://127.0.0.1:18182",
			wantURL: "http://127.0.0.1:18182/v1/embeddings",
		},
		{
			name:    "trailing slash without path",
			baseURL: "http://127.0.0.1:18182/",
			wantURL: "http://127.0.0.1:18182/v1/embeddings",
		},
		{
			name:    "with v1 prefix",
			baseURL: "http://127.0.0.1:18182/v1",
			wantURL: "http://127.0.0.1:18182/v1/embeddings",
		},
		{
			name:    "with v1 prefix and trailing slash",
			baseURL: "http://127.0.0.1:18182/proxy/v1/",
			wantURL: "http://127.0.0.1:18182/proxy/v1/embeddings",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var gotURL string
			var gotAuth string
			embedder, err := braingate.NewEmbedder(braingate.EmbedConfig{
				BaseURL: tc.baseURL,
				Model:   "embed-custom",
				APIKey:  "secret-token",
				HTTPClient: &http.Client{Transport: testTransport(func(r *http.Request) (*http.Response, error) {
					gotURL = r.URL.String()
					gotAuth = r.Header.Get("Authorization")
					return &http.Response{
						StatusCode: http.StatusOK,
						Body: io.NopCloser(strings.NewReader(`{
							"data": [{"index": 0, "embedding": [1.0, 0.0]}]
						}`)),
					}, nil
				})},
			})
			if err != nil {
				t.Fatalf("NewEmbedder error = %v", err)
			}

			vecs, err := embedder.Embed(t.Context(), []string{"hello"})
			if err != nil {
				t.Fatalf("Embed error = %v", err)
			}
			if gotURL != tc.wantURL {
				t.Fatalf("request URL = %q, want %q", gotURL, tc.wantURL)
			}
			if gotAuth != "Bearer secret-token" {
				t.Fatalf("Authorization = %q, want Bearer secret-token", gotAuth)
			}
			if !reflect.DeepEqual(vecs, [][]float64{{1, 0}}) {
				t.Fatalf("vecs = %+v", vecs)
			}
		})
	}
}

func TestEmbedReordersIndicesAndValidatesResponses(t *testing.T) {
	t.Run("reorders out-of-order indices", func(t *testing.T) {
		var gotModel string
		var gotInput []string
		embedder, err := braingate.NewEmbedder(braingate.EmbedConfig{
			HTTPClient: &http.Client{Transport: testTransport(func(r *http.Request) (*http.Response, error) {
				var req struct {
					Model string   `json:"model"`
					Input []string `json:"input"`
				}
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Fatal(err)
				}
				gotModel = req.Model
				gotInput = req.Input
				return &http.Response{
					StatusCode: http.StatusOK,
					Body: io.NopCloser(strings.NewReader(`{
						"data": [
							{"index": 1, "embedding": [0.0, 1.0]},
							{"index": 0, "embedding": [1.0, 0.0]}
						]
					}`)),
				}, nil
			})},
		})
		if err != nil {
			t.Fatal(err)
		}

		vecs, err := embedder.Embed(t.Context(), []string{"first", "second"})
		if err != nil {
			t.Fatalf("Embed error = %v", err)
		}
		if gotModel != braingate.DefaultEmbedModel || !reflect.DeepEqual(gotInput, []string{"first", "second"}) {
			t.Fatalf("request = (model=%q, input=%v)", gotModel, gotInput)
		}
		want := [][]float64{{1, 0}, {0, 1}}
		if !reflect.DeepEqual(vecs, want) {
			t.Fatalf("vecs = %+v, want %+v", vecs, want)
		}
	})

	t.Run("surfaces HTTP 500 JSON error message", func(t *testing.T) {
		embedder, err := braingate.NewEmbedder(braingate.EmbedConfig{
			HTTPClient: &http.Client{Transport: testTransport(func(*http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusInternalServerError,
					Status:     "500 Internal Server Error",
					Body:       io.NopCloser(strings.NewReader(`{"error":{"code":500,"message":"model name=embed failed to load","type":"server_error"}}`)),
				}, nil
			})},
		})
		if err != nil {
			t.Fatal(err)
		}

		_, err = embedder.Embed(t.Context(), []string{"hello"})
		if err == nil || !strings.Contains(err.Error(), "HTTP 500") || !strings.Contains(err.Error(), "model name=embed failed to load") {
			t.Fatalf("err = %v, want HTTP 500 with server error message", err)
		}
	})

	badPayloads := []struct {
		name    string
		payload string
		want    string
	}{
		{
			name:    "count mismatch",
			payload: `{"data":[{"index":0,"embedding":[1,0]}]}`,
			want:    "count mismatch",
		},
		{
			name:    "duplicate index",
			payload: `{"data":[{"index":0,"embedding":[1,0]},{"index":0,"embedding":[0,1]}]}`,
			want:    "duplicate embedding index",
		},
		{
			name:    "index out of range",
			payload: `{"data":[{"index":0,"embedding":[1,0]},{"index":5,"embedding":[0,1]}]}`,
			want:    "out of range",
		},
		{
			name:    "dimension mismatch",
			payload: `{"data":[{"index":0,"embedding":[1,0]},{"index":1,"embedding":[0,1,2]}]}`,
			want:    "dimension mismatch",
		},
		{
			name:    "empty embedding vector",
			payload: `{"data":[{"index":0,"embedding":[]},{"index":1,"embedding":[0,1]}]}`,
			want:    "empty embedding vector",
		},
	}

	for _, tc := range badPayloads {
		t.Run(tc.name, func(t *testing.T) {
			embedder, err := braingate.NewEmbedder(braingate.EmbedConfig{
				HTTPClient: &http.Client{Transport: testTransport(func(*http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode: http.StatusOK,
						Body:       io.NopCloser(strings.NewReader(tc.payload)),
					}, nil
				})},
			})
			if err != nil {
				t.Fatal(err)
			}
			_, err = embedder.Embed(t.Context(), []string{"a", "b"})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want substring %q", err, tc.want)
			}
		})
	}
}

func TestNewEmbedderRejectsInvalidURLAndEmptyBatch(t *testing.T) {
	if _, err := braingate.NewEmbedder(braingate.EmbedConfig{BaseURL: "ftp://localhost:18182"}); err == nil {
		t.Fatal("expected error for non-http scheme")
	}

	embedder, err := braingate.NewEmbedder(braingate.EmbedConfig{})
	if err != nil {
		t.Fatal(err)
	}
	vecs, err := embedder.Embed(t.Context(), nil)
	if err != nil || vecs != nil {
		t.Fatalf("empty Embed = (%v, %v), want (nil, nil)", vecs, err)
	}
	_ = math.NaN()
}
