package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	"github.com/snowmerak/braingate"
	"github.com/snowmerak/decisions"
)

func TestRunValidatesArgumentsAndEnv(t *testing.T) {
	exampleSpec := filepath.Join("..", "..", "examples", "immediate-response.json")

	t.Run("missing spec path", func(t *testing.T) {
		var out bytes.Buffer
		err := run(t.Context(), cliOptions{stimulus: "자극", noEmbed: true}, &out)
		if err == nil || !strings.Contains(err.Error(), "usage:") {
			t.Fatalf("err = %v, want usage error", err)
		}
	})

	t.Run("missing stimulus", func(t *testing.T) {
		var out bytes.Buffer
		err := run(t.Context(), cliOptions{specPath: exampleSpec, stimulus: "   ", noEmbed: true}, &out)
		if err == nil || !strings.Contains(err.Error(), "usage:") {
			t.Fatalf("err = %v, want usage error", err)
		}
	})

	t.Run("invalid depth", func(t *testing.T) {
		var out bytes.Buffer
		err := run(t.Context(), cliOptions{specPath: exampleSpec, stimulus: "자극", depth: -1, noEmbed: true}, &out)
		if err == nil || !strings.Contains(err.Error(), "depth") {
			t.Fatalf("err = %v, want depth error", err)
		}
	})

	t.Run("explicit zero depth", func(t *testing.T) {
		var out bytes.Buffer
		err := run(t.Context(), cliOptions{specPath: exampleSpec, stimulus: "자극", depth: 0, depthSet: true, noEmbed: true}, &out)
		if err == nil || !strings.Contains(err.Error(), "depth") {
			t.Fatalf("err = %v, want depth error", err)
		}
	})

	t.Run("missing api key", func(t *testing.T) {
		t.Setenv("TYPESAFE_API_KEY", "")
		var out bytes.Buffer
		err := run(t.Context(), cliOptions{specPath: exampleSpec, stimulus: "자극", noEmbed: true}, &out)
		if err == nil || !strings.Contains(err.Error(), "TYPESAFE_API_KEY") {
			t.Fatalf("err = %v, want TYPESAFE_API_KEY error", err)
		}
	})
}

func TestRunWithExecutesExampleSpecWithEmbedderAndSeed(t *testing.T) {
	exampleSpec := filepath.Join("..", "..", "examples", "immediate-response.json")
	client := mockScoreClient(t, func(r *http.Request) (float64, float64) {
		var req struct {
			State struct {
				Input struct {
					Stimulus string `json:"stimulus"`
					Context  string `json:"context"`
				} `json:"input"`
			} `json:"state"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		if req.State.Input.Stimulus != "따뜻한 격려" || req.State.Input.Context != "회의 직후" {
			t.Errorf("unexpected CLI input state: %+v", req.State.Input)
		}
		return 4, 4 // raw (+1.0, +1.0)
	})

	var gotEmbedTexts []string
	embedder := stubEmbedder(func(_ context.Context, texts []string) ([][]float64, error) {
		gotEmbedTexts = append([]string(nil), texts...)
		vecs := make([][]float64, len(texts))
		for i := range texts {
			vecs[i] = []float64{1, 0}
		}
		return vecs, nil
	})

	seed := int64(42)
	var out bytes.Buffer
	err := runWith(
		t.Context(),
		func() (*decisions.Client, error) { return client, nil },
		func() (braingate.Embedder, error) { return embedder, nil },
		cliOptions{
			specPath:        exampleSpec,
			stimulus:        "따뜻한 격려",
			previousContext: "회의 직후",
			depth:           2,
			seed:            &seed,
		},
		&out,
	)
	if err != nil {
		t.Fatalf("runWith error = %v", err)
	}
	if len(gotEmbedTexts) == 0 || gotEmbedTexts[0] != "회의 직후\n따뜻한 격려" {
		t.Fatalf("unexpected query embedding text: %v", gotEmbedTexts)
	}

	var result braingate.Result
	if err := json.Unmarshal(out.Bytes(), &result); err != nil {
		t.Fatalf("decode CLI stdout JSON: %v\noutput:\n%s", err, out.String())
	}
	if result.Final.StopReason != braingate.StopDepthReached || result.Final.Steps != 2 {
		t.Fatalf("final = %+v, want depth_reached after 2 steps", result.Final)
	}
	for i, step := range result.Steps {
		if step.Similarity == nil {
			t.Fatalf("step %d missing similarity when embedder enabled", i)
		}
	}
}

func TestRunWithNoEmbedSkipsEmbedder(t *testing.T) {
	exampleSpec := filepath.Join("..", "..", "examples", "immediate-response.json")
	client := mockScoreClient(t, func(*http.Request) (float64, float64) {
		return 4, 4
	})

	var out bytes.Buffer
	err := runWith(
		t.Context(),
		func() (*decisions.Client, error) { return client, nil },
		func() (braingate.Embedder, error) {
			t.Fatal("embedder factory should not be called when noEmbed is true")
			return nil, nil
		},
		cliOptions{
			specPath: exampleSpec,
			stimulus: "따뜻한 격려",
			depth:    1,
			noEmbed:  true,
		},
		&out,
	)
	if err != nil {
		t.Fatalf("runWith error = %v", err)
	}

	var result braingate.Result
	if err := json.Unmarshal(out.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Final.Steps != 1 || result.Steps[0].Similarity != nil {
		t.Fatalf("unexpected no-embed result: %+v", result)
	}
}

type stubEmbedder func(context.Context, []string) ([][]float64, error)

func (fn stubEmbedder) Embed(ctx context.Context, texts []string) ([][]float64, error) {
	return fn(ctx, texts)
}

type mockTransport func(*http.Request) (*http.Response, error)

func (transport mockTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return transport(r)
}

func mockScoreClient(t *testing.T, scores func(*http.Request) (float64, float64)) *decisions.Client {
	t.Helper()
	client, err := decisions.NewClient(decisions.Config{
		API:   decisions.SystemOneAPI,
		Model: "test",
		HTTPClient: &http.Client{Transport: mockTransport(func(r *http.Request) (*http.Response, error) {
			defer func() {
				if err := r.Body.Close(); err != nil {
					t.Errorf("close request body: %v", err)
				}
			}()
			v, a := scores(r)
			scoreObj := func(score float64) map[string]any {
				probs := map[string]float64{"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}
				switch int(score) {
				case 0:
					probs["0"] = 1
				case 1:
					probs["1"] = 1
				case 2:
					probs["2"] = 1
				case 3:
					probs["3"] = 1
				default:
					probs["4"] = 1
				}
				return map[string]any{
					"type":          "score",
					"score":         score,
					"confidence":    0.9,
					"probabilities": probs,
				}
			}
			body, err := json.Marshal(map[string]any{
				"model": "test",
				"usage": map[string]any{"input_tokens": 1, "output_tokens": 1},
				"answers": map[string]any{
					"valence":    scoreObj(v),
					"activation": scoreObj(a),
				},
			})
			if err != nil {
				return nil, err
			}
			return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(bytes.NewReader(body))}, nil
		})},
	})
	if err != nil {
		t.Fatal(err)
	}
	return client
}
