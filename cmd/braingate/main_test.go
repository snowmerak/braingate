package main

import (
	"bytes"
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
		err := run(t.Context(), "", "자극", "", &out)
		if err == nil || !strings.Contains(err.Error(), "usage:") {
			t.Fatalf("err = %v, want usage error", err)
		}
	})

	t.Run("missing stimulus", func(t *testing.T) {
		var out bytes.Buffer
		err := run(t.Context(), exampleSpec, "   ", "", &out)
		if err == nil || !strings.Contains(err.Error(), "usage:") {
			t.Fatalf("err = %v, want usage error", err)
		}
	})

	t.Run("missing api key", func(t *testing.T) {
		t.Setenv("TYPESAFE_API_KEY", "")
		var out bytes.Buffer
		err := run(t.Context(), exampleSpec, "자극", "", &out)
		if err == nil || !strings.Contains(err.Error(), "TYPESAFE_API_KEY") {
			t.Fatalf("err = %v, want TYPESAFE_API_KEY error", err)
		}
	})
}

func TestRunWithExecutesExampleSpecAndOutputsResultJSON(t *testing.T) {
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
		return 4, 4 // raw (+1.0, +1.0) -> initial_appraisal -> eager_memory -> stabilized_approach
	})

	var out bytes.Buffer
	err := runWith(
		t.Context(),
		func() (*decisions.Client, error) { return client, nil },
		exampleSpec,
		"따뜻한 격려",
		"회의 직후",
		&out,
	)
	if err != nil {
		t.Fatalf("runWith error = %v", err)
	}

	var result braingate.Result
	if err := json.Unmarshal(out.Bytes(), &result); err != nil {
		t.Fatalf("decode CLI stdout JSON: %v\noutput:\n%s", err, out.String())
	}
	if result.Final.StopReason != braingate.StopEndOfGraph || result.Final.Steps != 3 {
		t.Fatalf("final = %+v, want end_of_graph after 3 steps", result.Final)
	}
	wantGates := []string{"initial_appraisal", "eager_memory", "stabilized_approach"}
	for i, wantGate := range wantGates {
		if result.Steps[i].Gate != wantGate {
			t.Fatalf("step %d gate = %q, want %q", i, result.Steps[i].Gate, wantGate)
		}
	}
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
