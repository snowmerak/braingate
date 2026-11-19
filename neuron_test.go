package braingate_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/snowmerak/braingate"
	"github.com/snowmerak/decisions"
)

func TestRunPassesHistoryAlongSelectedRoute(t *testing.T) {
	finish := &braingate.Neuron{ID: "finish", Instructions: "Make the final choice"}
	review := &braingate.Neuron{ID: "review", Instructions: "Review the first choice"}
	start := &braingate.Neuron{ID: "start", Instructions: "Make the first choice"}
	finish.Choices = []braingate.Choice{
		{Choice: decisions.Choice{Value: "yes"}},
		{Choice: decisions.Choice{Value: "no"}},
	}
	review.Choices = []braingate.Choice{
		{Choice: decisions.Choice{Value: "yes"}},
		{Choice: decisions.Choice{Value: "no"}, Next: finish},
	}
	start.Choices = []braingate.Choice{
		{Choice: decisions.Choice{Value: "yes"}, Next: review},
		{Choice: decisions.Choice{Value: "no"}},
	}

	route := []*braingate.Neuron{start, review, finish}
	want := make([]braingate.Selection, 0, len(route))
	for i, choice := range []string{"yes", "no", "yes"} {
		want = append(want, braingate.Selection{
			NeuronID: route[i].ID, Choice: choice, Confidence: 0.8,
			Probabilities: map[string]float64{"yes": 0.6, "no": 0.4},
		})
	}
	const input = "Initial evidence"
	calls := 0
	client := testClient(t, func(r *http.Request) string {
		if calls >= len(route) {
			t.Fatal("made another decision after a terminal choice")
		}
		var request struct {
			State struct {
				Input   string
				History []braingate.Selection
			}
			Questions map[string]struct{ Instructions string }
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if got := request.Questions["answer"].Instructions; got != route[calls].Instructions {
			t.Errorf("call %d instructions = %q, want %q", calls, got, route[calls].Instructions)
		}
		if request.State.Input != input || !reflect.DeepEqual(request.State.History, want[:calls]) {
			t.Errorf("call %d state = %+v, want input %q and history %+v", calls, request.State, input, want[:calls])
		}
		choice := want[calls].Choice
		calls++
		return choice
	})

	got, err := start.Run(t.Context(), client, decisions.Input{State: input})
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("Run = (%+v, %v), want (%+v, nil)", got, err, want)
	}
}

func TestRunReportsCycle(t *testing.T) {
	neuron := &braingate.Neuron{ID: "repeat", Instructions: "Choose whether to repeat"}
	neuron.Choices = []braingate.Choice{
		{Choice: decisions.Choice{Value: "yes"}, Next: neuron},
		{Choice: decisions.Choice{Value: "no"}},
	}
	calls := 0
	client := testClient(t, func(*http.Request) string {
		calls++
		if calls > 1 {
			t.Fatal("called Decisions again for a visited neuron")
		}
		return "yes"
	})

	history, err := neuron.Run(t.Context(), client, decisions.Input{})
	if !errors.Is(err, braingate.ErrCycle) || len(history) != 1 || history[0].NeuronID != neuron.ID {
		t.Fatalf("Run = (%+v, %v), want one completed selection and ErrCycle", history, err)
	}
}

type testTransport func(*http.Request) (*http.Response, error)

func (transport testTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return transport(r)
}

func testClient(t *testing.T, choose func(*http.Request) string) *decisions.Client {
	t.Helper()
	client, err := decisions.NewClient(decisions.Config{
		API:   decisions.SystemOneAPI,
		Model: "test",
		HTTPClient: &http.Client{Transport: testTransport(func(r *http.Request) (*http.Response, error) {
			defer func() {
				if err := r.Body.Close(); err != nil {
					t.Errorf("close request body: %v", err)
				}
			}()
			body := fmt.Sprintf(`{
				"model":"test", "usage":{"input_tokens":1,"output_tokens":1},
				"answers":{"answer":{"type":"choice","choice":%q,"confidence":0.8,"probabilities":{"yes":0.6,"no":0.4}}}
			}`, choose(r))
			return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(body))}, nil
		})},
	})
	if err != nil {
		t.Fatal(err)
	}
	return client
}
