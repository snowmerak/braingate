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
	"github.com/snowmerak/braingate/internal/incidentdemo"
	"github.com/snowmerak/decisions"
)

func TestRunPassesHistoryAndComposesSelectedText(t *testing.T) {
	start := incidentdemo.NewGraph()
	want := []braingate.Selection{
		{NeuronID: "observation", Choice: "yes", Text: "배포 직후 오류율이 올랐습니다."},
		{NeuronID: "cause", Choice: "no", Text: "배포가 원인인지 확인하려면"},
		{NeuronID: "next_check", Choice: "yes", Text: "첫 오류 로그의 서비스와 버전을 이번 배포 대상과 대조하세요."},
	}
	for i := range want {
		want[i].Confidence = 0.8
		want[i].Probabilities = map[string]float64{"yes": 0.6, "no": 0.4}
	}
	instructions := []string{"shortly after deployment", "caused the errors", "affected service and its version"}
	const input = incidentdemo.SampleInput
	calls := 0
	client := testClient(t, func(r *http.Request) string {
		if calls >= len(want) {
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
		if got := request.Questions["answer"].Instructions; !strings.Contains(got, instructions[calls]) {
			t.Errorf("call %d instructions = %q, want %q in the question", calls, got, instructions[calls])
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
	const wantText = "배포 직후 오류율이 올랐습니다. 배포가 원인인지 확인하려면 첫 오류 로그의 서비스와 버전을 이번 배포 대상과 대조하세요."
	if text := braingate.ComposeText(got); text != wantText {
		t.Fatalf("ComposeText = %q, want %q", text, wantText)
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
