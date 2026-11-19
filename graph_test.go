package braingate_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/snowmerak/braingate"
	"github.com/snowmerak/decisions"
)

func TestRunRawRoutingInertiaAndStressFromSecondGate(t *testing.T) {
	gate3 := &braingate.Gate{
		ID:         "gate3",
		Experience: "세 번째 기억: 급격히 경계심이 올라온다.",
		Alpha:      0.25,
	}
	gate2 := &braingate.Gate{
		ID:         "gate2",
		Experience: "두 번째 기억: 미묘한 불호가 감지된다.",
		Routes: []braingate.Route{
			{
				Name:       "blended_trap",
				Valence:    braingate.Bounds{Min: 0.0, Max: 1.0},
				Activation: braingate.Bounds{Min: -1.0, Max: 1.0},
				Stop:       true,
			},
			{
				Name:       "raw_negative",
				Valence:    braingate.Bounds{Min: -1.0, Max: -0.01},
				Activation: braingate.Bounds{Min: -1.0, Max: 1.0},
				Next:       gate3,
			},
		},
	}
	gate1 := &braingate.Gate{
		ID:         "gate1",
		Experience: "첫 번째 기억: 강한 호감과 접근 충동이 일어난다.",
		Routes: []braingate.Route{
			{
				Name:       "positive_active",
				Valence:    braingate.Bounds{Min: 0.5, Max: 1.0},
				Activation: braingate.Bounds{Min: 0.5, Max: 1.0},
				Next:       gate2,
			},
		},
	}

	graph := &braingate.Graph{
		Start: gate1,
		Alpha: 0.5,
	}

	type reading struct {
		valenceScore    float64
		activationScore float64
		valenceConf     float64
		activationConf  float64
	}
	script := []reading{
		{valenceScore: 4, activationScore: 4, valenceConf: 0.91, activationConf: 0.87}, // raw: (+1.0, +1.0)
		{valenceScore: 1, activationScore: 2, valenceConf: 0.78, activationConf: 0.72}, // raw: (-0.5, 0.0), blended: (0.0, +0.25)
		{valenceScore: 0, activationScore: 4, valenceConf: 0.84, activationConf: 0.88}, // raw: (-1.0, +1.0)
	}
	expectedGates := []*braingate.Gate{gate1, gate2, gate3}
	const wantInput = "반가운 척 다가오지만 말투가 어딘가 불편한 동료"

	calls := 0
	client := testScoreClient(t, func(r *http.Request) scoreResponse {
		if calls >= len(script) {
			t.Fatalf("unexpected extra Evaluate call %d", calls+1)
		}
		var req struct {
			State struct {
				Input      string           `json:"input"`
				GateID     string           `json:"gate_id"`
				Visit      int              `json:"visit"`
				Experience string           `json:"experience"`
				State      braingate.Affect `json:"state"`
				History    []braingate.Step `json:"history"`
			} `json:"state"`
			Assets map[string]struct {
				Format string `json:"format"`
			} `json:"assets"`
			Questions map[string]struct {
				Type     string `json:"type"`
				Criteria []any  `json:"criteria"`
			} `json:"questions"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode Evaluate request: %v", err)
		}

		wantGate := expectedGates[calls]
		if req.State.Input != wantInput {
			t.Errorf("call %d input = %q, want %q", calls, req.State.Input, wantInput)
		}
		if req.State.GateID != wantGate.ID || req.State.Visit != 1 || req.State.Experience != wantGate.Experience {
			t.Errorf("call %d gate metadata = (%q, %d, %q), want (%q, 1, %q)",
				calls, req.State.GateID, req.State.Visit, req.State.Experience, wantGate.ID, wantGate.Experience)
		}
		if len(req.State.History) != calls {
			t.Errorf("call %d history len = %d, want %d", calls, len(req.State.History), calls)
		}
		if calls > 0 {
			prev := req.State.History[calls-1].State
			if !almostEqual(req.State.State.Valence, prev.Valence) ||
				!almostEqual(req.State.State.Activation, prev.Activation) ||
				!almostEqual(req.State.State.Stress, prev.Stress) {
				t.Errorf("call %d state = %+v, want previous step state %+v", calls, req.State.State, prev)
			}
		}
		if _, ok := req.Assets["snapshot"]; !ok {
			t.Errorf("call %d missing expected image asset", calls)
		}
		for _, qName := range []string{"valence", "activation"} {
			q, ok := req.Questions[qName]
			if !ok || q.Type != "score" || len(q.Criteria) != 5 {
				t.Errorf("call %d question %q = %+v, want 5-level score question", calls, qName, q)
			}
		}

		item := script[calls]
		calls++
		return scoreResponse{
			valence:    &scoreAnswerSpec{score: item.valenceScore, confidence: item.valenceConf},
			activation: &scoreAnswerSpec{score: item.activationScore, confidence: item.activationConf},
		}
	})

	input := decisions.Input{
		State: wantInput,
		Assets: map[string]decisions.ImageAsset{
			"snapshot": {Format: "png", DataBase64: "iVBORw0KGgo="},
		},
	}
	got, err := graph.Run(t.Context(), client, input)
	if err != nil {
		t.Fatalf("Run error = %v", err)
	}
	if calls != 3 || len(got.Steps) != 3 {
		t.Fatalf("calls=%d, steps=%d, want 3", calls, len(got.Steps))
	}

	delta2 := math.Hypot(0.0-0.5, 0.25-0.5)
	delta3 := math.Hypot(-0.25-0.0, 0.4375-0.25)
	wantStress := delta2 + delta3

	// Step 1: stress and delta must be 0.
	s0 := got.Steps[0]
	if s0.Gate != "gate1" || s0.Route != "positive_active" || s0.Next != "gate2" ||
		!almostEqual(s0.Raw.Valence, 1.0) || !almostEqual(s0.Raw.Activation, 1.0) ||
		!almostEqual(s0.State.Valence, 0.5) || !almostEqual(s0.State.Activation, 0.5) ||
		s0.State.Delta != 0 || s0.State.Stress != 0 {
		t.Fatalf("step 0 = %+v", s0)
	}

	// Step 2: blended valence is 0.0, raw valence is -0.5 -> must route via raw_negative, not blended_trap.
	s1 := got.Steps[1]
	if s1.Gate != "gate2" || s1.Route != "raw_negative" || s1.Next != "gate3" ||
		!almostEqual(s1.Raw.Valence, -0.5) || !almostEqual(s1.Raw.Activation, 0.0) ||
		!almostEqual(s1.State.Valence, 0.0) || !almostEqual(s1.State.Activation, 0.25) ||
		!almostEqual(s1.State.Delta, delta2) || !almostEqual(s1.State.Stress, delta2) {
		t.Fatalf("step 1 = %+v", s1)
	}

	// Step 3: gate3 overrides alpha to 0.25 and reaches end of graph.
	s2 := got.Steps[2]
	if s2.Gate != "gate3" || s2.Route != "" || s2.Next != "" ||
		!almostEqual(s2.Raw.Valence, -1.0) || !almostEqual(s2.Raw.Activation, 1.0) ||
		!almostEqual(s2.State.Valence, -0.25) || !almostEqual(s2.State.Activation, 0.4375) ||
		!almostEqual(s2.State.Delta, delta3) || !almostEqual(s2.State.Stress, wantStress) {
		t.Fatalf("step 2 = %+v", s2)
	}

	if got.Final.StopReason != braingate.StopEndOfGraph || got.Final.Steps != 3 ||
		!almostEqual(got.Final.Valence, -0.25) || !almostEqual(got.Final.Activation, 0.4375) ||
		!almostEqual(got.Final.Stress, wantStress) {
		t.Fatalf("final = %+v", got.Final)
	}
}

func TestRunStopReasons(t *testing.T) {
	t.Run("explicit_stop", func(t *testing.T) {
		start := &braingate.Gate{
			ID:         "start",
			Experience: "호감은 있으나 멈춰야 한다.",
			Routes: []braingate.Route{
				{
					Name:       "hold",
					Valence:    braingate.Bounds{Min: -1, Max: 1},
					Activation: braingate.Bounds{Min: -1, Max: 1},
					Stop:       true,
				},
			},
		}
		client := fixedScoreClient(t, 3, 1)
		got, err := (&braingate.Graph{Start: start}).Run(t.Context(), client, decisions.Input{State: "stimulus"})
		if err != nil {
			t.Fatalf("Run error = %v", err)
		}
		if got.Final.StopReason != braingate.StopExplicit || got.Final.Steps != 1 {
			t.Fatalf("final = %+v, want explicit_stop at 1 step", got.Final)
		}
		if got.Steps[0].Route != "hold" || got.Steps[0].Next != "" {
			t.Fatalf("step 0 = %+v, want route hold and empty next", got.Steps[0])
		}
	})

	t.Run("end_of_graph_nil_next", func(t *testing.T) {
		start := &braingate.Gate{
			ID:         "start",
			Experience: "단일 기억 후 자연 종료.",
			Routes: []braingate.Route{
				{
					Name:       "terminal_branch",
					Valence:    braingate.Bounds{Min: -1, Max: 1},
					Activation: braingate.Bounds{Min: -1, Max: 1},
					Next:       nil,
				},
			},
		}
		client := fixedScoreClient(t, 2, 2)
		got, err := (&braingate.Graph{Start: start}).Run(t.Context(), client, decisions.Input{State: "stimulus"})
		if err != nil {
			t.Fatalf("Run error = %v", err)
		}
		if got.Final.StopReason != braingate.StopEndOfGraph || got.Steps[0].Route != "terminal_branch" {
			t.Fatalf("got = %+v", got)
		}
	})

	t.Run("end_of_graph_unmatched_region", func(t *testing.T) {
		start := &braingate.Gate{
			ID:         "start",
			Experience: "특정 영역 외에는 이어지지 않는다.",
			Routes: []braingate.Route{
				{
					Name:       "only_high_positive",
					Valence:    braingate.Bounds{Min: 0.7, Max: 1.0},
					Activation: braingate.Bounds{Min: 0.7, Max: 1.0},
					Stop:       true,
				},
			},
		}
		client := fixedScoreClient(t, 2, 2) // raw (0, 0) does not match
		got, err := (&braingate.Graph{Start: start}).Run(t.Context(), client, decisions.Input{State: "stimulus"})
		if err != nil {
			t.Fatalf("Run error = %v", err)
		}
		if got.Final.StopReason != braingate.StopEndOfGraph || got.Steps[0].Route != "" || got.Steps[0].Next != "" {
			t.Fatalf("got = %+v", got)
		}
	})

	t.Run("max_visits_oscillation", func(t *testing.T) {
		gateA := &braingate.Gate{ID: "A", Experience: "다가가고 싶은 충동"}
		gateB := &braingate.Gate{ID: "B", Experience: "물러서고 싶은 경계"}
		full := braingate.Bounds{Min: -1, Max: 1}
		gateA.Routes = []braingate.Route{{Name: "to_B", Valence: full, Activation: full, Next: gateB}}
		gateB.Routes = []braingate.Route{{Name: "to_A", Valence: full, Activation: full, Next: gateA}}

		calls := 0
		client := testScoreClient(t, func(*http.Request) scoreResponse {
			calls++
			if calls%2 == 1 {
				return scoreResponse{
					valence:    &scoreAnswerSpec{score: 4, confidence: 0.9},
					activation: &scoreAnswerSpec{score: 4, confidence: 0.9},
				}
			}
			return scoreResponse{
				valence:    &scoreAnswerSpec{score: 0, confidence: 0.9},
				activation: &scoreAnswerSpec{score: 0, confidence: 0.9},
			}
		})

		got, err := (&braingate.Graph{Start: gateA, Alpha: 0.5, MaxVisits: 2}).Run(t.Context(), client, decisions.Input{State: "conflict"})
		if err != nil {
			t.Fatalf("Run error = %v", err)
		}
		if got.Final.StopReason != braingate.StopMaxVisits || got.Final.Steps != 4 {
			t.Fatalf("final = %+v, want max_visits after 4 steps", got.Final)
		}
		if got.Steps[0].Visit != 1 || got.Steps[2].Visit != 2 || got.Steps[3].Visit != 2 || got.Steps[3].Next != "A" {
			t.Fatalf("unexpected visits or next in steps: %+v", got.Steps)
		}
		if got.Steps[0].State.Stress != 0 || got.Final.Stress <= got.Steps[1].State.Stress {
			t.Fatalf("expected stress to rise across oscillation steps, got steps=%+v", got.Steps)
		}
	})

	t.Run("max_steps", func(t *testing.T) {
		g3 := &braingate.Gate{ID: "g3", Experience: "3단계"}
		g2 := &braingate.Gate{
			ID:         "g2",
			Experience: "2단계",
			Routes:     []braingate.Route{{Name: "to_g3", Valence: braingate.Bounds{Min: -1, Max: 1}, Activation: braingate.Bounds{Min: -1, Max: 1}, Next: g3}},
		}
		g1 := &braingate.Gate{
			ID:         "g1",
			Experience: "1단계",
			Routes:     []braingate.Route{{Name: "to_g2", Valence: braingate.Bounds{Min: -1, Max: 1}, Activation: braingate.Bounds{Min: -1, Max: 1}, Next: g2}},
		}

		client := fixedScoreClient(t, 3, 3)
		got, err := (&braingate.Graph{Start: g1, MaxSteps: 2}).Run(t.Context(), client, decisions.Input{State: "stimulus"})
		if err != nil {
			t.Fatalf("Run error = %v", err)
		}
		if got.Final.StopReason != braingate.StopMaxSteps || got.Final.Steps != 2 || got.Steps[1].Next != "g3" {
			t.Fatalf("got = %+v", got)
		}
	})
}

func TestRunPreservesPartialStepsOnRefusal(t *testing.T) {
	gate2 := &braingate.Gate{ID: "gate2", Experience: "두 번째 게이트"}
	gate1 := &braingate.Gate{
		ID:         "gate1",
		Experience: "첫 번째 게이트",
		Routes: []braingate.Route{
			{
				Name:       "next",
				Valence:    braingate.Bounds{Min: -1, Max: 1},
				Activation: braingate.Bounds{Min: -1, Max: 1},
				Next:       gate2,
			},
		},
	}

	calls := 0
	client := testScoreClient(t, func(*http.Request) scoreResponse {
		calls++
		if calls == 1 {
			return scoreResponse{
				valence:    &scoreAnswerSpec{score: 4, confidence: 0.9},
				activation: &scoreAnswerSpec{score: 0, confidence: 0.8},
			}
		}
		return scoreResponse{
			valence:           &scoreAnswerSpec{score: 2, confidence: 0.7},
			activationRefused: true,
		}
	})

	got, err := (&braingate.Graph{Start: gate1, Alpha: 0.5}).Run(t.Context(), client, decisions.Input{State: "stimulus"})
	if !errors.Is(err, decisions.ErrRefused) {
		t.Fatalf("err = %v, want decisions.ErrRefused", err)
	}
	if len(got.Steps) != 1 || got.Final.Steps != 1 {
		t.Fatalf("expected 1 preserved step on refusal, got %+v", got)
	}
	if !almostEqual(got.Final.Valence, 0.5) || !almostEqual(got.Final.Activation, -0.5) || got.Final.Stress != 0 {
		t.Fatalf("unexpected partial final state: %+v", got.Final)
	}
	if got.Final.StopReason != "" {
		t.Fatalf("expected empty StopReason on error, got %q", got.Final.StopReason)
	}
}

func TestRunRejectsInvalidInputs(t *testing.T) {
	client := fixedScoreClient(t, 2, 2)
	validGate := &braingate.Gate{ID: "start", Experience: "경험"}

	cases := []struct {
		name   string
		graph  *braingate.Graph
		client *decisions.Client
		want   string
	}{
		{name: "nil graph", graph: nil, client: client, want: "starting gate"},
		{name: "nil start", graph: &braingate.Graph{}, client: client, want: "starting gate"},
		{name: "nil client", graph: &braingate.Graph{Start: validGate}, client: nil, want: "client"},
		{name: "negative alpha", graph: &braingate.Graph{Start: validGate, Alpha: -0.1}, client: client, want: "alpha"},
		{name: "alpha > 1", graph: &braingate.Graph{Start: validGate, Alpha: 1.1}, client: client, want: "alpha"},
		{name: "nan alpha", graph: &braingate.Graph{Start: validGate, Alpha: math.NaN()}, client: client, want: "alpha"},
		{name: "negative max_visits", graph: &braingate.Graph{Start: validGate, MaxVisits: -1}, client: client, want: "max_visits"},
		{name: "negative max_steps", graph: &braingate.Graph{Start: validGate, MaxSteps: -1}, client: client, want: "max_steps"},
		{name: "blank gate id", graph: &braingate.Graph{Start: &braingate.Gate{ID: " ", Experience: "x"}}, client: client, want: "gate ID"},
		{name: "blank experience", graph: &braingate.Graph{Start: &braingate.Gate{ID: "g", Experience: " "}}, client: client, want: "experience"},
		{name: "invalid gate alpha", graph: &braingate.Graph{Start: &braingate.Gate{ID: "g", Experience: "x", Alpha: 1.5}}, client: client, want: "alpha"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := tc.graph.Run(t.Context(), tc.client, decisions.Input{State: "stimulus"})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Run error = %v, want substring %q", err, tc.want)
			}
		})
	}
}

func TestRunKeepsHistoryOrderedAcrossSteps(t *testing.T) {
	g2 := &braingate.Gate{ID: "g2", Experience: "둘째 기억"}
	g1 := &braingate.Gate{
		ID:         "g1",
		Experience: "첫째 기억",
		Routes: []braingate.Route{
			{Name: "step", Valence: braingate.Bounds{Min: -1, Max: 1}, Activation: braingate.Bounds{Min: -1, Max: 1}, Next: g2},
		},
	}

	var seenHistory []string
	client := testScoreClient(t, func(r *http.Request) scoreResponse {
		var req struct {
			State struct {
				History []braingate.Step `json:"history"`
			} `json:"state"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		ids := make([]string, 0, len(req.State.History))
		for _, h := range req.State.History {
			ids = append(ids, h.Gate)
		}
		seenHistory = ids
		return scoreResponse{
			valence:    &scoreAnswerSpec{score: 3, confidence: 0.8},
			activation: &scoreAnswerSpec{score: 3, confidence: 0.8},
		}
	})

	if _, err := (&braingate.Graph{Start: g1}).Run(t.Context(), client, decisions.Input{State: "s"}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(seenHistory, []string{"g1"}) {
		t.Fatalf("second call history gates = %v, want [g1]", seenHistory)
	}
}

type scoreAnswerSpec struct {
	score      float64
	confidence float64
}

type scoreResponse struct {
	valence           *scoreAnswerSpec
	activation        *scoreAnswerSpec
	activationRefused bool
}

type testTransport func(*http.Request) (*http.Response, error)

func (transport testTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return transport(r)
}

func fixedScoreClient(t *testing.T, valenceScore, activationScore float64) *decisions.Client {
	t.Helper()
	return testScoreClient(t, func(*http.Request) scoreResponse {
		return scoreResponse{
			valence:    &scoreAnswerSpec{score: valenceScore, confidence: 0.85},
			activation: &scoreAnswerSpec{score: activationScore, confidence: 0.85},
		}
	})
}

func testScoreClient(t *testing.T, respond func(*http.Request) scoreResponse) *decisions.Client {
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
			spec := respond(r)
			answers := map[string]any{
				"valence": encodeScoreAnswer(spec.valence),
			}
			if spec.activationRefused {
				answers["activation"] = map[string]any{"type": "refusal"}
			} else {
				answers["activation"] = encodeScoreAnswer(spec.activation)
			}
			payload, err := json.Marshal(map[string]any{
				"model":   "test",
				"usage":   map[string]any{"input_tokens": 1, "output_tokens": 1},
				"answers": answers,
			})
			if err != nil {
				return nil, fmt.Errorf("marshal mock response: %w", err)
			}
			return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(string(payload)))}, nil
		})},
	})
	if err != nil {
		t.Fatal(err)
	}
	return client
}

func encodeScoreAnswer(spec *scoreAnswerSpec) map[string]any {
	probs := map[string]float64{"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}
	idx := fmt.Sprintf("%d", int(math.Round(spec.score)))
	probs[idx] = 1.0
	return map[string]any{
		"type":          "score",
		"score":         spec.score,
		"confidence":    spec.confidence,
		"probabilities": probs,
	}
}

func almostEqual(got, want float64) bool {
	return math.Abs(got-want) <= 1e-9
}
