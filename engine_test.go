package braingate_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/snowmerak/braingate"
	"github.com/snowmerak/decisions"
)

func TestEngineRunWithEmbedderAndRandomPoolSampling(t *testing.T) {
	engine := &braingate.Engine{
		Temperament: braingate.Temperament{
			Depth:     2,
			Alpha:     0.5,
			Deadzone:  0.15,
			PoolSize:  2,
			MaxVisits: 2,
		},
		Memories: []braingate.Memory{
			{
				ID:         "mem_preembedded",
				Cue:        "사전 임베딩된 반가움",
				Position:   braingate.Point{Valence: 0.9, Activation: 0.9},
				Experience: "기분 좋게 다가갔던 기억",
				Embedding:  []float64{1, 0},
			},
			{
				ID:         "mem_semantic",
				Cue:        "발표 칭찬의 보람",
				Position:   braingate.Point{Valence: 0.7, Activation: 0.6},
				Experience: "발표를 인정받고 기뻤던 기억",
			},
			{
				ID:         "mem_far_q1",
				Cue:        "먼 1사분면 기억",
				Position:   braingate.Point{Valence: 0.1, Activation: 0.1},
				Experience: "사분면 내 거리가 멀고 의미도 먼 기억",
			},
			{
				ID:         "mem_wrong_quadrant",
				Cue:        "다른 사분면 방어 기억",
				Position:   braingate.Point{Valence: -0.9, Activation: 0.9},
				Experience: "같은 주제여도 사분면이 다르면 풀에서 제외되어야 함",
			},
		},
	}

	embedCalls := 0
	var gotBatch []string
	embedder := fakeEmbedder(func(_ context.Context, texts []string) ([][]float64, error) {
		embedCalls++
		gotBatch = append([]string(nil), texts...)
		// Batch order: [query, mem_semantic, mem_far_q1, mem_wrong_quadrant]
		return [][]float64{
			{1, 0},   // query
			{0.9, 0}, // mem_semantic (cosine 1.0)
			{-1, 0},  // mem_far_q1 (cosine -1.0)
			{1, 0},   // mem_wrong_quadrant (high cosine, filtered out by quadrant!)
		}, nil
	})

	// Script: Step 0 -> (+1.0, +1.0); Depth 1 -> (+0.5, +0.5); Depth 2 -> (-0.5, +1.0).
	scores := [][2]float64{
		{4, 4}, // initial: (+1.0, +1.0)
		{3, 3}, // depth 1: (+0.5, +0.5) -> stays in Q1 (+/+)
		{1, 4}, // depth 2: (-0.5, +1.0)
	}
	calls := 0
	var evalMemories []string
	client := testScoreClient(t, func(r *http.Request) scoreResponse {
		if calls >= len(scores) {
			t.Fatalf("unexpected extra Evaluate call %d", calls+1)
		}
		var req struct {
			State struct {
				Input      string           `json:"input"`
				Depth      int              `json:"depth"`
				MemoryID   string           `json:"memory_id"`
				Visit      int              `json:"visit"`
				Cue        string           `json:"cue"`
				Experience string           `json:"experience"`
				State      braingate.Affect `json:"state"`
				History    []braingate.Step `json:"history"`
			} `json:"state"`
			Assets map[string]any `json:"assets"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		if _, ok := req.Assets["img"]; !ok {
			t.Errorf("call %d missing forwarded asset", calls)
		}
		if calls == 0 {
			if req.State.MemoryID != "" || req.State.Depth != 0 {
				t.Errorf("initial call unexpectedly carried memory metadata: %+v", req.State)
			}
		} else {
			evalMemories = append(evalMemories, req.State.MemoryID)
			if req.State.Depth != calls || len(req.State.History) != calls-1 {
				t.Errorf("call %d depth=%d history=%d", calls, req.State.Depth, len(req.State.History))
			}
		}
		pair := scores[calls]
		calls++
		return scoreResponse{
			valence:    &scoreAnswerSpec{score: pair[0], confidence: 0.9},
			activation: &scoreAnswerSpec{score: pair[1], confidence: 0.85},
		}
	})

	rng := rand.New(rand.NewPCG(1, 2))
	input := decisions.Input{
		State:  "발표 잘 들었어요",
		Assets: map[string]decisions.ImageAsset{"img": {Format: "png", DataBase64: "iVBORw0KGgo="}},
	}
	got, err := engine.Run(t.Context(), client, input, braingate.RunOptions{
		Embedder: embedder,
		Rand:     rng,
	})
	if err != nil {
		t.Fatalf("Run error = %v", err)
	}

	if embedCalls != 1 {
		t.Fatalf("embedCalls = %d, want 1 lazy batch call", embedCalls)
	}
	if len(gotBatch) != 4 || gotBatch[0] != "발표 잘 들었어요" {
		t.Fatalf("unexpected embed batch (preembedded memory should be skipped): %v", gotBatch)
	}
	if got.Initial.Source != "evaluated" || !almostEqual(got.Initial.Valence, 1.0) || !almostEqual(got.Initial.Activation, 1.0) {
		t.Fatalf("initial = %+v", got.Initial)
	}
	if got.Final.StopReason != braingate.StopDepthReached || got.Final.Steps != 2 {
		t.Fatalf("final = %+v", got.Final)
	}

	// Verify pool contains only top PoolSize (2) from Q1, never mem_wrong_quadrant.
	step1 := got.Steps[0]
	if !reflect.DeepEqual(step1.Pool, []string{"mem_preembedded", "mem_semantic"}) {
		t.Fatalf("step 1 pool = %v, want [mem_preembedded mem_semantic]", step1.Pool)
	}
	if step1.Similarity == nil || !almostEqual(*step1.Similarity, 1.0) {
		t.Fatalf("step 1 similarity = %v, want 1.0", step1.Similarity)
	}

	// Verify immediate repeat cooldown: step 2 must not include step 1's chosen memory in its pool.
	step2 := got.Steps[1]
	if step2.Memory == step1.Memory {
		t.Fatalf("step 2 picked same memory %q immediately after step 1", step2.Memory)
	}
	for _, id := range step2.Pool {
		if id == step1.Memory {
			t.Fatalf("step 2 pool %v should exclude immediate previous memory %q", step2.Pool, step1.Memory)
		}
	}
	if !reflect.DeepEqual(evalMemories, []string{step1.Memory, step2.Memory}) {
		t.Fatalf("evaluated memories %v do not match step records", evalMemories)
	}

	// Verify cumulative state & stress math:
	// Initial = (1.0, 1.0), Step 1 raw = (0.5, 0.5), alpha = 0.5 => State = (0.75, 0.75)
	delta1 := math.Hypot(0.75-1.0, 0.75-1.0)
	// Step 2 raw = (-0.5, 1.0), alpha = 0.5 => State = (0.125, 0.875)
	delta2 := math.Hypot(0.125-0.75, 0.875-0.75)
	if !almostEqual(step1.State.Valence, 0.75) || !almostEqual(step1.State.Activation, 0.75) || !almostEqual(step1.State.Delta, delta1) {
		t.Fatalf("step 1 state = %+v", step1.State)
	}
	if !almostEqual(got.Final.Valence, 0.125) || !almostEqual(got.Final.Activation, 0.875) || !almostEqual(got.Final.Stress, delta1+delta2) {
		t.Fatalf("final = %+v, want (0.125, 0.875, %v)", got.Final, delta1+delta2)
	}
}

func TestEngineStopDeadzone(t *testing.T) {
	memories := []braingate.Memory{
		{ID: "m1", Position: braingate.Point{Valence: 0.6, Activation: 0.6}, Experience: "호의 기억"},
	}

	t.Run("stops at initial deadzone with zero embedder calls", func(t *testing.T) {
		embedCalls := 0
		embedder := fakeEmbedder(func(context.Context, []string) ([][]float64, error) {
			embedCalls++
			return nil, errors.New("should not be called")
		})
		client := fixedScoreClient(t, 2, 2) // raw (0, 0)
		engine := &braingate.Engine{
			Temperament: braingate.Temperament{Depth: 4, Alpha: 0.5, Deadzone: 0.15, PoolSize: 3, MaxVisits: 2},
			Memories:    memories,
		}
		got, err := engine.Run(t.Context(), client, decisions.Input{State: "담담한 인사"}, braingate.RunOptions{Embedder: embedder})
		if err != nil {
			t.Fatalf("Run error = %v", err)
		}
		if embedCalls != 0 {
			t.Fatalf("embedCalls = %d, want 0", embedCalls)
		}
		if got.Final.StopReason != braingate.StopDeadzone || got.Final.Steps != 0 || len(got.Steps) != 0 {
			t.Fatalf("got = %+v", got)
		}
	})

	t.Run("stops after step lands in deadzone while keeping residual affect", func(t *testing.T) {
		client := fixedScoreClient(t, 2, 2) // memory reading raw (0, 0) lands inside deadzone
		engine := &braingate.Engine{
			Temperament: braingate.Temperament{Depth: 4, Alpha: 0.5, Deadzone: 0.15, PoolSize: 3, MaxVisits: 2},
			Memories:    memories,
		}
		got, err := engine.Run(t.Context(), client, decisions.Input{State: "자극"}, braingate.RunOptions{
			Initial: &braingate.Point{Valence: 0.8, Activation: 0.8},
		})
		if err != nil {
			t.Fatalf("Run error = %v", err)
		}
		if got.Final.StopReason != braingate.StopDeadzone || got.Final.Steps != 1 {
			t.Fatalf("final = %+v", got.Final)
		}
		// Residual affect after blending (0.8, 0.8) toward (0, 0) with alpha 0.5 is (0.4, 0.4).
		wantDelta := math.Hypot(-0.4, -0.4)
		if !almostEqual(got.Final.Valence, 0.4) || !almostEqual(got.Final.Activation, 0.4) || !almostEqual(got.Final.Stress, wantDelta) {
			t.Fatalf("residual final = %+v, want (0.4, 0.4, %v)", got.Final, wantDelta)
		}
	})
}

func TestEngineStopNoMemory(t *testing.T) {
	engine := &braingate.Engine{
		Temperament: braingate.Temperament{Depth: 4, Alpha: 0.5, Deadzone: 0.15, PoolSize: 3, MaxVisits: 2},
		Memories: []braingate.Memory{
			{ID: "q3_only", Position: braingate.Point{Valence: -0.7, Activation: -0.7}, Experience: "회피 기억"},
		},
	}
	client := fixedScoreClient(t, 4, 4) // Q1 (+/+)
	got, err := engine.Run(t.Context(), client, decisions.Input{State: "자극"}, braingate.RunOptions{
		Initial: &braingate.Point{Valence: 0.5, Activation: 0.5},
	})
	if err != nil {
		t.Fatalf("Run error = %v", err)
	}
	if got.Final.StopReason != braingate.StopNoMemory || got.Final.Steps != 0 {
		t.Fatalf("got = %+v", got)
	}
}

func TestEngineRunSuppliedInitialMoodSkipsInitialEvaluate(t *testing.T) {
	engine := &braingate.Engine{
		Temperament: braingate.Temperament{Depth: 1, Alpha: 0.5, Deadzone: 0.1, PoolSize: 2, MaxVisits: 1},
		Memories: []braingate.Memory{
			{ID: "q1", Position: braingate.Point{Valence: 0.5, Activation: 0.5}, Experience: "긍정 기억", Alpha: 0.2},
			{ID: "q3", Position: braingate.Point{Valence: -0.5, Activation: -0.5}, Experience: "부정 기억"},
		},
	}

	calls := 0
	client := testScoreClient(t, func(*http.Request) scoreResponse {
		calls++
		return scoreResponse{
			valence:    &scoreAnswerSpec{score: 4, confidence: 0.9},
			activation: &scoreAnswerSpec{score: 4, confidence: 0.9},
		}
	})

	got, err := engine.Run(t.Context(), client, decisions.Input{State: "농담"}, braingate.RunOptions{
		Initial: &braingate.Point{Valence: 0.3, Activation: 0.2},
	})
	if err != nil {
		t.Fatalf("Run error = %v", err)
	}
	if calls != 1 {
		t.Fatalf("Evaluate calls = %d, want 1 (Step 0 should be skipped)", calls)
	}
	if got.Initial.Source != "supplied" || got.Steps[0].Memory != "q1" {
		t.Fatalf("got = %+v", got)
	}
	// Memory-level alpha = 0.2 from (0.3, 0.2) toward (1.0, 1.0): (0.44, 0.36)
	if !almostEqual(got.Final.Valence, 0.44) || !almostEqual(got.Final.Activation, 0.36) {
		t.Fatalf("final = %+v, want (0.44, 0.36)", got.Final)
	}
}

func TestEnginePreservesPartialStepsOnRefusal(t *testing.T) {
	engine := &braingate.Engine{
		Temperament: braingate.Temperament{Depth: 3, Alpha: 0.5, Deadzone: 0.1, PoolSize: 2, MaxVisits: 2},
		Memories: []braingate.Memory{
			{ID: "m1", Position: braingate.Point{Valence: 0.7, Activation: 0.7}, Experience: "첫 기억"},
			{ID: "m2", Position: braingate.Point{Valence: 0.4, Activation: 0.4}, Experience: "둘째 기억"},
		},
	}

	calls := 0
	client := testScoreClient(t, func(*http.Request) scoreResponse {
		calls++
		if calls <= 2 {
			return scoreResponse{
				valence:    &scoreAnswerSpec{score: 4, confidence: 0.9},
				activation: &scoreAnswerSpec{score: 4, confidence: 0.9},
			}
		}
		return scoreResponse{
			valence:           &scoreAnswerSpec{score: 3, confidence: 0.8},
			activationRefused: true,
		}
	})

	got, err := engine.Run(t.Context(), client, decisions.Input{State: "자극"}, braingate.RunOptions{})
	if !errors.Is(err, decisions.ErrRefused) {
		t.Fatalf("err = %v, want decisions.ErrRefused", err)
	}
	if got.Initial.Source != "evaluated" || len(got.Steps) != 1 || got.Final.Steps != 1 {
		t.Fatalf("expected initial + 1 completed step preserved, got %+v", got)
	}
	if got.Final.StopReason != "" {
		t.Fatalf("expected empty StopReason on error, got %q", got.Final.StopReason)
	}
}

func TestEngineRejectsInvalidInputs(t *testing.T) {
	client := fixedScoreClient(t, 2, 2)
	validMem := []braingate.Memory{{ID: "m1", Position: braingate.Point{Valence: 0.5, Activation: 0.5}, Experience: "기억"}}

	cases := []struct {
		name   string
		engine *braingate.Engine
		client *decisions.Client
		opts   braingate.RunOptions
		want   string
	}{
		{name: "nil engine", engine: nil, client: client, want: "engine"},
		{name: "nil client", engine: &braingate.Engine{Memories: validMem}, client: nil, want: "client"},
		{name: "negative depth", engine: &braingate.Engine{Temperament: braingate.Temperament{Depth: -1}, Memories: validMem}, client: client, want: "depth"},
		{name: "invalid alpha", engine: &braingate.Engine{Temperament: braingate.Temperament{Alpha: 1.5}, Memories: validMem}, client: client, want: "alpha"},
		{name: "invalid deadzone", engine: &braingate.Engine{Temperament: braingate.Temperament{Deadzone: 1.0}, Memories: validMem}, client: client, want: "deadzone"},
		{name: "negative radius", engine: &braingate.Engine{Temperament: braingate.Temperament{Radius: -0.1}, Memories: validMem}, client: client, want: "radius"},
		{name: "duplicate memory id", engine: &braingate.Engine{Memories: []braingate.Memory{validMem[0], validMem[0]}}, client: client, want: "duplicate memory ID"},
		{name: "out of range initial", engine: &braingate.Engine{Memories: validMem}, client: client, opts: braingate.RunOptions{Initial: &braingate.Point{Valence: 1.2}}, want: "initial point"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := tc.engine.Run(t.Context(), tc.client, decisions.Input{State: "s"}, tc.opts)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Run error = %v, want substring %q", err, tc.want)
			}
		})
	}
}

type fakeEmbedder func(context.Context, []string) ([][]float64, error)

func (fn fakeEmbedder) Embed(ctx context.Context, texts []string) ([][]float64, error) {
	return fn(ctx, texts)
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
