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
	"time"

	"github.com/snowmerak/braingate"
	"github.com/snowmerak/decisions"
)

func TestEngineRunWithEmbedderAndRandomPoolSampling(t *testing.T) {
	engine := &braingate.Engine{
		Temperament: braingate.Temperament{
			Depth:         2,
			Alpha:         0.5,
			Deadzone:      0.15,
			Radius:        1.0,
			MinSimilarity: 0.3,
			PoolSize:      2,
			MaxVisits:     2,
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
				ID:         "mem_too_far",
				Cue:        "정서 반경 밖 기억",
				Position:   braingate.Point{Valence: -0.8, Activation: -0.8},
				Experience: "의미는 같아도 정서 거리가 너무 먼 기억",
			},
			{
				ID:         "mem_low_sim",
				Cue:        "의미 유사도 미달 기억",
				Position:   braingate.Point{Valence: 0.95, Activation: 0.95},
				Experience: "위치는 가까워도 의미 유사도가 기준 미만인 기억",
			},
		},
	}

	embedCalls := 0
	var gotBatch []string
	embedder := fakeEmbedder(func(_ context.Context, texts []string) ([][]float64, error) {
		embedCalls++
		gotBatch = append([]string(nil), texts...)
		// Batch order: [query, mem_semantic, mem_too_far, mem_low_sim]
		return [][]float64{
			{1, 0},   // query
			{0.9, 0}, // mem_semantic (cosine 1.0)
			{1, 0},   // mem_too_far (high cosine, filtered out by Radius!)
			{-1, 0},  // mem_low_sim (close position, filtered out by MinSimilarity!)
		}, nil
	})

	// Script: Step 0 -> (+1.0, +1.0); Depth 1 -> (+0.5, +0.5); Depth 2 -> (-0.5, +1.0).
	scores := [][2]float64{
		{4, 4}, // initial: (+1.0, +1.0)
		{3, 3}, // depth 1: (+0.5, +0.5)
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
	if got.Temperament != engine.Temperament {
		t.Fatalf("result temperament = %+v, want %+v", got.Temperament, engine.Temperament)
	}
	if got.Initial.Source != "evaluated" || !almostEqual(got.Initial.Valence, 1.0) || !almostEqual(got.Initial.Activation, 1.0) {
		t.Fatalf("initial = %+v", got.Initial)
	}
	if got.Final.StopReason != braingate.StopDepthReached || got.Final.Steps != 2 {
		t.Fatalf("final = %+v", got.Final)
	}

	// Verify pool contains only top PoolSize (2) passing Radius and MinSimilarity filters.
	step1 := got.Steps[0]
	if !reflect.DeepEqual(step1.Pool, []string{"mem_preembedded", "mem_semantic"}) {
		t.Fatalf("step 1 pool = %v, want [mem_preembedded mem_semantic]", step1.Pool)
	}
	if step1.Similarity == nil || !almostEqual(*step1.Similarity, 1.0) || step1.Score <= 0 || step1.Score > 1 {
		t.Fatalf("step 1 similarity=%v score=%v", step1.Similarity, step1.Score)
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

func TestEngineCrossAxisNeighborAndCutoffs(t *testing.T) {
	memories := []braingate.Memory{
		{
			ID:         "cross_axis_close",
			Position:   braingate.Point{Valence: -0.05, Activation: 0.62},
			Experience: "축 너머 바로 인접한 기억",
			Embedding:  []float64{1, 0},
		},
		{
			ID:         "same_quadrant_far",
			Position:   braingate.Point{Valence: 0.80, Activation: 0.80},
			Experience: "같은 사분면이지만 반경 밖인 기억",
			Embedding:  []float64{1, 0},
		},
		{
			ID:         "close_but_unrelated",
			Position:   braingate.Point{Valence: 0.06, Activation: 0.60},
			Experience: "위치는 아주 가깝지만 의미가 무관한 기억",
			Embedding:  []float64{0, 1},
		},
	}

	embedder := fakeEmbedder(func(_ context.Context, texts []string) ([][]float64, error) {
		if len(texts) != 1 {
			t.Fatalf("expected only query to be embedded, got %d texts", len(texts))
		}
		return [][]float64{{1, 0}}, nil
	})
	client := fixedScoreClient(t, 3, 3)

	engine := &braingate.Engine{
		Temperament: braingate.Temperament{
			Depth:         1,
			Alpha:         0.5,
			Deadzone:      0.1,
			Radius:        0.5,
			MinSimilarity: 0.4,
			PoolSize:      1,
			MaxVisits:     1,
		},
		Memories: memories,
	}

	got, err := engine.Run(t.Context(), client, decisions.Input{State: "자극"}, braingate.RunOptions{
		Embedder: embedder,
		Initial:  &braingate.Point{Valence: 0.05, Activation: 0.60},
	})
	if err != nil {
		t.Fatalf("Run error = %v", err)
	}
	if len(got.Steps) != 1 || got.Steps[0].Memory != "cross_axis_close" || !reflect.DeepEqual(got.Steps[0].Pool, []string{"cross_axis_close"}) {
		t.Fatalf("expected cross_axis_close to win over far same-quadrant and unrelated close memories, got %+v", got.Steps)
	}

	t.Run("stops with no_memory when min_similarity filters all candidates", func(t *testing.T) {
		strictEngine := *engine
		strictEngine.Temperament.MinSimilarity = 0.99
		strictEngine.Memories = []braingate.Memory{memories[2]} // cos = 0
		res, err := strictEngine.Run(t.Context(), client, decisions.Input{State: "자극"}, braingate.RunOptions{
			Embedder: embedder,
			Initial:  &braingate.Point{Valence: 0.05, Activation: 0.60},
		})
		if err != nil {
			t.Fatalf("Run error = %v", err)
		}
		if res.Final.StopReason != braingate.StopNoMemory || res.Final.Steps != 0 {
			t.Fatalf("got = %+v, want StopNoMemory", res)
		}
	})
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

func TestEngineWithExternalStoreOnlyEmbedsQuery(t *testing.T) {
	store := braingate.NewHNSWStore(braingate.HNSWConfig{})
	if err := store.Upsert(t.Context(), []braingate.Memory{
		{
			ID:         "ext_1",
			Cue:        "발표 칭찬",
			Position:   braingate.Point{Valence: 0.8, Activation: 0.7},
			Experience: "발표를 인정받았던 기억",
			Embedding:  []float64{1, 0},
		},
		{
			ID:         "ext_2",
			Cue:        "따뜻한 화답",
			Position:   braingate.Point{Valence: 0.6, Activation: 0.5},
			Experience: "기분 좋게 응대한 기억",
			Embedding:  []float64{0.9, 0.1},
		},
	}); err != nil {
		t.Fatal(err)
	}

	embedCalls := 0
	embedder := fakeEmbedder(func(_ context.Context, texts []string) ([][]float64, error) {
		embedCalls++
		if !reflect.DeepEqual(texts, []string{"발표 칭찬"}) {
			t.Fatalf("expected query-only embedding batch, got %v", texts)
		}
		return [][]float64{{1, 0}}, nil
	})
	client := fixedScoreClient(t, 4, 4)

	engine := &braingate.Engine{
		Temperament: braingate.Temperament{Depth: 2, Alpha: 0.5, Deadzone: 0.1, Radius: 1.0, MinSimilarity: 0.3, PoolSize: 1, MaxVisits: 1},
		Store:       store,
	}
	got, err := engine.Run(t.Context(), client, decisions.Input{State: "발표 칭찬"}, braingate.RunOptions{Embedder: embedder})
	if err != nil {
		t.Fatalf("Run error = %v", err)
	}
	if embedCalls != 1 || got.Final.Steps != 2 || got.Final.StopReason != braingate.StopDepthReached {
		t.Fatalf("embedCalls=%d got=%+v", embedCalls, got)
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
		{name: "invalid min_similarity", engine: &braingate.Engine{Temperament: braingate.Temperament{MinSimilarity: 1.5}, Memories: validMem}, client: client, want: "min_similarity"},
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

func TestEngineTimeDecayAndSalienceAlpha(t *testing.T) {
	now := time.Date(2026, 11, 20, 15, 0, 0, 0, time.UTC)
	initial := &braingate.Point{Valence: 0.4, Activation: 0.4}

	t.Run("recent memory outranks closer faded memory", func(t *testing.T) {
		freshAt := now.Add(-6 * time.Hour)
		fadedAt := now.Add(-72 * time.Hour)
		engine := &braingate.Engine{
			Temperament: braingate.Temperament{
				Depth:         1,
				Alpha:         0.5,
				Deadzone:      0.1,
				Radius:        1.0,
				PoolSize:      2,
				MaxVisits:     1,
				HalfLifeHours: 24,
			},
			Memories: []braingate.Memory{
				{ID: "faded", Position: braingate.Point{Valence: 0.4, Activation: 0.5}, Experience: "오래전 기억", CreatedAt: &fadedAt},
				{ID: "fresh", Position: braingate.Point{Valence: 0.4, Activation: 0.6}, Experience: "방금 전 기억", CreatedAt: &freshAt},
			},
		}
		got, err := engine.Run(t.Context(), fixedScoreClient(t, 4, 4), decisions.Input{State: "자극"}, braingate.RunOptions{
			Initial: initial,
			Now:     now,
			Rand:    rand.New(rand.NewPCG(1, 2)),
		})
		if err != nil {
			t.Fatalf("Run error = %v", err)
		}
		if len(got.Steps) != 1 || !reflect.DeepEqual(got.Steps[0].Pool, []string{"fresh", "faded"}) {
			t.Fatalf("pool = %+v, want [fresh faded]", got.Steps)
		}
	})

	t.Run("salience boosts alpha and buffers decay with floor", func(t *testing.T) {
		runSingle := func(t *testing.T, mem braingate.Memory, halfLife float64) braingate.Step {
			t.Helper()
			engine := &braingate.Engine{
				Temperament: braingate.Temperament{
					Depth:         1,
					Alpha:         0.5,
					Deadzone:      0.1,
					Radius:        1.0,
					PoolSize:      1,
					MaxVisits:     1,
					HalfLifeHours: halfLife,
				},
				Memories: []braingate.Memory{mem},
			}
			got, err := engine.Run(t.Context(), fixedScoreClient(t, 4, 4), decisions.Input{State: "자극"}, braingate.RunOptions{
				Initial: initial,
				Now:     now,
			})
			if err != nil {
				t.Fatalf("Run error = %v", err)
			}
			if len(got.Steps) != 1 {
				t.Fatalf("steps = %d, want 1", len(got.Steps))
			}
			return got.Steps[0]
		}

		// Timeless salient memory: decay=1.0, alpha = 0.5 + 0.5*0.5*1.0 = 0.75, nextV = 0.25*0.4 + 0.75*1.0 = 0.85
		timelessStep := runSingle(t, braingate.Memory{
			ID: "core", Position: braingate.Point{Valence: 0.4, Activation: 0.4}, Experience: "핵심 가치", Salience: 1.0,
		}, 24)
		if !almostEqual(timelessStep.Decay, 1.0) || !almostEqual(timelessStep.Alpha, 0.75) || !almostEqual(timelessStep.State.Valence, 0.85) {
			t.Fatalf("timeless salient step = %+v, want decay=1 alpha=0.75 valence=0.85", timelessStep)
		}

		// Elapsed 1x half-life with salience=0: decay=0.5, alpha=0.25, nextV = 0.75*0.4 + 0.25*1.0 = 0.55
		t24 := now.Add(-24 * time.Hour)
		attenuatedStep := runSingle(t, braingate.Memory{
			ID: "plain", Position: braingate.Point{Valence: 0.4, Activation: 0.4}, Experience: "보통 기억", CreatedAt: &t24,
		}, 24)
		if !almostEqual(attenuatedStep.Decay, 0.5) || !almostEqual(attenuatedStep.Alpha, 0.25) || !almostEqual(attenuatedStep.State.Valence, 0.55) {
			t.Fatalf("attenuated step = %+v, want decay=0.5 alpha=0.25 valence=0.55", attenuatedStep)
		}

		// Salience=1.0 stretches half-life 5x (24 -> 120h) and keeps floor 0.3: at 120h decay = 0.3 + 0.7*0.5 = 0.65, alpha = 0.75*0.65 = 0.4875
		t120 := now.Add(-120 * time.Hour)
		stretchedStep := runSingle(t, braingate.Memory{
			ID: "salient_120h", Position: braingate.Point{Valence: 0.4, Activation: 0.4}, Experience: "강한 기억", Salience: 1.0, CreatedAt: &t120,
		}, 24)
		if !almostEqual(stretchedStep.Decay, 0.65) || !almostEqual(stretchedStep.Alpha, 0.4875) {
			t.Fatalf("stretched step = %+v, want decay=0.65 alpha=0.4875", stretchedStep)
		}

		// Very old salient memory retains >= 0.3 floor.
		ancient := now.Add(-24000 * time.Hour)
		floorStep := runSingle(t, braingate.Memory{
			ID: "ancient_salient", Position: braingate.Point{Valence: 0.4, Activation: 0.4}, Experience: "각인된 기억", Salience: 1.0, CreatedAt: &ancient,
		}, 24)
		if !almostEqual(floorStep.Decay, 0.3) || floorStep.Decay < 0.3 {
			t.Fatalf("floor step decay = %v, want 0.3", floorStep.Decay)
		}

		// Disabled halfLifeHours=0 keeps ancient zero-salience memory at decay=1.0, alpha=0.5.
		disabledStep := runSingle(t, braingate.Memory{
			ID: "old_disabled", Position: braingate.Point{Valence: 0.4, Activation: 0.4}, Experience: "감쇄 비활성", CreatedAt: &ancient,
		}, 0)
		if !almostEqual(disabledStep.Decay, 1.0) || !almostEqual(disabledStep.Alpha, 0.5) {
			t.Fatalf("disabled step = %+v, want decay=1 alpha=0.5", disabledStep)
		}

		// Future CreatedAt clamps decay at 1.0.
		future := now.Add(10 * time.Hour)
		futureStep := runSingle(t, braingate.Memory{
			ID: "future", Position: braingate.Point{Valence: 0.4, Activation: 0.4}, Experience: "미래 타임스탬프", CreatedAt: &future,
		}, 24)
		if !almostEqual(futureStep.Decay, 1.0) || !almostEqual(futureStep.Alpha, 0.5) {
			t.Fatalf("future step = %+v, want decay=1 alpha=0.5", futureStep)
		}
	})

	t.Run("rejects invalid decay parameters", func(t *testing.T) {
		client := fixedScoreClient(t, 2, 2)
		badHalfLife := &braingate.Engine{
			Temperament: braingate.Temperament{HalfLifeHours: -1},
			Memories:    []braingate.Memory{{ID: "m1", Position: braingate.Point{}, Experience: "exp"}},
		}
		if _, err := badHalfLife.Run(t.Context(), client, decisions.Input{State: "s"}, braingate.RunOptions{}); err == nil || !strings.Contains(err.Error(), "half_life_hours") {
			t.Fatalf("expected half_life_hours error, got %v", err)
		}

		badSalience := &braingate.Engine{
			Memories: []braingate.Memory{{ID: "m1", Position: braingate.Point{}, Experience: "exp", Salience: 1.5}},
		}
		if _, err := badSalience.Run(t.Context(), client, decisions.Input{State: "s"}, braingate.RunOptions{}); err == nil || !strings.Contains(err.Error(), "salience") {
			t.Fatalf("expected salience error, got %v", err)
		}
	})
}

func TestEngineAssociativeChainingUsesRecalledMemoryContext(t *testing.T) {
	t.Run("chains stored memory vector at depth 2 without extra embed calls", func(t *testing.T) {
		engine := &braingate.Engine{
			Temperament: braingate.Temperament{
				Depth:         2,
				Alpha:         0.5,
				Deadzone:      0.1,
				Radius:        1.0,
				MinSimilarity: 0.55,
				PoolSize:      1,
				MaxVisits:     1,
			},
			Memories: []braingate.Memory{
				{
					ID:         "bridge",
					Cue:        "중간 연상",
					Position:   braingate.Point{Valence: 0.5, Activation: 0.5},
					Experience: "초기 자극과 이어진 중간 기억",
					Embedding:  []float64{0.8, 0.6, 0},
				},
				{
					ID:         "chained_from_bridge",
					Cue:        "꼬리 연상",
					Position:   braingate.Point{Valence: 0.5, Activation: 0.5},
					Experience: "중간 기억에서만 이어지는 기억",
					Embedding:  []float64{0, 1, 0},
				},
				{
					ID:         "initial_echo",
					Cue:        "초기 유사",
					Position:   braingate.Point{Valence: 0.5, Activation: 0.5},
					Experience: "초기 자극과만 조금 비슷한 기억",
					Embedding:  []float64{0.6, 0, 0.8},
				},
			},
		}

		embedCalls := 0
		embedder := fakeEmbedder(func(_ context.Context, texts []string) ([][]float64, error) {
			embedCalls++
			if !reflect.DeepEqual(texts, []string{"초기 상황"}) {
				t.Fatalf("unexpected embed batch: %v", texts)
			}
			return [][]float64{{1, 0, 0}}, nil
		})

		got, err := engine.Run(t.Context(), fixedScoreClient(t, 3, 3), decisions.Input{State: "초기 상황"}, braingate.RunOptions{
			Embedder: embedder,
			Initial:  &braingate.Point{Valence: 0.5, Activation: 0.5},
		})
		if err != nil {
			t.Fatalf("Run error = %v", err)
		}
		if embedCalls != 1 {
			t.Fatalf("embedCalls = %d, want 1", embedCalls)
		}
		if len(got.Steps) != 2 || got.Steps[0].Memory != "bridge" || got.Steps[1].Memory != "chained_from_bridge" {
			t.Fatalf("steps = %+v, want [bridge chained_from_bridge]", got.Steps)
		}
		if got.Steps[1].Similarity == nil || !almostEqual(*got.Steps[1].Similarity, 0.6) {
			t.Fatalf("step 2 similarity = %v, want 0.6", got.Steps[1].Similarity)
		}
	})

	t.Run("embeds recalled memory text when custom store omits embedding on search hit", func(t *testing.T) {
		inner := braingate.NewHNSWStore(braingate.HNSWConfig{})
		if err := inner.Upsert(t.Context(), []braingate.Memory{
			{
				ID:         "m1",
				Cue:        "회의실 정적",
				Position:   braingate.Point{Valence: 0.5, Activation: 0.5},
				Experience: "발표 직후 어색했던 기억",
				Embedding:  []float64{1, 0},
			},
			{
				ID:         "m2",
				Cue:        "꼬리 기억",
				Position:   braingate.Point{Valence: 0.5, Activation: 0.5},
				Experience: "정적 뒤에 몰려온 기억",
				Embedding:  []float64{0, 1},
			},
		}); err != nil {
			t.Fatal(err)
		}

		store := &strippingMemoryStore{inner: inner}
		var batches [][]string
		embedder := fakeEmbedder(func(_ context.Context, texts []string) ([][]float64, error) {
			batches = append(batches, append([]string(nil), texts...))
			switch len(batches) {
			case 1:
				return [][]float64{{1, 0}}, nil
			case 2:
				return [][]float64{{0, 1}}, nil
			default:
				return nil, fmt.Errorf("unexpected embed call %d: %v", len(batches), texts)
			}
		})

		engine := &braingate.Engine{
			Temperament: braingate.Temperament{
				Depth:         2,
				Alpha:         0.5,
				Deadzone:      0.1,
				Radius:        1.0,
				MinSimilarity: 0.5,
				PoolSize:      1,
				MaxVisits:     1,
			},
			Store: store,
		}
		got, err := engine.Run(t.Context(), fixedScoreClient(t, 3, 3), decisions.Input{State: "초기 자극"}, braingate.RunOptions{
			Embedder: embedder,
			Initial:  &braingate.Point{Valence: 0.5, Activation: 0.5},
		})
		if err != nil {
			t.Fatalf("Run error = %v", err)
		}
		wantBatches := [][]string{
			{"초기 자극"},
			{"회의실 정적\n발표 직후 어색했던 기억"},
		}
		if !reflect.DeepEqual(batches, wantBatches) {
			t.Fatalf("embed batches = %v, want %v", batches, wantBatches)
		}
		wantQueries := [][]float64{{1, 0}, {0, 1}}
		if !reflect.DeepEqual(store.queries, wantQueries) {
			t.Fatalf("store queries = %v, want %v", store.queries, wantQueries)
		}
		if len(got.Steps) != 2 || got.Steps[0].Memory != "m1" || got.Steps[1].Memory != "m2" {
			t.Fatalf("steps = %+v, want [m1 m2]", got.Steps)
		}
	})
}

type strippingMemoryStore struct {
	inner   braingate.MemoryStore
	queries [][]float64
}

func (s *strippingMemoryStore) Upsert(ctx context.Context, memories []braingate.Memory) error {
	return s.inner.Upsert(ctx, memories)
}

func (s *strippingMemoryStore) Delete(ctx context.Context, ids []string) error {
	return s.inner.Delete(ctx, ids)
}

func (s *strippingMemoryStore) Search(ctx context.Context, query braingate.SearchQuery) ([]braingate.ScoredMemory, error) {
	s.queries = append(s.queries, append([]float64(nil), query.Vector...))
	hits, err := s.inner.Search(ctx, query)
	if err != nil {
		return nil, err
	}
	for i := range hits {
		hits[i].Memory.Embedding = nil
	}
	return hits, nil
}

func TestEngineMinPAndWeightedPoolSampling(t *testing.T) {
	initial := &braingate.Point{Valence: 0.2, Activation: 0}

	t.Run("min_p prunes low-scoring tail candidates from pool", func(t *testing.T) {
		memories := []braingate.Memory{
			{ID: "top", Position: braingate.Point{Valence: 0.3, Activation: 0}, Experience: "1등"},
			{ID: "close_second", Position: braingate.Point{Valence: 0.45, Activation: 0}, Experience: "2등"},
			{ID: "weak_third", Position: braingate.Point{Valence: 0.8, Activation: 0}, Experience: "약한 3등"},
		}
		client := fixedScoreClient(t, 3, 3)

		engine70 := &braingate.Engine{
			Temperament: braingate.Temperament{Depth: 1, Alpha: 0.5, Deadzone: 0.1, Radius: 1.0, PoolSize: 3, MinP: 0.7, MaxVisits: 1},
			Memories:    memories,
		}
		got70, err := engine70.Run(t.Context(), client, decisions.Input{State: "자극"}, braingate.RunOptions{
			Initial: initial,
			Rand:    rand.New(rand.NewPCG(1, 2)),
		})
		if err != nil {
			t.Fatalf("Run (min_p=0.7) error = %v", err)
		}
		if len(got70.Steps) != 1 || !reflect.DeepEqual(got70.Steps[0].Pool, []string{"top", "close_second"}) {
			t.Fatalf("pool (min_p=0.7) = %+v, want [top close_second]", got70.Steps)
		}

		engine90 := &braingate.Engine{
			Temperament: braingate.Temperament{Depth: 1, Alpha: 0.5, Deadzone: 0.1, Radius: 1.0, PoolSize: 3, MinP: 0.9, MaxVisits: 1},
			Memories:    memories,
		}
		got90, err := engine90.Run(t.Context(), client, decisions.Input{State: "자극"}, braingate.RunOptions{
			Initial: initial,
			Rand:    rand.New(rand.NewPCG(1, 2)),
		})
		if err != nil {
			t.Fatalf("Run (min_p=0.9) error = %v", err)
		}
		if len(got90.Steps) != 1 || !reflect.DeepEqual(got90.Steps[0].Pool, []string{"top"}) {
			t.Fatalf("pool (min_p=0.9) = %+v, want [top]", got90.Steps)
		}
	})

	t.Run("weighted sampling favors higher score over uniform 50-50", func(t *testing.T) {
		engine := &braingate.Engine{
			Temperament: braingate.Temperament{Depth: 1, Alpha: 0.5, Deadzone: 0.1, Radius: 1.0, PoolSize: 2, MinP: 0, MaxVisits: 1},
			Memories: []braingate.Memory{
				{ID: "high", Position: braingate.Point{Valence: 0.3, Activation: 0}, Experience: "높은 점수"},
				{ID: "low", Position: braingate.Point{Valence: -0.7, Activation: 0}, Experience: "낮은 점수"},
			},
		}
		client := fixedScoreClient(t, 3, 3)
		rng := rand.New(rand.NewPCG(7, 8))

		highCount := 0
		for range 300 {
			got, err := engine.Run(t.Context(), client, decisions.Input{State: "자극"}, braingate.RunOptions{
				Initial: initial,
				Rand:    rng,
			})
			if err != nil {
				t.Fatalf("Run error = %v", err)
			}
			if len(got.Steps) != 1 {
				t.Fatalf("steps = %d, want 1", len(got.Steps))
			}
			if got.Steps[0].Memory == "high" {
				highCount++
			}
		}
		if highCount <= 230 {
			t.Fatalf("highCount = %d / 300, want > 230 (~90%% weighted)", highCount)
		}
	})

	t.Run("rejects out-of-range min_p", func(t *testing.T) {
		client := fixedScoreClient(t, 2, 2)
		mem := []braingate.Memory{{ID: "m1", Position: braingate.Point{}, Experience: "exp"}}
		for _, v := range []float64{-0.1, 1.1} {
			bad := &braingate.Engine{
				Temperament: braingate.Temperament{MinP: v},
				Memories:    mem,
			}
			if _, err := bad.Run(t.Context(), client, decisions.Input{State: "s"}, braingate.RunOptions{}); err == nil || !strings.Contains(err.Error(), "min_p") {
				t.Fatalf("MinP=%v expected error, got %v", v, err)
			}
		}
	})
}

func TestEnginePersonaInjectedIntoDecisionsAndExcludedFromEmbeddings(t *testing.T) {
	memories := []braingate.Memory{
		{ID: "m1", Cue: "칭찬 기억", Position: braingate.Point{Valence: 0.7, Activation: 0.6}, Experience: "인정받았던 기억"},
		{ID: "m2", Cue: "경계 기억", Position: braingate.Point{Valence: 0.6, Activation: 0.5}, Experience: "속내를 의심했던 기억"},
	}

	assertPersonaFlow := func(t *testing.T, enginePersona, optPersona, wantPersona, forbiddenSubstr string) {
		t.Helper()
		var embeddedTexts []string
		embedder := fakeEmbedder(func(_ context.Context, texts []string) ([][]float64, error) {
			embeddedTexts = append(embeddedTexts, texts...)
			vecs := make([][]float64, len(texts))
			for i := range texts {
				vecs[i] = []float64{1, 0}
			}
			return vecs, nil
		})

		type questionReq struct {
			Instructions string `json:"instructions"`
		}
		var seenPersonas []string
		var seenValenceInstr []string
		var seenActivationInstr []string

		client, err := decisions.NewClient(decisions.Config{
			API:   decisions.SystemOneAPI,
			Model: "test",
			HTTPClient: &http.Client{Transport: testTransport(func(r *http.Request) (*http.Response, error) {
				defer func() { _ = r.Body.Close() }()
				var req struct {
					State struct {
						Persona string `json:"persona"`
					} `json:"state"`
					Questions struct {
						Valence    questionReq `json:"valence"`
						Activation questionReq `json:"activation"`
					} `json:"questions"`
				}
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					return nil, err
				}
				seenPersonas = append(seenPersonas, req.State.Persona)
				seenValenceInstr = append(seenValenceInstr, req.Questions.Valence.Instructions)
				seenActivationInstr = append(seenActivationInstr, req.Questions.Activation.Instructions)

				payload, err := json.Marshal(map[string]any{
					"model": "test",
					"usage": map[string]any{"input_tokens": 1, "output_tokens": 1},
					"answers": map[string]any{
						"valence":    encodeScoreAnswer(&scoreAnswerSpec{score: 4, confidence: 0.9}),
						"activation": encodeScoreAnswer(&scoreAnswerSpec{score: 4, confidence: 0.9}),
					},
				})
				if err != nil {
					return nil, err
				}
				return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(string(payload)))}, nil
			})},
		})
		if err != nil {
			t.Fatal(err)
		}

		engine := &braingate.Engine{
			Persona:     enginePersona,
			Temperament: braingate.Temperament{Depth: 2, Alpha: 0.5, Deadzone: 0.1, Radius: 1.0, MinSimilarity: 0.3, PoolSize: 1, MaxVisits: 1},
			Memories:    memories,
		}
		got, err := engine.Run(t.Context(), client, decisions.Input{State: "팀장님이 칭찬을 건넸다"}, braingate.RunOptions{
			Persona:  optPersona,
			Embedder: embedder,
		})
		if err != nil {
			t.Fatalf("Run error = %v", err)
		}
		if got.Persona != wantPersona {
			t.Fatalf("result.Persona = %q, want %q", got.Persona, wantPersona)
		}
		if len(seenPersonas) != 3 { // initial + depth 1 + depth 2
			t.Fatalf("seenPersonas count = %d, want 3", len(seenPersonas))
		}
		for i, p := range seenPersonas {
			if p != wantPersona {
				t.Fatalf("decisions call %d state.persona = %q, want %q", i, p, wantPersona)
			}
			if !strings.Contains(seenValenceInstr[i], "persona") || !strings.Contains(seenActivationInstr[i], "persona") {
				t.Fatalf("decisions call %d instructions did not reference persona: %q / %q", i, seenValenceInstr[i], seenActivationInstr[i])
			}
		}
		for _, text := range embeddedTexts {
			if strings.Contains(text, wantPersona) || strings.Contains(text, forbiddenSubstr) {
				t.Fatalf("persona leaked into embedder text %q", text)
			}
		}
	}

	t.Run("uses trimmed Engine.Persona without leaking to embeddings", func(t *testing.T) {
		assertPersonaFlow(t, "  칭찬을 경계하는 냉소적 성향  ", "", "칭찬을 경계하는 냉소적 성향", "냉소적")
	})

	t.Run("RunOptions.Persona overrides Engine.Persona", func(t *testing.T) {
		assertPersonaFlow(t, "기본 성향", "  직설적 피드백에서 안도하는 성향  ", "직설적 피드백에서 안도하는 성향", "안도하는")
	})
}

func TestEngineDualEmbeddingGeometricCalibrationAndChaining(t *testing.T) {
	t.Run("secondary embedder disambiguates compressed primary similarity and chains both vectors", func(t *testing.T) {
		engine := &braingate.Engine{
			Temperament: braingate.Temperament{
				Depth:         2,
				Alpha:         0.5,
				Deadzone:      0.1,
				Radius:        1.0,
				MinSimilarity: 0.3,
				PoolSize:      2,
				MinP:          0.7,
				MaxVisits:     1,
			},
			Memories: []braingate.Memory{
				{
					ID:                 "distractor",
					Position:           braingate.Point{Valence: 0.4, Activation: 0},
					Experience:         "가깝지만 2차 임베딩상 무관한 기억",
					Embedding:          []float64{0.74, -0.6726},
					SecondaryEmbedding: []float64{0.2, 0, 0.9798},
				},
				{
					ID:                 "true_match",
					Position:           braingate.Point{Valence: 0.85, Activation: 0},
					Experience:         "핵심 일치 기억",
					Embedding:          []float64{0.76, 0.6499},
					SecondaryEmbedding: []float64{0.85, 0.5268, 0},
				},
				{
					ID:                 "chained_follow",
					Position:           braingate.Point{Valence: 0.5, Activation: 0.5},
					Experience:         "핵심 기억에서 이어지는 후속 기억",
					Embedding:          []float64{0, 1},
					SecondaryEmbedding: []float64{0, 1, 0},
				},
			},
		}

		primaryCalls := 0
		primaryEmbedder := fakeEmbedder(func(_ context.Context, texts []string) ([][]float64, error) {
			primaryCalls++
			if !reflect.DeepEqual(texts, []string{"초기 사건"}) {
				t.Fatalf("unexpected primary embedder batch: %v", texts)
			}
			return [][]float64{{1, 0}}, nil
		})
		secondaryCalls := 0
		secondaryEmbedder := fakeEmbedder(func(_ context.Context, texts []string) ([][]float64, error) {
			secondaryCalls++
			if !reflect.DeepEqual(texts, []string{"초기 사건"}) {
				t.Fatalf("unexpected secondary embedder batch: %v", texts)
			}
			return [][]float64{{1, 0, 0}}, nil
		})

		got, err := engine.Run(t.Context(), fixedScoreClient(t, 3, 3), decisions.Input{State: "초기 사건"}, braingate.RunOptions{
			Embedder:          primaryEmbedder,
			SecondaryEmbedder: secondaryEmbedder,
			Initial:           &braingate.Point{Valence: 0.2, Activation: 0},
		})
		if err != nil {
			t.Fatalf("Run error = %v", err)
		}
		if primaryCalls != 1 || secondaryCalls != 1 {
			t.Fatalf("embedder calls = (%d, %d), want (1, 1)", primaryCalls, secondaryCalls)
		}
		if len(got.Steps) != 2 || got.Steps[0].Memory != "true_match" || got.Steps[1].Memory != "chained_follow" {
			t.Fatalf("steps = %+v, want [true_match chained_follow]", got.Steps)
		}
		if !reflect.DeepEqual(got.Steps[0].Pool, []string{"true_match"}) {
			t.Fatalf("step 1 pool = %v, want [true_match] (distractor pruned by dual min_p)", got.Steps[0].Pool)
		}
		if got.Steps[0].Similarity == nil || got.Steps[0].SecondarySimilarity == nil || got.Steps[1].Similarity == nil || got.Steps[1].SecondarySimilarity == nil {
			t.Fatalf("expected both primary and secondary similarities recorded on steps: %+v", got.Steps)
		}
	})

	t.Run("rejects SecondaryEmbedder without primary Embedder", func(t *testing.T) {
		engine := &braingate.Engine{
			Memories: []braingate.Memory{{ID: "m1", Position: braingate.Point{Valence: 0.5}, Experience: "exp"}},
		}
		_, err := engine.Run(t.Context(), fixedScoreClient(t, 3, 3), decisions.Input{State: "s"}, braingate.RunOptions{
			SecondaryEmbedder: fakeEmbedder(func(context.Context, []string) ([][]float64, error) {
				return [][]float64{{1, 0}}, nil
			}),
		})
		if err == nil || !strings.Contains(err.Error(), "primary embedder is required") {
			t.Fatalf("err = %v, want primary embedder error", err)
		}
	})
}
