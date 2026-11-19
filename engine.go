package braingate

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"

	"github.com/snowmerak/decisions"
)

const (
	// DefaultDepth is the default maximum number of memory recall steps per run.
	DefaultDepth = 4
	// DefaultAlpha is the default blending rate applied to raw coordinates from the first recalled memory onward.
	DefaultAlpha = 0.5
	// DefaultDeadzone is the default neutral stop radius around (0, 0) used by Spec.Build when omitted.
	DefaultDeadzone = 0.15
	// DefaultPoolSize is the default candidate pool size sampled uniformly at each depth step.
	DefaultPoolSize = 3
	// DefaultMaxVisits is the default maximum number of times a single memory may be recalled in one run.
	DefaultMaxVisits = 2
)

// StopReason describes why descent execution stopped normally.
type StopReason string

const (
	StopDeadzone     StopReason = "deadzone"
	StopDepthReached StopReason = "depth_reached"
	StopNoMemory     StopReason = "no_memory"
)

// Point represents a 2D coordinate on the (Valence, Activation) affect plane in [-1, 1] x [-1, 1].
type Point struct {
	Valence    float64 `json:"valence"`
	Activation float64 `json:"activation"`
}

// Temperament configures how deeply and sensitively the engine descends through memories.
type Temperament struct {
	Depth     int     `json:"depth"`
	Alpha     float64 `json:"alpha"`
	Deadzone  float64 `json:"deadzone"`
	PoolSize  int     `json:"pool_size"`
	MaxVisits int     `json:"max_visits"`
	Radius    float64 `json:"radius,omitempty"`
}

// Memory represents one experience positioned on the 2D affect plane without static edges.
type Memory struct {
	ID         string    `json:"id"`
	Cue        string    `json:"cue,omitempty"`
	Position   Point     `json:"position"`
	Experience string    `json:"experience"`
	Alpha      float64   `json:"alpha,omitempty"`
	Embedding  []float64 `json:"embedding,omitempty"`
}

// Engine evaluates stimuli against a quadrant-positioned memory pool according to a Temperament.
type Engine struct {
	Temperament Temperament
	Memories    []Memory
}

// RunOptions supplies optional embedding ranking, initial mood positioning, and deterministic randomness.
type RunOptions struct {
	Embedder Embedder
	Query    string
	Initial  *Point
	Rand     *rand.Rand
}

// Affect represents cumulative valence, activation, and accumulated stress.
type Affect struct {
	Valence    float64 `json:"valence"`
	Activation float64 `json:"activation"`
	Stress     float64 `json:"stress"`
}

// InitialState records the starting point before memory descent begins.
type InitialState struct {
	Source               string  `json:"source"`
	Valence              float64 `json:"valence"`
	Activation           float64 `json:"activation"`
	ValenceConfidence    float64 `json:"valence_confidence,omitempty"`
	ActivationConfidence float64 `json:"activation_confidence,omitempty"`
}

// RawReading captures the unblended evaluation emitted at a single step.
type RawReading struct {
	Valence              float64 `json:"valence"`
	Activation           float64 `json:"activation"`
	ValenceConfidence    float64 `json:"valence_confidence"`
	ActivationConfidence float64 `json:"activation_confidence"`
}

// StepState captures the cumulative state and change delta after blending a recalled memory's reading.
type StepState struct {
	Valence    float64 `json:"valence"`
	Activation float64 `json:"activation"`
	Delta      float64 `json:"delta"`
	Stress     float64 `json:"stress"`
}

// Step records one memory recall, its candidate pool, raw evaluation, and cumulative state update.
type Step struct {
	Depth      int        `json:"depth"`
	Memory     string     `json:"memory"`
	Visit      int        `json:"visit"`
	Pool       []string   `json:"pool"`
	Distance   float64    `json:"distance"`
	Similarity *float64   `json:"similarity,omitempty"`
	Raw        RawReading `json:"raw"`
	State      StepState  `json:"state"`
}

// Summary holds the final affect snapshot and termination metadata for a run.
type Summary struct {
	Valence    float64    `json:"valence"`
	Activation float64    `json:"activation"`
	Stress     float64    `json:"stress"`
	StopReason StopReason `json:"stop_reason"`
	Steps      int        `json:"steps"`
}

// Result contains the initial baseline, final affect summary, and ordered recall steps of a run.
type Result struct {
	Initial InitialState `json:"initial"`
	Final   Summary      `json:"final"`
	Steps   []Step       `json:"steps"`
}

type resolvedConfig struct {
	depth     int
	alpha     float64
	deadzone  float64
	poolSize  int
	maxVisits int
	radius    float64
}

type evaluationState struct {
	Input      any    `json:"input"`
	Depth      int    `json:"depth"`
	MemoryID   string `json:"memory_id"`
	Visit      int    `json:"visit"`
	Cue        string `json:"cue,omitempty"`
	Experience string `json:"experience"`
	State      Affect `json:"state"`
	History    []Step `json:"history"`
}

type candidate struct {
	index  int
	memory Memory
	dist   float64
	cosPtr *float64
	score  float64
}

// Run establishes the initial affect coordinate, optionally embeds the query and unembedded memories
// in one batch, and descends up to Temperament.Depth times by sampling from quadrant-matched pools.
func (e *Engine) Run(ctx context.Context, client *decisions.Client, input decisions.Input, opts RunOptions) (Result, error) {
	if e == nil {
		return Result{}, errors.New("braingate: engine is required")
	}
	if client == nil {
		return Result{}, errors.New("braingate: decisions client is required")
	}

	cfg, err := resolveTemperament(e.Temperament)
	if err != nil {
		return Result{}, err
	}
	preDim, err := validateMemories(e.Memories)
	if err != nil {
		return Result{}, err
	}
	if opts.Initial != nil {
		if err := validatePoint(*opts.Initial); err != nil {
			return Result{}, fmt.Errorf("braingate: initial point: %w", err)
		}
	}

	initial, err := resolveInitial(ctx, client, input, opts.Initial)
	if err != nil {
		return Result{}, err
	}

	current := Affect{
		Valence:    initial.Valence,
		Activation: initial.Activation,
		Stress:     0,
	}
	probe := Point{
		Valence:    initial.Valence,
		Activation: initial.Activation,
	}
	steps := []Step{}

	if math.Hypot(probe.Valence, probe.Activation) <= cfg.deadzone {
		return Result{
			Initial: initial,
			Final: Summary{
				Valence:    current.Valence,
				Activation: current.Activation,
				Stress:     0,
				StopReason: StopDeadzone,
				Steps:      0,
			},
			Steps: steps,
		}, nil
	}

	var queryVec []float64
	var memVecs [][]float64
	if opts.Embedder != nil && len(e.Memories) > 0 {
		queryVec, memVecs, err = prepareEmbeddings(ctx, opts.Embedder, opts.Query, input.State, e.Memories, preDim)
		if err != nil {
			return partialResult(initial, current, steps), err
		}
	}

	visits := make(map[string]int, len(e.Memories))
	lastMemoryID := ""
	var stopReason StopReason

	for d := 1; d <= cfg.depth; d++ {
		candidates := rankCandidates(e.Memories, probe, lastMemoryID, visits, cfg, queryVec, memVecs)
		if len(candidates) == 0 {
			stopReason = StopNoMemory
			break
		}

		poolLimit := min(cfg.poolSize, len(candidates))
		pool := candidates[:poolLimit]
		poolIDs := make([]string, len(pool))
		for i, c := range pool {
			poolIDs[i] = c.memory.ID
		}

		chosen := pool[pickIndex(opts.Rand, len(pool))]
		visits[chosen.memory.ID]++
		visit := visits[chosen.memory.ID]

		evalState := evaluationState{
			Input:      input.State,
			Depth:      d,
			MemoryID:   chosen.memory.ID,
			Visit:      visit,
			Cue:        chosen.memory.Cue,
			Experience: chosen.memory.Experience,
			State:      current,
			History:    steps,
		}
		resp, err := client.Evaluate(ctx, decisions.Request{
			Input:     decisions.Input{State: evalState, Assets: input.Assets},
			Questions: memoryQuestions(),
		})
		if err != nil {
			return partialResult(initial, current, steps), fmt.Errorf("braingate: depth %d memory %q: %w", d, chosen.memory.ID, err)
		}

		raw, err := parseRawReading(resp, chosen.memory.ID)
		if err != nil {
			return partialResult(initial, current, steps), err
		}

		stepAlpha := cfg.alpha
		if chosen.memory.Alpha > 0 {
			stepAlpha = chosen.memory.Alpha
		}
		nextV := clamp((1-stepAlpha)*current.Valence+stepAlpha*raw.Valence, -1, 1)
		nextA := clamp((1-stepAlpha)*current.Activation+stepAlpha*raw.Activation, -1, 1)
		delta := math.Hypot(nextV-current.Valence, nextA-current.Activation)
		nextStress := current.Stress + delta
		current = Affect{
			Valence:    nextV,
			Activation: nextA,
			Stress:     nextStress,
		}

		steps = append(steps, Step{
			Depth:      d,
			Memory:     chosen.memory.ID,
			Visit:      visit,
			Pool:       poolIDs,
			Distance:   chosen.dist,
			Similarity: chosen.cosPtr,
			Raw:        raw,
			State: StepState{
				Valence:    nextV,
				Activation: nextA,
				Delta:      delta,
				Stress:     nextStress,
			},
		})

		probe = Point{Valence: raw.Valence, Activation: raw.Activation}
		lastMemoryID = chosen.memory.ID

		if math.Hypot(probe.Valence, probe.Activation) <= cfg.deadzone {
			stopReason = StopDeadzone
			break
		}
		if d == cfg.depth {
			stopReason = StopDepthReached
			break
		}
	}

	return Result{
		Initial: initial,
		Final: Summary{
			Valence:    current.Valence,
			Activation: current.Activation,
			Stress:     current.Stress,
			StopReason: stopReason,
			Steps:      len(steps),
		},
		Steps: steps,
	}, nil
}

func resolveTemperament(t Temperament) (resolvedConfig, error) {
	depth := t.Depth
	if depth == 0 {
		depth = DefaultDepth
	}
	if depth < 0 {
		return resolvedConfig{}, fmt.Errorf("braingate: depth must be >= 1, got %d", t.Depth)
	}

	alpha := t.Alpha
	if alpha == 0 {
		alpha = DefaultAlpha
	}
	if !validAlpha(alpha) {
		return resolvedConfig{}, fmt.Errorf("braingate: alpha must be in (0, 1], got %v", t.Alpha)
	}

	if !finite(t.Deadzone) || t.Deadzone < 0 || t.Deadzone >= 1 {
		return resolvedConfig{}, fmt.Errorf("braingate: deadzone must be in [0, 1), got %v", t.Deadzone)
	}

	poolSize := t.PoolSize
	if poolSize == 0 {
		poolSize = DefaultPoolSize
	}
	if poolSize < 0 {
		return resolvedConfig{}, fmt.Errorf("braingate: pool_size must be >= 1, got %d", t.PoolSize)
	}

	maxVisits := t.MaxVisits
	if maxVisits == 0 {
		maxVisits = DefaultMaxVisits
	}
	if maxVisits < 0 {
		return resolvedConfig{}, fmt.Errorf("braingate: max_visits must be >= 1, got %d", t.MaxVisits)
	}

	if !finite(t.Radius) || t.Radius < 0 {
		return resolvedConfig{}, fmt.Errorf("braingate: radius must be >= 0, got %v", t.Radius)
	}

	return resolvedConfig{
		depth:     depth,
		alpha:     alpha,
		deadzone:  t.Deadzone,
		poolSize:  poolSize,
		maxVisits: maxVisits,
		radius:    t.Radius,
	}, nil
}

func validateMemories(memories []Memory) (int, error) {
	seen := make(map[string]struct{}, len(memories))
	dim := 0
	for _, m := range memories {
		if strings.TrimSpace(m.ID) == "" {
			return 0, errors.New("braingate: memory ID is required")
		}
		if _, exists := seen[m.ID]; exists {
			return 0, fmt.Errorf("braingate: duplicate memory ID %q", m.ID)
		}
		seen[m.ID] = struct{}{}

		if strings.TrimSpace(m.Experience) == "" {
			return 0, fmt.Errorf("braingate: memory %q requires an experience", m.ID)
		}
		if err := validatePoint(m.Position); err != nil {
			return 0, fmt.Errorf("braingate: memory %q position: %w", m.ID, err)
		}
		if m.Alpha != 0 && !validAlpha(m.Alpha) {
			return 0, fmt.Errorf("braingate: memory %q alpha must be in (0, 1], got %v", m.ID, m.Alpha)
		}
		if len(m.Embedding) > 0 {
			if dim == 0 {
				dim = len(m.Embedding)
			} else if len(m.Embedding) != dim {
				return 0, fmt.Errorf("braingate: memory %q embedding dimension %d does not match %d", m.ID, len(m.Embedding), dim)
			}
			for _, v := range m.Embedding {
				if !finite(v) {
					return 0, fmt.Errorf("braingate: memory %q embedding contains non-finite value", m.ID)
				}
			}
		}
	}
	return dim, nil
}

func validatePoint(p Point) error {
	if !finite(p.Valence) || !finite(p.Activation) {
		return errors.New("coordinates must be finite numbers")
	}
	if p.Valence < -1 || p.Valence > 1 || p.Activation < -1 || p.Activation > 1 {
		return fmt.Errorf("coordinates (%v, %v) must be within [-1, 1]", p.Valence, p.Activation)
	}
	return nil
}

func resolveInitial(ctx context.Context, client *decisions.Client, input decisions.Input, supplied *Point) (InitialState, error) {
	if supplied != nil {
		return InitialState{
			Source:     "supplied",
			Valence:    supplied.Valence,
			Activation: supplied.Activation,
		}, nil
	}

	evalInput := struct {
		Input any `json:"input"`
	}{Input: input.State}

	resp, err := client.Evaluate(ctx, decisions.Request{
		Input:     decisions.Input{State: evalInput, Assets: input.Assets},
		Questions: initialQuestions(),
	})
	if err != nil {
		return InitialState{}, fmt.Errorf("braingate: initial appraisal: %w", err)
	}

	raw, err := parseRawReading(resp, "initial")
	if err != nil {
		return InitialState{}, err
	}
	return InitialState{
		Source:               "evaluated",
		Valence:              raw.Valence,
		Activation:           raw.Activation,
		ValenceConfidence:    raw.ValenceConfidence,
		ActivationConfidence: raw.ActivationConfidence,
	}, nil
}

func prepareEmbeddings(ctx context.Context, embedder Embedder, queryOverride string, inputState any, memories []Memory, preDim int) ([]float64, [][]float64, error) {
	queryText, err := resolveQueryText(queryOverride, inputState)
	if err != nil {
		return nil, nil, err
	}

	batch := make([]string, 1, len(memories)+1)
	batch[0] = queryText
	unembeddedIndices := make([]int, 0, len(memories))
	for i, m := range memories {
		if len(m.Embedding) == 0 {
			batch = append(batch, memoryText(m))
			unembeddedIndices = append(unembeddedIndices, i)
		}
	}

	vecs, err := embedder.Embed(ctx, batch)
	if err != nil {
		return nil, nil, err
	}
	if len(vecs) != len(batch) {
		return nil, nil, fmt.Errorf("braingate: embedder returned %d vectors for %d texts", len(vecs), len(batch))
	}

	queryVec := vecs[0]
	if len(queryVec) == 0 {
		return nil, nil, errors.New("braingate: query embedding vector is empty")
	}
	if preDim > 0 && len(queryVec) != preDim {
		return nil, nil, fmt.Errorf("braingate: query embedding dimension %d does not match precomputed memory dimension %d", len(queryVec), preDim)
	}

	memVecs := make([][]float64, len(memories))
	for i, m := range memories {
		if len(m.Embedding) > 0 {
			memVecs[i] = m.Embedding
		}
	}
	for offset, memIdx := range unembeddedIndices {
		memVecs[memIdx] = vecs[offset+1]
		if len(memVecs[memIdx]) != len(queryVec) {
			return nil, nil, fmt.Errorf("braingate: memory %q embedding dimension %d does not match query dimension %d", memories[memIdx].ID, len(memVecs[memIdx]), len(queryVec))
		}
	}
	return queryVec, memVecs, nil
}

func resolveQueryText(override string, state any) (string, error) {
	if trimmed := strings.TrimSpace(override); trimmed != "" {
		return trimmed, nil
	}
	if s, ok := state.(string); ok {
		if trimmed := strings.TrimSpace(s); trimmed != "" {
			return trimmed, nil
		}
	}
	if state != nil {
		encoded, err := json.Marshal(state)
		if err != nil {
			return "", fmt.Errorf("braingate: marshal query state: %w", err)
		}
		if trimmed := strings.TrimSpace(string(encoded)); trimmed != "" && trimmed != "null" {
			return trimmed, nil
		}
	}
	return "", errors.New("braingate: query text is required when embedder is configured")
}

func memoryText(m Memory) string {
	if cue := strings.TrimSpace(m.Cue); cue != "" {
		return cue + "\n" + m.Experience
	}
	return m.Experience
}

func rankCandidates(memories []Memory, probe Point, lastID string, visits map[string]int, cfg resolvedConfig, queryVec []float64, memVecs [][]float64) []candidate {
	candidates := make([]candidate, 0, len(memories))
	maxDiag := 2.0 * math.Sqrt2

	for i, m := range memories {
		if m.ID == lastID {
			continue
		}
		if visits[m.ID] >= cfg.maxVisits {
			continue
		}
		if !sameQuadrant(probe, m.Position) {
			continue
		}
		dist := math.Hypot(probe.Valence-m.Position.Valence, probe.Activation-m.Position.Activation)
		if cfg.radius > 0 && dist > cfg.radius {
			continue
		}

		spatial := clamp(1.0-dist/maxDiag, 0, 1)
		score := spatial
		var cosPtr *float64
		if len(queryVec) > 0 {
			cos := cosineSimilarity(queryVec, memVecs[i])
			sem := clamp((cos+1.0)/2.0, 0, 1)
			score = 0.5*spatial + 0.5*sem
			cosPtr = &cos
		}

		candidates = append(candidates, candidate{
			index:  i,
			memory: m,
			dist:   dist,
			cosPtr: cosPtr,
			score:  score,
		})
	}

	slices.SortFunc(candidates, func(a, b candidate) int {
		if a.score > b.score {
			return -1
		}
		if a.score < b.score {
			return 1
		}
		return strings.Compare(a.memory.ID, b.memory.ID)
	})
	return candidates
}

func sameQuadrant(a, b Point) bool {
	return (a.Valence >= 0) == (b.Valence >= 0) && (a.Activation >= 0) == (b.Activation >= 0)
}

func cosineSimilarity(a, b []float64) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return clamp(dot/(math.Sqrt(normA)*math.Sqrt(normB)), -1, 1)
}

func pickIndex(rng *rand.Rand, n int) int {
	if n <= 1 {
		return 0
	}
	if rng != nil {
		return rng.IntN(n)
	}
	return rand.IntN(n)
}

func validAlpha(v float64) bool {
	return finite(v) && v > 0 && v <= 1
}

func finite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

func partialResult(initial InitialState, current Affect, steps []Step) Result {
	return Result{
		Initial: initial,
		Final: Summary{
			Valence:    current.Valence,
			Activation: current.Activation,
			Stress:     current.Stress,
			Steps:      len(steps),
		},
		Steps: steps,
	}
}

func parseRawReading(resp *decisions.Response, label string) (RawReading, error) {
	valenceAns, err := extractScore(resp, label, "valence")
	if err != nil {
		return RawReading{}, err
	}
	activationAns, err := extractScore(resp, label, "activation")
	if err != nil {
		return RawReading{}, err
	}
	return RawReading{
		Valence:              clamp((valenceAns.Score-2.0)/2.0, -1, 1),
		Activation:           clamp((activationAns.Score-2.0)/2.0, -1, 1),
		ValenceConfidence:    valenceAns.Confidence,
		ActivationConfidence: activationAns.Confidence,
	}, nil
}

func extractScore(resp *decisions.Response, label, name string) (decisions.ScoreAnswer, error) {
	ans, ok := resp.Answers[name]
	if !ok {
		return decisions.ScoreAnswer{}, fmt.Errorf("braingate: %q: missing %q answer", label, name)
	}
	if _, refused := ans.(decisions.RefusalAnswer); refused {
		return decisions.ScoreAnswer{}, fmt.Errorf("braingate: %q: %w", label, &decisions.RefusalError{Name: name})
	}
	scoreAns, ok := ans.(decisions.ScoreAnswer)
	if !ok {
		return decisions.ScoreAnswer{}, fmt.Errorf("braingate: %q: unexpected %q answer type %T", label, name, ans)
	}
	return scoreAns, nil
}

func clamp(v, minVal, maxVal float64) float64 {
	if v < minVal {
		return minVal
	}
	if v > maxVal {
		return maxVal
	}
	return v
}

func initialQuestions() decisions.Questions {
	return scoreQuestionPair(
		"주어진 자극과 문맥(input)만 보고 기억을 떠올리기 전에 즉각 느껴지는 원초적 선호-불호(valence) 반응 수준을 평가하라.",
		"주어진 자극과 문맥(input)만 보고 기억을 떠올리기 전에 즉각 느껴지는 행동 에너지의 활성-억제(activation) 반응 수준을 평가하라.",
	)
}

func memoryQuestions() decisions.Questions {
	return scoreQuestionPair(
		"주어진 자극과 문맥(input), 현재 누적 정서 상태(state)와 이전 회상 경로(history), 그리고 지금 떠오른 기억(cue, experience)을 바탕으로 즉각 유도되는 선호-불호(valence) 반응 수준을 평가하라.",
		"주어진 자극과 문맥(input), 현재 누적 정서 상태(state)와 이전 회상 경로(history), 그리고 지금 떠오른 기억(cue, experience)을 바탕으로 즉각 유도되는 행동 에너지의 활성-억제(activation) 반응 수준을 평가하라.",
	)
}

func scoreQuestionPair(valenceInstructions, activationInstructions string) decisions.Questions {
	return decisions.Questions{
		"valence": decisions.ScoreQuestion{
			Instructions: valenceInstructions,
			Levels: []decisions.Level{
				{Label: "strong_dislike", Description: "강한 불호, 거부감, 불쾌감 (-1.0)"},
				{Label: "dislike", Description: "약한 불호, 불편함, 꺼림 (-0.5)"},
				{Label: "neutral", Description: "선호도 불호도 아닌 중립 (0.0)"},
				{Label: "like", Description: "약한 선호, 호감, 긍정감 (+0.5)"},
				{Label: "strong_like", Description: "강한 선호, 반가움, 끌림 (+1.0)"},
			},
		},
		"activation": decisions.ScoreQuestion{
			Instructions: activationInstructions,
			Levels: []decisions.Level{
				{Label: "strong_inhibit", Description: "강한 억제, 위축, 멈춤, 참음 (-1.0)"},
				{Label: "inhibit", Description: "약한 억제, 망설임, 유보 (-0.5)"},
				{Label: "neutral", Description: "활성도 억제도 아닌 평정/중립 (0.0)"},
				{Label: "activate", Description: "약한 활성, 반응 준비, 관여 (+0.5)"},
				{Label: "strong_activate", Description: "강한 활성, 즉각적 행동 충동, 실행 (+1.0)"},
			},
		},
	}
}
