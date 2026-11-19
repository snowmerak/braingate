package braingate

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/snowmerak/decisions"
)

const (
	// DefaultAlpha is the default blending rate applied to raw coordinates when Graph.Alpha is zero.
	DefaultAlpha = 0.5
	// DefaultMaxVisits is the default per-gate visit limit when Graph.MaxVisits is zero.
	DefaultMaxVisits = 2
	// DefaultMaxSteps is the default total step limit when Graph.MaxSteps is zero.
	DefaultMaxSteps = 16
)

// StopReason describes why graph execution stopped normally.
type StopReason string

const (
	StopExplicit   StopReason = "explicit_stop"
	StopEndOfGraph StopReason = "end_of_graph"
	StopMaxVisits  StopReason = "max_visits"
	StopMaxSteps   StopReason = "max_steps"
)

// Bounds defines an inclusive range [Min, Max] along one affect axis.
type Bounds struct {
	Min float64
	Max float64
}

// Contains reports whether v lies within [Min, Max].
func (b Bounds) Contains(v float64) bool {
	return v >= b.Min && v <= b.Max
}

// Route maps a region of raw (Valence, Activation) coordinates to a transition or explicit stop.
type Route struct {
	Name       string
	Valence    Bounds
	Activation Bounds
	Next       *Gate
	Stop       bool
}

// Gate evaluates one experience/memory against the stimulus and cumulative affect state,
// then routes based on the raw evaluation coordinates.
type Gate struct {
	ID         string
	Experience string
	Alpha      float64
	Routes     []Route
}

// Graph configures a reflexive experience-gate circuit and its execution limits.
type Graph struct {
	Start     *Gate
	Alpha     float64
	MaxVisits int
	MaxSteps  int
}

// Affect represents cumulative valence, activation, and accumulated stress.
type Affect struct {
	Valence    float64 `json:"valence"`
	Activation float64 `json:"activation"`
	Stress     float64 `json:"stress"`
}

// RawReading captures the unblended evaluation emitted at a single gate.
type RawReading struct {
	Valence              float64 `json:"valence"`
	Activation           float64 `json:"activation"`
	ValenceConfidence    float64 `json:"valence_confidence"`
	ActivationConfidence float64 `json:"activation_confidence"`
}

// StepState captures the cumulative state and change delta after blending a gate's raw reading.
type StepState struct {
	Valence    float64 `json:"valence"`
	Activation float64 `json:"activation"`
	Delta      float64 `json:"delta"`
	Stress     float64 `json:"stress"`
}

// Step records one gate evaluation, cumulative state update, and selected transition.
type Step struct {
	Gate  string     `json:"gate"`
	Visit int        `json:"visit"`
	Raw   RawReading `json:"raw"`
	State StepState  `json:"state"`
	Route string     `json:"route,omitempty"`
	Next  string     `json:"next,omitempty"`
}

// Summary holds the final affect snapshot and termination metadata for a run.
type Summary struct {
	Valence    float64    `json:"valence"`
	Activation float64    `json:"activation"`
	Stress     float64    `json:"stress"`
	StopReason StopReason `json:"stop_reason"`
	Steps      int        `json:"steps"`
}

// Result contains the final affect summary and ordered step-by-step trace of a run.
type Result struct {
	Final Summary `json:"final"`
	Steps []Step  `json:"steps"`
}

type evaluationState struct {
	Input      any    `json:"input"`
	GateID     string `json:"gate_id"`
	Visit      int    `json:"visit"`
	Experience string `json:"experience"`
	State      Affect `json:"state"`
	History    []Step `json:"history"`
}

// Run executes the graph starting at g.Start until an explicit stop, graph end,
// visit limit, or step limit is reached. Any error returns the completed steps
// and state prior to the failure.
func (g *Graph) Run(ctx context.Context, client *decisions.Client, input decisions.Input) (Result, error) {
	if g == nil || g.Start == nil {
		return Result{}, errors.New("braingate: starting gate is required")
	}
	if client == nil {
		return Result{}, errors.New("braingate: decisions client is required")
	}
	alpha, maxVisits, maxSteps, err := g.resolveConfig()
	if err != nil {
		return Result{}, err
	}

	var current Affect
	steps := []Step{}
	visits := make(map[string]int)
	var stopReason StopReason

	for gate := g.Start; gate != nil; {
		stepAlpha, err := validateGate(gate, alpha)
		if err != nil {
			return partialResult(current, steps), err
		}

		visits[gate.ID]++
		visit := visits[gate.ID]

		evalState := evaluationState{
			Input:      input.State,
			GateID:     gate.ID,
			Visit:      visit,
			Experience: gate.Experience,
			State:      current,
			History:    steps,
		}
		resp, err := client.Evaluate(ctx, decisions.Request{
			Input:     decisions.Input{State: evalState, Assets: input.Assets},
			Questions: gateQuestions(),
		})
		if err != nil {
			return partialResult(current, steps), fmt.Errorf("braingate: gate %q: %w", gate.ID, err)
		}

		raw, err := parseRawReading(resp, gate.ID)
		if err != nil {
			return partialResult(current, steps), err
		}

		nextV := clamp((1-stepAlpha)*current.Valence+stepAlpha*raw.Valence, -1, 1)
		nextA := clamp((1-stepAlpha)*current.Activation+stepAlpha*raw.Activation, -1, 1)

		var delta float64
		nextStress := current.Stress
		if len(steps) >= 1 {
			delta = math.Hypot(nextV-current.Valence, nextA-current.Activation)
			nextStress += delta
		}
		current = Affect{
			Valence:    nextV,
			Activation: nextA,
			Stress:     nextStress,
		}

		matched, ok := matchRoute(gate.Routes, raw.Valence, raw.Activation)
		step := Step{
			Gate:  gate.ID,
			Visit: visit,
			Raw:   raw,
			State: StepState{
				Valence:    nextV,
				Activation: nextA,
				Delta:      delta,
				Stress:     nextStress,
			},
		}
		if ok {
			step.Route = matched.Name
			if matched.Next != nil {
				step.Next = matched.Next.ID
			}
		}
		steps = append(steps, step)

		if !ok {
			stopReason = StopEndOfGraph
			break
		}
		if matched.Stop {
			stopReason = StopExplicit
			break
		}
		if matched.Next == nil {
			stopReason = StopEndOfGraph
			break
		}
		if visits[matched.Next.ID] >= maxVisits {
			stopReason = StopMaxVisits
			break
		}
		if len(steps) >= maxSteps {
			stopReason = StopMaxSteps
			break
		}
		gate = matched.Next
	}

	return Result{
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

func (g *Graph) resolveConfig() (alpha float64, maxVisits int, maxSteps int, err error) {
	alpha = g.Alpha
	if alpha == 0 {
		alpha = DefaultAlpha
	}
	if !validAlpha(alpha) {
		return 0, 0, 0, fmt.Errorf("braingate: alpha must be in (0, 1], got %v", g.Alpha)
	}

	maxVisits = g.MaxVisits
	if maxVisits == 0 {
		maxVisits = DefaultMaxVisits
	}
	if maxVisits < 0 {
		return 0, 0, 0, fmt.Errorf("braingate: max_visits must be >= 1, got %d", g.MaxVisits)
	}

	maxSteps = g.MaxSteps
	if maxSteps == 0 {
		maxSteps = DefaultMaxSteps
	}
	if maxSteps < 0 {
		return 0, 0, 0, fmt.Errorf("braingate: max_steps must be >= 1, got %d", g.MaxSteps)
	}

	return alpha, maxVisits, maxSteps, nil
}

func validateGate(gate *Gate, defaultAlpha float64) (float64, error) {
	if strings.TrimSpace(gate.ID) == "" {
		return 0, errors.New("braingate: gate ID is required")
	}
	if strings.TrimSpace(gate.Experience) == "" {
		return 0, fmt.Errorf("braingate: gate %q requires an experience", gate.ID)
	}
	if gate.Alpha == 0 {
		return defaultAlpha, nil
	}
	if !validAlpha(gate.Alpha) {
		return 0, fmt.Errorf("braingate: gate %q alpha must be in (0, 1], got %v", gate.ID, gate.Alpha)
	}
	return gate.Alpha, nil
}

func validAlpha(v float64) bool {
	return finite(v) && v > 0 && v <= 1
}

func partialResult(current Affect, steps []Step) Result {
	return Result{
		Final: Summary{
			Valence:    current.Valence,
			Activation: current.Activation,
			Stress:     current.Stress,
			Steps:      len(steps),
		},
		Steps: steps,
	}
}

func parseRawReading(resp *decisions.Response, gateID string) (RawReading, error) {
	valenceAns, err := extractScore(resp, gateID, "valence")
	if err != nil {
		return RawReading{}, err
	}
	activationAns, err := extractScore(resp, gateID, "activation")
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

func extractScore(resp *decisions.Response, gateID, name string) (decisions.ScoreAnswer, error) {
	ans, ok := resp.Answers[name]
	if !ok {
		return decisions.ScoreAnswer{}, fmt.Errorf("braingate: gate %q: missing %q answer", gateID, name)
	}
	if _, refused := ans.(decisions.RefusalAnswer); refused {
		return decisions.ScoreAnswer{}, fmt.Errorf("braingate: gate %q: %w", gateID, &decisions.RefusalError{Name: name})
	}
	scoreAns, ok := ans.(decisions.ScoreAnswer)
	if !ok {
		return decisions.ScoreAnswer{}, fmt.Errorf("braingate: gate %q: unexpected %q answer type %T", gateID, name, ans)
	}
	return scoreAns, nil
}

func matchRoute(routes []Route, valence, activation float64) (Route, bool) {
	for _, route := range routes {
		if route.Valence.Contains(valence) && route.Activation.Contains(activation) {
			return route, true
		}
	}
	return Route{}, false
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

func gateQuestions() decisions.Questions {
	return decisions.Questions{
		"valence": decisions.ScoreQuestion{
			Instructions: "주어진 자극과 문맥(input), 현재 누적 정서 상태(state)와 이전 경로(history), 그리고 지금 활성화된 경험/기억(experience)을 바탕으로 즉각 유도되는 선호-불호(valence) 반응 수준을 평가하라.",
			Levels: []decisions.Level{
				{Label: "strong_dislike", Description: "강한 불호, 거부감, 불쾌감 (-1.0)"},
				{Label: "dislike", Description: "약한 불호, 불편함, 꺼림 (-0.5)"},
				{Label: "neutral", Description: "선호도 불호도 아닌 중립 (0.0)"},
				{Label: "like", Description: "약한 선호, 호감, 긍정감 (+0.5)"},
				{Label: "strong_like", Description: "강한 선호, 반가움, 끌림 (+1.0)"},
			},
		},
		"activation": decisions.ScoreQuestion{
			Instructions: "주어진 자극과 문맥(input), 현재 누적 정서 상태(state)와 이전 경로(history), 그리고 지금 활성화된 경험/기억(experience)을 바탕으로 즉각 유도되는 행동 에너지의 활성-억제(activation) 반응 수준을 평가하라.",
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
