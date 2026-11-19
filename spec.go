package braingate

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// Spec defines a temperament configuration and quadrant-positioned memory pool in JSON form.
type Spec struct {
	Temperament TemperamentSpec `json:"temperament,omitempty"`
	Memories    []MemorySpec    `json:"memories"`
}

// TemperamentSpec configures optional temperament overrides. Nil fields receive package defaults.
type TemperamentSpec struct {
	Depth     *int     `json:"depth,omitempty"`
	Alpha     *float64 `json:"alpha,omitempty"`
	Deadzone  *float64 `json:"deadzone,omitempty"`
	PoolSize  *int     `json:"pool_size,omitempty"`
	MaxVisits *int     `json:"max_visits,omitempty"`
	Radius    *float64 `json:"radius,omitempty"`
}

// MemorySpec defines one memory entry on the 2D affect plane.
type MemorySpec struct {
	ID         string    `json:"id"`
	Cue        string    `json:"cue,omitempty"`
	Position   Point     `json:"position"`
	Experience string    `json:"experience"`
	Alpha      *float64  `json:"alpha,omitempty"`
	Embedding  []float64 `json:"embedding,omitempty"`
}

// ParseSpec decodes a single JSON Spec object (rejecting unknown fields and trailing data)
// and builds a validated Engine.
func ParseSpec(data []byte) (*Engine, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var spec Spec
	if err := decoder.Decode(&spec); err != nil {
		return nil, fmt.Errorf("braingate: parse spec: %w", err)
	}
	if err := decoder.Decode(new(json.RawMessage)); err != io.EOF {
		return nil, errors.New("braingate: parse spec: expected one JSON object")
	}
	return spec.Build()
}

// Build validates the specification and constructs a runnable Engine.
func (spec Spec) Build() (*Engine, error) {
	temperament, err := spec.Temperament.build()
	if err != nil {
		return nil, err
	}

	if len(spec.Memories) == 0 {
		return nil, errors.New("braingate: at least one memory is required")
	}

	memories := make([]Memory, 0, len(spec.Memories))
	for _, m := range spec.Memories {
		var memAlpha float64
		if m.Alpha != nil {
			if !validAlpha(*m.Alpha) {
				return nil, fmt.Errorf("braingate: memory %q alpha must be in (0, 1], got %v", m.ID, *m.Alpha)
			}
			memAlpha = *m.Alpha
		}
		var emb []float64
		if len(m.Embedding) > 0 {
			emb = append([]float64(nil), m.Embedding...)
		}
		memories = append(memories, Memory{
			ID:         m.ID,
			Cue:        m.Cue,
			Position:   m.Position,
			Experience: m.Experience,
			Alpha:      memAlpha,
			Embedding:  emb,
		})
	}

	if _, err := validateMemories(memories); err != nil {
		return nil, err
	}

	return &Engine{
		Temperament: temperament,
		Memories:    memories,
	}, nil
}

func (t TemperamentSpec) build() (Temperament, error) {
	depth := DefaultDepth
	if t.Depth != nil {
		if *t.Depth < 1 {
			return Temperament{}, fmt.Errorf("braingate: depth must be >= 1, got %d", *t.Depth)
		}
		depth = *t.Depth
	}

	alpha := DefaultAlpha
	if t.Alpha != nil {
		if !validAlpha(*t.Alpha) {
			return Temperament{}, fmt.Errorf("braingate: alpha must be in (0, 1], got %v", *t.Alpha)
		}
		alpha = *t.Alpha
	}

	deadzone := DefaultDeadzone
	if t.Deadzone != nil {
		if !finite(*t.Deadzone) || *t.Deadzone < 0 || *t.Deadzone >= 1 {
			return Temperament{}, fmt.Errorf("braingate: deadzone must be in [0, 1), got %v", *t.Deadzone)
		}
		deadzone = *t.Deadzone
	}

	poolSize := DefaultPoolSize
	if t.PoolSize != nil {
		if *t.PoolSize < 1 {
			return Temperament{}, fmt.Errorf("braingate: pool_size must be >= 1, got %d", *t.PoolSize)
		}
		poolSize = *t.PoolSize
	}

	maxVisits := DefaultMaxVisits
	if t.MaxVisits != nil {
		if *t.MaxVisits < 1 {
			return Temperament{}, fmt.Errorf("braingate: max_visits must be >= 1, got %d", *t.MaxVisits)
		}
		maxVisits = *t.MaxVisits
	}

	var radius float64
	if t.Radius != nil {
		if !finite(*t.Radius) || *t.Radius < 0 {
			return Temperament{}, fmt.Errorf("braingate: radius must be >= 0, got %v", *t.Radius)
		}
		radius = *t.Radius
	}

	return Temperament{
		Depth:     depth,
		Alpha:     alpha,
		Deadzone:  deadzone,
		PoolSize:  poolSize,
		MaxVisits: maxVisits,
		Radius:    radius,
	}, nil
}
