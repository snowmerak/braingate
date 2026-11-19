package braingate

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
)

// Spec defines a reflexive experience-gate graph in JSON-serializable form.
type Spec struct {
	Start     string     `json:"start"`
	Alpha     *float64   `json:"alpha,omitempty"`
	MaxVisits *int       `json:"max_visits,omitempty"`
	MaxSteps  *int       `json:"max_steps,omitempty"`
	Gates     []GateSpec `json:"gates"`
}

// GateSpec defines one experience/memory gate and its raw-coordinate routing table.
type GateSpec struct {
	ID         string      `json:"id"`
	Experience string      `json:"experience"`
	Alpha      *float64    `json:"alpha,omitempty"`
	Routes     []RouteSpec `json:"routes,omitempty"`
}

// RouteSpec defines one region in (valence, activation) space and its target gate or stop rule.
type RouteSpec struct {
	Name       string      `json:"name,omitempty"`
	Valence    *BoundsSpec `json:"valence,omitempty"`
	Activation *BoundsSpec `json:"activation,omitempty"`
	Next       string      `json:"next,omitempty"`
	Stop       bool        `json:"stop,omitempty"`
}

// BoundsSpec defines optional inclusive bounds [Min, Max]. Nil fields default to [-1, 1].
type BoundsSpec struct {
	Min *float64 `json:"min,omitempty"`
	Max *float64 `json:"max,omitempty"`
}

// ParseSpec decodes a single JSON Spec object (rejecting unknown fields and trailing data)
// and builds a validated Graph.
func ParseSpec(data []byte) (*Graph, error) {
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

// Build validates the specification, resolves named gate references, and returns a runnable Graph.
func (spec Spec) Build() (*Graph, error) {
	alpha := DefaultAlpha
	if spec.Alpha != nil {
		if !validAlpha(*spec.Alpha) {
			return nil, fmt.Errorf("braingate: alpha must be in (0, 1], got %v", *spec.Alpha)
		}
		alpha = *spec.Alpha
	}

	maxVisits := DefaultMaxVisits
	if spec.MaxVisits != nil {
		if *spec.MaxVisits < 1 {
			return nil, fmt.Errorf("braingate: max_visits must be >= 1, got %d", *spec.MaxVisits)
		}
		maxVisits = *spec.MaxVisits
	}

	maxSteps := DefaultMaxSteps
	if spec.MaxSteps != nil {
		if *spec.MaxSteps < 1 {
			return nil, fmt.Errorf("braingate: max_steps must be >= 1, got %d", *spec.MaxSteps)
		}
		maxSteps = *spec.MaxSteps
	}

	if len(spec.Gates) == 0 {
		return nil, errors.New("braingate: at least one gate is required")
	}

	gates := make(map[string]*Gate, len(spec.Gates))
	for _, gateSpec := range spec.Gates {
		if strings.TrimSpace(gateSpec.ID) == "" {
			return nil, errors.New("braingate: gate ID is required")
		}
		if _, exists := gates[gateSpec.ID]; exists {
			return nil, fmt.Errorf("braingate: duplicate gate ID %q", gateSpec.ID)
		}
		if strings.TrimSpace(gateSpec.Experience) == "" {
			return nil, fmt.Errorf("braingate: gate %q requires an experience", gateSpec.ID)
		}
		var gateAlpha float64
		if gateSpec.Alpha != nil {
			if !validAlpha(*gateSpec.Alpha) {
				return nil, fmt.Errorf("braingate: gate %q alpha must be in (0, 1], got %v", gateSpec.ID, *gateSpec.Alpha)
			}
			gateAlpha = *gateSpec.Alpha
		}
		gates[gateSpec.ID] = &Gate{
			ID:         gateSpec.ID,
			Experience: gateSpec.Experience,
			Alpha:      gateAlpha,
		}
	}

	start, exists := gates[spec.Start]
	if !exists {
		return nil, fmt.Errorf("braingate: unknown starting gate %q", spec.Start)
	}

	for _, gateSpec := range spec.Gates {
		routes := make([]Route, 0, len(gateSpec.Routes))
		seenNames := make(map[string]struct{}, len(gateSpec.Routes))
		for i, routeSpec := range gateSpec.Routes {
			if routeSpec.Name != "" {
				if _, exists := seenNames[routeSpec.Name]; exists {
					return nil, fmt.Errorf("braingate: gate %q has duplicate route name %q", gateSpec.ID, routeSpec.Name)
				}
				seenNames[routeSpec.Name] = struct{}{}
			}
			if routeSpec.Stop && routeSpec.Next != "" {
				return nil, fmt.Errorf("braingate: gate %q route %d cannot set both stop and next", gateSpec.ID, i)
			}

			var nextGate *Gate
			if routeSpec.Next != "" {
				nextGate = gates[routeSpec.Next]
				if nextGate == nil {
					return nil, fmt.Errorf("braingate: gate %q route %d references unknown gate %q", gateSpec.ID, i, routeSpec.Next)
				}
			}

			valence, err := resolveBounds(routeSpec.Valence)
			if err != nil {
				return nil, fmt.Errorf("braingate: gate %q route %d valence: %w", gateSpec.ID, i, err)
			}
			activation, err := resolveBounds(routeSpec.Activation)
			if err != nil {
				return nil, fmt.Errorf("braingate: gate %q route %d activation: %w", gateSpec.ID, i, err)
			}

			routes = append(routes, Route{
				Name:       routeSpec.Name,
				Valence:    valence,
				Activation: activation,
				Next:       nextGate,
				Stop:       routeSpec.Stop,
			})
		}
		gates[gateSpec.ID].Routes = routes
	}

	return &Graph{
		Start:     start,
		Alpha:     alpha,
		MaxVisits: maxVisits,
		MaxSteps:  maxSteps,
	}, nil
}

func resolveBounds(spec *BoundsSpec) (Bounds, error) {
	minVal, maxVal := -1.0, 1.0
	if spec != nil {
		if spec.Min != nil {
			minVal = *spec.Min
		}
		if spec.Max != nil {
			maxVal = *spec.Max
		}
	}
	if !finite(minVal) || !finite(maxVal) {
		return Bounds{}, errors.New("bounds must be finite numbers")
	}
	if minVal < -1.0 || maxVal > 1.0 || minVal > maxVal {
		return Bounds{}, fmt.Errorf("bounds [%v, %v] must satisfy -1 <= min <= max <= 1", minVal, maxVal)
	}
	return Bounds{Min: minVal, Max: maxVal}, nil
}

func finite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}
