package braingate_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/snowmerak/braingate"
)

func TestParseSpecBuildsDefaultsAndMemories(t *testing.T) {
	const definition = `{
		"memories": [
			{
				"id": "m1",
				"cue": "반가운 인사",
				"position": { "valence": 0.6, "activation": 0.5 },
				"experience": "기분 좋게 응대했던 기억",
				"alpha": 0.75,
				"embedding": [0.1, 0.2]
			},
			{
				"id": "m2",
				"position": { "valence": -0.4, "activation": -0.3 },
				"experience": "조용히 물러선 기억"
			}
		]
	}`

	engine, err := braingate.ParseSpec([]byte(definition))
	if err != nil {
		t.Fatalf("ParseSpec error = %v", err)
	}

	wantTemp := braingate.Temperament{
		Depth:     braingate.DefaultDepth,
		Alpha:     braingate.DefaultAlpha,
		Deadzone:  braingate.DefaultDeadzone,
		PoolSize:  braingate.DefaultPoolSize,
		MaxVisits: braingate.DefaultMaxVisits,
		Radius:    0,
	}
	if engine.Temperament != wantTemp {
		t.Fatalf("temperament = %+v, want %+v", engine.Temperament, wantTemp)
	}
	if len(engine.Memories) != 2 || engine.Memories[0].Alpha != 0.75 || len(engine.Memories[0].Embedding) != 2 {
		t.Fatalf("unexpected memories: %+v", engine.Memories)
	}
}

func TestParseSpecPreservesExplicitZeroDeadzone(t *testing.T) {
	const definition = `{
		"temperament": { "deadzone": 0, "depth": 2, "radius": 1.5 },
		"memories": [
			{ "id": "m1", "position": { "valence": 0.5, "activation": 0.5 }, "experience": "기억" }
		]
	}`
	engine, err := braingate.ParseSpec([]byte(definition))
	if err != nil {
		t.Fatalf("ParseSpec error = %v", err)
	}
	if engine.Temperament.Deadzone != 0 || engine.Temperament.Depth != 2 || engine.Temperament.Radius != 1.5 {
		t.Fatalf("unexpected temperament: %+v", engine.Temperament)
	}
}

func TestParseSpecRejectsUnknownFieldsAndTrailingData(t *testing.T) {
	cases := []struct {
		name string
		raw  string
	}{
		{
			name: "unknown field",
			raw:  `{"extra":1,"memories":[{"id":"m1","position":{"valence":0,"activation":0},"experience":"exp"}]}`,
		},
		{
			name: "trailing json object",
			raw:  `{"memories":[{"id":"m1","position":{"valence":0,"activation":0},"experience":"exp"}]}{"extra":true}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := braingate.ParseSpec([]byte(tc.raw)); err == nil {
				t.Fatalf("expected ParseSpec error for %s", tc.name)
			}
		})
	}
}

func TestSpecBuildValidationErrors(t *testing.T) {
	f := func(v float64) *float64 { return &v }
	i := func(v int) *int { return &v }
	validMem := braingate.MemorySpec{
		ID:         "m1",
		Position:   braingate.Point{Valence: 0.4, Activation: 0.4},
		Experience: "기억",
	}

	cases := []struct {
		name string
		spec braingate.Spec
		want string
	}{
		{
			name: "no memories",
			spec: braingate.Spec{},
			want: "at least one memory",
		},
		{
			name: "depth < 1",
			spec: braingate.Spec{Temperament: braingate.TemperamentSpec{Depth: i(0)}, Memories: []braingate.MemorySpec{validMem}},
			want: "depth",
		},
		{
			name: "alpha zero",
			spec: braingate.Spec{Temperament: braingate.TemperamentSpec{Alpha: f(0)}, Memories: []braingate.MemorySpec{validMem}},
			want: "alpha",
		},
		{
			name: "alpha > 1",
			spec: braingate.Spec{Temperament: braingate.TemperamentSpec{Alpha: f(1.2)}, Memories: []braingate.MemorySpec{validMem}},
			want: "alpha",
		},
		{
			name: "deadzone >= 1",
			spec: braingate.Spec{Temperament: braingate.TemperamentSpec{Deadzone: f(1.0)}, Memories: []braingate.MemorySpec{validMem}},
			want: "deadzone",
		},
		{
			name: "pool_size < 1",
			spec: braingate.Spec{Temperament: braingate.TemperamentSpec{PoolSize: i(0)}, Memories: []braingate.MemorySpec{validMem}},
			want: "pool_size",
		},
		{
			name: "max_visits < 1",
			spec: braingate.Spec{Temperament: braingate.TemperamentSpec{MaxVisits: i(0)}, Memories: []braingate.MemorySpec{validMem}},
			want: "max_visits",
		},
		{
			name: "negative radius",
			spec: braingate.Spec{Temperament: braingate.TemperamentSpec{Radius: f(-0.1)}, Memories: []braingate.MemorySpec{validMem}},
			want: "radius",
		},
		{
			name: "blank memory id",
			spec: braingate.Spec{Memories: []braingate.MemorySpec{{ID: " ", Position: validMem.Position, Experience: "e"}}},
			want: "memory ID",
		},
		{
			name: "duplicate memory id",
			spec: braingate.Spec{Memories: []braingate.MemorySpec{validMem, validMem}},
			want: "duplicate memory ID",
		},
		{
			name: "blank experience",
			spec: braingate.Spec{Memories: []braingate.MemorySpec{{ID: "m1", Position: validMem.Position, Experience: " "}}},
			want: "experience",
		},
		{
			name: "out of range position",
			spec: braingate.Spec{Memories: []braingate.MemorySpec{{ID: "m1", Position: braingate.Point{Valence: 1.5, Activation: 0}, Experience: "e"}}},
			want: "within [-1, 1]",
		},
		{
			name: "memory alpha zero",
			spec: braingate.Spec{Memories: []braingate.MemorySpec{{ID: "m1", Position: validMem.Position, Experience: "e", Alpha: f(0)}}},
			want: "alpha",
		},
		{
			name: "inconsistent embedding dimensions",
			spec: braingate.Spec{Memories: []braingate.MemorySpec{
				{ID: "m1", Position: validMem.Position, Experience: "e1", Embedding: []float64{1, 0}},
				{ID: "m2", Position: validMem.Position, Experience: "e2", Embedding: []float64{1, 0, 0}},
			}},
			want: "embedding dimension",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := tc.spec.Build()
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Build error = %v, want substring %q", err, tc.want)
			}
		})
	}
}

func TestExampleImmediateResponseSpecIsValid(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("examples", "immediate-response.json"))
	if err != nil {
		t.Fatalf("read example spec: %v", err)
	}
	engine, err := braingate.ParseSpec(data)
	if err != nil {
		t.Fatalf("ParseSpec example error = %v", err)
	}
	if len(engine.Memories) < 4 {
		t.Fatalf("expected memories across quadrants, got %d", len(engine.Memories))
	}
	quadrants := map[[2]bool]bool{}
	for _, m := range engine.Memories {
		quadrants[[2]bool{m.Position.Valence >= 0, m.Position.Activation >= 0}] = true
	}
	if len(quadrants) != 4 {
		t.Fatalf("expected example memories in all 4 quadrants, covered %d", len(quadrants))
	}
}
