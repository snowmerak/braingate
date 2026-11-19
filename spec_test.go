package braingate_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/snowmerak/braingate"
)

func TestParseSpecBuildsDefaultsAndMemories(t *testing.T) {
	const definition = `{
		"persona": "  신중하고 사려 깊은 성향  ",
		"memories": [
			{
				"id": "m1",
				"cue": "반가운 인사",
				"position": { "valence": 0.6, "activation": 0.5 },
				"experience": "기분 좋게 응대했던 기억",
				"alpha": 0.75,
				"salience": 0.8,
				"created_at": "2026-11-10T09:30:00Z",
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
	if engine.Persona != "신중하고 사려 깊은 성향" {
		t.Fatalf("persona = %q, want %q", engine.Persona, "신중하고 사려 깊은 성향")
	}

	wantTemp := braingate.Temperament{
		Depth:         braingate.DefaultDepth,
		Alpha:         braingate.DefaultAlpha,
		Deadzone:      braingate.DefaultDeadzone,
		Radius:        braingate.DefaultRadius,
		MinSimilarity: braingate.DefaultMinSimilarity,
		PoolSize:      braingate.DefaultPoolSize,
		MinP:          braingate.DefaultMinP,
		MaxVisits:     braingate.DefaultMaxVisits,
		HalfLifeHours: braingate.DefaultHalfLifeHours,
	}
	if engine.Temperament != wantTemp {
		t.Fatalf("temperament = %+v, want %+v", engine.Temperament, wantTemp)
	}
	wantCreated := time.Date(2026, 11, 10, 9, 30, 0, 0, time.UTC)
	if len(engine.Memories) != 2 || engine.Memories[0].Alpha != 0.75 || engine.Memories[0].Salience != 0.8 || engine.Memories[0].CreatedAt == nil || !engine.Memories[0].CreatedAt.Equal(wantCreated) || engine.Memories[1].CreatedAt != nil || len(engine.Memories[0].Embedding) != 2 {
		t.Fatalf("unexpected memories: %+v", engine.Memories)
	}
}

func TestParseSpecPreservesExplicitZeroDeadzone(t *testing.T) {
	const definition = `{
		"temperament": { "deadzone": 0, "depth": 2, "radius": 1.5, "min_similarity": -0.2, "min_p": 0, "half_life_hours": 0 },
		"memories": [
			{ "id": "m1", "position": { "valence": 0.5, "activation": 0.5 }, "experience": "기억" }
		]
	}`
	engine, err := braingate.ParseSpec([]byte(definition))
	if err != nil {
		t.Fatalf("ParseSpec error = %v", err)
	}
	if engine.Temperament.Deadzone != 0 || engine.Temperament.Depth != 2 || engine.Temperament.Radius != 1.5 || engine.Temperament.MinSimilarity != -0.2 || engine.Temperament.MinP != 0 || engine.Temperament.HalfLifeHours != 0 {
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
			name: "radius zero",
			spec: braingate.Spec{Temperament: braingate.TemperamentSpec{Radius: f(0)}, Memories: []braingate.MemorySpec{validMem}},
			want: "radius",
		},
		{
			name: "negative radius",
			spec: braingate.Spec{Temperament: braingate.TemperamentSpec{Radius: f(-0.1)}, Memories: []braingate.MemorySpec{validMem}},
			want: "radius",
		},
		{
			name: "min_similarity > 1",
			spec: braingate.Spec{Temperament: braingate.TemperamentSpec{MinSimilarity: f(1.1)}, Memories: []braingate.MemorySpec{validMem}},
			want: "min_similarity",
		},
		{
			name: "min_similarity < -1",
			spec: braingate.Spec{Temperament: braingate.TemperamentSpec{MinSimilarity: f(-1.1)}, Memories: []braingate.MemorySpec{validMem}},
			want: "min_similarity",
		},
		{
			name: "negative half_life_hours",
			spec: braingate.Spec{Temperament: braingate.TemperamentSpec{HalfLifeHours: f(-1)}, Memories: []braingate.MemorySpec{validMem}},
			want: "half_life_hours",
		},
		{
			name: "min_p < 0",
			spec: braingate.Spec{Temperament: braingate.TemperamentSpec{MinP: f(-0.1)}, Memories: []braingate.MemorySpec{validMem}},
			want: "min_p",
		},
		{
			name: "min_p > 1",
			spec: braingate.Spec{Temperament: braingate.TemperamentSpec{MinP: f(1.1)}, Memories: []braingate.MemorySpec{validMem}},
			want: "min_p",
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
			name: "negative salience",
			spec: braingate.Spec{Memories: []braingate.MemorySpec{{ID: "m1", Position: validMem.Position, Experience: "e", Salience: f(-0.1)}}},
			want: "salience",
		},
		{
			name: "salience > 1",
			spec: braingate.Spec{Memories: []braingate.MemorySpec{{ID: "m1", Position: validMem.Position, Experience: "e", Salience: f(1.1)}}},
			want: "salience",
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
