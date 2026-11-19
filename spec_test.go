package braingate_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/snowmerak/braingate"
)

func TestParseSpecBuildsDefaultsAndRoutes(t *testing.T) {
	const definition = `{
		"start": "first",
		"gates": [
			{
				"id": "leaf",
				"experience": "마무리 기억"
			},
			{
				"id": "first",
				"experience": "초기 반응 기억",
				"alpha": 0.75,
				"routes": [
					{
						"name": "positive_high",
						"valence": { "min": 0.2 },
						"activation": { "max": 0.8 },
						"next": "leaf"
					},
					{
						"name": "halt",
						"stop": true
					}
				]
			}
		]
	}`

	graph, err := braingate.ParseSpec([]byte(definition))
	if err != nil {
		t.Fatalf("ParseSpec error = %v", err)
	}
	if graph.Alpha != braingate.DefaultAlpha || graph.MaxVisits != braingate.DefaultMaxVisits || graph.MaxSteps != braingate.DefaultMaxSteps {
		t.Fatalf("unexpected graph defaults: %+v", graph)
	}
	if graph.Start == nil || graph.Start.ID != "first" || graph.Start.Alpha != 0.75 || len(graph.Start.Routes) != 2 {
		t.Fatalf("unexpected start gate: %+v", graph.Start)
	}

	firstRoute := graph.Start.Routes[0]
	if firstRoute.Name != "positive_high" ||
		firstRoute.Valence != (braingate.Bounds{Min: 0.2, Max: 1.0}) ||
		firstRoute.Activation != (braingate.Bounds{Min: -1.0, Max: 0.8}) ||
		firstRoute.Next == nil || firstRoute.Next.ID != "leaf" || len(firstRoute.Next.Routes) != 0 {
		t.Fatalf("unexpected first route: %+v", firstRoute)
	}

	secondRoute := graph.Start.Routes[1]
	if secondRoute.Name != "halt" || !secondRoute.Stop ||
		secondRoute.Valence != (braingate.Bounds{Min: -1.0, Max: 1.0}) ||
		secondRoute.Activation != (braingate.Bounds{Min: -1.0, Max: 1.0}) {
		t.Fatalf("unexpected second route: %+v", secondRoute)
	}
}

func TestParseSpecRejectsUnknownFieldsAndTrailingData(t *testing.T) {
	cases := []struct {
		name string
		raw  string
	}{
		{
			name: "unknown field",
			raw:  `{"start":"g","extra":1,"gates":[{"id":"g","experience":"exp"}]}`,
		},
		{
			name: "trailing json object",
			raw:  `{"start":"g","gates":[{"id":"g","experience":"exp"}]}{"extra":true}`,
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

	cases := []struct {
		name string
		spec braingate.Spec
		want string
	}{
		{
			name: "no gates",
			spec: braingate.Spec{Start: "a"},
			want: "at least one gate",
		},
		{
			name: "alpha zero",
			spec: braingate.Spec{Start: "a", Alpha: f(0), Gates: []braingate.GateSpec{{ID: "a", Experience: "e"}}},
			want: "alpha",
		},
		{
			name: "alpha > 1",
			spec: braingate.Spec{Start: "a", Alpha: f(1.2), Gates: []braingate.GateSpec{{ID: "a", Experience: "e"}}},
			want: "alpha",
		},
		{
			name: "max_visits < 1",
			spec: braingate.Spec{Start: "a", MaxVisits: i(0), Gates: []braingate.GateSpec{{ID: "a", Experience: "e"}}},
			want: "max_visits",
		},
		{
			name: "max_steps < 1",
			spec: braingate.Spec{Start: "a", MaxSteps: i(0), Gates: []braingate.GateSpec{{ID: "a", Experience: "e"}}},
			want: "max_steps",
		},
		{
			name: "blank gate id",
			spec: braingate.Spec{Start: " ", Gates: []braingate.GateSpec{{ID: " ", Experience: "e"}}},
			want: "gate ID",
		},
		{
			name: "duplicate gate id",
			spec: braingate.Spec{
				Start: "a",
				Gates: []braingate.GateSpec{
					{ID: "a", Experience: "e1"},
					{ID: "a", Experience: "e2"},
				},
			},
			want: "duplicate gate ID",
		},
		{
			name: "blank experience",
			spec: braingate.Spec{Start: "a", Gates: []braingate.GateSpec{{ID: "a", Experience: " "}}},
			want: "experience",
		},
		{
			name: "gate alpha zero",
			spec: braingate.Spec{Start: "a", Gates: []braingate.GateSpec{{ID: "a", Experience: "e", Alpha: f(0)}}},
			want: "alpha",
		},
		{
			name: "unknown start",
			spec: braingate.Spec{Start: "missing", Gates: []braingate.GateSpec{{ID: "a", Experience: "e"}}},
			want: "unknown starting gate",
		},
		{
			name: "duplicate route name",
			spec: braingate.Spec{
				Start: "a",
				Gates: []braingate.GateSpec{{
					ID:         "a",
					Experience: "e",
					Routes:     []braingate.RouteSpec{{Name: "dup"}, {Name: "dup"}},
				}},
			},
			want: "duplicate route name",
		},
		{
			name: "stop and next together",
			spec: braingate.Spec{
				Start: "a",
				Gates: []braingate.GateSpec{{
					ID:         "a",
					Experience: "e",
					Routes:     []braingate.RouteSpec{{Stop: true, Next: "a"}},
				}},
			},
			want: "cannot set both stop and next",
		},
		{
			name: "unknown next gate",
			spec: braingate.Spec{
				Start: "a",
				Gates: []braingate.GateSpec{{
					ID:         "a",
					Experience: "e",
					Routes:     []braingate.RouteSpec{{Next: "missing"}},
				}},
			},
			want: "unknown gate",
		},
		{
			name: "min > max",
			spec: braingate.Spec{
				Start: "a",
				Gates: []braingate.GateSpec{{
					ID:         "a",
					Experience: "e",
					Routes:     []braingate.RouteSpec{{Valence: &braingate.BoundsSpec{Min: f(0.5), Max: f(0.2)}}},
				}},
			},
			want: "-1 <= min <= max <= 1",
		},
		{
			name: "out of range bounds",
			spec: braingate.Spec{
				Start: "a",
				Gates: []braingate.GateSpec{{
					ID:         "a",
					Experience: "e",
					Routes:     []braingate.RouteSpec{{Activation: &braingate.BoundsSpec{Min: f(-1.5)}}},
				}},
			},
			want: "-1 <= min <= max <= 1",
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
	graph, err := braingate.ParseSpec(data)
	if err != nil {
		t.Fatalf("ParseSpec example error = %v", err)
	}
	if graph.Start == nil || graph.Start.ID != "initial_appraisal" {
		t.Fatalf("unexpected start gate in example: %+v", graph.Start)
	}
}
