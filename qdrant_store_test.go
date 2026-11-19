package braingate_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/snowmerak/braingate"
)

func TestQdrantStoreLifecycleAndCircularRefinement(t *testing.T) {
	var seenCalls []string
	var createdIndexes []string
	var upsertedIDs []string
	var deletedIDs []string

	store, err := braingate.NewQdrantStore(braingate.QdrantConfig{
		BaseURL:    "http://localhost:6333",
		Collection: "test_memories",
		APIKey:     "qdrant-secret",
		HTTPClient: &http.Client{Transport: testTransport(func(r *http.Request) (*http.Response, error) {
			if r.Header.Get("api-key") != "qdrant-secret" {
				t.Errorf("missing api-key header on %s %s", r.Method, r.URL.Path)
			}
			key := r.Method + " " + r.URL.Path
			seenCalls = append(seenCalls, key)

			switch {
			case r.Method == http.MethodGet && r.URL.Path == "/collections/test_memories":
				return jsonHTTPResponse(http.StatusNotFound, `{"status":{"error":"not found"}}`), nil

			case r.Method == http.MethodPut && r.URL.Path == "/collections/test_memories":
				var req struct {
					Vectors struct {
						Size     int    `json:"size"`
						Distance string `json:"distance"`
					} `json:"vectors"`
				}
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Fatal(err)
				}
				if req.Vectors.Size != 3 || req.Vectors.Distance != "Cosine" {
					t.Fatalf("unexpected create collection payload: %+v", req)
				}
				return jsonHTTPResponse(http.StatusOK, `{"result":true,"status":"ok"}`), nil

			case r.Method == http.MethodPut && r.URL.Path == "/collections/test_memories/index":
				var req struct {
					FieldName   string `json:"field_name"`
					FieldSchema string `json:"field_schema"`
				}
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Fatal(err)
				}
				if req.FieldSchema != "float" {
					t.Fatalf("unexpected index schema: %+v", req)
				}
				createdIndexes = append(createdIndexes, req.FieldName)
				return jsonHTTPResponse(http.StatusOK, `{"result":{"status":"completed"},"status":"ok"}`), nil

			case r.Method == http.MethodPut && r.URL.Path == "/collections/test_memories/points":
				var req struct {
					Points []struct {
						ID      string `json:"id"`
						Payload struct {
							MemoryID string `json:"memory_id"`
						} `json:"payload"`
					} `json:"points"`
				}
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Fatal(err)
				}
				for _, p := range req.Points {
					if len(p.ID) != 36 || p.ID[14] != '5' {
						t.Fatalf("expected deterministic UUIDv5, got %q", p.ID)
					}
					upsertedIDs = append(upsertedIDs, p.ID)
				}
				return jsonHTTPResponse(http.StatusOK, `{"result":{"status":"completed"},"status":"ok"}`), nil

			case r.Method == http.MethodPost && r.URL.Path == "/collections/test_memories/points/delete":
				var req struct {
					Points []string `json:"points"`
				}
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Fatal(err)
				}
				deletedIDs = append(deletedIDs, req.Points...)
				return jsonHTTPResponse(http.StatusOK, `{"result":{"status":"completed"},"status":"ok"}`), nil

			case r.Method == http.MethodPost && r.URL.Path == "/collections/test_memories/points/query":
				var req struct {
					Filter struct {
						Must []struct {
							Key   string             `json:"key"`
							Range map[string]float64 `json:"range"`
						} `json:"must"`
						MustNot []struct {
							HasID []string `json:"has_id"`
						} `json:"must_not"`
					} `json:"filter"`
					ScoreThreshold float64 `json:"score_threshold"`
				}
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Fatal(err)
				}
				if len(req.Filter.Must) != 2 || len(req.Filter.MustNot) != 1 || len(req.Filter.MustNot[0].HasID) != 1 {
					t.Fatalf("unexpected query filter: %+v", req.Filter)
				}
				if req.Filter.Must[0].Range["gte"] != -1 || req.Filter.Must[0].Range["lte"] != 1 {
					t.Fatalf("unexpected bounding box range: %+v", req.Filter.Must)
				}
				// Return 3 records: corner (0.8, 0.8) outside circle, valid (-0.6, 0.6), and excluded fallback.
				return jsonHTTPResponse(http.StatusOK, `{
					"result": {
						"points": [
							{"id":"uuid-corner","score":0.95,"payload":{"memory_id":"corner_outside","valence":0.8,"activation":0.8,"experience":"모서리"}},
							{"id":"uuid-valid","score":0.90,"payload":{"memory_id":"inside_circle","cue":"단서","valence":-0.6,"activation":0.6,"experience":"원 안","alpha":0.4,"salience":0.85,"created_at":"2026-11-10T12:00:00Z"}},
							{"id":"uuid-excl","score":0.99,"payload":{"memory_id":"excluded","valence":0.1,"activation":0.1,"experience":"제외"}}
						]
					},
					"status": "ok"
				}`), nil

			case r.Method == http.MethodPost && r.URL.Path == "/collections/test_memories/points/scroll":
				return jsonHTTPResponse(http.StatusOK, `{
					"result": {
						"points": [
							{"payload":{"memory_id":"corner_outside","valence":0.8,"activation":0.8,"experience":"모서리"}},
							{"payload":{"memory_id":"near","valence":0.1,"activation":0.2,"experience":"가까움"}},
							{"payload":{"memory_id":"farther","valence":0.5,"activation":0.5,"experience":"조금 멈"}}
						]
					},
					"status": "ok"
				}`), nil

			default:
				t.Fatalf("unexpected request: %s", key)
				return nil, nil
			}
		})},
	})
	if err != nil {
		t.Fatalf("NewQdrantStore error = %v", err)
	}

	createdAt := time.Date(2026, 11, 10, 12, 0, 0, 0, time.UTC)
	memories := []braingate.Memory{
		{ID: "inside_circle", Cue: "단서", Position: braingate.Point{Valence: -0.6, Activation: 0.6}, Experience: "원 안", Alpha: 0.4, Salience: 0.85, CreatedAt: &createdAt, Embedding: []float64{1, 0, 0}},
		{ID: "corner_outside", Position: braingate.Point{Valence: 0.8, Activation: 0.8}, Experience: "모서리", Embedding: []float64{1, 0, 0}},
	}
	if err := store.Upsert(t.Context(), memories); err != nil {
		t.Fatalf("Upsert error = %v", err)
	}
	if !reflect.DeepEqual(createdIndexes, []string{"valence", "activation"}) {
		t.Fatalf("createdIndexes = %v, want [valence activation]", createdIndexes)
	}

	// Second EnsureCollection should use cached dimension with zero HTTP requests.
	callsBefore := len(seenCalls)
	if err := store.EnsureCollection(t.Context(), 3); err != nil {
		t.Fatalf("cached EnsureCollection error = %v", err)
	}
	if len(seenCalls) != callsBefore {
		t.Fatalf("expected cached EnsureCollection to make 0 HTTP calls")
	}

	hits, err := store.Search(t.Context(), braingate.SearchQuery{
		Probe:         braingate.Point{Valence: 0, Activation: 0},
		Radius:        1.0,
		Vector:        []float64{1, 0, 0},
		MinSimilarity: 0.35,
		ExcludeIDs:    map[string]bool{"excluded": true},
		Limit:         10,
	})
	if err != nil {
		t.Fatalf("Search error = %v", err)
	}
	if len(hits) != 1 || hits[0].Memory.ID != "inside_circle" || hits[0].Memory.Alpha != 0.4 || hits[0].Memory.Salience != 0.85 || hits[0].Memory.CreatedAt == nil || !hits[0].Memory.CreatedAt.Equal(createdAt) || hits[0].Similarity == nil || !almostEqual(*hits[0].Similarity, 0.90) {
		t.Fatalf("hits = %+v, want only [inside_circle]", hits)
	}

	// Vectorless scroll fallback should also filter corners and sort by Distance.
	scrollHits, err := store.Search(t.Context(), braingate.SearchQuery{
		Probe:  braingate.Point{Valence: 0, Activation: 0},
		Radius: 1.0,
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("scroll Search error = %v", err)
	}
	if len(scrollHits) != 2 || scrollHits[0].Memory.ID != "near" || scrollHits[1].Memory.ID != "farther" {
		t.Fatalf("scrollHits = %+v, want [near farther]", scrollHits)
	}

	if err := store.Delete(t.Context(), []string{"inside_circle"}); err != nil {
		t.Fatalf("Delete error = %v", err)
	}
	if len(deletedIDs) != 1 || deletedIDs[0] != upsertedIDs[0] {
		t.Fatalf("deleted UUID %v does not match upserted UUID %v", deletedIDs, upsertedIDs)
	}
}

func TestEmbedMemoriesPopulatesMissingEmbeddingsOnly(t *testing.T) {
	var embeddedTexts []string
	embedder := fakeEmbedder(func(_ context.Context, texts []string) ([][]float64, error) {
		embeddedTexts = append([]string(nil), texts...)
		return [][]float64{{0, 1}}, nil
	})

	input := []braingate.Memory{
		{ID: "pre", Position: braingate.Point{Valence: 0.5, Activation: 0.5}, Experience: "기존", Embedding: []float64{1, 0}},
		{ID: "need", Cue: "단서", Position: braingate.Point{Valence: 0.2, Activation: 0.2}, Experience: "새 기억"},
	}

	out, err := braingate.EmbedMemories(t.Context(), embedder, input)
	if err != nil {
		t.Fatalf("EmbedMemories error = %v", err)
	}
	if !reflect.DeepEqual(embeddedTexts, []string{"단서\n새 기억"}) {
		t.Fatalf("embeddedTexts = %v", embeddedTexts)
	}
	if len(input[1].Embedding) != 0 {
		t.Fatal("EmbedMemories should not mutate caller slice in place")
	}
	if !reflect.DeepEqual(out[0].Embedding, []float64{1, 0}) || !reflect.DeepEqual(out[1].Embedding, []float64{0, 1}) {
		t.Fatalf("unexpected embeddings: %+v", out)
	}
}

func jsonHTTPResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Status:     http.StatusText(status),
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}
