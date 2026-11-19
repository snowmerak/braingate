package braingate_test

import (
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/snowmerak/braingate"
)

func TestHNSWStoreUpsertSearchDeleteAndSaveLoad(t *testing.T) {
	createdAt := time.Date(2026, 11, 10, 12, 0, 0, 0, time.UTC)
	store := braingate.NewHNSWStore(braingate.HNSWConfig{})
	err := store.Upsert(t.Context(), []braingate.Memory{
		{
			ID:                 "inside_circle",
			Cue:                "원 안 기억",
			Position:           braingate.Point{Valence: 0.6, Activation: 0.6}, // dist from (0,0) = 0.8485 <= 1.0
			Experience:         "반경 내 호의 기억",
			Salience:           0.75,
			CreatedAt:          &createdAt,
			Embedding:          []float64{1, 0, 0},
			SecondaryEmbedding: []float64{0, 1, 0, 0},
		},
		{
			ID:         "corner_outside",
			Cue:        "사각형 모서리 기억",
			Position:   braingate.Point{Valence: 0.8, Activation: 0.8}, // dist from (0,0) = 1.131 > 1.0
			Experience: "바운딩 박스 안이지만 원 밖인 기억",
			Embedding:  []float64{1, 0, 0},
		},
		{
			ID:         "low_sim",
			Cue:        "유사도 낮은 기억",
			Position:   braingate.Point{Valence: 0.1, Activation: 0.1},
			Experience: "가깝지만 의미가 다른 기억",
			Embedding:  []float64{0, 1, 0},
		},
		{
			ID:         "excluded",
			Cue:        "제외 대상 기억",
			Position:   braingate.Point{Valence: 0.2, Activation: 0.2},
			Experience: "직전 쿨다운 기억",
			Embedding:  []float64{1, 0, 0},
		},
	})
	if err != nil {
		t.Fatalf("Upsert error = %v", err)
	}
	if store.Len() != 4 {
		t.Fatalf("Len = %d, want 4", store.Len())
	}

	hits, err := store.Search(t.Context(), braingate.SearchQuery{
		Probe:           braingate.Point{Valence: 0, Activation: 0},
		Radius:          1.0,
		Vector:          []float64{1, 0, 0},
		SecondaryVector: []float64{0, 1, 0, 0},
		MinSimilarity:   0.5,
		ExcludeIDs:      map[string]bool{"excluded": true},
		Limit:           10,
	})
	if err != nil {
		t.Fatalf("Search error = %v", err)
	}
	if len(hits) != 1 || hits[0].Memory.ID != "inside_circle" || hits[0].Memory.Salience != 0.75 || hits[0].Memory.CreatedAt == nil || !hits[0].Memory.CreatedAt.Equal(createdAt) || hits[0].Similarity == nil || !almostEqual(*hits[0].Similarity, 1.0) || hits[0].SecondarySimilarity == nil || !almostEqual(*hits[0].SecondarySimilarity, 1.0) {
		t.Fatalf("hits = %+v, want only [inside_circle]", hits)
	}

	// Spatial-only search when Vector is omitted.
	spatialHits, err := store.Search(t.Context(), braingate.SearchQuery{
		Probe:      braingate.Point{Valence: 0, Activation: 0},
		Radius:     1.0,
		ExcludeIDs: map[string]bool{"excluded": true},
		Limit:      10,
	})
	if err != nil {
		t.Fatalf("spatial Search error = %v", err)
	}
	if len(spatialHits) != 2 || spatialHits[0].Memory.ID != "low_sim" || spatialHits[1].Memory.ID != "inside_circle" {
		t.Fatalf("spatialHits = %+v, want [low_sim inside_circle]", spatialHits)
	}

	// Atomic file save and load on Windows.
	snapshotPath := filepath.Join(t.TempDir(), "snapshots", "memories.hnsw")
	if err := store.SaveFile(snapshotPath); err != nil {
		t.Fatalf("SaveFile error = %v", err)
	}

	reloaded := braingate.NewHNSWStore(braingate.HNSWConfig{})
	if err := reloaded.LoadFile(snapshotPath); err != nil {
		t.Fatalf("LoadFile error = %v", err)
	}
	reloadedHits, err := reloaded.Search(t.Context(), braingate.SearchQuery{
		Probe:           braingate.Point{Valence: 0, Activation: 0},
		Radius:          1.0,
		Vector:          []float64{1, 0, 0},
		SecondaryVector: []float64{0, 1, 0, 0},
		MinSimilarity:   0.5,
		ExcludeIDs:      map[string]bool{"excluded": true},
		Limit:           10,
	})
	if err != nil {
		t.Fatalf("reloaded Search error = %v", err)
	}
	if len(reloadedHits) != 1 || reloadedHits[0].Memory.ID != "inside_circle" || reloadedHits[0].Memory.Salience != 0.75 || reloadedHits[0].Memory.CreatedAt == nil || !reloadedHits[0].Memory.CreatedAt.Equal(createdAt) || reloadedHits[0].SecondarySimilarity == nil || !almostEqual(*reloadedHits[0].SecondarySimilarity, 1.0) {
		t.Fatalf("reloadedHits = %+v", reloadedHits)
	}

	if err := reloaded.Delete(t.Context(), []string{"inside_circle"}); err != nil {
		t.Fatalf("Delete error = %v", err)
	}
	if reloaded.Len() != 3 {
		t.Fatalf("reloaded Len after Delete = %d, want 3", reloaded.Len())
	}

	// Dimension mismatch checks.
	if err := store.Upsert(t.Context(), []braingate.Memory{{
		ID: "bad_dim", Position: braingate.Point{}, Experience: "x", Embedding: []float64{1, 0},
	}}); err == nil || !strings.Contains(err.Error(), "dimension") {
		t.Fatalf("expected dimension error on Upsert, got %v", err)
	}
	if _, err := store.Search(t.Context(), braingate.SearchQuery{
		Probe: braingate.Point{}, Radius: 1.0, Vector: []float64{1, 0},
	}); err == nil || !strings.Contains(err.Error(), "dimension") {
		t.Fatalf("expected dimension error on Search, got %v", err)
	}
}
