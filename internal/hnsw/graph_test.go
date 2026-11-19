package hnsw_test

import (
	"path/filepath"
	"testing"

	"github.com/snowmerak/braingate/internal/hnsw"
)

func TestGraphAddSearchDeleteAndFiltered(t *testing.T) {
	g := hnsw.NewGraph[string]()
	g.Add(
		hnsw.MakeNode("a", []float32{1, 0, 0}),
		hnsw.MakeNode("b", []float32{0.95, 0.05, 0}),
		hnsw.MakeNode("c", []float32{0.8, 0.2, 0}),
		hnsw.MakeNode("d", []float32{0, 1, 0}),
	)

	if g.Len() != 4 || g.Dims() != 3 {
		t.Fatalf("Len=%d Dims=%d, want 4 and 3", g.Len(), g.Dims())
	}

	got := g.Search([]float32{1, 0, 0}, 2)
	if len(got) != 2 || got[0].Key != "a" || got[1].Key != "b" {
		t.Fatalf("Search = %+v, want [a b]", got)
	}

	// Reject entry node "a" (lexicographically smallest key is layer-0 entry) and "b".
	filtered := g.SearchFiltered([]float32{1, 0, 0}, 2, func(key string) bool {
		return key != "a" && key != "b"
	})
	if len(filtered) != 2 || filtered[0].Key != "c" || filtered[1].Key != "d" {
		t.Fatalf("SearchFiltered = %+v, want [c d]", filtered)
	}

	// Replace existing key "c" with vector near "d".
	g.Add(hnsw.MakeNode("c", []float32{0.05, 0.99, 0}))
	if vec, ok := g.Lookup("c"); !ok || vec[1] < 0.9 {
		t.Fatalf("Lookup(c) after replace = (%v, %v)", vec, ok)
	}

	if !g.Delete("d") || g.Delete("d") {
		t.Fatal("expected first Delete(d)=true and second Delete(d)=false")
	}
	if g.Len() != 3 {
		t.Fatalf("Len after Delete = %d, want 3", g.Len())
	}
}

func TestGraphSaveFileAndLoadFile(t *testing.T) {
	g := hnsw.NewGraph[string]()
	g.Add(
		hnsw.MakeNode("alpha", []float32{1, 0}),
		hnsw.MakeNode("beta", []float32{0, 1}),
	)

	path := filepath.Join(t.TempDir(), "nested", "index.hnsw")
	if err := g.SaveFile(path); err != nil {
		t.Fatalf("SaveFile error = %v", err)
	}

	loaded := hnsw.NewGraph[string]()
	if err := loaded.LoadFile(path); err != nil {
		t.Fatalf("LoadFile error = %v", err)
	}
	if loaded.Len() != 2 {
		t.Fatalf("loaded.Len() = %d, want 2", loaded.Len())
	}
	res := loaded.Search([]float32{0.9, 0.1}, 1)
	if len(res) != 1 || res[0].Key != "alpha" {
		t.Fatalf("loaded Search = %+v, want [alpha]", res)
	}
}
