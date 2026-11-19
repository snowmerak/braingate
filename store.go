package braingate

import (
	"context"
	"errors"
	"fmt"
	"slices"
)

const defaultSearchLimit = 32

// SearchQuery specifies affect-space radius filtering and optional vector similarity ranking.
type SearchQuery struct {
	Probe         Point
	Radius        float64
	Vector        []float64
	MinSimilarity float64
	ExcludeIDs    map[string]bool
	Limit         int
}

// ScoredMemory pairs a stored Memory with its Euclidean affect distance and optional cosine similarity.
type ScoredMemory struct {
	Memory     Memory
	Distance   float64
	Similarity *float64
}

// MemoryStore persists and queries Memory records for Engine.Run.
type MemoryStore interface {
	Upsert(ctx context.Context, memories []Memory) error
	Delete(ctx context.Context, ids []string) error
	Search(ctx context.Context, query SearchQuery) ([]ScoredMemory, error)
}

// EmbedMemories validates memories and computes embeddings in one batch for any entries lacking one.
func EmbedMemories(ctx context.Context, embedder Embedder, memories []Memory) ([]Memory, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	preDim, err := validateMemories(memories)
	if err != nil {
		return nil, err
	}
	out := make([]Memory, len(memories))
	for i, m := range memories {
		out[i] = cloneMemory(m)
	}
	if len(out) == 0 {
		return out, nil
	}

	batch := make([]string, 0, len(out))
	indices := make([]int, 0, len(out))
	for i, m := range out {
		if len(m.Embedding) == 0 {
			batch = append(batch, memoryText(m))
			indices = append(indices, i)
		}
	}
	if len(batch) == 0 {
		return out, nil
	}
	if embedder == nil {
		return nil, errors.New("braingate: embedder is required to embed memories")
	}

	vecs, err := embedder.Embed(ctx, batch)
	if err != nil {
		return nil, err
	}
	if len(vecs) != len(batch) {
		return nil, fmt.Errorf("braingate: embedder returned %d vectors for %d memories", len(vecs), len(batch))
	}

	dim := preDim
	for offset, idx := range indices {
		vec := vecs[offset]
		if len(vec) == 0 {
			return nil, fmt.Errorf("braingate: memory %q received empty embedding", out[idx].ID)
		}
		if dim == 0 {
			dim = len(vec)
		} else if len(vec) != dim {
			return nil, fmt.Errorf("braingate: memory %q embedding dimension %d does not match %d", out[idx].ID, len(vec), dim)
		}
		for _, v := range vec {
			if !finite(v) {
				return nil, fmt.Errorf("braingate: memory %q embedding contains non-finite value", out[idx].ID)
			}
		}
		out[idx].Embedding = slices.Clone(vec)
	}
	return out, nil
}

func cloneMemory(m Memory) Memory {
	cloned := m
	if len(m.Embedding) > 0 {
		cloned.Embedding = slices.Clone(m.Embedding)
	}
	if m.CreatedAt != nil {
		ts := *m.CreatedAt
		cloned.CreatedAt = &ts
	}
	return cloned
}

func validateSearchQuery(q SearchQuery) (int, error) {
	if err := validatePoint(q.Probe); err != nil {
		return 0, fmt.Errorf("braingate: search probe: %w", err)
	}
	if !finite(q.Radius) || q.Radius <= 0 {
		return 0, fmt.Errorf("braingate: search radius must be > 0, got %v", q.Radius)
	}
	if len(q.Vector) > 0 {
		if !finite(q.MinSimilarity) || q.MinSimilarity < -1 || q.MinSimilarity > 1 {
			return 0, fmt.Errorf("braingate: search min_similarity must be in [-1, 1], got %v", q.MinSimilarity)
		}
		for _, v := range q.Vector {
			if !finite(v) {
				return 0, errors.New("braingate: search vector contains non-finite value")
			}
		}
	}
	limit := q.Limit
	if limit <= 0 {
		limit = defaultSearchLimit
	}
	return limit, nil
}
