package braingate

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"github.com/snowmerak/braingate/internal/hnsw"
)

const hnswStoreMagic = "BGHNSW01"

// HNSWConfig configures the in-memory filtered HNSW index parameters.
type HNSWConfig struct {
	M        int
	Ml       float64
	EfSearch int
}

// HNSWStore is a concurrency-safe MemoryStore backed by internal/hnsw with
// circular affect-space filtering and atomic file persistence.
type HNSWStore struct {
	mu       sync.RWMutex
	cfg      HNSWConfig
	memories map[string]Memory
	index    *hnsw.Graph[string]
	dims     int
}

// NewHNSWStore creates an empty HNSWStore using cfg (zero fields use defaults).
func NewHNSWStore(cfg HNSWConfig) *HNSWStore {
	return &HNSWStore{
		cfg:      cfg,
		memories: make(map[string]Memory),
		index:    newHNSWGraph(cfg),
	}
}

func newHNSWGraph(cfg HNSWConfig) *hnsw.Graph[string] {
	g := hnsw.NewGraph[string]()
	if cfg.M > 0 {
		g.M = cfg.M
	}
	if cfg.Ml > 0 {
		g.Ml = cfg.Ml
	}
	if cfg.EfSearch > 0 {
		g.EfSearch = cfg.EfSearch
	}
	return g
}

// Len returns the number of memories currently stored.
func (s *HNSWStore) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.memories)
}

// Upsert validates and stores memories, updating the HNSW vector index for embedded entries.
func (s *HNSWStore) Upsert(ctx context.Context, memories []Memory) error {
	if s == nil {
		return errors.New("braingate: HNSWStore is nil")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	batchDim, err := validateMemories(memories)
	if err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.index.Len() > 0 && batchDim > 0 && batchDim != s.dims {
		return fmt.Errorf("braingate: embedding dimension %d does not match store dimension %d", batchDim, s.dims)
	}

	for _, m := range memories {
		cloned := cloneMemory(m)
		s.memories[cloned.ID] = cloned
		if len(cloned.Embedding) > 0 {
			if s.dims == 0 {
				s.dims = len(cloned.Embedding)
			}
			s.index.Add(hnsw.MakeNode(cloned.ID, toFloat32(cloned.Embedding)))
		} else {
			s.index.Delete(cloned.ID)
			if s.index.Len() == 0 {
				s.dims = 0
			}
		}
	}
	return nil
}

// Delete removes memories by ID from both metadata storage and the HNSW graph.
func (s *HNSWStore) Delete(ctx context.Context, ids []string) error {
	if s == nil {
		return errors.New("braingate: HNSWStore is nil")
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	for _, id := range ids {
		delete(s.memories, id)
		s.index.Delete(id)
	}
	if s.index.Len() == 0 {
		s.dims = 0
	}
	return nil
}

// Search queries memories within query.Radius around query.Probe, optionally filtering
// and ordering by cosine similarity when query.Vector is provided.
func (s *HNSWStore) Search(ctx context.Context, query SearchQuery) ([]ScoredMemory, error) {
	if s == nil {
		return nil, errors.New("braingate: HNSWStore is nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	limit, err := validateSearchQuery(query)
	if err != nil {
		return nil, err
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(s.memories) == 0 {
		return nil, nil
	}

	if len(query.Vector) == 0 {
		results := make([]ScoredMemory, 0, len(s.memories))
		for id, m := range s.memories {
			if query.ExcludeIDs[id] {
				continue
			}
			dist := math.Hypot(query.Probe.Valence-m.Position.Valence, query.Probe.Activation-m.Position.Activation)
			if dist > query.Radius {
				continue
			}
			results = append(results, ScoredMemory{
				Memory:   cloneMemory(m),
				Distance: dist,
			})
		}
		slices.SortFunc(results, func(a, b ScoredMemory) int {
			if a.Distance < b.Distance {
				return -1
			}
			if a.Distance > b.Distance {
				return 1
			}
			return strings.Compare(a.Memory.ID, b.Memory.ID)
		})
		if len(results) > limit {
			results = results[:limit]
		}
		return results, nil
	}

	if s.index.Len() == 0 {
		return nil, nil
	}
	if len(query.Vector) != s.dims {
		return nil, fmt.Errorf("braingate: search vector dimension %d does not match store dimension %d", len(query.Vector), s.dims)
	}

	hits := s.index.SearchFiltered(toFloat32(query.Vector), limit, func(id string) bool {
		if query.ExcludeIDs[id] {
			return false
		}
		m, ok := s.memories[id]
		if !ok || len(m.Embedding) == 0 {
			return false
		}
		dist := math.Hypot(query.Probe.Valence-m.Position.Valence, query.Probe.Activation-m.Position.Activation)
		if dist > query.Radius {
			return false
		}
		return cosineSimilarity(query.Vector, m.Embedding) >= query.MinSimilarity
	})

	results := make([]ScoredMemory, 0, len(hits))
	for _, hit := range hits {
		m, ok := s.memories[hit.Key]
		if !ok {
			continue
		}
		dist := math.Hypot(query.Probe.Valence-m.Position.Valence, query.Probe.Activation-m.Position.Activation)
		sim := cosineSimilarity(query.Vector, m.Embedding)
		results = append(results, ScoredMemory{
			Memory:     cloneMemory(m),
			Distance:   dist,
			Similarity: &sim,
		})
	}
	slices.SortFunc(results, func(a, b ScoredMemory) int {
		if *a.Similarity > *b.Similarity {
			return -1
		}
		if *a.Similarity < *b.Similarity {
			return 1
		}
		if a.Distance < b.Distance {
			return -1
		}
		if a.Distance > b.Distance {
			return 1
		}
		return strings.Compare(a.Memory.ID, b.Memory.ID)
	})
	return results, nil
}

// Export writes the store's memories and HNSW graph snapshot to w.
func (s *HNSWStore) Export(w io.Writer) error {
	if s == nil {
		return errors.New("braingate: HNSWStore is nil")
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	bw := bufio.NewWriter(w)
	if _, err := bw.WriteString(hnswStoreMagic); err != nil {
		return err
	}

	memories := make([]Memory, 0, len(s.memories))
	for _, m := range s.memories {
		memories = append(memories, m)
	}
	slices.SortFunc(memories, func(a, b Memory) int {
		return strings.Compare(a.ID, b.ID)
	})

	encoded, err := json.Marshal(memories)
	if err != nil {
		return fmt.Errorf("braingate: marshal memories: %w", err)
	}

	var lenBuf [binary.MaxVarintLen64]byte
	n := binary.PutVarint(lenBuf[:], int64(s.dims))
	if _, err := bw.Write(lenBuf[:n]); err != nil {
		return err
	}
	n = binary.PutVarint(lenBuf[:], int64(len(encoded)))
	if _, err := bw.Write(lenBuf[:n]); err != nil {
		return err
	}
	if _, err := bw.Write(encoded); err != nil {
		return err
	}
	if err := s.index.Export(bw); err != nil {
		return err
	}
	return bw.Flush()
}

// Import restores the store's memories and HNSW graph snapshot from r.
func (s *HNSWStore) Import(r io.Reader) error {
	if s == nil {
		return errors.New("braingate: HNSWStore is nil")
	}

	br := bufio.NewReader(r)
	magic := make([]byte, len(hnswStoreMagic))
	if _, err := io.ReadFull(br, magic); err != nil {
		return fmt.Errorf("braingate: read store header: %w", err)
	}
	if string(magic) != hnswStoreMagic {
		return fmt.Errorf("braingate: invalid HNSWStore header %q", string(magic))
	}

	dims, err := binary.ReadVarint(br)
	if err != nil {
		return err
	}
	payloadLen, err := binary.ReadVarint(br)
	if err != nil {
		return err
	}
	if dims < 0 || payloadLen < 0 {
		return errors.New("braingate: negative store metadata length")
	}

	encoded := make([]byte, payloadLen)
	if _, err := io.ReadFull(br, encoded); err != nil {
		return err
	}
	var memories []Memory
	if err := json.Unmarshal(encoded, &memories); err != nil {
		return fmt.Errorf("braingate: decode memories: %w", err)
	}
	memDim, err := validateMemories(memories)
	if err != nil {
		return err
	}
	if memDim > 0 && int(dims) != memDim {
		return fmt.Errorf("braingate: store dimension %d does not match memories dimension %d", dims, memDim)
	}

	idx := newHNSWGraph(s.cfg)
	if err := idx.Import(br); err != nil {
		return err
	}

	memMap := make(map[string]Memory, len(memories))
	for _, m := range memories {
		memMap[m.ID] = m
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.memories = memMap
	s.index = idx
	s.dims = int(dims)
	return nil
}

// SaveFile atomically writes a binary snapshot of the store to path.
func (s *HNSWStore) SaveFile(path string) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".hnsw-store-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	cleanup := true
	defer func() {
		if cleanup {
			_ = tmp.Close()
			_ = os.Remove(tmpPath)
		}
	}()

	if err := s.Export(tmp); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return err
	}
	cleanup = false
	return nil
}

// LoadFile loads a binary snapshot previously saved with SaveFile from path.
func (s *HNSWStore) LoadFile(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer func() {
		_ = f.Close()
	}()
	return s.Import(f)
}

func toFloat32(src []float64) []float32 {
	out := make([]float32, len(src))
	for i, v := range src {
		out[i] = float32(v)
	}
	return out
}
