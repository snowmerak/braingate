package braingate

import (
	"bytes"
	"context"
	"crypto/sha1"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"sync"
	"time"
)

const (
	// DefaultQdrantBaseURL is the default local Qdrant REST server address.
	DefaultQdrantBaseURL = "http://localhost:6333"
	// DefaultQdrantCollection is the default Qdrant collection name for braingate memories.
	DefaultQdrantCollection = "braingate_memories"

	maxQdrantResponseBytes = 16 << 20
)

// Fixed namespace UUID ("6ba7b814-9dad-11d1-80b4-00c04fd430c8" XOR tag) for deterministic UUIDv5 mapping.
var qdrantNamespaceUUID = [16]byte{
	0x9d, 0x3b, 0x2f, 0x71,
	0x4a, 0x8e, 0x41, 0x9c,
	0x82, 0x17, 0x5c, 0x3a,
	0xe1, 0x09, 0x6b, 0x42,
}

// QdrantConfig configures a QdrantStore REST client.
type QdrantConfig struct {
	BaseURL    string
	Collection string
	APIKey     string
	HTTPClient *http.Client
}

// QdrantStore persists and queries Memory records in a Qdrant collection using
// bounding-box payload prefiltering, circular affect refinement, and cosine ranking.
type QdrantStore struct {
	baseURL    string
	collection string
	apiKey     string
	client     *http.Client

	mu          sync.Mutex
	ensuredSize int
}

// NewQdrantStore validates configuration and constructs a QdrantStore REST client.
func NewQdrantStore(cfg QdrantConfig) (*QdrantStore, error) {
	rawBase := strings.TrimSpace(cfg.BaseURL)
	if rawBase == "" {
		rawBase = DefaultQdrantBaseURL
	}
	parsed, err := url.Parse(rawBase)
	if err != nil {
		return nil, fmt.Errorf("braingate: invalid qdrant base URL: %w", err)
	}
	if (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return nil, fmt.Errorf("braingate: qdrant base URL %q must use http or https with a host", rawBase)
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/")

	collection := strings.TrimSpace(cfg.Collection)
	if collection == "" {
		collection = DefaultQdrantCollection
	}

	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{
			Timeout: 30 * time.Second,
			CheckRedirect: func(*http.Request, []*http.Request) error {
				return http.ErrUseLastResponse
			},
		}
	}

	return &QdrantStore{
		baseURL:    parsed.String(),
		collection: collection,
		apiKey:     strings.TrimSpace(cfg.APIKey),
		client:     httpClient,
	}, nil
}

// EnsureCollection creates the collection with Cosine vectors and float payload indexes
// on valence and activation if it does not already exist.
func (s *QdrantStore) EnsureCollection(ctx context.Context, vectorSize int) error {
	if s == nil {
		return errors.New("braingate: QdrantStore is nil")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if vectorSize <= 0 {
		return fmt.Errorf("braingate: qdrant vector size must be > 0, got %d", vectorSize)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.ensuredSize == vectorSize {
		return nil
	}
	if s.ensuredSize > 0 && s.ensuredSize != vectorSize {
		return fmt.Errorf("braingate: qdrant collection %q already configured for dimension %d, got %d", s.collection, s.ensuredSize, vectorSize)
	}

	colPath := "/collections/" + url.PathEscape(s.collection)
	status, body, err := s.doJSON(ctx, http.MethodGet, colPath, nil)
	if err != nil {
		return err
	}
	if status >= 200 && status < 300 {
		existingDim := parseCollectionVectorSize(body)
		if existingDim > 0 && existingDim != vectorSize {
			return fmt.Errorf("braingate: qdrant collection %q has dimension %d, want %d", s.collection, existingDim, vectorSize)
		}
		s.ensuredSize = vectorSize
		return nil
	}
	if status != http.StatusNotFound {
		return fmt.Errorf("braingate: qdrant inspect collection HTTP %d: %s", status, extractQdrantError(body, http.StatusText(status)))
	}

	createReq := map[string]any{
		"vectors": map[string]any{
			"size":     vectorSize,
			"distance": "Cosine",
		},
	}
	status, body, err = s.doJSON(ctx, http.MethodPut, colPath, createReq)
	if err != nil {
		return err
	}
	if status < 200 || status >= 300 {
		return fmt.Errorf("braingate: qdrant create collection HTTP %d: %s", status, extractQdrantError(body, http.StatusText(status)))
	}

	for _, field := range []string{"valence", "activation"} {
		idxReq := map[string]any{
			"field_name":   field,
			"field_schema": "float",
		}
		status, body, err = s.doJSON(ctx, http.MethodPut, colPath+"/index?wait=true", idxReq)
		if err != nil {
			return err
		}
		if status < 200 || status >= 300 {
			return fmt.Errorf("braingate: qdrant create %s index HTTP %d: %s", field, status, extractQdrantError(body, http.StatusText(status)))
		}
	}

	s.ensuredSize = vectorSize
	return nil
}

// Upsert validates memories, ensures the collection exists, and writes points into Qdrant.
func (s *QdrantStore) Upsert(ctx context.Context, memories []Memory) error {
	if s == nil {
		return errors.New("braingate: QdrantStore is nil")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(memories) == 0 {
		return nil
	}

	dim, _, err := validateMemories(memories)
	if err != nil {
		return err
	}
	if dim == 0 {
		return errors.New("braingate: qdrant upsert requires embeddings on every memory")
	}
	for _, m := range memories {
		if len(m.Embedding) == 0 {
			return fmt.Errorf("braingate: memory %q requires an embedding for qdrant upsert", m.ID)
		}
	}

	if err := s.EnsureCollection(ctx, dim); err != nil {
		return err
	}

	type qdrantPoint struct {
		ID      string        `json:"id"`
		Vector  []float64     `json:"vector"`
		Payload qdrantPayload `json:"payload"`
	}
	points := make([]qdrantPoint, 0, len(memories))
	for _, m := range memories {
		var createdAt string
		if m.CreatedAt != nil {
			createdAt = m.CreatedAt.UTC().Format(time.RFC3339Nano)
		}
		points = append(points, qdrantPoint{
			ID:     qdrantPointUUID(m.ID),
			Vector: m.Embedding,
			Payload: qdrantPayload{
				MemoryID:           m.ID,
				Cue:                m.Cue,
				Valence:            m.Position.Valence,
				Activation:         m.Position.Activation,
				Experience:         m.Experience,
				Alpha:              m.Alpha,
				Salience:           m.Salience,
				CreatedAt:          createdAt,
				SecondaryEmbedding: slices.Clone(m.SecondaryEmbedding),
			},
		})
	}

	path := "/collections/" + url.PathEscape(s.collection) + "/points?wait=true"
	status, body, err := s.doJSON(ctx, http.MethodPut, path, map[string]any{"points": points})
	if err != nil {
		return err
	}
	if status < 200 || status >= 300 {
		return fmt.Errorf("braingate: qdrant upsert HTTP %d: %s", status, extractQdrantError(body, http.StatusText(status)))
	}
	return nil
}

// Delete removes memories from Qdrant by their Memory.ID values.
func (s *QdrantStore) Delete(ctx context.Context, ids []string) error {
	if s == nil {
		return errors.New("braingate: QdrantStore is nil")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(ids) == 0 {
		return nil
	}

	pointIDs := make([]string, len(ids))
	for i, id := range ids {
		pointIDs[i] = qdrantPointUUID(id)
	}
	path := "/collections/" + url.PathEscape(s.collection) + "/points/delete?wait=true"
	status, body, err := s.doJSON(ctx, http.MethodPost, path, map[string]any{"points": pointIDs})
	if err != nil {
		return err
	}
	if status < 200 || status >= 300 {
		return fmt.Errorf("braingate: qdrant delete HTTP %d: %s", status, extractQdrantError(body, http.StatusText(status)))
	}
	return nil
}

// Search queries Qdrant using a bounding-box payload filter around query.Probe and
// refines results on the client to the exact circular affect radius query.Radius.
func (s *QdrantStore) Search(ctx context.Context, query SearchQuery) ([]ScoredMemory, error) {
	if s == nil {
		return nil, errors.New("braingate: QdrantStore is nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	limit, err := validateSearchQuery(query)
	if err != nil {
		return nil, err
	}

	filter := buildQdrantFilter(query)
	colPath := "/collections/" + url.PathEscape(s.collection)

	if len(query.Vector) > 0 {
		overfetch := max(limit*3, 32)
		reqBody := map[string]any{
			"query":           query.Vector,
			"filter":          filter,
			"score_threshold": query.MinSimilarity,
			"limit":           overfetch,
			"with_payload":    true,
			"with_vector":     true,
		}
		status, body, err := s.doJSON(ctx, http.MethodPost, colPath+"/points/query", reqBody)
		if err != nil {
			return nil, err
		}
		if status < 200 || status >= 300 {
			return nil, fmt.Errorf("braingate: qdrant query HTTP %d: %s", status, extractQdrantError(body, http.StatusText(status)))
		}

		records, err := decodeQdrantPoints(body)
		if err != nil {
			return nil, err
		}
		results := make([]ScoredMemory, 0, len(records))
		for _, rec := range records {
			m, err := rec.Payload.toMemory()
			if err != nil {
				return nil, err
			}
			if m.ID == "" || query.ExcludeIDs[m.ID] {
				continue
			}
			dist := math.Hypot(query.Probe.Valence-m.Position.Valence, query.Probe.Activation-m.Position.Activation)
			if dist > query.Radius {
				continue
			}
			if rec.Score < query.MinSimilarity {
				continue
			}
			if vec := parseQdrantVector(rec.Vector); len(vec) > 0 {
				m.Embedding = vec
			}
			var secSimPtr *float64
			if len(query.SecondaryVector) > 0 && len(m.SecondaryEmbedding) > 0 {
				if len(m.SecondaryEmbedding) != len(query.SecondaryVector) {
					return nil, fmt.Errorf("braingate: memory %q secondary embedding dimension %d does not match query secondary dimension %d", m.ID, len(m.SecondaryEmbedding), len(query.SecondaryVector))
				}
				secSim := cosineSimilarity(query.SecondaryVector, m.SecondaryEmbedding)
				secSimPtr = &secSim
			}
			sim := clamp(rec.Score, -1, 1)
			results = append(results, ScoredMemory{
				Memory:              m,
				Distance:            dist,
				Similarity:          &sim,
				SecondarySimilarity: secSimPtr,
			})
		}
		if len(results) > limit {
			results = results[:limit]
		}
		return results, nil
	}

	scrollLimit := max(limit*4, 64)
	reqBody := map[string]any{
		"filter":       filter,
		"limit":        scrollLimit,
		"with_payload": true,
	}
	status, body, err := s.doJSON(ctx, http.MethodPost, colPath+"/points/scroll", reqBody)
	if err != nil {
		return nil, err
	}
	if status < 200 || status >= 300 {
		return nil, fmt.Errorf("braingate: qdrant scroll HTTP %d: %s", status, extractQdrantError(body, http.StatusText(status)))
	}

	records, err := decodeQdrantPoints(body)
	if err != nil {
		return nil, err
	}
	results := make([]ScoredMemory, 0, len(records))
	for _, rec := range records {
		m, err := rec.Payload.toMemory()
		if err != nil {
			return nil, err
		}
		if m.ID == "" || query.ExcludeIDs[m.ID] {
			continue
		}
		dist := math.Hypot(query.Probe.Valence-m.Position.Valence, query.Probe.Activation-m.Position.Activation)
		if dist > query.Radius {
			continue
		}
		results = append(results, ScoredMemory{
			Memory:   m,
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

type qdrantPayload struct {
	MemoryID           string    `json:"memory_id"`
	Cue                string    `json:"cue,omitempty"`
	Valence            float64   `json:"valence"`
	Activation         float64   `json:"activation"`
	Experience         string    `json:"experience"`
	Alpha              float64   `json:"alpha,omitempty"`
	Salience           float64   `json:"salience,omitempty"`
	CreatedAt          string    `json:"created_at,omitempty"`
	SecondaryEmbedding []float64 `json:"secondary_embedding,omitempty"`
}

func (p qdrantPayload) toMemory() (Memory, error) {
	var createdAt *time.Time
	if strings.TrimSpace(p.CreatedAt) != "" {
		parsed, err := time.Parse(time.RFC3339Nano, p.CreatedAt)
		if err != nil {
			return Memory{}, fmt.Errorf("braingate: memory %q invalid qdrant created_at %q: %w", p.MemoryID, p.CreatedAt, err)
		}
		createdAt = &parsed
	}
	return Memory{
		ID:                 p.MemoryID,
		Cue:                p.Cue,
		Position:           Point{Valence: p.Valence, Activation: p.Activation},
		Experience:         p.Experience,
		Alpha:              p.Alpha,
		Salience:           p.Salience,
		CreatedAt:          createdAt,
		SecondaryEmbedding: slices.Clone(p.SecondaryEmbedding),
	}, nil
}

type qdrantPointRecord struct {
	Score   float64         `json:"score"`
	Vector  json.RawMessage `json:"vector,omitempty"`
	Payload qdrantPayload   `json:"payload"`
}

func parseQdrantVector(raw json.RawMessage) []float64 {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '[' {
		return nil
	}
	var vec []float64
	if err := json.Unmarshal(trimmed, &vec); err != nil {
		return nil
	}
	for _, v := range vec {
		if !finite(v) {
			return nil
		}
	}
	return vec
}

func decodeQdrantPoints(body []byte) ([]qdrantPointRecord, error) {
	var wrapper struct {
		Result json.RawMessage `json:"result"`
	}
	if err := json.Unmarshal(body, &wrapper); err != nil {
		return nil, fmt.Errorf("braingate: decode qdrant response: %w", err)
	}
	rawResult := bytes.TrimSpace(wrapper.Result)
	if len(rawResult) == 0 || bytes.Equal(rawResult, []byte("null")) {
		return nil, nil
	}
	if rawResult[0] == '[' {
		var points []qdrantPointRecord
		if err := json.Unmarshal(rawResult, &points); err != nil {
			return nil, fmt.Errorf("braingate: decode qdrant result points: %w", err)
		}
		return points, nil
	}
	var obj struct {
		Points []qdrantPointRecord `json:"points"`
	}
	if err := json.Unmarshal(rawResult, &obj); err != nil {
		return nil, fmt.Errorf("braingate: decode qdrant result object: %w", err)
	}
	return obj.Points, nil
}

func buildQdrantFilter(query SearchQuery) map[string]any {
	filter := map[string]any{
		"must": []any{
			map[string]any{
				"key": "valence",
				"range": map[string]float64{
					"gte": max(-1.0, query.Probe.Valence-query.Radius),
					"lte": min(1.0, query.Probe.Valence+query.Radius),
				},
			},
			map[string]any{
				"key": "activation",
				"range": map[string]float64{
					"gte": max(-1.0, query.Probe.Activation-query.Radius),
					"lte": min(1.0, query.Probe.Activation+query.Radius),
				},
			},
		},
	}
	if len(query.ExcludeIDs) > 0 {
		ids := make([]string, 0, len(query.ExcludeIDs))
		for id := range query.ExcludeIDs {
			ids = append(ids, id)
		}
		slices.Sort(ids)
		uuids := make([]string, len(ids))
		for i, id := range ids {
			uuids[i] = qdrantPointUUID(id)
		}
		filter["must_not"] = []any{
			map[string]any{
				"has_id": uuids,
			},
		}
	}
	return filter
}

func (s *QdrantStore) doJSON(ctx context.Context, method, path string, reqBody any) (int, []byte, error) {
	var reader io.Reader
	if reqBody != nil {
		encoded, err := json.Marshal(reqBody)
		if err != nil {
			return 0, nil, fmt.Errorf("braingate: marshal qdrant request: %w", err)
		}
		reader = bytes.NewReader(encoded)
	}

	req, err := http.NewRequestWithContext(ctx, method, s.baseURL+path, reader)
	if err != nil {
		return 0, nil, fmt.Errorf("braingate: build qdrant request: %w", err)
	}
	if reqBody != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if s.apiKey != "" {
		req.Header.Set("api-key", s.apiKey)
	}

	resp, err := s.client.Do(req)
	if err != nil {
		return 0, nil, fmt.Errorf("braingate: qdrant request failed: %w", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxQdrantResponseBytes+1))
	if err != nil {
		return 0, nil, fmt.Errorf("braingate: read qdrant response: %w", err)
	}
	if len(body) > maxQdrantResponseBytes {
		return 0, nil, fmt.Errorf("braingate: qdrant response exceeded %d bytes", maxQdrantResponseBytes)
	}
	return resp.StatusCode, body, nil
}

func parseCollectionVectorSize(body []byte) int {
	var decoded struct {
		Result struct {
			Config struct {
				Params struct {
					Vectors struct {
						Size int `json:"size"`
					} `json:"vectors"`
				} `json:"params"`
			} `json:"config"`
		} `json:"result"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		return 0
	}
	return decoded.Result.Config.Params.Vectors.Size
}

func extractQdrantError(body []byte, fallback string) string {
	var qErr struct {
		Status struct {
			Error string `json:"error"`
		} `json:"status"`
	}
	if err := json.Unmarshal(body, &qErr); err == nil && strings.TrimSpace(qErr.Status.Error) != "" {
		return strings.TrimSpace(qErr.Status.Error)
	}
	return extractEmbedError(body, fallback)
}

func qdrantPointUUID(memoryID string) string {
	h := sha1.New()
	_, _ = h.Write(qdrantNamespaceUUID[:])
	_, _ = h.Write([]byte(memoryID))
	sum := h.Sum(nil)
	sum[6] = (sum[6] & 0x0f) | 0x50
	sum[8] = (sum[8] & 0x3f) | 0x80
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x", sum[0:4], sum[4:6], sum[6:8], sum[8:10], sum[10:16])
}
