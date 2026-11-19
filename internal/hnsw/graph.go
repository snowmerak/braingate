package hnsw

import (
	"cmp"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
)

// Vector is the float32 dense vector representation stored in an HNSW graph.
type Vector = []float32

// DistanceFunc computes a non-negative dissimilarity score between two equal-length vectors.
type DistanceFunc func(a, b Vector) float32

// CosineDistance computes 1 - cosine_similarity(a, b) in float64 precision.
// When either vector has zero norm, it returns 1.
func CosineDistance(a, b Vector) float32 {
	var dot, normA, normB float64
	for i := range a {
		fa := float64(a[i])
		fb := float64(b[i])
		dot += fa * fb
		normA += fa * fa
		normB += fb * fb
	}
	if normA == 0 || normB == 0 {
		return 1
	}
	cos := dot / (math.Sqrt(normA) * math.Sqrt(normB))
	if cos > 1 {
		cos = 1
	} else if cos < -1 {
		cos = -1
	}
	return float32(1 - cos)
}

// EuclideanDistance computes L2 distance between two equal-length vectors.
func EuclideanDistance(a, b Vector) float32 {
	var sum float64
	for i := range a {
		diff := float64(a[i]) - float64(b[i])
		sum += diff * diff
	}
	return float32(math.Sqrt(sum))
}

var distanceFuncs = map[string]DistanceFunc{
	"cosine":    CosineDistance,
	"euclidean": EuclideanDistance,
}

func distanceFuncName(fn DistanceFunc) (string, bool) {
	switch fmt.Sprintf("%p", fn) {
	case fmt.Sprintf("%p", CosineDistance):
		return "cosine", true
	case fmt.Sprintf("%p", EuclideanDistance):
		return "euclidean", true
	default:
		for name, registered := range distanceFuncs {
			if fmt.Sprintf("%p", registered) == fmt.Sprintf("%p", fn) {
				return name, true
			}
		}
		return "", false
	}
}

// Node pairs an ordered key with its embedding vector.
type Node[K cmp.Ordered] struct {
	Key   K
	Value Vector
}

// MakeNode constructs a Node for insertion into a Graph.
func MakeNode[K cmp.Ordered](key K, vec Vector) Node[K] {
	return Node[K]{Key: key, Value: vec}
}

// SearchResult pairs a matched Node with its computed distance from the query vector.
type SearchResult[K cmp.Ordered] struct {
	Node[K]
	Distance float32
}

type layerNode[K cmp.Ordered] struct {
	Node[K]
	neighbors map[K]*layerNode[K]
}

func (n *layerNode[K]) addNeighbor(newNode *layerNode[K], m int, dist DistanceFunc) {
	if n.neighbors == nil {
		n.neighbors = make(map[K]*layerNode[K], m)
	}
	n.neighbors[newNode.Key] = newNode
	if len(n.neighbors) <= m {
		return
	}

	worstDist := float32(math.Inf(-1))
	var worst *layerNode[K]
	for _, neighbor := range n.neighbors {
		d := dist(neighbor.Value, n.Value)
		if d > worstDist || worst == nil {
			worstDist = d
			worst = neighbor
		}
	}

	delete(n.neighbors, worst.Key)
	delete(worst.neighbors, n.Key)
	worst.replenish(m, dist)
}

func (n *layerNode[K]) replenish(m int, dist DistanceFunc) {
	if len(n.neighbors) >= m {
		return
	}
	for _, neighbor := range n.neighbors {
		for key, candidate := range neighbor.neighbors {
			if _, ok := n.neighbors[key]; ok || candidate == n {
				continue
			}
			n.addNeighbor(candidate, m, dist)
			if len(n.neighbors) >= m {
				return
			}
		}
	}
}

func (n *layerNode[K]) isolate(m int, dist DistanceFunc) {
	for _, neighbor := range n.neighbors {
		delete(neighbor.neighbors, n.Key)
		neighbor.replenish(m, dist)
	}
}

type searchCandidate[K cmp.Ordered] struct {
	node *layerNode[K]
	dist float32
}

func (s searchCandidate[K]) Less(o searchCandidate[K]) bool {
	return s.dist < o.dist
}

func (n *layerNode[K]) search(
	k int,
	efSearch int,
	target Vector,
	distance DistanceFunc,
	accept func(K) bool,
) []searchCandidate[K] {
	candidates := Heap[searchCandidate[K]]{}
	candidates.Init(make([]searchCandidate[K], 0, max(efSearch, 1)))

	entryDist := distance(n.Value, target)
	candidates.Push(searchCandidate[K]{node: n, dist: entryDist})

	result := Heap[searchCandidate[K]]{}
	result.Init(make([]searchCandidate[K], 0, k))
	if accept == nil || accept(n.Key) {
		result.Push(searchCandidate[K]{node: n, dist: entryDist})
	}

	visited := map[K]bool{n.Key: true}
	neighborKeys := make([]K, 0, 16)

	for candidates.Len() > 0 {
		current := candidates.Pop()
		if result.Len() >= k && current.dist > result.Max().dist {
			break
		}

		neighborKeys = neighborKeys[:0]
		for key := range current.node.neighbors {
			neighborKeys = append(neighborKeys, key)
		}
		slices.Sort(neighborKeys)

		for _, neighborID := range neighborKeys {
			if visited[neighborID] {
				continue
			}
			visited[neighborID] = true

			neighbor := current.node.neighbors[neighborID]
			dist := distance(neighbor.Value, target)

			if accept == nil || accept(neighborID) {
				if result.Len() < k {
					result.Push(searchCandidate[K]{node: neighbor, dist: dist})
				} else if dist < result.Max().dist {
					result.PopLast()
					result.Push(searchCandidate[K]{node: neighbor, dist: dist})
				}
			}

			if candidates.Len() < efSearch || dist < candidates.Max().dist {
				candidates.Push(searchCandidate[K]{node: neighbor, dist: dist})
				if candidates.Len() > efSearch {
					candidates.PopLast()
				}
			}
		}
	}

	out := slices.Clone(result.Slice())
	slices.SortFunc(out, func(a, b searchCandidate[K]) int {
		if a.dist < b.dist {
			return -1
		}
		if a.dist > b.dist {
			return 1
		}
		return cmp.Compare(a.node.Key, b.node.Key)
	})
	return out
}

type layer[K cmp.Ordered] struct {
	nodes map[K]*layerNode[K]
}

func (l *layer[K]) entry() *layerNode[K] {
	if l == nil || len(l.nodes) == 0 {
		return nil
	}
	var bestKey K
	var bestNode *layerNode[K]
	first := true
	for k, node := range l.nodes {
		if first || k < bestKey {
			bestKey = k
			bestNode = node
			first = false
		}
	}
	return bestNode
}

func (l *layer[K]) size() int {
	if l == nil {
		return 0
	}
	return len(l.nodes)
}

// Graph is a Hierarchical Navigable Small World approximate nearest neighbor index.
type Graph[K cmp.Ordered] struct {
	Distance DistanceFunc
	Rng      *rand.Rand
	M        int
	Ml       float64
	EfSearch int

	layers []*layer[K]
}

// NewGraph constructs an HNSW graph with default cosine distance and M=16, Ml=0.25, EfSearch=64.
func NewGraph[K cmp.Ordered]() *Graph[K] {
	return &Graph[K]{
		M:        16,
		Ml:       0.25,
		Distance: CosineDistance,
		EfSearch: 64,
		Rng:      rand.New(rand.NewPCG(1, 2)),
	}
}

func maxLevel(ml float64, numNodes int) int {
	if ml <= 0 {
		panic("hnsw: Ml must be greater than 0")
	}
	if numNodes == 0 {
		return 1
	}
	l := math.Log(float64(numNodes)) / math.Log(1/ml)
	return int(math.Round(l)) + 1
}

func (g *Graph[K]) randomLevel() int {
	maxLvl := 1
	if len(g.layers) > 0 {
		maxLvl = maxLevel(g.Ml, g.layers[0].size())
	}
	for level := range maxLvl {
		var r float64
		if g.Rng != nil {
			r = g.Rng.Float64()
		} else {
			r = rand.Float64()
		}
		if r > g.Ml {
			return level
		}
	}
	return maxLvl
}

func (g *Graph[K]) assertDims(n Vector) {
	if len(n) == 0 {
		panic("hnsw: vector must not be empty")
	}
	if len(g.layers) == 0 || g.layers[0].size() == 0 {
		return
	}
	hasDims := g.Dims()
	if hasDims != len(n) {
		panic(fmt.Sprintf("hnsw: embedding dimension mismatch: %d != %d", hasDims, len(n)))
	}
}

// Dims returns the dimensionality of stored vectors, or 0 if the graph is empty.
func (g *Graph[K]) Dims() int {
	if len(g.layers) == 0 || g.layers[0].size() == 0 {
		return 0
	}
	return len(g.layers[0].entry().Value)
}

// Len returns the number of nodes in the base layer.
func (g *Graph[K]) Len() int {
	if len(g.layers) == 0 {
		return 0
	}
	return g.layers[0].size()
}

// Add inserts or replaces nodes in the graph.
func (g *Graph[K]) Add(nodes ...Node[K]) {
	if g.Distance == nil {
		g.Distance = CosineDistance
	}
	if g.M <= 0 {
		g.M = 16
	}
	if g.Ml <= 0 {
		g.Ml = 0.25
	}
	if g.EfSearch <= 0 {
		g.EfSearch = 64
	}

	for _, node := range nodes {
		key := node.Key
		vec := slices.Clone(node.Value)
		g.Delete(key)
		g.assertDims(vec)

		insertLevel := g.randomLevel()
		for insertLevel >= len(g.layers) {
			g.layers = append(g.layers, &layer[K]{})
		}

		var elevator *K
		for i := len(g.layers) - 1; i >= 0; i-- {
			l := g.layers[i]
			if l.entry() == nil {
				if insertLevel >= i {
					l.nodes = map[K]*layerNode[K]{
						key: {Node: Node[K]{Key: key, Value: vec}},
					}
				}
				continue
			}

			searchPoint := l.entry()
			if elevator != nil {
				if elevated, ok := l.nodes[*elevator]; ok {
					searchPoint = elevated
				}
			}

			neighborhood := searchPoint.search(g.M, g.EfSearch, vec, g.Distance, nil)
			if len(neighborhood) > 0 {
				nextKey := neighborhood[0].node.Key
				elevator = &nextKey
			}

			if insertLevel >= i {
				newNode := &layerNode[K]{Node: Node[K]{Key: key, Value: vec}}
				l.nodes[key] = newNode
				for _, neighbor := range neighborhood {
					neighbor.node.addNeighbor(newNode, g.M, g.Distance)
					newNode.addNeighbor(neighbor.node, g.M, g.Distance)
				}
			}
		}
	}
}

// Delete removes the node with key from every layer, returning true if it existed.
func (g *Graph[K]) Delete(key K) bool {
	if len(g.layers) == 0 {
		return false
	}
	if g.Distance == nil {
		g.Distance = CosineDistance
	}

	var deleted bool
	for _, l := range g.layers {
		node, ok := l.nodes[key]
		if !ok {
			continue
		}
		delete(l.nodes, key)
		node.isolate(g.M, g.Distance)
		deleted = true
	}
	for len(g.layers) > 0 && g.layers[len(g.layers)-1].size() == 0 {
		g.layers = g.layers[:len(g.layers)-1]
	}
	return deleted
}

// Lookup retrieves the stored vector for key.
func (g *Graph[K]) Lookup(key K) (Vector, bool) {
	if len(g.layers) == 0 {
		return nil, false
	}
	node, ok := g.layers[0].nodes[key]
	if !ok {
		return nil, false
	}
	return slices.Clone(node.Value), true
}

// Search returns up to k nearest neighbors ordered by ascending distance.
func (g *Graph[K]) Search(near Vector, k int) []Node[K] {
	results := g.SearchFiltered(near, k, nil)
	out := make([]Node[K], len(results))
	for i, r := range results {
		out[i] = r.Node
	}
	return out
}

// SearchWithDistance returns up to k nearest neighbors with their distances.
func (g *Graph[K]) SearchWithDistance(near Vector, k int) []SearchResult[K] {
	return g.SearchFiltered(near, k, nil)
}

// SearchFiltered returns up to k nearest neighbors whose keys satisfy accept.
// Upper layers descend unfiltered for fast routing while base-layer exploration
// expands the beam through rejected nodes so filtered regions are still discovered.
func (g *Graph[K]) SearchFiltered(near Vector, k int, accept func(K) bool) []SearchResult[K] {
	if k <= 0 || len(g.layers) == 0 || g.layers[0].size() == 0 {
		return nil
	}
	g.assertDims(near)
	distFn := g.Distance
	if distFn == nil {
		distFn = CosineDistance
	}

	ef := max(g.EfSearch, k*4, 64)
	var elevator *K
	for layerIdx := len(g.layers) - 1; layerIdx >= 0; layerIdx-- {
		l := g.layers[layerIdx]
		searchPoint := l.entry()
		if searchPoint == nil {
			continue
		}
		if elevator != nil {
			if elevated, ok := l.nodes[*elevator]; ok {
				searchPoint = elevated
			}
		}

		if layerIdx > 0 {
			nodes := searchPoint.search(1, ef, near, distFn, nil)
			if len(nodes) > 0 {
				nextKey := nodes[0].node.Key
				elevator = &nextKey
			}
			continue
		}

		nodes := searchPoint.search(k, ef, near, distFn, accept)
		out := make([]SearchResult[K], len(nodes))
		for i, c := range nodes {
			out[i] = SearchResult[K]{
				Node:     Node[K]{Key: c.node.Key, Value: slices.Clone(c.node.Value)},
				Distance: c.dist,
			}
		}
		return out
	}

	return nil
}
