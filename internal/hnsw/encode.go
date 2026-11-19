package hnsw

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

const encodingVersion = 1

var byteOrder = binary.LittleEndian

// Export serializes the graph configuration, layers, vectors, and adjacency edges to w.
func (g *Graph[K]) Export(w io.Writer) error {
	distName, ok := distanceFuncName(g.Distance)
	if !ok {
		return errors.New("hnsw: distance function is not registered for export")
	}

	bw := bufio.NewWriter(w)
	if err := writeValues(bw, encodingVersion, g.M, g.Ml, g.EfSearch, distName, len(g.layers)); err != nil {
		return err
	}

	for _, l := range g.layers {
		if err := writeValue(bw, len(l.nodes)); err != nil {
			return err
		}
		for key, node := range l.nodes {
			if err := writeValues(bw, key, node.Value, len(node.neighbors)); err != nil {
				return err
			}
			for neighborKey := range node.neighbors {
				if err := writeValue(bw, neighborKey); err != nil {
					return err
				}
			}
		}
	}
	return bw.Flush()
}

// Import restores a Graph previously serialized with Export from r.
func (g *Graph[K]) Import(r io.Reader) error {
	br := bufio.NewReader(r)

	var version int
	if err := readValue(br, &version); err != nil {
		return err
	}
	if version != encodingVersion {
		return fmt.Errorf("hnsw: incompatible encoding version %d", version)
	}

	var distName string
	var nLayers int
	if err := readValues(br, &g.M, &g.Ml, &g.EfSearch, &distName, &nLayers); err != nil {
		return err
	}
	distFn, ok := distanceFuncs[distName]
	if !ok {
		return fmt.Errorf("hnsw: unknown distance function %q", distName)
	}
	g.Distance = distFn
	if nLayers < 0 {
		return fmt.Errorf("hnsw: invalid layer count %d", nLayers)
	}

	g.layers = make([]*layer[K], nLayers)
	for i := range nLayers {
		var nNodes int
		if err := readValue(br, &nNodes); err != nil {
			return err
		}
		if nNodes < 0 {
			return fmt.Errorf("hnsw: invalid node count %d", nNodes)
		}

		nodes := make(map[K]*layerNode[K], nNodes)
		pendingNeighbors := make(map[K][]K, nNodes)
		for range nNodes {
			var key K
			var vec Vector
			var nNeighbors int
			if err := readValues(br, &key, &vec, &nNeighbors); err != nil {
				return err
			}
			if nNeighbors < 0 {
				return fmt.Errorf("hnsw: invalid neighbor count %d", nNeighbors)
			}
			neighbors := make([]K, nNeighbors)
			for k := range nNeighbors {
				if err := readValue(br, &neighbors[k]); err != nil {
					return err
				}
			}
			nodes[key] = &layerNode[K]{
				Node:      Node[K]{Key: key, Value: vec},
				neighbors: make(map[K]*layerNode[K], nNeighbors),
			}
			pendingNeighbors[key] = neighbors
		}

		for key, neighbors := range pendingNeighbors {
			node := nodes[key]
			for _, neighborKey := range neighbors {
				neighbor, exists := nodes[neighborKey]
				if !exists {
					return fmt.Errorf("hnsw: missing neighbor key %v", neighborKey)
				}
				node.neighbors[neighborKey] = neighbor
			}
		}
		g.layers[i] = &layer[K]{nodes: nodes}
	}

	return nil
}

// SaveFile atomically writes the exported graph to path using a temp file in the same directory.
func (g *Graph[K]) SaveFile(path string) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".hnsw-*")
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

	if err := g.Export(tmp); err != nil {
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

// LoadFile loads a graph previously saved with SaveFile from path.
func (g *Graph[K]) LoadFile(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer func() {
		_ = f.Close()
	}()
	return g.Import(f)
}

func writeValues(w io.Writer, values ...any) error {
	for _, v := range values {
		if err := writeValue(w, v); err != nil {
			return err
		}
	}
	return nil
}

func writeValue(w io.Writer, v any) error {
	switch val := v.(type) {
	case int:
		var buf [binary.MaxVarintLen64]byte
		n := binary.PutVarint(buf[:], int64(val))
		_, err := w.Write(buf[:n])
		return err
	case string:
		if err := writeValue(w, len(val)); err != nil {
			return err
		}
		_, err := io.WriteString(w, val)
		return err
	case Vector:
		if err := writeValue(w, len(val)); err != nil {
			return err
		}
		return binary.Write(w, byteOrder, val)
	default:
		return binary.Write(w, byteOrder, v)
	}
}

func readValues(r *bufio.Reader, targets ...any) error {
	for _, target := range targets {
		if err := readValue(r, target); err != nil {
			return err
		}
	}
	return nil
}

func readValue(r *bufio.Reader, target any) error {
	switch dst := target.(type) {
	case *int:
		v, err := binary.ReadVarint(r)
		if err != nil {
			return err
		}
		*dst = int(v)
		return nil
	case *string:
		var ln int
		if err := readValue(r, &ln); err != nil {
			return err
		}
		if ln < 0 {
			return fmt.Errorf("hnsw: negative string length %d", ln)
		}
		buf := make([]byte, ln)
		if _, err := io.ReadFull(r, buf); err != nil {
			return err
		}
		*dst = string(buf)
		return nil
	case *Vector:
		var ln int
		if err := readValue(r, &ln); err != nil {
			return err
		}
		if ln < 0 {
			return fmt.Errorf("hnsw: negative vector length %d", ln)
		}
		vec := make(Vector, ln)
		if err := binary.Read(r, byteOrder, vec); err != nil {
			return err
		}
		*dst = vec
		return nil
	default:
		return binary.Read(r, byteOrder, target)
	}
}
