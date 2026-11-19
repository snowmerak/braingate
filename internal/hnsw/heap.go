// Package hnsw provides an in-memory Hierarchical Navigable Small World (HNSW)
// vector index adapted from github.com/coder/hnsw (CC0 1.0 Universal), with
// cross-platform atomic persistence and base-layer predicate filtering.
package hnsw

import "container/heap"

// Lessable orders elements inside Heap.
type Lessable[T any] interface {
	Less(T) bool
}

type innerHeap[T Lessable[T]] struct {
	data []T
}

func (h *innerHeap[T]) Len() int           { return len(h.data) }
func (h *innerHeap[T]) Less(i, j int) bool { return h.data[i].Less(h.data[j]) }
func (h *innerHeap[T]) Swap(i, j int)      { h.data[i], h.data[j] = h.data[j], h.data[i] }

func (h *innerHeap[T]) Push(x any) {
	h.data = append(h.data, x.(T))
}

func (h *innerHeap[T]) Pop() any {
	n := len(h.data)
	x := h.data[n-1]
	h.data = h.data[:n-1]
	return x
}

// Heap is a min-heap backed by container/heap that also supports bounded max removal.
type Heap[T Lessable[T]] struct {
	inner innerHeap[T]
}

// Init establishes heap invariants over the supplied slice.
func (h *Heap[T]) Init(d []T) {
	h.inner.data = d
	heap.Init(&h.inner)
}

// Len returns the number of elements in the heap.
func (h *Heap[T]) Len() int {
	return h.inner.Len()
}

// Push inserts x into the heap.
func (h *Heap[T]) Push(x T) {
	heap.Push(&h.inner, x)
}

// Pop removes and returns the minimum element.
func (h *Heap[T]) Pop() T {
	return heap.Pop(&h.inner).(T)
}

// PopLast removes and returns the maximum element.
func (h *Heap[T]) PopLast() T {
	return heap.Remove(&h.inner, h.maxIndex()).(T)
}

// Min returns the minimum element without removing it.
func (h *Heap[T]) Min() T {
	return h.inner.data[0]
}

// Max returns the maximum element without removing it.
func (h *Heap[T]) Max() T {
	return h.inner.data[h.maxIndex()]
}

// Slice returns the underlying slice backing the heap.
func (h *Heap[T]) Slice() []T {
	return h.inner.data
}

func (h *Heap[T]) maxIndex() int {
	n := h.inner.Len()
	best := n / 2
	for i := best + 1; i < n; i++ {
		if h.inner.data[best].Less(h.inner.data[i]) {
			best = i
		}
	}
	return best
}
