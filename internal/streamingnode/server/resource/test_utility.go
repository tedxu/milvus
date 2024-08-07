//go:build test
// +build test

package resource

import "github.com/milvus-io/milvus/internal/streamingnode/server/resource/idalloc"

// InitForTest initializes the singleton of resources for test.
func InitForTest(opts ...optResourceInit) {
	r = &resourceImpl{}
	for _, opt := range opts {
		opt(r)
	}
	if r.rootCoordClient != nil {
		r.timestampAllocator = idalloc.NewTSOAllocator(r.rootCoordClient)
	}
}
