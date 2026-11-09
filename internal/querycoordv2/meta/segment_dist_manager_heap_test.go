// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package meta

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
)

const (
	heapFixtureSegments          = 10000
	heapFixtureIndexesPerSegment = 2
)

// newHeapFixtureSegments builds n distribution entries the way the dist
// handler does from a QueryNode report: a projected proto, and per index
// only the loaded build the handler keeps. The baseline before trimming,
// with the full FieldIndexInfo a node echoes back (8 params and 4 file paths
// of 120 bytes per index), measured 4613 heap bytes per segment.
func newHeapFixtureSegments(n, indexesPerSegment int) []*Segment {
	segments := make([]*Segment, 0, n)
	for i := 0; i < n; i++ {
		id := int64(i + 1)
		builds := make(map[int64]int64, indexesPerSegment)
		for j := 0; j < indexesPerSegment; j++ {
			builds[int64(1000+j)] = id*10 + int64(j)
		}
		segments = append(segments, &Segment{
			SegmentInfo: &datapb.SegmentInfo{
				ID:            id,
				CollectionID:  1,
				PartitionID:   2,
				InsertChannel: fmt.Sprintf("channel-%d", i%16),
				NumOfRows:     1000,
				Level:         datapb.SegmentLevel_L1,
				IsSorted:      true,
			},
			Version:          1,
			IndexBuilds:      builds,
			JSONStatsFormats: map[int64]int64{200: 2},
		})
	}
	return segments
}

// TestSegmentDistManagerHeapFootprint logs the live heap the distribution
// retains per segment. It is the before/after measurement of trimming the
// distribution record; run it with -run TestSegmentDistManagerHeapFootprint
// and read the heapB/segment line.
func TestSegmentDistManagerHeapFootprint(t *testing.T) {
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)

	manager := NewSegmentDistManager()
	manager.Update(1, newHeapFixtureSegments(heapFixtureSegments, heapFixtureIndexesPerSegment)...)

	runtime.GC()
	runtime.ReadMemStats(&after)
	perSegment := (int64(after.HeapAlloc) - int64(before.HeapAlloc)) / heapFixtureSegments
	runtime.KeepAlive(manager)

	t.Logf("heapB/segment=%d segments=%d indexesPerSegment=%d", perSegment, heapFixtureSegments, heapFixtureIndexesPerSegment)
}
