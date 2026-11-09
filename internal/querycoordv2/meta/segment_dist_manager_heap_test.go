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

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
)

const (
	heapFixtureSegments          = 10000
	heapFixtureIndexesPerSegment = 2
	heapFixtureParams            = 8
	heapFixtureFilePaths         = 4
	heapFixtureFilePathLen       = 120
)

// newHeapFixtureSegments builds n distribution entries the way the dist
// handler does from a QueryNode report: a projected proto and, per index,
// the FieldIndexInfo the node echoes back. Strings are allocated per
// segment, as decoding each report does.
func newHeapFixtureSegments(n, indexesPerSegment int) []*Segment {
	segments := make([]*Segment, 0, n)
	for i := 0; i < n; i++ {
		id := int64(i + 1)
		indexInfo := make(map[int64]*querypb.FieldIndexInfo, indexesPerSegment)
		for j := 0; j < indexesPerSegment; j++ {
			indexID := int64(1000 + j)
			params := make([]*commonpb.KeyValuePair, 0, heapFixtureParams)
			for p := 0; p < heapFixtureParams; p++ {
				params = append(params, &commonpb.KeyValuePair{
					Key:   fmt.Sprintf("key%05d", p),
					Value: fmt.Sprintf("value-%010d", i),
				})
			}
			paths := make([]string, 0, heapFixtureFilePaths)
			for f := 0; f < heapFixtureFilePaths; f++ {
				path := fmt.Sprintf("files/index_files/%d/%d/%d/%d/", id, indexID, f, i)
				for len(path) < heapFixtureFilePathLen {
					path += "x"
				}
				paths = append(paths, path)
			}
			indexInfo[indexID] = &querypb.FieldIndexInfo{
				FieldID:        int64(100 + j),
				EnableIndex:    true,
				IndexName:      fmt.Sprintf("index_%d", j),
				IndexID:        indexID,
				BuildID:        id*10 + int64(j),
				IndexParams:    params,
				IndexFilePaths: paths,
				IndexSize:      1 << 20,
				NumRows:        1000,
			}
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
			Version:   1,
			IndexInfo: indexInfo,
			JSONStatsField: map[int64]*querypb.JsonStatsInfo{
				200: {FieldID: 200, DataFormatVersion: 2},
			},
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
