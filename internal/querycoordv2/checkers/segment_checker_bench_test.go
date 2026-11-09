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

package checkers

import (
	"context"
	"flag"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/mock"

	"github.com/milvus-io/milvus/internal/metacache"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// BenchmarkSealedSegmentDiff diffs one replica's distribution against a
// next target holding the same segments plus 1% new ones. Phase 3 of the
// shared-store design reuses it.
func BenchmarkSealedSegmentDiff(b *testing.B) {
	for _, entries := range sealedDiffBenchEntries {
		b.Run(fmt.Sprintf("entries=%d", entries), sealedSegmentDiffBench(entries))
	}
}

var sealedDiffBenchEntries = []int{100000, 1000000}

func sealedSegmentDiffBench(entries int) func(*testing.B) {
	return func(b *testing.B) {
		const collectionID, nodes = int64(1), 100
		dist := make([]*meta.Segment, 0, entries)
		target := make(map[int64]*datapb.SegmentInfo, entries+entries/100)
		ids := typeutil.NewUniqueSet()
		for i := 0; i < entries+entries/100; i++ {
			info := &datapb.SegmentInfo{ID: int64(i) + 1, CollectionID: collectionID, InsertChannel: "ch", NumOfRows: 1000}
			target[info.ID] = info
			ids.Insert(info.ID)
			if i < entries {
				dist = append(dist, &meta.Segment{SegmentInfo: info, Node: int64(i%nodes) + 1, Version: int64(i)})
			}
		}
		targetMgr := meta.NewMockTargetManager(b)
		targetMgr.EXPECT().IsNextTargetExist(mock.Anything, collectionID).Return(true).Maybe()
		targetMgr.EXPECT().IsCurrentTargetExist(mock.Anything, collectionID, mock.Anything).Return(true).Maybe()
		targetMgr.EXPECT().GetSealedSegmentsByCollection(mock.Anything, collectionID, mock.Anything).Return(target).Maybe()
		targetMgr.EXPECT().GetSealedSegmentIDsByCollection(mock.Anything, collectionID, mock.Anything).Return(ids).Maybe()
		checker := &SegmentChecker{
			meta:      &meta.Meta{CollectionManager: meta.NewCollectionManager(nil, metacache.NewMetaStore(nil))},
			targetMgr: targetMgr,
		}
		replica := meta.NewReplica(&querypb.Replica{ID: 1, CollectionID: collectionID})
		ctx := context.Background()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			checker.getSealedSegmentDiff(ctx, collectionID, replica, dist)
		}
	}
}

// TestSealedSegmentDiffBaseline runs BenchmarkSealedSegmentDiff and logs
// its result, for runners that cannot pass benchmark flags. It runs only
// when -run names it.
func TestSealedSegmentDiffBaseline(t *testing.T) {
	if f := flag.Lookup("test.run"); f == nil || !strings.Contains(f.Value.String(), t.Name()) {
		t.Skip("baseline runs only when -run names it")
	}
	for _, entries := range sealedDiffBenchEntries {
		r := testing.Benchmark(sealedSegmentDiffBench(entries))
		t.Logf("Baseline: BenchmarkSealedSegmentDiff/entries=%d %d ns/op %d B/op %d allocs/op (N=%d)",
			entries, r.NsPerOp(), r.AllocedBytesPerOp(), r.AllocsPerOp(), r.N)
	}
}
