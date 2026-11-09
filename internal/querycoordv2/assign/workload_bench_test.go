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

package assign

import (
	"flag"
	"fmt"
	"strings"
	"testing"

	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/internal/querycoordv2/session"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
)

// workloadBenchEntries are the distribution sizes the read-path
// benchmarks run at. Phase 3 of the shared-store design reuses them.
var workloadBenchEntries = []int{100000, 1000000}

// newWorkloadBenchDist builds a distribution with entries sealed segments
// over 100 nodes and 10 collections, 1000 rows each.
func newWorkloadBenchDist(entries int) *meta.DistributionManager {
	const nodes, collections = 100, 10
	dist := meta.NewDistributionManager(session.NewNodeManager())
	byNode := make(map[int64][]*meta.SegmentFixture, nodes)
	for i := 0; i < entries; i++ {
		node := int64(i%nodes) + 1
		collection := int64(i%collections) + 1
		byNode[node] = append(byNode[node], &meta.SegmentFixture{
			SegmentInfo: &datapb.SegmentInfo{
				ID:            int64(i) + 1,
				CollectionID:  collection,
				PartitionID:   collection * 10,
				InsertChannel: fmt.Sprintf("ch-%d", collection),
				NumOfRows:     1000,
			},
			Node: node,
		})
	}
	for node, segments := range byNode {
		dist.SegmentDistManager.Update(node, meta.SeedInto(dist, segments...)...)
	}
	return dist
}

func scoreWorkloadBench(entries int) func(*testing.B) {
	return func(b *testing.B) {
		policy := newScoreBasedAssignPolicy(session.NewNodeManager(), nil, newWorkloadBenchDist(entries), nil)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			policy.version = -1
			policy.getWorkloadStatus()
		}
	}
}

func rowCountWorkloadBench(entries int) func(*testing.B) {
	return func(b *testing.B) {
		policy := newRowCountBasedAssignPolicy(session.NewNodeManager(), nil, newWorkloadBenchDist(entries))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			policy.version = -1
			policy.getWorkloadStatus()
		}
	}
}

func BenchmarkScoreWorkloadStatus(b *testing.B) {
	for _, entries := range workloadBenchEntries {
		b.Run(fmt.Sprintf("entries=%d", entries), scoreWorkloadBench(entries))
	}
}

func BenchmarkRowCountWorkloadStatus(b *testing.B) {
	for _, entries := range workloadBenchEntries {
		b.Run(fmt.Sprintf("entries=%d", entries), rowCountWorkloadBench(entries))
	}
}

// runBaselineOnlyWhenNamed skips a baseline test unless -run names it, so
// the package's normal test run does not pay for the benchmarks.
func runBaselineOnlyWhenNamed(t *testing.T) {
	if f := flag.Lookup("test.run"); f == nil || !strings.Contains(f.Value.String(), t.Name()) {
		t.Skip("baseline runs only when -run names it")
	}
}

// TestWorkloadStatusBaseline runs the workload benchmarks and logs their
// results, for runners that cannot pass benchmark flags.
func TestWorkloadStatusBaseline(t *testing.T) {
	runBaselineOnlyWhenNamed(t)
	for _, entries := range workloadBenchEntries {
		for name, bench := range map[string]func(int) func(*testing.B){
			"BenchmarkScoreWorkloadStatus":    scoreWorkloadBench,
			"BenchmarkRowCountWorkloadStatus": rowCountWorkloadBench,
		} {
			r := testing.Benchmark(bench(entries))
			t.Logf("Baseline: %s/entries=%d %d ns/op %d B/op %d allocs/op (N=%d)",
				name, entries, r.NsPerOp(), r.AllocedBytesPerOp(), r.AllocsPerOp(), r.N)
		}
	}
}
