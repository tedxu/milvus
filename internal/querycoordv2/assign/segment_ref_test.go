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
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/internal/metacache"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/internal/querycoordv2/session"
	"github.com/milvus-io/milvus/internal/querycoordv2/task"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
)

func newTieTestNodes(t *testing.T, n int64) (*session.NodeManager, *task.MockScheduler) {
	nodeManager := session.NewNodeManager()
	scheduler := task.NewMockScheduler(t)
	scheduler.EXPECT().GetSegmentTaskDeltaSnapshot(mock.Anything, mock.Anything).Return(task.NewSegmentTaskDeltaSnapshot(nil, nil)).Maybe()
	scheduler.EXPECT().GetChannelTaskDelta(mock.Anything, mock.Anything).Return(0).Maybe()
	for i := int64(1); i <= n; i++ {
		nodeManager.Add(session.NewNodeInfo(session.ImmutableNodeInfo{NodeID: i, Version: common.Version, Address: "localhost", Hostname: "node"}))
		nodeManager.Get(i).SetState(session.NodeStateNormal)
	}
	return nodeManager, scheduler
}

// refsOf builds the refs that fixtures resolve to.
func refsOf(fixtures []*meta.SegmentFixture) []SegmentRef {
	refs := make([]SegmentRef, 0, len(fixtures))
	for _, f := range fixtures {
		refs = append(refs, SegmentRefFromInfo(f.SegmentInfo, f.Node))
	}
	return refs
}

func TestSegmentRefsFromDist(t *testing.T) {
	store := metacache.NewMetaStore(nil)
	records := meta.SeedSegments(store,
		&meta.SegmentFixture{SegmentInfo: &datapb.SegmentInfo{ID: 1, CollectionID: 10, PartitionID: 100, NumOfRows: 7, InsertChannel: "ch1", Level: datapb.SegmentLevel_L1}, Node: 1},
		&meta.SegmentFixture{SegmentInfo: &datapb.SegmentInfo{ID: 2, CollectionID: 10, PartitionID: 101, NumOfRows: 9, InsertChannel: "ch2", Level: datapb.SegmentLevel_L2}, Node: 2},
	)
	// A record the store no longer has is dropped.
	records = append(records, &meta.LoadedSegment{ID: 3, CollectionID: 10, Channel: "ch1", Node: 3})
	refs := SegmentRefsFromDist(store, records)
	assert.Equal(t, []SegmentRef{
		{ID: 1, CollectionID: 10, PartitionID: 100, NumOfRows: 7, Channel: "ch1", Level: datapb.SegmentLevel_L1, From: 1},
		{ID: 2, CollectionID: 10, PartitionID: 101, NumOfRows: 9, Channel: "ch2", Level: datapb.SegmentLevel_L2, From: 2},
	}, refs)
}

// TestScoreAssignMovingSegmentReducesSource pins that a ref's From selects
// the source node, whose score each planned move then reduces.
func TestScoreAssignMovingSegmentReducesSource(t *testing.T) {
	nodeManager, scheduler := newTieTestNodes(t, 2)
	dist := meta.NewDistributionManager(nodeManager)
	onNode1 := []*meta.SegmentFixture{
		{SegmentInfo: &datapb.SegmentInfo{ID: 1, CollectionID: 100, NumOfRows: 1000}, Node: 1},
		{SegmentInfo: &datapb.SegmentInfo{ID: 2, CollectionID: 100, NumOfRows: 1000}, Node: 1},
	}
	dist.SegmentDistManager.Update(1, meta.SeedInto(dist, onNode1...)...)
	policy := newScoreBasedAssignPolicy(nodeManager, scheduler, dist, meta.NewMeta(nil, nil, nodeManager, metacache.NewMetaStore(nil)))
	plans := policy.AssignSegment(context.Background(), 100, refsOf(onNode1), []int64{1, 2}, true)
	require.Len(t, plans, 2)
	for _, plan := range plans {
		assert.Equal(t, int64(1), plan.From)
	}
	assert.InDelta(t, plans[0].FromScore-plans[0].SegmentScore, plans[1].FromScore, 1)
}

// TestBalancePlansStableForTies pins the plans for segments that all tie on
// rows, on nodes that all have zero rows: the plan count (capped by the
// balance batch size) and the segment order, both captured from the code
// before segment refs. Which node each segment goes to is not pinned: node
// items come from a map, so tied nodes enter the queue in random order.
func TestBalancePlansStableForTies(t *testing.T) {
	nodeManager, scheduler := newTieTestNodes(t, 3)
	policy := newRowCountBasedAssignPolicy(nodeManager, scheduler, meta.NewDistributionManager(nodeManager))
	segments := make([]SegmentRef, 0, 7)
	for i := int64(1); i <= 7; i++ {
		segments = append(segments, SegmentRef{ID: i, CollectionID: 100})
	}
	plans := policy.AssignSegment(context.Background(), 100, segments, []int64{1, 2, 3}, false)
	got := make([]int64, 0, len(plans))
	for _, plan := range plans {
		got = append(got, plan.Segment.ID)
		assert.Contains(t, []int64{1, 2, 3}, plan.To)
	}
	assert.Equal(t, []int64{1, 2, 3, 4, 5}, got)
}

// TestWorkloadStatusStoreMiss pins that an entry the store no longer has
// counts 0 rows on its node.
func TestWorkloadStatusStoreMiss(t *testing.T) {
	nodeManager, scheduler := newTieTestNodes(t, 1)
	dist := meta.NewDistributionManager(nodeManager)
	dist.SegmentDistManager.Update(1, append(
		meta.SeedInto(dist, &meta.SegmentFixture{SegmentInfo: &datapb.SegmentInfo{ID: 1, CollectionID: 100, NumOfRows: 100}, Node: 1}),
		&meta.LoadedSegment{ID: 2, CollectionID: 100, Channel: "ch", Node: 1})...)

	rowCount := newRowCountBasedAssignPolicy(nodeManager, scheduler, dist).getWorkloadStatus()
	assert.Equal(t, 100, rowCount.nodeGlobalRowCount[1])
	score := newScoreBasedAssignPolicy(nodeManager, scheduler, dist, nil).getWorkloadStatus()
	assert.Equal(t, 100, score.nodeGlobalRowCount[1])
	assert.Equal(t, 100, score.nodeCollectionRowCount[1][100])
}

// TestDroppedSegmentStillResolved pins that a segment compacted away while
// still loaded, kept in the store as Dropped until GC, resolves with its
// real fields and is not treated as a miss.
func TestDroppedSegmentStillResolved(t *testing.T) {
	store := metacache.NewMetaStore(nil)
	records := meta.SeedSegments(store, &meta.SegmentFixture{
		SegmentInfo: &datapb.SegmentInfo{ID: 1, CollectionID: 10, PartitionID: 100, NumOfRows: 7, InsertChannel: "ch", State: commonpb.SegmentState_Dropped},
		Node:        1,
	})
	refs := SegmentRefsFromDist(store, records)
	require.Len(t, refs, 1)
	assert.Equal(t, int64(7), refs[0].NumOfRows)
	assert.Equal(t, int64(100), refs[0].PartitionID)
}

// TestWorkloadStatusFromAggregates pins the workload status on a known
// distribution, with the totals a full scan gives: per node and per
// (node, collection), a store miss counting 0.
func TestWorkloadStatusFromAggregates(t *testing.T) {
	nodeManager, scheduler := newTieTestNodes(t, 2)
	dist := meta.NewDistributionManager(nodeManager)
	fixture := func(id, collection, rows, node int64) *meta.SegmentFixture {
		return &meta.SegmentFixture{SegmentInfo: &datapb.SegmentInfo{ID: id, CollectionID: collection, NumOfRows: rows}, Node: node}
	}
	dist.SegmentDistManager.Update(1, append(meta.SeedInto(dist, fixture(1, 100, 10, 1), fixture(2, 200, 20, 1)),
		&meta.LoadedSegment{ID: 9, CollectionID: 100, Channel: "ch", Node: 1})...)
	dist.SegmentDistManager.Update(2, meta.SeedInto(dist, fixture(3, 100, 30, 2))...)

	rowCount := newRowCountBasedAssignPolicy(nodeManager, scheduler, dist).getWorkloadStatus()
	assert.Equal(t, map[int64]int{1: 30, 2: 30}, rowCount.nodeGlobalRowCount)
	score := newScoreBasedAssignPolicy(nodeManager, scheduler, dist, nil).getWorkloadStatus()
	assert.Equal(t, map[int64]int{1: 30, 2: 30}, score.nodeGlobalRowCount)
	assert.Equal(t, map[int64]map[int64]int{1: {100: 10, 200: 20}, 2: {100: 30}}, score.nodeCollectionRowCount)
}
