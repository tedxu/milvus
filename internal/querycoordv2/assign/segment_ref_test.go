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
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

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

func TestSegmentRefsFromDist(t *testing.T) {
	refs := SegmentRefsFromDist([]*meta.Segment{
		{SegmentInfo: &datapb.SegmentInfo{ID: 1, CollectionID: 10, PartitionID: 100, NumOfRows: 7, InsertChannel: "ch1", Level: datapb.SegmentLevel_L1}, Node: 1},
		{SegmentInfo: &datapb.SegmentInfo{ID: 2, CollectionID: 10, PartitionID: 101, NumOfRows: 9, InsertChannel: "ch2", Level: datapb.SegmentLevel_L2}, Node: 2},
	})
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
	onNode1 := []*meta.Segment{
		{SegmentInfo: &datapb.SegmentInfo{ID: 1, CollectionID: 100, NumOfRows: 1000}, Node: 1},
		{SegmentInfo: &datapb.SegmentInfo{ID: 2, CollectionID: 100, NumOfRows: 1000}, Node: 1},
	}
	dist.SegmentDistManager.Update(1, onNode1...)
	policy := newScoreBasedAssignPolicy(nodeManager, scheduler, dist, meta.NewMeta(nil, nil, nodeManager, metacache.NewMetaStore(nil)))
	plans := policy.AssignSegment(context.Background(), 100, SegmentRefsFromDist(onNode1), []int64{1, 2}, true)
	require.Len(t, plans, 2)
	for _, plan := range plans {
		assert.Equal(t, int64(1), plan.From)
	}
	assert.InDelta(t, plans[0].FromScore-plans[0].SegmentScore, plans[1].FromScore, 1)
}

// TestBalancePlansStableForTies pins the plans for segments that all tie on
// rows, on nodes that all have zero rows. The expected order was captured
// from the code before segment refs.
func TestBalancePlansStableForTies(t *testing.T) {
	nodeManager, scheduler := newTieTestNodes(t, 3)
	policy := newRowCountBasedAssignPolicy(nodeManager, scheduler, meta.NewDistributionManager(nodeManager))
	segments := make([]SegmentRef, 0, 7)
	for i := int64(1); i <= 7; i++ {
		segments = append(segments, SegmentRef{ID: i, CollectionID: 100})
	}
	plans := policy.AssignSegment(context.Background(), 100, segments, []int64{1, 2, 3}, false)
	got := make([]string, 0, len(plans))
	for _, plan := range plans {
		got = append(got, fmt.Sprintf("%d->%d", plan.Segment.ID, plan.To))
	}
	assert.Equal(t, "1->1 2->3 3->1 4->3 5->1", strings.Join(got, " "))
}
