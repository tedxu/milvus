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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/msgpb"
	"github.com/milvus-io/milvus/internal/metacache"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/internal/querycoordv2/session"
	"github.com/milvus-io/milvus/internal/querycoordv2/task"
	"github.com/milvus-io/milvus/internal/storagev2/packed"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// TestGrowingDiffNilPosition pins that a growing segment reported with no
// start position compares as timestamp 0 against the target's seek position.
func TestGrowingDiffNilPosition(t *testing.T) {
	targetMgr := meta.NewMockTargetManager(t)
	targetMgr.EXPECT().GetCollectionTargetVersion(mock.Anything, int64(1), meta.CurrentTarget).Return(int64(10))
	targetMgr.EXPECT().IsNextTargetExist(mock.Anything, int64(1)).Return(true)
	targetMgr.EXPECT().GetGrowingSegmentsByCollection(mock.Anything, int64(1), mock.Anything).Return(typeutil.NewUniqueSet())
	targetMgr.EXPECT().GetDmChannelsByCollection(mock.Anything, int64(1), meta.CurrentTarget).Return(map[string]*meta.DmChannel{
		"ch": {VchannelInfo: &datapb.VchannelInfo{ChannelName: "ch", SeekPosition: &msgpb.MsgPosition{Timestamp: 5}}},
	})
	checker := &SegmentChecker{targetMgr: targetMgr}
	view := &meta.LeaderView{
		ID:              2,
		CollectionID:    1,
		Channel:         "ch",
		TargetVersion:   10,
		GrowingSegments: map[int64]*msgpb.MsgPosition{9: nil, 10: {Timestamp: 6}},
	}

	_, toRelease := checker.getGrowingSegmentDiff(context.Background(), 1,
		meta.NewReplica(&querypb.Replica{ID: 1, CollectionID: 1}), []*meta.DmChannel{{Node: 2, View: view}})

	require.Len(t, toRelease, 1)
	assert.Equal(t, int64(9), toRelease[0].GetID())
	assert.Equal(t, int64(1), toRelease[0].GetCollectionID())
	assert.Equal(t, "ch", toRelease[0].GetInsertChannel())
	assert.Equal(t, int64(2), toRelease[0].Node)
}

func newSealedDiffTestChecker(t *testing.T, store metacache.MetaStore, target map[int64]*datapb.SegmentInfo) *SegmentChecker {
	ids := typeutil.NewUniqueSet()
	for id := range target {
		ids.Insert(id)
	}
	targetMgr := meta.NewMockTargetManager(t)
	targetMgr.EXPECT().IsNextTargetExist(mock.Anything, int64(1)).Return(true).Maybe()
	targetMgr.EXPECT().IsCurrentTargetExist(mock.Anything, int64(1), mock.Anything).Return(true).Maybe()
	targetMgr.EXPECT().GetSealedSegmentsByCollection(mock.Anything, int64(1), mock.Anything).Return(target).Maybe()
	targetMgr.EXPECT().GetSealedSegmentIDsByCollection(mock.Anything, int64(1), mock.Anything).Return(ids).Maybe()
	return &SegmentChecker{
		meta:      &meta.Meta{CollectionManager: meta.NewCollectionManager(nil, store)},
		dist:      meta.NewDistributionManager(session.NewNodeManager(), meta.WithMetaView(store)),
		targetMgr: targetMgr,
	}
}

// TestSegmentCheckerReleasesStoreMiss pins that a loaded segment absent from
// both the store and the target is released on its record's channel.
func TestSegmentCheckerReleasesStoreMiss(t *testing.T) {
	ctx := context.Background()
	checker := newSealedDiffTestChecker(t, metacache.NewMetaStore(nil), map[int64]*datapb.SegmentInfo{})
	replica := meta.NewReplica(&querypb.Replica{ID: 1, CollectionID: 1})
	miss := &meta.LoadedSegment{ID: 7, CollectionID: 1, Channel: "ch", Node: 2}

	_, _, toRelease, _ := checker.getSealedSegmentDiff(ctx, 1, replica, []*meta.LoadedSegment{miss})
	require.Equal(t, []*meta.LoadedSegment{miss}, toRelease)

	tasks := checker.createSegmentReduceTasks(ctx, toRelease, replica, querypb.DataScope_Historical)
	require.Len(t, tasks, 1)
	action := tasks[0].Actions()[0].(*task.SegmentAction)
	assert.Equal(t, "ch", action.GetShard())
	assert.Equal(t, int64(2), action.Node())
	assert.Equal(t, int64(7), action.GetSegmentID())
}

// TestReopenComparesLoadedWithTarget pins that the loaded manifest path and
// data version are compared with the target's, and that a nil data version
// (an old QueryNode) skips the data version comparison.
func TestReopenComparesLoadedWithTarget(t *testing.T) {
	ctx := context.Background()
	replica := meta.NewReplica(&querypb.Replica{ID: 1, CollectionID: 1})
	older, newer := packed.MarshalManifestPath("base", 1), packed.MarshalManifestPath("base", 2)
	behind := int32(1)
	cases := []struct {
		name        string
		loaded      *meta.LoadedSegment
		target      *datapb.SegmentInfo
		wantReopens int
	}{
		{
			"older manifest", &meta.LoadedSegment{ID: 7, CollectionID: 1, Channel: "ch", Node: 2, ManifestPath: older},
			&datapb.SegmentInfo{ID: 7, CollectionID: 1, InsertChannel: "ch", ManifestPath: newer}, 1,
		},
		{
			"same manifest", &meta.LoadedSegment{ID: 7, CollectionID: 1, Channel: "ch", Node: 2, ManifestPath: newer},
			&datapb.SegmentInfo{ID: 7, CollectionID: 1, InsertChannel: "ch", ManifestPath: newer}, 0,
		},
		{
			"data version behind", &meta.LoadedSegment{ID: 7, CollectionID: 1, Channel: "ch", Node: 2, ManifestPath: newer, DataVersion: &behind},
			&datapb.SegmentInfo{ID: 7, CollectionID: 1, InsertChannel: "ch", ManifestPath: newer, DataVersion: 2}, 1,
		},
		{
			"nil data version", &meta.LoadedSegment{ID: 7, CollectionID: 1, Channel: "ch", Node: 2, ManifestPath: newer},
			&datapb.SegmentInfo{ID: 7, CollectionID: 1, InsertChannel: "ch", ManifestPath: newer, DataVersion: 2}, 0,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			store := metacache.NewMetaStore(nil)
			store.PutSegment(tc.target)
			checker := newSealedDiffTestChecker(t, store, map[int64]*datapb.SegmentInfo{7: tc.target})
			_, _, _, toUpdate := checker.getSealedSegmentDiff(ctx, 1, replica, []*meta.LoadedSegment{tc.loaded})
			assert.Len(t, toUpdate, tc.wantReopens)
		})
	}
}

// TestGrowingReleaseHasNoWorkloadEffect pins that releasing a growing
// segment carries 0 rows even when the store has a flushed segment with
// the same ID: sealed row totals never counted growing rows.
func TestGrowingReleaseHasNoWorkloadEffect(t *testing.T) {
	store := metacache.NewMetaStore(nil)
	store.PutSegment(&datapb.SegmentInfo{ID: 9, CollectionID: 1, InsertChannel: "ch", NumOfRows: 100})
	checker := newSealedDiffTestChecker(t, store, map[int64]*datapb.SegmentInfo{})
	replica := meta.NewReplica(&querypb.Replica{ID: 1, CollectionID: 1})
	segments := []*meta.LoadedSegment{{ID: 9, CollectionID: 1, Channel: "ch", Node: 2}}

	growing := checker.createSegmentReduceTasks(context.Background(), segments, replica, querypb.DataScope_Streaming)
	require.Len(t, growing, 1)
	assert.Zero(t, growing[0].Actions()[0].WorkLoadEffect())

	sealed := checker.createSegmentReduceTasks(context.Background(), segments, replica, querypb.DataScope_Historical)
	require.Len(t, sealed, 1)
	assert.Equal(t, -100, sealed[0].Actions()[0].WorkLoadEffect())
}
