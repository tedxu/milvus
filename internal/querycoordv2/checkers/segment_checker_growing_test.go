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
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
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
