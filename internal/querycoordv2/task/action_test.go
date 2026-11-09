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

package task

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus-proto/go-api/v3/msgpb"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/internal/querycoordv2/session"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
)

func TestSegmentInDistGrowing(t *testing.T) {
	dist := meta.NewDistributionManager(session.NewNodeManager())
	dist.ChannelDistManager.Update(2, &meta.DmChannel{
		VchannelInfo: &datapb.VchannelInfo{CollectionID: 1, ChannelName: "ch"},
		Node:         2,
		View: &meta.LeaderView{
			ID:              2,
			CollectionID:    1,
			Channel:         "ch",
			GrowingSegments: map[int64]*msgpb.MsgPosition{7: {Timestamp: 1}},
		},
	})
	assert.True(t, segmentInDist(dist, 2, "ch", 7, querypb.DataScope_Streaming))
	assert.False(t, segmentInDist(dist, 2, "ch", 8, querypb.DataScope_Streaming))
}
