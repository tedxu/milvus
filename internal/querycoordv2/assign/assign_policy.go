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

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/internal/metacache"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
)

// SegmentRef is a transient view of one segment for one plan computation.
// It is built per plan and never stored.
type SegmentRef struct {
	ID, CollectionID, PartitionID, NumOfRows int64
	Channel                                  string
	Level                                    datapb.SegmentLevel
	// From is the node the segment is on; 0 when placing a target segment.
	From int64
}

// SegmentRefFromInfo builds the ref of a segment on node from, or of a
// target segment when from is 0.
func SegmentRefFromInfo(info *datapb.SegmentInfo, from int64) SegmentRef {
	return SegmentRef{
		ID:           info.GetID(),
		CollectionID: info.GetCollectionID(),
		PartitionID:  info.GetPartitionID(),
		NumOfRows:    info.GetNumOfRows(),
		Channel:      info.GetInsertChannel(),
		Level:        info.GetLevel(),
		From:         from,
	}
}

// SegmentRefsFromDist builds the refs of distribution entries from the
// store, with From set to each entry's node, in one batched read. An entry
// the store no longer has is dropped: there is nothing to plan for it, and
// the segment checker releases it.
func SegmentRefsFromDist(view metacache.MetaView, segments []*meta.LoadedSegment) []SegmentRef {
	refs := make([]SegmentRef, 0, len(segments))
	if len(segments) == 0 {
		return refs
	}
	ids := make([]int64, len(segments))
	for i, segment := range segments {
		ids[i] = segment.ID
	}
	i := 0
	view.ForEachSegment(ids, func(_ int64, info *datapb.SegmentInfo, ok bool) {
		if ok {
			refs = append(refs, SegmentRefFromInfo(info, segments[i].Node))
		}
		i++
	})
	return refs
}

// SegmentAssignPlan represents a plan to assign a segment to a node
type SegmentAssignPlan struct {
	Segment      SegmentRef
	Replica      *meta.Replica
	From         int64 // -1 if empty
	To           int64
	FromScore    int64
	ToScore      int64
	SegmentScore int64
	LoadPriority commonpb.LoadPriority
}

func (segPlan *SegmentAssignPlan) String() string {
	return fmt.Sprintf("SegmentPlan:[collectionID: %d, replicaID: %d, segmentID: %d, from: %d, to: %d, fromScore: %d, toScore: %d, segmentScore: %d]\n",
		segPlan.Segment.CollectionID, segPlan.Replica.GetID(), segPlan.Segment.ID, segPlan.From, segPlan.To, segPlan.FromScore, segPlan.ToScore, segPlan.SegmentScore)
}

// ChannelAssignPlan represents a plan to assign a channel to a node
type ChannelAssignPlan struct {
	Channel      *meta.DmChannel
	Replica      *meta.Replica
	From         int64
	To           int64
	FromScore    int64
	ToScore      int64
	ChannelScore int64
}

func (chanPlan *ChannelAssignPlan) String() string {
	return fmt.Sprintf("ChannelPlan:[collectionID: %d, channel: %s, replicaID: %d, from: %d, to: %d]\n",
		chanPlan.Channel.CollectionID, chanPlan.Channel.ChannelName, chanPlan.Replica.GetID(), chanPlan.From, chanPlan.To)
}

// AssignPolicy defines the unified policy for assigning both segments and channels to nodes
// This interface abstracts the common logic of resource assignment across different balancers
type AssignPolicy interface {
	// AssignSegment assigns segments to nodes based on the policy
	// Returns a list of segment assignment plans
	AssignSegment(ctx context.Context, collectionID int64, segments []SegmentRef, nodes []int64, forceAssign bool) []SegmentAssignPlan

	// AssignChannel assigns channels to nodes based on the policy
	// Returns a list of channel assignment plans
	AssignChannel(ctx context.Context, collectionID int64, channels []*meta.DmChannel, nodes []int64, forceAssign bool) []ChannelAssignPlan
}

// ScoreAwareAssignPolicy extends AssignPolicy with score-based node conversion
// and score calculation methods. This interface eliminates the need for type
// assertions when accessing score-specific functionality.
type ScoreAwareAssignPolicy interface {
	AssignPolicy
	ConvertToNodeItemsBySegment(collectionID int64, nodes []int64) map[int64]*NodeItem
	ConvertToNodeItemsByChannel(collectionID int64, nodes []int64) map[int64]*NodeItem
	CalculateSegmentScore(s SegmentRef) float64
	CalculateChannelScore(ch *meta.DmChannel, currentCollection int64) float64
}

// AssignPolicyConfig contains common configuration for assignment policies
type AssignPolicyConfig struct {
	// BatchSize limits the number of resources to assign in one batch
	BatchSize int

	// EnableBenefitCheck enables benefit evaluation before assignment
	EnableBenefitCheck bool
}
