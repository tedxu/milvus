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

package balance

import (
	"context"
	"fmt"
	"sort"
	"time"

	"github.com/milvus-io/milvus/internal/coordinator/snmanager"
	"github.com/milvus-io/milvus/internal/querycoordv2/assign"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/internal/querycoordv2/task"
	"github.com/milvus-io/milvus/internal/util/streamingutil"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
)

const (
	PlanInfoPrefix = "Balance-Plans:" // Prefix for balance plan log messages
)

// CreateSegmentTasksFromPlans converts segment assignment plans into executable tasks.
// Each plan is converted to a task with grow action (load segment) on the target node
// and reduce action (release segment) on the source node.
func CreateSegmentTasksFromPlans(ctx context.Context, source task.Source, timeout time.Duration, plans []assign.SegmentAssignPlan) []task.Task {
	ret := make([]task.Task, 0)
	for _, p := range plans {
		actions := make([]task.Action, 0)
		if p.To != -1 {
			action := task.NewSegmentActionWithScope(p.To, task.ActionTypeGrow, p.Segment.Channel, p.Segment.ID, querypb.DataScope_Historical, int(p.Segment.NumOfRows))
			actions = append(actions, action)
		}
		if p.From != -1 {
			action := task.NewSegmentActionWithScope(p.From, task.ActionTypeReduce, p.Segment.Channel, p.Segment.ID, querypb.DataScope_Historical, int(p.Segment.NumOfRows))
			actions = append(actions, action)
		}
		t, err := task.NewSegmentTask(
			ctx,
			timeout,
			source,
			p.Segment.CollectionID,
			p.Replica,
			p.LoadPriority,
			actions...,
		)
		if err != nil {
			mlog.Warn(ctx, "create segment task from plan failed",
				mlog.Int64("collection", p.Segment.CollectionID),
				mlog.FieldSegmentID(p.Segment.ID),
				mlog.Int64("replica", p.Replica.GetID()),
				mlog.String("channel", p.Segment.Channel),
				mlog.Int64("from", p.From),
				mlog.Int64("to", p.To),
				mlog.Err(err),
			)
			continue
		}

		mlog.Info(ctx, "create segment task",
			mlog.Int64("collection", p.Segment.CollectionID),
			mlog.FieldSegmentID(p.Segment.ID),
			mlog.Int64("replica", p.Replica.GetID()),
			mlog.String("channel", p.Segment.Channel),
			mlog.String("level", p.Segment.Level.String()),
			mlog.Int32("loadPriority", int32(p.LoadPriority)),
			mlog.Int64("from", p.From),
			mlog.Int64("to", p.To))
		if task.GetTaskType(t) == task.TaskTypeMove {
			// from balance checker
			t.SetPriority(task.TaskPriorityLow)
		} else {
			// from segment checker
			t.SetPriority(task.TaskPriorityNormal)
		}
		ret = append(ret, t)
	}
	return ret
}

// CreateChannelTasksFromPlans converts channel assignment plans into executable tasks.
// Each plan is converted to a task with grow action (subscribe channel) on the target node
// and reduce action (unsubscribe channel) on the source node.
func CreateChannelTasksFromPlans(ctx context.Context, source task.Source, timeout time.Duration, plans []assign.ChannelAssignPlan) []task.Task {
	ret := make([]task.Task, 0, len(plans))
	for _, p := range plans {
		actions := make([]task.Action, 0)
		if p.To != -1 {
			action := task.NewChannelAction(p.To, task.ActionTypeGrow, p.Channel.GetChannelName())
			actions = append(actions, action)
		}
		if p.From != -1 {
			action := task.NewChannelAction(p.From, task.ActionTypeReduce, p.Channel.GetChannelName())
			actions = append(actions, action)
		}
		t, err := task.NewChannelTask(ctx, timeout, source, p.Channel.GetCollectionID(), p.Replica, actions...)
		if err != nil {
			mlog.Warn(ctx, "create channel task failed",
				mlog.Int64("collection", p.Channel.GetCollectionID()),
				mlog.Int64("replica", p.Replica.GetID()),
				mlog.String("channel", p.Channel.GetChannelName()),
				mlog.Int64("from", p.From),
				mlog.Int64("to", p.To),
				mlog.Err(err),
			)
			continue
		}

		mlog.Info(ctx, "create channel task",
			mlog.Int64("collection", p.Channel.GetCollectionID()),
			mlog.Int64("replica", p.Replica.GetID()),
			mlog.String("channel", p.Channel.GetChannelName()),
			mlog.Int64("from", p.From),
			mlog.Int64("to", p.To))
		t.SetPriority(task.TaskPriorityHigh)
		ret = append(ret, t)
	}
	return ret
}

// PrintNewBalancePlans logs the newly generated balance plans for debugging and monitoring.
func PrintNewBalancePlans(collectionID int64, replicaID int64, segmentPlans []assign.SegmentAssignPlan,
	channelPlans []assign.ChannelAssignPlan,
) {
	balanceInfo := fmt.Sprintf("%s new plans:{collectionID:%d, replicaID:%d, ", PlanInfoPrefix, collectionID, replicaID)
	for _, segmentPlan := range segmentPlans {
		balanceInfo += segmentPlan.String()
	}
	for _, channelPlan := range channelPlans {
		balanceInfo += channelPlan.String()
	}
	balanceInfo += "}"
	mlog.Info(context.TODO(),
		balanceInfo)
}

// sortIfChannelAtWALLocated sorts channels based on WAL location when streaming service is enabled.
// Channels located at their WAL nodes are placed at the end of the list, making them less likely
// to be selected for movement. This optimization reduces unnecessary channel migrations when
// a channel is already at its optimal location for streaming.
func sortIfChannelAtWALLocated(channels []*meta.DmChannel) []*meta.DmChannel {
	if !streamingutil.IsStreamingServiceEnabled() {
		return channels
	}
	weighter := func(ch *meta.DmChannel) int {
		nodeID := snmanager.StaticStreamingNodeManager.GetWALLocated(ch.GetChannelName())
		if ch.Node == nodeID {
			// if node is the node where the WAL is located, put it to the tail of the channels.
			// so assign 1 to the node the WAL is located.
			return 1
		}
		return 0
	}
	sort.Slice(channels, func(i, j int) bool {
		return weighter(channels[i]) < weighter(channels[j])
	})
	return channels
}
