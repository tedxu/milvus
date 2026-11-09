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
	"time"

	"github.com/samber/lo"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/metacache"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/internal/querycoordv2/params"
	"github.com/milvus-io/milvus/internal/querycoordv2/session"
	"github.com/milvus-io/milvus/internal/querycoordv2/task"
	"github.com/milvus-io/milvus/internal/querycoordv2/utils"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

var _ Checker = (*IndexChecker)(nil)

// IndexChecker perform segment index check.
type IndexChecker struct {
	*checkerActivation
	meta    *meta.Meta
	dist    *meta.DistributionManager
	nodeMgr *session.NodeManager

	targetMgr meta.TargetManagerInterface
}

func NewIndexChecker(
	meta *meta.Meta,
	dist *meta.DistributionManager,
	nodeMgr *session.NodeManager,
	targetMgr meta.TargetManagerInterface,
) *IndexChecker {
	return &IndexChecker{
		checkerActivation: newCheckerActivation(),
		meta:              meta,
		dist:              dist,
		nodeMgr:           nodeMgr,
		targetMgr:         targetMgr,
	}
}

func (c *IndexChecker) ID() utils.CheckerType {
	return utils.IndexChecker
}

func (c *IndexChecker) Description() string {
	return "SegmentChecker checks index state change of segments and generates load index task"
}

func (c *IndexChecker) Check(ctx context.Context) []task.Task {
	if !c.IsActive() {
		return nil
	}
	collectionIDs := c.meta.GetAll(ctx)
	var tasks []task.Task

	for _, collectionID := range collectionIDs {
		indexInfos := c.meta.ListIndexInfos(collectionID)

		collection := c.meta.GetCollection(ctx, collectionID)
		if collection == nil {
			mlog.Warn(ctx, "collection released during check index", mlog.Int64("collection", collectionID))
			continue
		}
		// A nil schema means RootCoord no longer has the collection.
		schema := c.meta.GetCollectionSchema(ctx, collectionID)
		replicas := c.meta.GetByCollection(ctx, collectionID)
		for _, replica := range replicas {
			tasks = append(tasks, c.checkReplica(ctx, collection, replica, indexInfos, schema)...)
		}
	}

	return tasks
}

func (c *IndexChecker) checkReplica(ctx context.Context, collection *meta.Collection, replica *meta.Replica, indexInfos []*indexpb.IndexInfo, schema *schemapb.CollectionSchema) []task.Task {
	view := c.meta.MetaView()
	segments := c.dist.SegmentDistManager.GetByFilter(meta.WithCollectionID(replica.GetCollectionID()), meta.WithReplica(replica))
	roNodeSet := typeutil.NewUniqueSet(replica.GetRONodes()...)

	segmentsToUpdate := make(map[int64]*meta.LoadedSegment)
	segmentsStatsToUpdate := make(map[int64]*meta.LoadedSegment)
	for _, segment := range segments {
		// skip update index in read only node
		if roNodeSet.Contain(segment.Node) {
			continue
		}
		// A segment the store no longer has is on its way out; the segment
		// checker releases it.
		storeSegment, ok := view.GetSegment(segment.GetID())
		if !ok {
			continue
		}
		if missing, ready := c.checkSegment(view, segment, indexInfos); missing {
			if ready {
				segmentsToUpdate[segment.GetID()] = segment
			}
		} else {
			for _, fieldID := range c.checkSegmentStats(segment, schema, collection.LoadFields) {
				if _, ok := storeSegment.GetJsonKeyStats()[fieldID]; ok {
					segmentsStatsToUpdate[segment.GetID()] = segment
					break
				}
			}
		}

		if len(c.checkRedundantIndices(segment, indexInfos)) > 0 {
			segmentsToUpdate[segment.GetID()] = segment
		}
	}

	tasks := lo.FilterMap(lo.Values(segmentsToUpdate), func(segment *meta.LoadedSegment, _ int) (task.Task, bool) {
		return c.createSegmentUpdateTask(ctx, segment, replica)
	})
	tasksStats := lo.FilterMap(lo.Values(segmentsStatsToUpdate), func(segment *meta.LoadedSegment, _ int) (task.Task, bool) {
		return c.createSegmentStatsUpdateTask(ctx, segment, replica)
	})
	return append(tasks, tasksStats...)
}

// checkSegment reports whether the node lacks some live index of the
// segment (missing), and whether the store has a build of a missing index
// that can be loaded: finished, not deleted, and with index files (ready).
// Only absent index IDs count: a reopen loads the index IDs a segment lacks,
// so a newer build of a loaded index would load nothing.
func (c *IndexChecker) checkSegment(view metacache.MetaView, segment *meta.LoadedSegment, indexInfos []*indexpb.IndexInfo) (missing, ready bool) {
	for _, indexInfo := range indexInfos {
		if _, ok := segment.IndexBuilds[indexInfo.GetIndexID()]; ok {
			continue
		}
		missing = true
		record, ok := view.GetSegmentIndex(segment.GetID(), indexInfo.GetIndexID())
		if ok && record.IndexState == commonpb.IndexState_Finished && !record.IsDeleted && len(record.IndexFileKeys) > 0 {
			ready = true
		}
	}
	return missing, ready
}

// checkRedundantIndices returns redundant indexIDs for each segment
func (c *IndexChecker) checkRedundantIndices(segment *meta.LoadedSegment, indexInfos []*indexpb.IndexInfo) []int64 {
	var redundant []int64
	indexInfoMap := typeutil.NewSet[int64]()

	for _, indexInfo := range indexInfos {
		indexInfoMap.Insert(indexInfo.IndexID)
	}

	for indexID := range segment.IndexBuilds {
		if !indexInfoMap.Contain(indexID) {
			redundant = append(redundant, indexID)
		}
	}

	return redundant
}

func (c *IndexChecker) createSegmentUpdateTask(ctx context.Context, segment *meta.LoadedSegment, replica *meta.Replica) (task.Task, bool) {
	stored, _ := c.meta.MetaView().GetSegment(segment.GetID())
	action := task.NewSegmentActionWithScope(segment.Node, task.ActionTypeReopen, segment.GetInsertChannel(), segment.GetID(), querypb.DataScope_Historical, int(stored.GetNumOfRows()))
	t, err := task.NewSegmentTask(
		ctx,
		params.Params.QueryCoordCfg.SegmentTaskTimeout.GetAsDuration(time.Millisecond),
		c.ID(),
		segment.GetCollectionID(),
		replica,
		commonpb.LoadPriority_LOW, // Index update is not urgent
		action,
	)
	if err != nil {
		mlog.Warn(ctx, "create segment update task failed",
			mlog.Int64("collection", segment.GetCollectionID()),
			mlog.String("channel", segment.GetInsertChannel()),
			mlog.Int64("node", segment.Node),
			mlog.Err(err),
		)
		return nil, false
	}
	// index task shall have lower or equal priority than balance task
	t.SetPriority(task.TaskPriorityLow)
	t.SetReason("missing index")
	return t, true
}

func (c *IndexChecker) checkSegmentStats(segment *meta.LoadedSegment, schema *schemapb.CollectionSchema, loadField []int64) (missFieldIDs []int64) {
	var result []int64

	if paramtable.Get().CommonCfg.EnabledJSONKeyStats.GetAsBool() {
		if schema == nil {
			mlog.Warn(context.TODO(), "schema released during check index", mlog.Int64("collection", segment.GetCollectionID()))
			return result
		}
		loadFieldMap := make(map[int64]struct{})
		for _, v := range loadField {
			loadFieldMap[v] = struct{}{}
		}

		for _, field := range schema.GetFields() {
			// Check if the field exists in both loadFieldMap and jsonStatsFieldMap
			h := typeutil.CreateFieldSchemaHelper(field)
			fieldID := field.GetFieldID()
			if h.EnableJSONKeyStatsIndex() {
				if _, ok := loadFieldMap[fieldID]; ok {
					if format, ok := segment.JSONStatsFormats[fieldID]; !ok || format < common.JSONStatsDataFormatVersion {
						result = append(result, fieldID)
					}
				}
			}
		}
	}
	return result
}

func (c *IndexChecker) createSegmentStatsUpdateTask(ctx context.Context, segment *meta.LoadedSegment, replica *meta.Replica) (task.Task, bool) {
	stored, _ := c.meta.MetaView().GetSegment(segment.GetID())
	action := task.NewSegmentActionWithScope(segment.Node, task.ActionTypeReopen, segment.GetInsertChannel(), segment.GetID(), querypb.DataScope_Historical, int(stored.GetNumOfRows()))
	t, err := task.NewSegmentTask(
		ctx,
		params.Params.QueryCoordCfg.SegmentTaskTimeout.GetAsDuration(time.Millisecond),
		c.ID(),
		segment.GetCollectionID(),
		replica,
		commonpb.LoadPriority_LOW, // Stats update is not urgent
		action,
	)
	if err != nil {
		mlog.Warn(ctx, "create segment stats update task failed",
			mlog.Int64("collection", segment.GetCollectionID()),
			mlog.String("channel", segment.GetInsertChannel()),
			mlog.Int64("node", segment.Node),
			mlog.Err(err),
		)
		return nil, false
	}
	t.SetPriority(task.TaskPriorityLow)
	t.SetReason("missing json stats")
	return t, true
}
