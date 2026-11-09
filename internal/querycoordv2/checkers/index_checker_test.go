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

	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/internal/metacache"
	"github.com/milvus-io/milvus/internal/metastore/kv/querycoord"
	catalogmocks "github.com/milvus-io/milvus/internal/metastore/mocks"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/internal/querycoordv2/params"
	"github.com/milvus-io/milvus/internal/querycoordv2/session"
	"github.com/milvus-io/milvus/internal/querycoordv2/task"
	"github.com/milvus-io/milvus/internal/querycoordv2/utils"
	"github.com/milvus-io/milvus/pkg/v3/kv"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/etcd"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

type IndexCheckerSuite struct {
	suite.Suite
	kv        kv.MetaKv
	checker   *IndexChecker
	meta      *meta.Meta
	metaStore metacache.MetaStore
	nodeMgr   *session.NodeManager
	targetMgr *meta.MockTargetManager
}

func (suite *IndexCheckerSuite) SetupSuite() {
	paramtable.Init()
}

func (suite *IndexCheckerSuite) SetupTest() {
	var err error
	config := params.GenerateEtcdConfig()
	cli, err := etcd.GetEtcdClient(
		config.UseEmbedEtcd.GetAsBool(),
		config.EtcdUseSSL.GetAsBool(),
		config.Endpoints.GetAsStrings(),
		config.EtcdTLSCert.GetValue(),
		config.EtcdTLSKey.GetValue(),
		config.EtcdTLSCACert.GetValue(),
		config.EtcdTLSMinVersion.GetValue())
	suite.Require().NoError(err)
	suite.kv = etcdkv.NewEtcdKV(cli, config.MetaRootPath.GetValue())

	// meta
	store := querycoord.NewCatalog(suite.kv)
	idAllocator := params.RandomIncrementIDAllocator()
	suite.nodeMgr = session.NewNodeManager()
	suite.metaStore = metacache.NewMetaStore(nil)
	suite.meta = meta.NewMeta(idAllocator, store, suite.nodeMgr, suite.metaStore)
	distManager := meta.NewDistributionManager(suite.nodeMgr, meta.WithMetaView(suite.metaStore))

	suite.targetMgr = meta.NewMockTargetManager(suite.T())
	suite.checker = NewIndexChecker(suite.meta, distManager, suite.nodeMgr, suite.targetMgr)

	suite.targetMgr.EXPECT().GetSealedSegment(mock.Anything, mock.Anything, mock.Anything, mock.Anything).RunAndReturn(func(ctx context.Context, cid, sid int64, i3 int32) *datapb.SegmentInfo {
		return &datapb.SegmentInfo{
			ID:    sid,
			Level: datapb.SegmentLevel_L1,
		}
	}).Maybe()
}

// putBuild publishes a segment index record of collection 1 to the store.
func (suite *IndexCheckerSuite) putBuild(segmentID, indexID, buildID int64, state commonpb.IndexState) {
	suite.metaStore.PutSegmentIndexes(&model.SegmentIndex{
		CollectionID: 1, SegmentID: segmentID, IndexID: indexID, BuildID: buildID,
		IndexState: state, IndexFileKeys: []string{"index"},
	})
}

// putFinishedBuild publishes a finished build with files, ready to load.
func (suite *IndexCheckerSuite) putFinishedBuild(segmentID, indexID, buildID int64) {
	suite.putBuild(segmentID, indexID, buildID, commonpb.IndexState_Finished)
}

// putSegment publishes a segment of collection 1 to the store.
func (suite *IndexCheckerSuite) putSegment(segmentID int64, jsonKeyStats map[int64]*datapb.JsonKeyStats) {
	suite.metaStore.PutSegment(&datapb.SegmentInfo{
		ID: segmentID, CollectionID: 1, PartitionID: 1, InsertChannel: "test-insert-channel",
		JsonKeyStats: jsonKeyStats,
	})
}

func (suite *IndexCheckerSuite) TearDownTest() {
	suite.kv.Close()
}

func (suite *IndexCheckerSuite) TestLoadIndex() {
	checker := suite.checker
	ctx := context.Background()

	// meta
	coll := utils.CreateTestCollection(1, 1)
	coll.FieldIndexID = map[int64]int64{101: 1000}
	schema := &schemapb.CollectionSchema{
		Name: "test_loadJsonIndex",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 101, DataType: schemapb.DataType_JSON, Name: "JSON"},
		},
	}
	checker.meta.PutCollection(ctx, coll)
	suite.metaStore.PutCollection(&metacache.CollectionInfo{ID: coll.GetCollectionID(), Schema: schema})
	checker.meta.Put(ctx, utils.CreateTestReplica(200, 1, []int64{1, 2}))
	suite.nodeMgr.Add(session.NewNodeInfo(session.ImmutableNodeInfo{
		NodeID:   1,
		Address:  "localhost",
		Hostname: "localhost",
	}))
	suite.nodeMgr.Add(session.NewNodeInfo(session.ImmutableNodeInfo{
		NodeID:   2,
		Address:  "localhost",
		Hostname: "localhost",
	}))
	checker.meta.HandleNodeUp(ctx, 1)
	checker.meta.HandleNodeUp(ctx, 2)

	// dist
	checker.dist.SegmentDistManager.Update(1, meta.SeedInto(checker.dist, utils.CreateTestSegment(1, 1, 2, 1, 1, "test-insert-channel"))...)

	// store
	suite.putSegment(2, nil)
	suite.putFinishedBuild(2, 1000, 5)

	suite.metaStore.PutIndex(&model.Index{CollectionID: 1, FieldID: 101, IndexID: 1000})

	tasks := checker.Check(context.Background())
	suite.Require().Len(tasks, 1)

	t := tasks[0]
	suite.Require().Len(t.Actions(), 1)

	action, ok := t.Actions()[0].(*task.SegmentAction)
	suite.Require().True(ok)
	suite.EqualValues(200, t.ReplicaID())
	suite.Equal(task.ActionTypeReopen, action.Type())
	suite.EqualValues(2, action.GetSegmentID())

	// test skip load index for read only node
	suite.nodeMgr.Stopping(1)
	suite.nodeMgr.Stopping(2)
	suite.meta.HandleNodeStopping(ctx, 1)
	suite.meta.HandleNodeStopping(ctx, 2)
	utils.RecoverAllCollection(suite.meta)
	tasks = checker.Check(context.Background())
	suite.Require().Len(tasks, 0)
}

func (suite *IndexCheckerSuite) TestIndexInfoNotMatch() {
	checker := suite.checker
	ctx := context.Background()

	// meta
	coll := utils.CreateTestCollection(1, 1)
	coll.FieldIndexID = map[int64]int64{101: 1000}
	schema := &schemapb.CollectionSchema{
		Name: "test_loadJsonIndex",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 101, DataType: schemapb.DataType_JSON, Name: "JSON"},
		},
	}
	checker.meta.PutCollection(ctx, coll)
	suite.metaStore.PutCollection(&metacache.CollectionInfo{ID: coll.GetCollectionID(), Schema: schema})
	checker.meta.Put(ctx, utils.CreateTestReplica(200, 1, []int64{1, 2}))
	suite.nodeMgr.Add(session.NewNodeInfo(session.ImmutableNodeInfo{
		NodeID:   1,
		Address:  "localhost",
		Hostname: "localhost",
	}))
	suite.nodeMgr.Add(session.NewNodeInfo(session.ImmutableNodeInfo{
		NodeID:   2,
		Address:  "localhost",
		Hostname: "localhost",
	}))
	checker.meta.HandleNodeUp(ctx, 1)
	checker.meta.HandleNodeUp(ctx, 2)

	// dist
	checker.dist.SegmentDistManager.Update(1, meta.SeedInto(checker.dist, utils.CreateTestSegment(1, 1, 2, 1, 1, "test-insert-channel"))...)
	checker.dist.SegmentDistManager.Update(1, meta.SeedInto(checker.dist, utils.CreateTestSegment(1, 1, 3, 1, 1, "test-insert-channel"))...)

	// store
	suite.putSegment(2, nil)
	suite.putSegment(3, nil)
	suite.putBuild(2, 1000, 5, commonpb.IndexState_InProgress)
	// 1002 is not a live definition of the collection.
	suite.putFinishedBuild(3, 1002, 6)

	suite.metaStore.PutIndex(&model.Index{CollectionID: 1, FieldID: 101, IndexID: 1000})
	tasks := checker.Check(context.Background())
	suite.Require().Len(tasks, 0)
}

func (suite *IndexCheckerSuite) TestNoStoreRecordNoTask() {
	checker := suite.checker
	ctx := context.Background()

	// meta
	coll := utils.CreateTestCollection(1, 1)
	coll.FieldIndexID = map[int64]int64{101: 1000}
	schema := &schemapb.CollectionSchema{
		Name: "test_loadJsonIndex",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 101, DataType: schemapb.DataType_JSON, Name: "JSON"},
		},
	}
	checker.meta.PutCollection(ctx, coll)
	suite.metaStore.PutCollection(&metacache.CollectionInfo{ID: coll.GetCollectionID(), Schema: schema})
	checker.meta.Put(ctx, utils.CreateTestReplica(200, 1, []int64{1, 2}))
	suite.nodeMgr.Add(session.NewNodeInfo(session.ImmutableNodeInfo{
		NodeID:   1,
		Address:  "localhost",
		Hostname: "localhost",
	}))
	suite.nodeMgr.Add(session.NewNodeInfo(session.ImmutableNodeInfo{
		NodeID:   2,
		Address:  "localhost",
		Hostname: "localhost",
	}))
	checker.meta.HandleNodeUp(ctx, 1)
	checker.meta.HandleNodeUp(ctx, 2)

	// dist
	checker.dist.SegmentDistManager.Update(1, meta.SeedInto(checker.dist, utils.CreateTestSegment(1, 1, 2, 1, 1, "test-insert-channel"))...)
	checker.dist.SegmentDistManager.Update(1, meta.SeedInto(checker.dist, utils.CreateTestSegment(1, 1, 3, 1, 1, "test-insert-channel"))...)
	suite.putSegment(2, nil)
	suite.putSegment(3, nil)

	// store
	suite.metaStore.PutIndex(&model.Index{CollectionID: 1, FieldID: 101, IndexID: 1000})
	tasks := checker.Check(context.Background())
	suite.Require().Len(tasks, 0)
}

func (suite *IndexCheckerSuite) TestCreateNewIndex() {
	checker := suite.checker
	ctx := context.Background()

	// meta
	coll := utils.CreateTestCollection(1, 1)
	coll.FieldIndexID = map[int64]int64{101: 1000}
	schema := &schemapb.CollectionSchema{
		Name: "test_loadJsonIndex",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 101, DataType: schemapb.DataType_JSON, Name: "JSON"},
		},
	}
	checker.meta.PutCollection(ctx, coll)
	suite.metaStore.PutCollection(&metacache.CollectionInfo{ID: coll.GetCollectionID(), Schema: schema})
	checker.meta.Put(ctx, utils.CreateTestReplica(200, 1, []int64{1, 2}))
	suite.nodeMgr.Add(session.NewNodeInfo(session.ImmutableNodeInfo{
		NodeID:   1,
		Address:  "localhost",
		Hostname: "localhost",
	}))
	suite.nodeMgr.Add(session.NewNodeInfo(session.ImmutableNodeInfo{
		NodeID:   2,
		Address:  "localhost",
		Hostname: "localhost",
	}))
	checker.meta.HandleNodeUp(ctx, 1)
	checker.meta.HandleNodeUp(ctx, 2)

	// dist
	segment := utils.CreateTestSegment(1, 1, 2, 1, 1, "test-insert-channel")
	segment.IndexBuilds = map[int64]int64{1000: 0}
	checker.dist.SegmentDistManager.Update(1, meta.SeedInto(checker.dist, segment)...)

	// store
	suite.metaStore.PutIndex(&model.Index{CollectionID: 1, FieldID: 101, IndexID: 1000})
	suite.metaStore.PutIndex(&model.Index{CollectionID: 1, FieldID: 102, IndexID: 1001})
	suite.putSegment(2, nil)
	suite.putFinishedBuild(2, 1000, 0)
	suite.putFinishedBuild(2, 1001, 7)
	tasks := checker.Check(context.Background())
	suite.Len(tasks, 1)
	suite.Len(tasks[0].Actions(), 1)
	suite.Equal(tasks[0].Actions()[0].(*task.SegmentAction).Type(), task.ActionTypeReopen)
}

func TestRemoveRedundantIndex(t *testing.T) {
	paramtable.Init()
	ctx := context.Background()

	catalog := catalogmocks.NewQueryCoordCatalog(t)
	catalog.EXPECT().SaveCollection(mock.Anything, mock.Anything).Return(nil)
	catalog.EXPECT().SaveReplica(mock.Anything, mock.Anything).Return(nil)
	catalog.EXPECT().SaveResourceGroup(mock.Anything, mock.Anything).Return(nil).Maybe()

	nodeMgr := session.NewNodeManager()
	metaStore := metacache.NewMetaStore(nil)
	metaMgr := meta.NewMeta(params.RandomIncrementIDAllocator(), catalog, nodeMgr, metaStore)
	distManager := meta.NewDistributionManager(nodeMgr, meta.WithMetaView(metaStore))
	targetMgr := meta.NewMockTargetManager(t)
	checker := NewIndexChecker(metaMgr, distManager, nodeMgr, targetMgr)
	targetMgr.EXPECT().GetSealedSegment(mock.Anything, mock.Anything, mock.Anything, mock.Anything).RunAndReturn(func(ctx context.Context, cid, sid int64, i3 int32) *datapb.SegmentInfo {
		return &datapb.SegmentInfo{
			ID:    sid,
			Level: datapb.SegmentLevel_L1,
		}
	}).Maybe()

	// meta
	coll := utils.CreateTestCollection(1, 1)
	coll.FieldIndexID = map[int64]int64{101: 1000}
	schema := &schemapb.CollectionSchema{
		Name: "test_remove_redundant_index",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 101, DataType: schemapb.DataType_JSON, Name: "JSON"},
		},
	}
	require.NoError(t, checker.meta.PutCollection(ctx, coll))
	metaStore.PutCollection(&metacache.CollectionInfo{ID: coll.GetCollectionID(), Schema: schema})
	require.NoError(t, checker.meta.Put(ctx, utils.CreateTestReplica(200, 1, []int64{1, 2})))
	nodeMgr.Add(session.NewNodeInfo(session.ImmutableNodeInfo{
		NodeID:   1,
		Address:  "localhost",
		Hostname: "localhost",
	}))
	nodeMgr.Add(session.NewNodeInfo(session.ImmutableNodeInfo{
		NodeID:   2,
		Address:  "localhost",
		Hostname: "localhost",
	}))
	checker.meta.HandleNodeUp(ctx, 1)
	checker.meta.HandleNodeUp(ctx, 2)

	// dist
	segment := utils.CreateTestSegment(1, 1, 2, 1, 1, "test-insert-channel")
	segment.IndexBuilds = map[int64]int64{1000: 0, 1001: 0}
	checker.dist.SegmentDistManager.Update(1, meta.SeedInto(checker.dist, segment)...)

	// store
	metaStore.PutIndex(&model.Index{CollectionID: 1, FieldID: 101, IndexID: 1000})
	metaStore.PutSegment(&datapb.SegmentInfo{ID: 2, CollectionID: 1, PartitionID: 1, InsertChannel: "test-insert-channel"})

	tasks := checker.Check(context.Background())
	require.Len(t, tasks, 1)
	require.Len(t, tasks[0].Actions(), 1)
	action, ok := tasks[0].Actions()[0].(*task.SegmentAction)
	require.True(t, ok)
	require.EqualValues(t, 200, tasks[0].ReplicaID())
	require.Equal(t, task.ActionTypeReopen, action.Type())
	require.EqualValues(t, 2, action.GetSegmentID())
}

func (suite *IndexCheckerSuite) TestLoadJsonIndex() {
	checker := suite.checker
	ctx := context.Background()
	paramtable.Get().Save(paramtable.Get().CommonCfg.EnabledJSONKeyStats.Key, "true")
	defer paramtable.Get().Reset(paramtable.Get().CommonCfg.EnabledJSONKeyStats.Key)
	// meta
	coll := utils.CreateTestCollection(1, 1)
	coll.FieldIndexID = map[int64]int64{101: 1000}
	schema := &schemapb.CollectionSchema{
		Name: "test_loadJsonIndex",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 101, DataType: schemapb.DataType_JSON, Name: "JSON"},
		},
	}
	coll.LoadFields = []int64{101}
	checker.meta.PutCollection(ctx, coll)
	suite.metaStore.PutCollection(&metacache.CollectionInfo{ID: coll.GetCollectionID(), Schema: schema})
	checker.meta.Put(ctx, utils.CreateTestReplica(200, 1, []int64{1, 2}))
	suite.nodeMgr.Add(session.NewNodeInfo(session.ImmutableNodeInfo{
		NodeID:   1,
		Address:  "localhost",
		Hostname: "localhost",
	}))
	suite.nodeMgr.Add(session.NewNodeInfo(session.ImmutableNodeInfo{
		NodeID:   2,
		Address:  "localhost",
		Hostname: "localhost",
	}))
	checker.meta.HandleNodeUp(ctx, 1)
	checker.meta.HandleNodeUp(ctx, 2)

	// dist
	fieldIndexInfo := &querypb.FieldIndexInfo{
		FieldID:     101,
		IndexID:     1000,
		EnableIndex: true,
	}

	segment := utils.CreateTestSegment(1, 1, 2, 1, 1, "test-insert-channel")
	segment.IndexBuilds = map[int64]int64{fieldIndexInfo.IndexID: fieldIndexInfo.BuildID}
	checker.dist.SegmentDistManager.Update(1, meta.SeedInto(checker.dist, segment)...)

	// store
	suite.metaStore.PutIndex(&model.Index{CollectionID: 1, FieldID: 101, IndexID: 1000})
	mockJSONKeyStats := map[int64]*datapb.JsonKeyStats{
		101: {
			FieldID: 101,
		},
	}
	suite.putSegment(2, mockJSONKeyStats)
	tasks := checker.Check(context.Background())
	suite.Require().Len(tasks, 1)

	t := tasks[0]
	suite.Require().Len(t.Actions(), 1)

	action, ok := t.Actions()[0].(*task.SegmentAction)
	suite.Require().True(ok)
	suite.EqualValues(200, t.ReplicaID())
	suite.Equal(task.ActionTypeReopen, action.Type())
	suite.EqualValues(2, action.GetSegmentID())

	// test skip load json index for read only node
	suite.nodeMgr.Stopping(1)
	suite.nodeMgr.Stopping(2)
	suite.meta.HandleNodeStopping(ctx, 1)
	suite.meta.HandleNodeStopping(ctx, 2)
	utils.RecoverAllCollection(suite.meta)
	tasks = checker.Check(context.Background())
	suite.Require().Len(tasks, 0)
}

func (suite *IndexCheckerSuite) TestJsonIndexNotMatch() {
	checker := suite.checker
	ctx := context.Background()
	paramtable.Get().Save(paramtable.Get().CommonCfg.EnabledJSONKeyStats.Key, "true")
	defer paramtable.Get().Reset(paramtable.Get().CommonCfg.EnabledJSONKeyStats.Key)
	// meta
	coll := utils.CreateTestCollection(1, 1)
	coll.FieldIndexID = map[int64]int64{101: 1000}
	schema := &schemapb.CollectionSchema{
		Name: "test_loadJsonIndex",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 101, DataType: schemapb.DataType_JSON, Name: "JSON"},
		},
	}
	checker.meta.PutCollection(ctx, coll)
	suite.metaStore.PutCollection(&metacache.CollectionInfo{ID: coll.GetCollectionID(), Schema: schema})
	checker.meta.Put(ctx, utils.CreateTestReplica(200, 1, []int64{1, 2}))
	suite.nodeMgr.Add(session.NewNodeInfo(session.ImmutableNodeInfo{
		NodeID:   1,
		Address:  "localhost",
		Hostname: "localhost",
	}))
	suite.nodeMgr.Add(session.NewNodeInfo(session.ImmutableNodeInfo{
		NodeID:   2,
		Address:  "localhost",
		Hostname: "localhost",
	}))
	checker.meta.HandleNodeUp(ctx, 1)
	checker.meta.HandleNodeUp(ctx, 2)

	// dist
	checker.dist.SegmentDistManager.Update(1, meta.SeedInto(checker.dist, utils.CreateTestSegment(1, 1, 2, 1, 1, "test-insert-channel"))...)

	// store
	suite.metaStore.PutIndex(&model.Index{CollectionID: 1, FieldID: 101, IndexID: 1000})
	suite.putBuild(2, 1000, 5, commonpb.IndexState_InProgress)
	suite.putSegment(2, nil)
	tasks := checker.Check(context.Background())
	suite.Require().Len(tasks, 0)
}

func (suite *IndexCheckerSuite) TestCreateNewJsonIndex() {
	checker := suite.checker
	ctx := context.Background()
	paramtable.Get().Save(paramtable.Get().CommonCfg.EnabledJSONKeyStats.Key, "true")
	defer paramtable.Get().Reset(paramtable.Get().CommonCfg.EnabledJSONKeyStats.Key)
	// meta
	coll := utils.CreateTestCollection(1, 1)
	coll.FieldIndexID = map[int64]int64{101: 1000}
	coll.LoadFields = []int64{101}
	schema := &schemapb.CollectionSchema{
		Name: "test_loadJsonIndex",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 101, DataType: schemapb.DataType_JSON, Name: "JSON"},
		},
	}
	checker.meta.PutCollection(ctx, coll)
	suite.metaStore.PutCollection(&metacache.CollectionInfo{ID: coll.GetCollectionID(), Schema: schema})
	checker.meta.Put(ctx, utils.CreateTestReplica(200, 1, []int64{1, 2}))
	suite.nodeMgr.Add(session.NewNodeInfo(session.ImmutableNodeInfo{
		NodeID:   1,
		Address:  "localhost",
		Hostname: "localhost",
	}))
	suite.nodeMgr.Add(session.NewNodeInfo(session.ImmutableNodeInfo{
		NodeID:   2,
		Address:  "localhost",
		Hostname: "localhost",
	}))
	checker.meta.HandleNodeUp(ctx, 1)
	checker.meta.HandleNodeUp(ctx, 2)

	// dist
	fieldIndexInfo := &querypb.FieldIndexInfo{
		FieldID:     101,
		IndexID:     1000,
		EnableIndex: true,
	}

	segment := utils.CreateTestSegment(1, 1, 2, 1, 1, "test-insert-channel")
	segment.IndexBuilds = map[int64]int64{fieldIndexInfo.IndexID: fieldIndexInfo.BuildID}
	checker.dist.SegmentDistManager.Update(1, meta.SeedInto(checker.dist, segment)...)

	// store
	suite.metaStore.PutIndex(&model.Index{CollectionID: 1, FieldID: 101, IndexID: 1000})
	mockJSONKeyStats := map[int64]*datapb.JsonKeyStats{
		101: {
			FieldID: 101,
		},
	}
	suite.putSegment(2, mockJSONKeyStats)
	tasks := checker.Check(context.Background())
	suite.Len(tasks, 1)
	suite.Len(tasks[0].Actions(), 1)
	suite.Equal(tasks[0].Actions()[0].(*task.SegmentAction).Type(), task.ActionTypeReopen)
}

// setupLoadedIndex loads collection 1 on replica 200 (nodes 1 and 2) with
// the live index 1000 on field 101, and puts segment 2 into the store and
// onto node 1 with loaded builds.
func (suite *IndexCheckerSuite) setupLoadedIndex(builds map[int64]int64) {
	ctx := context.Background()
	coll := utils.CreateTestCollection(1, 1)
	coll.FieldIndexID = map[int64]int64{101: 1000}
	suite.checker.meta.PutCollection(ctx, coll)
	suite.metaStore.PutCollection(&metacache.CollectionInfo{ID: 1, Schema: &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{{FieldID: 101, DataType: schemapb.DataType_FloatVector, Name: "vec"}},
	}})
	suite.checker.meta.Put(ctx, utils.CreateTestReplica(200, 1, []int64{1, 2}))
	for _, node := range []int64{1, 2} {
		suite.nodeMgr.Add(session.NewNodeInfo(session.ImmutableNodeInfo{NodeID: node, Address: "localhost", Hostname: "localhost"}))
		suite.checker.meta.HandleNodeUp(ctx, node)
	}
	suite.metaStore.PutIndex(&model.Index{CollectionID: 1, FieldID: 101, IndexID: 1000})
	suite.putSegment(2, nil)
	segment := utils.CreateTestSegment(1, 1, 2, 1, 1, "test-insert-channel")
	segment.IndexBuilds = builds
	suite.checker.dist.SegmentDistManager.Update(1, meta.SeedInto(suite.checker.dist, segment)...)
}

func (suite *IndexCheckerSuite) TestLoadedCurrentBuildNoTask() {
	suite.setupLoadedIndex(map[int64]int64{1000: 9})
	suite.putFinishedBuild(2, 1000, 9)
	suite.Empty(suite.checker.Check(context.Background()))
	suite.Empty(suite.checker.Check(context.Background()), "a loaded current build never asks for a reopen")
}

func (suite *IndexCheckerSuite) TestMissingIndexWaitsForReadyBuild() {
	suite.setupLoadedIndex(nil)
	for _, state := range []commonpb.IndexState{commonpb.IndexState_Unissued, commonpb.IndexState_InProgress, commonpb.IndexState_Failed} {
		suite.putBuild(2, 1000, 9, state)
		suite.Empty(suite.checker.Check(context.Background()), state.String())
	}
	suite.metaStore.PutSegmentIndexes(&model.SegmentIndex{
		CollectionID: 1, SegmentID: 2, IndexID: 1000, BuildID: 9, IndexState: commonpb.IndexState_Finished,
	})
	suite.Empty(suite.checker.Check(context.Background()), "a finished build without files is not loadable")

	suite.putFinishedBuild(2, 1000, 9)
	tasks := suite.checker.Check(context.Background())
	suite.Require().Len(tasks, 1)
	suite.Equal(task.ActionTypeReopen, tasks[0].Actions()[0].(*task.SegmentAction).Type())
}

// A reopen loads only index IDs the segment lacks (segcore diffs by index
// ID), so a newer build of an index the node already loaded must not ask
// for one: it would load nothing and only relabel the loaded build.
func (suite *IndexCheckerSuite) TestNewerBuildOfLoadedIndexNoTask() {
	suite.setupLoadedIndex(map[int64]int64{1000: 7})
	suite.putFinishedBuild(2, 1000, 9)
	suite.Empty(suite.checker.Check(context.Background()))
}

// A newer build still in progress must not hold back a JSON stats reopen.
func (suite *IndexCheckerSuite) TestNewerBuildInProgressAllowsJSONStatsReopen() {
	paramtable.Get().Save(paramtable.Get().CommonCfg.EnabledJSONKeyStats.Key, "true")
	defer paramtable.Get().Reset(paramtable.Get().CommonCfg.EnabledJSONKeyStats.Key)
	suite.setupLoadedIndex(map[int64]int64{1000: 7})
	coll := utils.CreateTestCollection(1, 1)
	coll.FieldIndexID = map[int64]int64{101: 1000}
	coll.LoadFields = []int64{101}
	suite.checker.meta.PutCollection(context.Background(), coll)
	suite.metaStore.PutCollection(&metacache.CollectionInfo{ID: 1, Schema: &schemapb.CollectionSchema{
		Fields: []*schemapb.FieldSchema{{FieldID: 101, DataType: schemapb.DataType_JSON, Name: "JSON"}},
	}})
	suite.putSegment(2, map[int64]*datapb.JsonKeyStats{101: {FieldID: 101}})
	suite.putBuild(2, 1000, 9, commonpb.IndexState_InProgress)
	suite.Len(suite.checker.Check(context.Background()), 1)
}

func (suite *IndexCheckerSuite) TestDroppedIndexNotRequired() {
	suite.setupLoadedIndex(map[int64]int64{1000: 9, 1001: 4})
	suite.putFinishedBuild(2, 1000, 9)
	suite.metaStore.PutIndex(&model.Index{CollectionID: 1, FieldID: 102, IndexID: 1001, IsDeleted: true})
	suite.putFinishedBuild(2, 1001, 4)
	// 1001 is dropped: it is not required, and the loaded copy is redundant.
	tasks := suite.checker.Check(context.Background())
	suite.Require().Len(tasks, 1)
	suite.Equal("missing index", tasks[0].GetReason())
}

func (suite *IndexCheckerSuite) TestSegmentMissingFromStore() {
	suite.setupLoadedIndex(nil)
	suite.metaStore.RemoveSegment(2)
	suite.putFinishedBuild(2, 1000, 9)
	suite.Empty(suite.checker.Check(context.Background()))
}

func TestIndexChecker(t *testing.T) {
	suite.Run(t, new(IndexCheckerSuite))
}
