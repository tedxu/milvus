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

package meta

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/internal/metacache"
	"github.com/milvus-io/milvus/internal/metacache/metacachetest"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/metricsinfo"
)

type SegmentDistManagerSuite struct {
	suite.Suite
	dist       *SegmentDistManager
	collection int64
	partitions []int64
	nodes      []int64
	segments   map[int64]*LoadedSegment
}

type countingSegmentDistFilter struct {
	count *int
}

func (f countingSegmentDistFilter) Match(*LoadedSegment) bool {
	*f.count++
	return true
}

func (f countingSegmentDistFilter) AddFilter(criterion *segDistCriterion) {
	criterion.hasOtherFilter = true
}

func (suite *SegmentDistManagerSuite) SetupSuite() {
	const (
		shardNum = 2
	)
	// Replica 0: 0, 2
	// Replica 1: 1
	suite.collection = 10
	suite.partitions = []int64{1, 2}
	suite.nodes = []int64{0, 1, 2}
	suite.segments = map[int64]*LoadedSegment{
		1: {ID: 1, CollectionID: suite.collection, Channel: "dmc0"},
		2: {ID: 2, CollectionID: suite.collection, Channel: "dmc1"},
		3: {ID: 3, CollectionID: suite.collection, Channel: "dmc0"},
		4: {ID: 4, CollectionID: suite.collection, Channel: "dmc1"},
	}
}

func (suite *SegmentDistManagerSuite) SetupTest() {
	suite.dist = NewSegmentDistManager()
	// Distribution:
	// node 0 contains channel segment 1, 2
	// node 1 contains channel segment 1, 2, 3, 4
	// node 2 contains channel segment 3, 4
	suite.dist.Update(suite.nodes[0], cloneLoaded(suite.segments[1]), cloneLoaded(suite.segments[2]))
	suite.dist.Update(suite.nodes[1],
		cloneLoaded(suite.segments[1]),
		cloneLoaded(suite.segments[2]),
		cloneLoaded(suite.segments[3]),
		cloneLoaded(suite.segments[4]))
	suite.dist.Update(suite.nodes[2], cloneLoaded(suite.segments[3]), cloneLoaded(suite.segments[4]))
}

func (suite *SegmentDistManagerSuite) TestVersion() {
	dist := suite.dist
	v1 := dist.GetVersion()

	// Update with some new data
	dist.Update(suite.nodes[0], cloneLoaded(suite.segments[1]), cloneLoaded(suite.segments[2]), cloneLoaded(suite.segments[3]))
	v2 := dist.GetVersion()
	suite.Greater(v2, v1)
}

func (suite *SegmentDistManagerSuite) TestPatch() {
	suite.Run("empty delta", func() {
		suite.SetupTest()
		dist := suite.dist
		v1 := dist.GetVersion()
		before := dist.GetByFilter(WithNodeID(suite.nodes[0]))

		dist.Patch(suite.nodes[0], nil, nil)

		v2 := dist.GetVersion()
		after := dist.GetByFilter(WithNodeID(suite.nodes[0]))
		suite.Equal(v1, v2)
		suite.Equal(before, after)
	})

	suite.Run("remove last segment", func() {
		suite.SetupTest()
		dist := suite.dist
		v1 := dist.GetVersion()

		dist.Patch(suite.nodes[0], nil, []int64{1, 2})

		v2 := dist.GetVersion()
		segments := dist.GetByFilter(WithNodeID(suite.nodes[0]))
		suite.Greater(v2, v1)
		suite.Empty(segments)
	})

	suite.Run("upsert segments", func() {
		suite.SetupTest()
		dist := suite.dist
		v1 := dist.GetVersion()
		replaced := cloneLoaded(suite.segments[1])
		replaced.Version = 7
		added := &LoadedSegment{ID: 5, CollectionID: suite.collection, Channel: "dmc2"}

		dist.Patch(suite.nodes[0], []*LoadedSegment{replaced, added}, nil)

		v2 := dist.GetVersion()
		segments := dist.GetByFilter(WithNodeID(suite.nodes[0]))
		suite.Greater(v2, v1)
		suite.Len(segments, 3)
		suite.AssertIDs(segments, 1, 2, 5)
		suite.AssertNode(segments, suite.nodes[0])
		segment1 := dist.GetByFilter(WithNodeID(suite.nodes[0]), WithSegmentID(1))
		suite.Require().Len(segment1, 1)
		suite.Equal(int64(7), segment1[0].Version)
		segment5 := dist.GetByFilter(WithNodeID(suite.nodes[0]), WithSegmentID(5))
		suite.Require().Len(segment5, 1)
		suite.Equal("dmc2", segment5[0].GetInsertChannel())
	})
}

func (suite *SegmentDistManagerSuite) TestNodeOffline() {
	dist := suite.dist

	// Verify node 0 has segments
	segments := dist.GetByFilter(WithNodeID(suite.nodes[0]))
	suite.Len(segments, 2)

	// Simulate node offline by calling Update with empty segments
	dist.Update(suite.nodes[0])

	// Verify node 0's segments are removed
	segments = dist.GetByFilter(WithNodeID(suite.nodes[0]))
	suite.Len(segments, 0)

	// Verify other nodes are not affected
	segments = dist.GetByFilter(WithNodeID(suite.nodes[1]))
	suite.Len(segments, 4)
}

func (suite *SegmentDistManagerSuite) TestGetBy() {
	dist := suite.dist
	// Test GetByNode
	for _, node := range suite.nodes {
		segments := dist.GetByFilter(WithNodeID(node))
		suite.AssertNode(segments, node)
	}

	// Test GetByShard
	for _, shard := range []string{"dmc0", "dmc1"} {
		segments := dist.GetByFilter(WithChannel(shard))
		suite.AssertShard(segments, shard)
	}

	// Test GetByCollection
	segments := dist.GetByFilter(WithCollectionID(suite.collection))
	suite.Len(segments, 8)
	suite.AssertCollection(segments, suite.collection)
	segments = dist.GetByFilter(WithCollectionID(-1))
	suite.Len(segments, 0)

	// Test GetByNodeAndCollection
	// 1. Valid node and valid collection
	for _, node := range suite.nodes {
		segments := dist.GetByFilter(WithCollectionID(suite.collection), WithNodeID(node))
		suite.AssertNode(segments, node)
		suite.AssertCollection(segments, suite.collection)
	}

	// 2. Valid node and invalid collection
	segments = dist.GetByFilter(WithCollectionID(-1), WithNodeID(suite.nodes[1]))
	suite.Len(segments, 0)

	// 3. Invalid node and valid collection
	segments = dist.GetByFilter(WithCollectionID(suite.collection), WithNodeID(-1))
	suite.Len(segments, 0)

	// Test GetBy With Wrong Replica
	replica := newReplica(&querypb.Replica{
		ID:           1,
		CollectionID: suite.collection + 1,
		Nodes:        []int64{suite.nodes[0]},
	})
	segments = dist.GetByFilter(WithReplica(replica))
	suite.Len(segments, 0)

	// Test GetBy With Correct Replica
	replica = newReplica(&querypb.Replica{
		ID:           1,
		CollectionID: suite.collection,
		Nodes:        []int64{suite.nodes[0]},
	})
	segments = dist.GetByFilter(WithReplica(replica))
	suite.Len(segments, 2)

	// Test GetBySegment
	segments = dist.GetByFilter(WithSegmentID(1))
	suite.Len(segments, 2)
	suite.AssertIDs(segments, 1)

	segments = dist.GetByFilter(WithCollectionID(-1), WithSegmentID(1))
	suite.Len(segments, 0)

	segments = dist.GetByFilter(WithNodeID(suite.nodes[2]), WithSegmentID(1))
	suite.Len(segments, 0)
}

func (suite *SegmentDistManagerSuite) TestGetBySegmentIDUsesIndex() {
	visited := 0

	segments := suite.dist.GetByFilter(countingSegmentDistFilter{count: &visited}, WithSegmentID(4))

	suite.Len(segments, 2)
	suite.AssertIDs(segments, 4)
	suite.Equal(2, visited)
}

func (suite *SegmentDistManagerSuite) AssertIDs(segments []*LoadedSegment, ids ...int64) bool {
	for _, segment := range segments {
		hasSegment := false
		for _, id := range ids {
			if segment.ID == id {
				hasSegment = true
				break
			}
		}
		if !suite.True(hasSegment, "segment %v not in the given expected list %+v", segment.GetID(), ids) {
			return false
		}
	}
	return true
}

func (suite *SegmentDistManagerSuite) AssertNode(segments []*LoadedSegment, node int64) bool {
	for _, segment := range segments {
		if !suite.Equal(node, segment.Node) {
			return false
		}
	}
	return true
}

func (suite *SegmentDistManagerSuite) AssertCollection(segments []*LoadedSegment, collection int64) bool {
	for _, segment := range segments {
		if !suite.Equal(collection, segment.GetCollectionID()) {
			return false
		}
	}
	return true
}

func (suite *SegmentDistManagerSuite) AssertShard(segments []*LoadedSegment, shard string) bool {
	for _, segment := range segments {
		if !suite.Equal(shard, segment.GetInsertChannel()) {
			return false
		}
	}
	return true
}

func TestSegmentDistManager(t *testing.T) {
	suite.Run(t, new(SegmentDistManagerSuite))
}

func TestGetSegmentDistJSON(t *testing.T) {
	// Initialize SegmentDistManager
	manager := NewSegmentDistManager()

	// Add some segments to the SegmentDistManager, with their fields in the store
	store := metacachetest.NewMetaStore()
	records := SeedSegments(store,
		&SegmentFixture{SegmentInfo: &datapb.SegmentInfo{ID: 1, CollectionID: 100, PartitionID: 10, InsertChannel: "channel-1", NumOfRows: 1000, State: commonpb.SegmentState_Flushed}, Node: 1, Version: 1},
		&SegmentFixture{SegmentInfo: &datapb.SegmentInfo{ID: 2, CollectionID: 200, PartitionID: 20, InsertChannel: "channel-2", NumOfRows: 2000, State: commonpb.SegmentState_Flushed}, Node: 2, Version: 1},
	)

	manager.Update(1, records[0])
	manager.Update(2, records[1])

	segments := manager.GetSegmentDist(0, store)
	assert.Equal(t, 2, len(segments))

	checkResults := func(s *metricsinfo.Segment) {
		switch s.SegmentID {
		case 1:
			assert.Equal(t, int64(100), s.CollectionID)
			assert.Equal(t, int64(10), s.PartitionID)
			assert.Equal(t, "channel-1", s.Channel)
			assert.Equal(t, int64(1000), s.NumOfRows)
			assert.Equal(t, "Flushed", s.State)
			assert.Equal(t, int64(1), s.NodeID)
		case 2:
			assert.Equal(t, int64(200), s.CollectionID)
			assert.Equal(t, int64(20), s.PartitionID)
			assert.Equal(t, "channel-2", s.Channel)
			assert.Equal(t, int64(2000), s.NumOfRows)
			assert.Equal(t, "Flushed", s.State)
			assert.Equal(t, int64(2), s.NodeID)
		default:
			assert.Failf(t, "unexpected segment id", "unexpected segment id %d", s.SegmentID)
		}
	}

	for _, s := range segments {
		checkResults(s)
	}
}

func TestGetSegmentDistReadsIndexFromStore(t *testing.T) {
	store := metacachetest.NewMetaStore()
	store.PutIndex(&model.Index{CollectionID: 100, FieldID: 101, IndexID: 7})
	store.PutSegmentIndexes(&model.SegmentIndex{CollectionID: 100, SegmentID: 1, IndexID: 7, BuildID: 70, IndexMemSize: 4096})

	manager := NewSegmentDistManager()
	segment := &LoadedSegment{ID: 1, CollectionID: 100, Channel: "ch"}
	segment.IndexBuilds = map[int64]int64{7: 70, 8: 80}
	manager.Update(1, segment)

	rows := manager.GetSegmentDist(0, store)
	require.Len(t, rows, 1)
	fields := rows[0].IndexedFields
	require.Len(t, fields, 2)
	assert.Equal(t, metricsinfo.IndexedField{IndexFieldID: 101, IndexID: 7, BuildID: 70, IndexSize: 4096, IsLoaded: true}, *fields[0])
	// No definition or record in the store: the loaded build is still listed.
	assert.Equal(t, metricsinfo.IndexedField{IndexID: 8, BuildID: 80, IsLoaded: true}, *fields[1])
}

func TestLoadedFieldIndexInfos(t *testing.T) {
	store := metacachetest.NewMetaStore()
	store.PutIndex(&model.Index{CollectionID: 1, FieldID: 101, IndexID: 100, IndexName: "vec"})
	store.PutIndex(&model.Index{CollectionID: 1, FieldID: 102, IndexID: 300, IndexName: "no_record"})
	store.PutSegmentIndexes(&model.SegmentIndex{
		CollectionID: 1, SegmentID: 10, IndexID: 100, BuildID: 9,
		IndexMemSize: 4096, NumRows: 1000, IndexVersion: 2,
		CurrentIndexVersion: 5, CurrentScalarIndexVersion: 6,
		IndexFileKeys: []string{"f1"},
	})

	cases := []struct {
		name   string
		builds map[int64]int64
		want   []*querypb.FieldIndexInfo
	}{
		{
			name:   "definition and record",
			builds: map[int64]int64{100: 7},
			want: []*querypb.FieldIndexInfo{{
				FieldID: 101, EnableIndex: true, IndexName: "vec", IndexID: 100, BuildID: 7,
				IndexSize: 4096, NumRows: 1000, IndexVersion: 2,
				CurrentIndexVersion: 5, CurrentScalarIndexVersion: 6,
			}},
		},
		{
			name:   "definition missing",
			builds: map[int64]int64{200: 8},
			want:   []*querypb.FieldIndexInfo{{EnableIndex: true, IndexID: 200, BuildID: 8}},
		},
		{
			name:   "record missing",
			builds: map[int64]int64{300: 3},
			want:   []*querypb.FieldIndexInfo{{FieldID: 102, EnableIndex: true, IndexName: "no_record", IndexID: 300, BuildID: 3}},
		},
		{
			name:   "sorted by index id",
			builds: map[int64]int64{300: 3, 200: 8},
			want: []*querypb.FieldIndexInfo{
				{EnableIndex: true, IndexID: 200, BuildID: 8},
				{FieldID: 102, EnableIndex: true, IndexName: "no_record", IndexID: 300, BuildID: 3},
			},
		},
		{name: "nothing loaded", builds: nil, want: []*querypb.FieldIndexInfo{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := LoadedFieldIndexInfos(store, 1, 10, tc.builds)
			assert.Equal(t, len(tc.want), len(got))
			for i := range tc.want {
				assert.True(t, proto.Equal(tc.want[i], got[i]), "index %d: want %v, got %v", i, tc.want[i], got[i])
			}
		})
	}
}

func cloneLoaded(s *LoadedSegment) *LoadedSegment {
	c := *s
	return &c
}

// TestSegmentDistMetricsStoreMiss pins that a loaded segment the store no
// longer has reports only its identity and node.
func TestSegmentDistMetricsStoreMiss(t *testing.T) {
	dist := NewSegmentDistManager()
	dist.Update(3, &LoadedSegment{ID: 7, CollectionID: 1, Channel: "ch", Node: 3})

	metrics := dist.GetSegmentDist(1, metacache.NewMetaStore(nil))

	require.Len(t, metrics, 1)
	assert.Equal(t, int64(7), metrics[0].SegmentID)
	assert.Equal(t, int64(1), metrics[0].CollectionID)
	assert.Equal(t, "ch", metrics[0].Channel)
	assert.Equal(t, int64(3), metrics[0].NodeID)
	assert.Zero(t, metrics[0].NumOfRows)
	assert.Zero(t, metrics[0].PartitionID)
}

// TestSegmentDistMetricsReportsStoreFlags pins that metrics carry the store's
// state and flags, which the projected copy reported as zero.
func TestSegmentDistMetricsReportsStoreFlags(t *testing.T) {
	store := metacache.NewMetaStore(nil)
	store.PutSegment(&datapb.SegmentInfo{
		ID: 7, CollectionID: 1, PartitionID: 2, InsertChannel: "ch", NumOfRows: 100,
		State: commonpb.SegmentState_Flushed, IsImporting: true, Level: datapb.SegmentLevel_L1, IsSorted: true,
	})
	dist := NewSegmentDistManager()
	dist.Update(3, &LoadedSegment{ID: 7, CollectionID: 1, Channel: "ch", Node: 3})

	metrics := dist.GetSegmentDist(1, store)

	require.Len(t, metrics, 1)
	assert.Equal(t, commonpb.SegmentState_Flushed.String(), metrics[0].State)
	assert.True(t, metrics[0].IsImporting)
	assert.Equal(t, int64(2), metrics[0].PartitionID)
	assert.Equal(t, int64(100), metrics[0].NumOfRows)
	assert.True(t, metrics[0].IsSorted)
}
