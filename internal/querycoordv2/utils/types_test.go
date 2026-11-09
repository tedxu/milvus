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

package utils

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/msgpb"
	"github.com/milvus-io/milvus/internal/metacache/metacachetest"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/tsoutil"
)

func Test_packLoadSegmentRequest(t *testing.T) {
	mockVChannel := "fake-by-dev-rootcoord-dml-1-test-packLoadSegmentRequest-v0"
	mockPChannel := "fake-by-dev-rootcoord-dml-1"

	t0 := tsoutil.ComposeTSByTime(time.Now().Add(-20 * time.Minute))
	t1 := tsoutil.ComposeTSByTime(time.Now().Add(-8 * time.Minute))
	t2 := tsoutil.ComposeTSByTime(time.Now().Add(-5 * time.Minute))
	t3 := tsoutil.ComposeTSByTime(time.Now().Add(-1 * time.Minute))

	segmentInfo := &datapb.SegmentInfo{
		ID:            0,
		InsertChannel: mockVChannel,
		StartPosition: &msgpb.MsgPosition{
			ChannelName: mockPChannel,
			Timestamp:   t1,
		},
		DmlPosition: &msgpb.MsgPosition{
			ChannelName: mockPChannel,
			Timestamp:   t2,
		},
	}

	channel := &datapb.VchannelInfo{
		ChannelName: mockVChannel,
		SeekPosition: &msgpb.MsgPosition{
			ChannelName: mockPChannel,
			Timestamp:   t2,
		},
	}

	t.Run("test set deltaPosition from channel seek position", func(t *testing.T) {
		req := PackSegmentLoadInfo(segmentInfo, channel.GetSeekPosition(), nil)
		assert.NotNil(t, req.GetDeltaPosition())
		assert.Equal(t, mockPChannel, req.GetDeltaPosition().ChannelName)
		assert.Equal(t, t2, req.GetDeltaPosition().Timestamp)
	})

	t.Run("test channel cp after segment dml position", func(t *testing.T) {
		channel := proto.Clone(channel).(*datapb.VchannelInfo)
		channel.SeekPosition.Timestamp = t3
		req := PackSegmentLoadInfo(segmentInfo, channel.GetSeekPosition(), nil)
		assert.NotNil(t, req.GetDeltaPosition())
		assert.Equal(t, mockPChannel, req.GetDeltaPosition().ChannelName)
		assert.Equal(t, t3, req.GetDeltaPosition().Timestamp)
	})

	t.Run("test tsLag > 10minutes", func(t *testing.T) {
		channel := proto.Clone(channel).(*datapb.VchannelInfo)
		channel.SeekPosition.Timestamp = t0
		req := PackSegmentLoadInfo(segmentInfo, channel.GetSeekPosition(), nil)
		assert.NotNil(t, req.GetDeltaPosition())
		assert.Equal(t, mockPChannel, req.GetDeltaPosition().ChannelName)
		assert.Equal(t, channel.SeekPosition.Timestamp, req.GetDeltaPosition().GetTimestamp())
	})
}

func TestPackSegmentLoadInfo_ManifestPath(t *testing.T) {
	mockPChannel := "fake-by-dev-rootcoord-dml-1"
	checkpoint := &msgpb.MsgPosition{
		ChannelName: mockPChannel,
		Timestamp:   tsoutil.ComposeTSByTime(time.Now().Add(-1 * time.Minute)),
	}

	t.Run("manifest set clears legacy stats fields and keeps json stats placeholders", func(t *testing.T) {
		childManifestPaths := []string{`{"ver":4,"base_path":"files/insert_log/100/10/2001"}`}
		seg := &datapb.SegmentInfo{
			ID:           100,
			ManifestPath: "base/path@5",
			Stats: &datapb.Statistics{
				InsertBinlogSize: 1024,
				LoadResource: &datapb.LoadResourceStatistics{
					ColumnGroups: []*datapb.ColumnGroupStatistics{{GroupId: 0, FieldIds: []int64{10}, MemorySize: 1024}},
				},
			},
			ChildManifestPaths: childManifestPaths,
			Statslogs: []*datapb.FieldBinlog{
				{FieldID: 1},
			},
			Bm25Statslogs: []*datapb.FieldBinlog{
				{FieldID: 2},
			},
			TextStatsLogs: map[int64]*datapb.TextIndexStats{
				10: {FieldID: 10},
			},
			JsonKeyStats: map[int64]*datapb.JsonKeyStats{
				20: {FieldID: 20},
			},
			Deltalogs: []*datapb.FieldBinlog{
				{FieldID: 0, Binlogs: []*datapb.Binlog{{LogPath: "delta/1"}}},
			},
		}
		loadInfo := PackSegmentLoadInfo(seg, checkpoint, nil)

		assert.Equal(t, "base/path@5", loadInfo.GetManifestPath())
		assert.Equal(t, seg.GetStats(), loadInfo.GetStats())
		assert.Equal(t, childManifestPaths, loadInfo.GetChildManifestPaths())
		assert.Empty(t, loadInfo.GetStatslogs())
		assert.Empty(t, loadInfo.GetBm25Logs())
		assert.Equal(t, seg.GetTextStatsLogs(), loadInfo.GetTextStatsLogs())
		assert.Equal(t, seg.GetJsonKeyStats(), loadInfo.GetJsonKeyStatsLogs())
		// Deltalogs should always be populated
		assert.NotEmpty(t, loadInfo.GetDeltalogs())
	})

	t.Run("no manifest populates all legacy fields", func(t *testing.T) {
		seg := &datapb.SegmentInfo{
			ID: 200,
			Statslogs: []*datapb.FieldBinlog{
				{FieldID: 1},
			},
			Bm25Statslogs: []*datapb.FieldBinlog{
				{FieldID: 2},
			},
			TextStatsLogs: map[int64]*datapb.TextIndexStats{
				10: {FieldID: 10},
			},
			JsonKeyStats: map[int64]*datapb.JsonKeyStats{
				20: {FieldID: 20},
			},
			Deltalogs: []*datapb.FieldBinlog{
				{FieldID: 0, Binlogs: []*datapb.Binlog{{LogPath: "delta/1"}}},
			},
		}
		loadInfo := PackSegmentLoadInfo(seg, checkpoint, nil)

		assert.Empty(t, loadInfo.GetManifestPath())
		assert.NotEmpty(t, loadInfo.GetStatslogs())
		assert.NotEmpty(t, loadInfo.GetBm25Logs())
		assert.NotEmpty(t, loadInfo.GetTextStatsLogs())
		assert.NotEmpty(t, loadInfo.GetJsonKeyStatsLogs())
		assert.NotEmpty(t, loadInfo.GetDeltalogs())
	})
}

func TestPackSegmentLoadInfo_CommitTimestamp(t *testing.T) {
	const commitTs uint64 = 99999
	const dataVersion int32 = 7

	seg := &datapb.SegmentInfo{
		ID:              1,
		CollectionID:    10,
		PartitionID:     100,
		InsertChannel:   "ch1",
		CommitTimestamp: commitTs,
		DataVersion:     dataVersion,
		StartPosition:   &msgpb.MsgPosition{Timestamp: 1000},
	}

	checkpoint := &msgpb.MsgPosition{Timestamp: 2000}
	loadInfo := PackSegmentLoadInfo(seg, checkpoint, nil)
	assert.Equal(t, commitTs, loadInfo.GetCommitTimestamp())
	assert.Equal(t, dataVersion, loadInfo.GetDataVersion())
}

func TestMergeMetaSegmentIntoSegmentInfoReadsStore(t *testing.T) {
	store := metacachetest.NewMetaStore()
	store.PutIndex(&model.Index{CollectionID: 1, FieldID: 101, IndexID: 100, IndexName: "vec"})
	store.PutSegmentIndexes(&model.SegmentIndex{CollectionID: 1, SegmentID: 10, IndexID: 100, BuildID: 9})

	segment := meta.SeedSegments(store, &meta.SegmentFixture{
		SegmentInfo: &datapb.SegmentInfo{ID: 10, CollectionID: 1, InsertChannel: "ch"},
		Node:        3,
		IndexBuilds: map[int64]int64{100: 9},
	})[0]
	info := &querypb.SegmentInfo{}
	MergeMetaSegmentIntoSegmentInfo(store, info, segment)

	assert.Equal(t, "vec", info.GetIndexName())
	assert.Equal(t, int64(100), info.GetIndexID())
	assert.Len(t, info.GetIndexInfos(), 1)
	assert.Equal(t, int64(101), info.GetIndexInfos()[0].GetFieldID())
	assert.Empty(t, info.GetIndexInfos()[0].GetIndexFilePaths())
	assert.Equal(t, []int64{3}, info.GetNodeIds())
}

// TestGetSegmentInfoStoreMiss pins that a loaded segment the store no
// longer has reports only its identity and nodes.
func TestGetSegmentInfoStoreMiss(t *testing.T) {
	info := &querypb.SegmentInfo{}
	MergeMetaSegmentIntoSegmentInfo(metacachetest.NewMetaStore(), info, &meta.LoadedSegment{ID: 7, CollectionID: 1, Channel: "ch", Node: 3})

	assert.Equal(t, int64(7), info.GetSegmentID())
	assert.Equal(t, int64(1), info.GetCollectionID())
	assert.Equal(t, "ch", info.GetDmChannel())
	assert.Equal(t, []int64{3}, info.GetNodeIds())
	assert.Zero(t, info.GetNumRows())
	assert.Zero(t, info.GetPartitionID())
}

// TestGetSegmentInfoReportsStoreFlags pins that segment fields come from
// the store, not from the distribution record.
func TestGetSegmentInfoReportsStoreFlags(t *testing.T) {
	store := metacachetest.NewMetaStore()
	records := meta.SeedSegments(store, &meta.SegmentFixture{
		SegmentInfo: &datapb.SegmentInfo{ID: 7, CollectionID: 1, PartitionID: 2, InsertChannel: "ch", NumOfRows: 100, Level: datapb.SegmentLevel_L1, IsSorted: true},
		Node:        3,
	})
	info := &querypb.SegmentInfo{}
	MergeMetaSegmentIntoSegmentInfo(store, info, records...)

	assert.Equal(t, int64(2), info.GetPartitionID())
	assert.Equal(t, int64(100), info.GetNumRows())
	assert.Equal(t, datapb.SegmentLevel_L1, info.GetLevel())
	assert.True(t, info.GetIsSorted())
}
