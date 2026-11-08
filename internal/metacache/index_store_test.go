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

package metacache

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/metastore/model"
)

func segIdx(segmentID, indexID, buildID int64) *model.SegmentIndex {
	return &model.SegmentIndex{CollectionID: 1, SegmentID: segmentID, IndexID: indexID, BuildID: buildID}
}

func TestIndexStorePutGetRemoveIndex(t *testing.T) {
	s := NewMetaStore(nil)
	live := &model.Index{CollectionID: 1, IndexID: 100}
	deleted := &model.Index{CollectionID: 1, IndexID: 101, IsDeleted: true}

	assert.Nil(t, s.PutIndex(live))
	assert.Nil(t, s.PutIndex(deleted))
	replacement := &model.Index{CollectionID: 1, IndexID: 100, IndexName: "renamed"}
	assert.Same(t, live, s.PutIndex(replacement))

	got, ok := s.GetIndex(1, 100)
	require.True(t, ok)
	assert.Same(t, replacement, got)
	assert.ElementsMatch(t, []*model.Index{replacement, deleted}, s.GetIndexes(1), "deleted definitions are included")
	assert.ElementsMatch(t, []*model.Index{replacement, deleted}, s.ListAllIndexes())

	assert.Same(t, deleted, s.RemoveIndex(1, 101))
	assert.Nil(t, s.RemoveIndex(1, 101))
	assert.Same(t, replacement, s.RemoveIndex(1, 100))
	assert.Empty(t, s.GetIndexes(1))
	assert.Empty(t, s.ListAllIndexes())
	_, ok = s.GetIndex(1, 100)
	assert.False(t, ok)
}

func TestIndexStoreSupersededBuild(t *testing.T) {
	s := NewMetaStore(nil)
	build1, build2 := segIdx(10, 100, 1), segIdx(10, 100, 2)
	s.PutSegmentIndexes(build1)
	s.PutSegmentIndexes(build2)

	current, ok := s.GetSegmentIndex(10, 100)
	require.True(t, ok)
	assert.Equal(t, int64(2), current.BuildID)
	old, ok := s.GetSegmentIndexByBuildID(1)
	require.True(t, ok, "a superseded build stays findable by build ID for GC")
	assert.Same(t, build1, old)
	assert.ElementsMatch(t, []*model.SegmentIndex{build1, build2}, s.ListSegmentIndexBuilds())

	assert.True(t, s.RemoveSegmentIndex(build1))
	current, ok = s.GetSegmentIndex(10, 100)
	require.True(t, ok, "removing a superseded build keeps the current one")
	assert.Equal(t, int64(2), current.BuildID)
	_, ok = s.GetSegmentIndexByBuildID(1)
	assert.False(t, ok)

	assert.True(t, s.RemoveSegmentIndex(build2))
	assert.False(t, s.RemoveSegmentIndex(build2))
	assert.Empty(t, s.GetSegmentIndexes(10))
	assert.Empty(t, s.ListSegmentIndexBuilds())
}

func TestIndexStoreLoadPicksNewestBuild(t *testing.T) {
	s := NewMetaStore(nil)
	b7, b9, b8 := segIdx(10, 100, 7), segIdx(10, 100, 9), segIdx(10, 100, 8)
	other := segIdx(11, 100, 3)
	def := &model.Index{CollectionID: 1, IndexID: 100}
	s.LoadIndexes([]*model.Index{def}, []*model.SegmentIndex{b7, b9, b8, other})

	current, ok := s.GetSegmentIndex(10, 100)
	require.True(t, ok)
	assert.Equal(t, int64(9), current.BuildID)
	for _, b := range []*model.SegmentIndex{b7, b8, b9, other} {
		got, ok := s.GetSegmentIndexByBuildID(b.BuildID)
		require.True(t, ok, "build %d", b.BuildID)
		assert.Same(t, b, got)
	}
	got, ok := s.GetIndex(1, 100)
	require.True(t, ok)
	assert.Same(t, def, got)
}

func TestIndexStorePutSegmentIndexesReturnsOlds(t *testing.T) {
	s := NewMetaStore(nil)
	first := segIdx(10, 100, 5)
	assert.Equal(t, []*model.SegmentIndex{nil, nil}, s.PutSegmentIndexes(first, segIdx(10, 101, 6)))

	updated := segIdx(10, 100, 5)
	updated.NodeID = 3
	olds := s.PutSegmentIndexes(segIdx(11, 100, 7), updated)
	require.Len(t, olds, 2)
	assert.Nil(t, olds[0])
	assert.Same(t, first, olds[1])
	got, _ := s.GetSegmentIndexByBuildID(5)
	assert.Same(t, updated, got)
	assert.Len(t, s.GetSegmentIndexes(10), 2)
}

func TestIndexStoreSnapshot(t *testing.T) {
	s := NewMetaStore(nil)
	live := &model.Index{CollectionID: 1, IndexID: 100}
	deleted := &model.Index{CollectionID: 1, IndexID: 101, IsDeleted: true}
	otherColl := &model.Index{CollectionID: 2, IndexID: 200}
	for _, idx := range []*model.Index{live, deleted, otherColl} {
		s.PutIndex(idx)
	}
	r10, r11, r12 := segIdx(10, 100, 1), segIdx(11, 100, 2), segIdx(12, 100, 3)
	s.PutSegmentIndexes(r10, r11, r12)

	defs, records := s.GetIndexSnapshot(1, []int64{10, 11, 99})
	assert.ElementsMatch(t, []*model.Index{live, deleted}, defs)
	assert.Equal(t, map[int64]map[int64]*model.SegmentIndex{
		10: {100: r10},
		11: {100: r11},
	}, records)
}

func TestIndexStoreUpdateSegmentIndexBuild(t *testing.T) {
	cases := []struct {
		name        string
		updateBuild int64
		wantCurrent int64
	}{
		{name: "current build replaces the current record", updateBuild: 2, wantCurrent: 2},
		{name: "superseded build does not become current", updateBuild: 1, wantCurrent: 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := NewMetaStore(nil)
			s.PutSegmentIndexes(segIdx(10, 100, 1))
			s.PutSegmentIndexes(segIdx(10, 100, 2))

			updated := segIdx(10, 100, tc.updateBuild)
			updated.ManifestPublished = true
			s.UpdateSegmentIndexBuild(updated)

			byBuild, ok := s.GetSegmentIndexByBuildID(tc.updateBuild)
			require.True(t, ok)
			assert.Same(t, updated, byBuild, "the build lookup always takes the update")
			current, ok := s.GetSegmentIndex(10, 100)
			require.True(t, ok)
			assert.Equal(t, tc.wantCurrent, current.BuildID)
			assert.Equal(t, tc.updateBuild == tc.wantCurrent, current.ManifestPublished)
		})
	}
}

// DataCoord retries recovery against the same store, so a load must replace
// whatever an earlier, failed attempt left behind.
func TestIndexStoreLoadReplacesPreviousContents(t *testing.T) {
	s := NewMetaStore(nil)
	s.LoadIndexes(
		[]*model.Index{{CollectionID: 1, IndexID: 100}, {CollectionID: 2, IndexID: 200}},
		[]*model.SegmentIndex{segIdx(10, 100, 1), segIdx(20, 200, 2)},
	)
	s.PutSegmentIndexes(segIdx(30, 100, 3))

	s.LoadIndexes([]*model.Index{{CollectionID: 1, IndexID: 100}}, []*model.SegmentIndex{segIdx(10, 100, 1)})

	assert.Len(t, s.ListAllIndexes(), 1)
	_, ok := s.GetIndex(2, 200)
	assert.False(t, ok, "a definition from the earlier attempt must not survive")
	assert.Len(t, s.ListSegmentIndexBuilds(), 1)
	_, ok = s.GetSegmentIndexByBuildID(2)
	assert.False(t, ok)
	_, ok = s.GetSegmentIndexByBuildID(3)
	assert.False(t, ok)
	assert.Empty(t, s.GetSegmentIndexes(20))
	assert.Empty(t, s.GetSegmentIndexes(30))
	current, ok := s.GetSegmentIndex(10, 100)
	require.True(t, ok)
	assert.Equal(t, int64(1), current.BuildID)
}
