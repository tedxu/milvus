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

//go:build test

package meta

import (
	"github.com/milvus-io/milvus/internal/metacache"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
)

// SegmentFixture is a test's description of a loaded segment: the store
// proto it resolves to and what the node reports. SeedSegments turns
// fixtures into distribution records.
type SegmentFixture struct {
	*datapb.SegmentInfo
	Node               int64
	Version            int64
	LastDeltaTimestamp uint64
	IndexBuilds        map[int64]int64
	JSONStatsFormats   map[int64]int64
	ManifestPath       string
	DataVersion        *int32
	// KeepStored keeps a proto the store already has for the ID, such as a
	// target segment, instead of replacing it with this fixture's.
	KeepStored bool
}

// SeedSegments puts each fixture's proto into store, and returns the
// distribution records. A fixture with KeepStored set leaves a proto the
// store already has in place.
func SeedSegments(store metacache.MetaStore, fixtures ...*SegmentFixture) []*LoadedSegment {
	records := make([]*LoadedSegment, 0, len(fixtures))
	for _, f := range fixtures {
		if _, ok := store.GetSegment(f.GetID()); !ok || !f.KeepStored {
			store.PutSegment(f.SegmentInfo)
		}
		records = append(records, &LoadedSegment{
			ID:                 f.GetID(),
			CollectionID:       f.GetCollectionID(),
			Channel:            f.GetInsertChannel(),
			Node:               f.Node,
			Version:            f.Version,
			LastDeltaTimestamp: f.LastDeltaTimestamp,
			IndexBuilds:        f.IndexBuilds,
			JSONStatsFormats:   f.JSONStatsFormats,
			ManifestPath:       f.ManifestPath,
			DataVersion:        f.DataVersion,
		})
	}
	return records
}

// SeedInto seeds fixtures into the store dm resolves entries in, giving dm
// a store of its own if it has none, and returns the records.
func SeedInto(dm *DistributionManager, fixtures ...*SegmentFixture) []*LoadedSegment {
	if dm.view == nil {
		dm.view = metacache.NewMetaStore(nil)
		if sdm, ok := dm.SegmentDistManager.(*SegmentDistManager); ok {
			sdm.view = dm.view
		}
	}
	store, ok := dm.view.(metacache.MetaStore)
	if !ok {
		panic("the distribution's meta view is not a MetaStore")
	}
	return SeedSegments(store, fixtures...)
}

// FixtureFromInfo describes a segment with the proto info and no node.
func FixtureFromInfo(info *datapb.SegmentInfo) *SegmentFixture {
	return &SegmentFixture{SegmentInfo: info}
}
