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
	"sync"

	"github.com/milvus-io/milvus/internal/metastore/model"
)

// IndexView reads the index metadata the store holds: collection index
// definitions and segment index records. Values are immutable once published:
// readers get the store's own pointers and must not modify them.
type IndexView interface {
	GetIndex(collectionID, indexID int64) (*model.Index, bool)
	// GetIndexes returns the collection's definitions, including logically
	// deleted ones, in no particular order.
	GetIndexes(collectionID int64) []*model.Index
	// ListAllIndexes returns every definition of every collection, including
	// logically deleted ones.
	ListAllIndexes() []*model.Index
	// GetSegmentIndex returns the current build of (segment, index).
	GetSegmentIndex(segmentID, indexID int64) (*model.SegmentIndex, bool)
	// GetSegmentIndexes returns the current build of each index of the
	// segment, keyed by index ID, in a new map.
	GetSegmentIndexes(segmentID int64) map[int64]*model.SegmentIndex
	// GetSegmentIndexByBuildID finds any build, including one superseded by a
	// newer build but not yet removed.
	GetSegmentIndexByBuildID(buildID int64) (*model.SegmentIndex, bool)
	// ListSegmentIndexBuilds returns every build, including superseded ones.
	ListSegmentIndexBuilds() []*model.SegmentIndex
	// GetIndexSnapshot reads the collection's definitions (including deleted
	// ones) and the current builds of segmentIDs under one read lock.
	GetIndexSnapshot(collectionID int64, segmentIDs []int64) ([]*model.Index, map[int64]map[int64]*model.SegmentIndex)
}

// IndexStore is the writable index section. DataCoord's IndexMeta is its only
// writer, and publishes a value only after the catalog write succeeded.
type IndexStore interface {
	IndexView
	// PutIndex installs a definition and returns the one it replaced.
	PutIndex(index *model.Index) (old *model.Index)
	// RemoveIndex drops a definition and returns it.
	RemoveIndex(collectionID, indexID int64) (old *model.Index)
	// PutSegmentIndexes makes each record the current build of its
	// (segment, index) and adds it to the build lookup, under one write lock.
	// A superseded build stays in the build lookup until RemoveSegmentIndex.
	// olds[i] is the record that held records[i].BuildID before, or nil.
	PutSegmentIndexes(records ...*model.SegmentIndex) (olds []*model.SegmentIndex)
	// UpdateSegmentIndexBuild replaces the build's record in the build lookup.
	// It replaces the current record of (segment, index) only if that is the
	// same build, so an update of a superseded build never becomes current.
	UpdateSegmentIndexBuild(record *model.SegmentIndex)
	// RemoveSegmentIndex removes the build from the build lookup, and clears
	// the current record of (segment, index) only if it is the same build.
	// It reports whether the build was present.
	RemoveSegmentIndex(record *model.SegmentIndex) (removed bool)
	// LoadIndexes replaces the section with reloaded metadata. Every record
	// enters the build lookup; the highest build ID of each (segment, index)
	// becomes current, whatever the input order. DataCoord retries recovery
	// against the same store, so nothing a failed attempt loaded survives.
	LoadIndexes(indexes []*model.Index, records []*model.SegmentIndex)
}

// indexSection holds the index metadata under its own lock, separate from
// the segment lock. Nothing is called back while mu is held.
type indexSection struct {
	mu sync.RWMutex
	// collectionID -> indexID -> definition
	defs map[int64]map[int64]*model.Index
	// segmentID -> indexID -> current build
	current map[int64]map[int64]*model.SegmentIndex
	// buildID -> record, including superseded builds
	builds map[int64]*model.SegmentIndex
}

func newIndexSection() indexSection {
	return indexSection{
		defs:    make(map[int64]map[int64]*model.Index),
		current: make(map[int64]map[int64]*model.SegmentIndex),
		builds:  make(map[int64]*model.SegmentIndex),
	}
}

func (s *metaStore) GetIndex(collectionID, indexID int64) (*model.Index, bool) {
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()
	index, ok := s.index.defs[collectionID][indexID]
	return index, ok
}

func (s *metaStore) GetIndexes(collectionID int64) []*model.Index {
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()
	return s.index.collectionIndexesLocked(collectionID)
}

func (s *metaStore) ListAllIndexes() []*model.Index {
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()
	var out []*model.Index
	for _, indexes := range s.index.defs {
		for _, index := range indexes {
			out = append(out, index)
		}
	}
	return out
}

func (s *metaStore) GetSegmentIndex(segmentID, indexID int64) (*model.SegmentIndex, bool) {
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()
	record, ok := s.index.current[segmentID][indexID]
	return record, ok
}

func (s *metaStore) GetSegmentIndexes(segmentID int64) map[int64]*model.SegmentIndex {
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()
	return s.index.segmentIndexesLocked(segmentID)
}

func (s *metaStore) GetSegmentIndexByBuildID(buildID int64) (*model.SegmentIndex, bool) {
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()
	record, ok := s.index.builds[buildID]
	return record, ok
}

func (s *metaStore) ListSegmentIndexBuilds() []*model.SegmentIndex {
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()
	out := make([]*model.SegmentIndex, 0, len(s.index.builds))
	for _, record := range s.index.builds {
		out = append(out, record)
	}
	return out
}

func (s *metaStore) GetIndexSnapshot(collectionID int64, segmentIDs []int64) ([]*model.Index, map[int64]map[int64]*model.SegmentIndex) {
	s.index.mu.RLock()
	defer s.index.mu.RUnlock()
	records := make(map[int64]map[int64]*model.SegmentIndex, len(segmentIDs))
	for _, segmentID := range segmentIDs {
		if indexes := s.index.segmentIndexesLocked(segmentID); len(indexes) > 0 {
			records[segmentID] = indexes
		}
	}
	return s.index.collectionIndexesLocked(collectionID), records
}

func (s *metaStore) PutIndex(index *model.Index) *model.Index {
	s.index.mu.Lock()
	defer s.index.mu.Unlock()
	indexes, ok := s.index.defs[index.CollectionID]
	if !ok {
		indexes = make(map[int64]*model.Index)
		s.index.defs[index.CollectionID] = indexes
	}
	old := indexes[index.IndexID]
	indexes[index.IndexID] = index
	return old
}

func (s *metaStore) RemoveIndex(collectionID, indexID int64) *model.Index {
	s.index.mu.Lock()
	defer s.index.mu.Unlock()
	indexes, ok := s.index.defs[collectionID]
	if !ok {
		return nil
	}
	old := indexes[indexID]
	delete(indexes, indexID)
	if len(indexes) == 0 {
		delete(s.index.defs, collectionID)
	}
	return old
}

func (s *metaStore) PutSegmentIndexes(records ...*model.SegmentIndex) []*model.SegmentIndex {
	s.index.mu.Lock()
	defer s.index.mu.Unlock()
	olds := make([]*model.SegmentIndex, len(records))
	for i, record := range records {
		olds[i] = s.index.builds[record.BuildID]
		s.index.builds[record.BuildID] = record
		s.index.setCurrentLocked(record)
	}
	return olds
}

func (s *metaStore) UpdateSegmentIndexBuild(record *model.SegmentIndex) {
	s.index.mu.Lock()
	defer s.index.mu.Unlock()
	s.index.builds[record.BuildID] = record
	if current, ok := s.index.current[record.SegmentID][record.IndexID]; ok && current.BuildID == record.BuildID {
		s.index.current[record.SegmentID][record.IndexID] = record
	}
}

func (s *metaStore) RemoveSegmentIndex(record *model.SegmentIndex) bool {
	s.index.mu.Lock()
	defer s.index.mu.Unlock()
	_, removed := s.index.builds[record.BuildID]
	delete(s.index.builds, record.BuildID)
	indexes, ok := s.index.current[record.SegmentID]
	if !ok {
		return removed
	}
	if current, ok := indexes[record.IndexID]; ok && current.BuildID == record.BuildID {
		delete(indexes, record.IndexID)
	}
	if len(indexes) == 0 {
		delete(s.index.current, record.SegmentID)
	}
	return removed
}

func (s *metaStore) LoadIndexes(indexes []*model.Index, records []*model.SegmentIndex) {
	s.index.mu.Lock()
	defer s.index.mu.Unlock()
	s.index.defs = make(map[int64]map[int64]*model.Index)
	s.index.current = make(map[int64]map[int64]*model.SegmentIndex)
	s.index.builds = make(map[int64]*model.SegmentIndex)
	for _, index := range indexes {
		collection, ok := s.index.defs[index.CollectionID]
		if !ok {
			collection = make(map[int64]*model.Index)
			s.index.defs[index.CollectionID] = collection
		}
		collection[index.IndexID] = index
	}
	for _, record := range records {
		s.index.builds[record.BuildID] = record
		if current, ok := s.index.current[record.SegmentID][record.IndexID]; ok && current.BuildID > record.BuildID {
			continue
		}
		s.index.setCurrentLocked(record)
	}
}

func (s *indexSection) setCurrentLocked(record *model.SegmentIndex) {
	indexes, ok := s.current[record.SegmentID]
	if !ok {
		indexes = make(map[int64]*model.SegmentIndex)
		s.current[record.SegmentID] = indexes
	}
	indexes[record.IndexID] = record
}

func (s *indexSection) collectionIndexesLocked(collectionID int64) []*model.Index {
	indexes := s.defs[collectionID]
	out := make([]*model.Index, 0, len(indexes))
	for _, index := range indexes {
		out = append(out, index)
	}
	return out
}

func (s *indexSection) segmentIndexesLocked(segmentID int64) map[int64]*model.SegmentIndex {
	indexes := s.current[segmentID]
	out := make(map[int64]*model.SegmentIndex, len(indexes))
	for indexID, record := range indexes {
		out[indexID] = record
	}
	return out
}
