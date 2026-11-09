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
	"context"
	"sync"

	"golang.org/x/sync/errgroup"

	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/pkg/v3/metrics"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/util/conc"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/timerecord"
)

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

// LoadIndexesFromCatalog replaces the index section with the definitions
// and the segment index records of collectionIDs read from the catalog.
func (s *metaStore) LoadIndexesFromCatalog(ctx context.Context, collectionIDs []int64) error {
	record := timerecord.NewTimeRecorder("metaStore-loadIndexes")
	var indexes []*model.Index
	collectionRecords := make([][]*model.SegmentIndex, len(collectionIDs))
	g, gctx := errgroup.WithContext(ctx)
	g.Go(func() error {
		var err error
		indexes, err = s.catalog.ListIndexes(gctx)
		return err
	})
	g.Go(func() error {
		pool := conc.NewPool[any](paramtable.Get().MetaStoreCfg.ReadConcurrency.GetAsInt())
		defer pool.Release()
		futures := make([]*conc.Future[any], 0, len(collectionIDs))
		for i, collectionID := range collectionIDs {
			futures = append(futures, pool.Submit(func() (any, error) {
				records, err := s.catalog.ListSegmentIndexes(gctx, collectionID)
				collectionRecords[i] = records
				return nil, err
			}))
		}
		return conc.AwaitAll(futures...)
	})
	if err := g.Wait(); err != nil {
		mlog.Warn(ctx, "metaStore failed to load indexes", mlog.Err(err))
		return err
	}

	memSizeMultiplier := paramtable.Get().DataCoordCfg.IndexMemSizeEstimateMultiplier.GetAsUint64()
	records := make([]*model.SegmentIndex, 0)
	for _, collection := range collectionRecords {
		for _, segIdx := range collection {
			// A catalog row is durable in the catalog, not in a manifest.
			segIdx.ManifestPublished = false
			if segIdx.IndexMemSize == 0 {
				segIdx.IndexMemSize = segIdx.IndexSerializedSize * memSizeMultiplier
			}
			records = append(records, segIdx)
		}
	}
	s.loadIndexes(indexes, records)

	// Observe the file counts off the startup path.
	go func() {
		for _, segIdx := range records {
			metrics.FlushedSegmentFileNum.WithLabelValues(metrics.IndexFileLabel).Observe(float64(len(segIdx.IndexFileKeys)))
		}
	}()
	mlog.Info(ctx, "metaStore loaded indexes",
		mlog.Int("numIndexes", len(indexes)),
		mlog.Int("numSegmentIndexes", len(records)),
		mlog.Duration("duration", record.ElapseSpan()))
	return nil
}

// loadIndexes replaces the section with reloaded metadata. Every record
// enters the build lookup; the highest build ID of each (segment, index)
// becomes current, whatever the input order. DataCoord retries recovery
// against the same store, so nothing a failed attempt loaded survives.
func (s *metaStore) loadIndexes(indexes []*model.Index, records []*model.SegmentIndex) {
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
