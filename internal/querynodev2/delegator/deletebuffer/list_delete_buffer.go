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

package deletebuffer

import (
	"context"
	"sync"

	"github.com/cockroachdb/errors"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/querynodev2/segments"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/metrics"
	"github.com/milvus-io/milvus/pkg/v2/util/tsoutil"
)

func NewListDeleteBuffer[T timed](startTs uint64, sizePerBlock int64, labels []string) DeleteBuffer[T] {
	return &listDeleteBuffer[T]{
		safeTs:           startTs,
		sizePerBlock:     sizePerBlock,
		list:             []*cacheBlock[T]{newCacheBlock[T](startTs, sizePerBlock)},
		labels:           labels,
		l0Segments:       make([]segments.Segment, 0),
		pinnedTimestamps: make(map[uint64]map[int64]struct{}),
	}
}

// listDeleteBuffer implements DeleteBuffer with a list.
// head points to the earliest block.
// tail points to the latest block which shall be written into.
type listDeleteBuffer[T timed] struct {
	mut sync.RWMutex

	list []*cacheBlock[T]

	safeTs       uint64
	sizePerBlock int64

	// cached metrics
	rowNum int64
	size   int64

	// metrics labels
	labels []string

	// maintain l0 segment list
	l0Segments []segments.Segment

	// track pinned timestamps to prevent cleanup
	// map[timestamp]map[segmentID]struct{} - tracks which segments pin which timestamps
	pinnedTimestamps map[uint64]map[int64]struct{}
}

func (b *listDeleteBuffer[T]) RegisterL0(segmentList ...segments.Segment) {
	b.mut.Lock()
	defer b.mut.Unlock()
	// Filter out nil segments
	for _, seg := range segmentList {
		if seg != nil {
			b.l0Segments = append(b.l0Segments, seg)
			log.Info("register l0 from delete buffer",
				zap.Int64("segmentID", seg.ID()),
				zap.Time("startPosition", tsoutil.PhysicalTime(seg.StartPosition().GetTimestamp())),
			)
		}
	}

	b.updateMetrics()
}

func (b *listDeleteBuffer[T]) ListL0() []segments.Segment {
	b.mut.RLock()
	defer b.mut.RUnlock()
	return b.l0Segments
}

func (b *listDeleteBuffer[T]) UnRegister(ts uint64) {
	b.mut.Lock()
	defer b.mut.Unlock()
	if b.isPinned(ts) {
		return
	}
	var newSegments []segments.Segment
	for _, s := range b.l0Segments {
		if s.StartPosition().GetTimestamp() >= ts {
			newSegments = append(newSegments, s)
		} else {
			s.Release(context.TODO())
			log.Info("unregister l0 from delete buffer",
				zap.Int64("segmentID", s.ID()),
				zap.Time("startPosition", tsoutil.PhysicalTime(s.StartPosition().GetTimestamp())),
				zap.Time("cleanTs", tsoutil.PhysicalTime(ts)),
			)
		}
	}
	b.l0Segments = newSegments
	b.tryCleanDelete(ts)
	b.updateMetrics()
}

func (b *listDeleteBuffer[T]) Clear() {
	b.mut.Lock()
	defer b.mut.Unlock()

	// clean l0 segments
	for _, s := range b.l0Segments {
		s.Release(context.TODO())
	}
	b.l0Segments = nil

	// reset cache block
	b.list = []*cacheBlock[T]{newCacheBlock[T](b.safeTs, b.sizePerBlock)}
	b.updateMetrics()
}

func (b *listDeleteBuffer[T]) updateMetrics() {
	metrics.QueryNodeDeleteBufferRowNum.WithLabelValues(b.labels...).Set(float64(b.rowNum))
	metrics.QueryNodeDeleteBufferSize.WithLabelValues(b.labels...).Set(float64(b.size))
}

func (b *listDeleteBuffer[T]) Put(entry T) {
	b.mut.Lock()
	defer b.mut.Unlock()

	tail := b.list[len(b.list)-1]
	err := tail.Put(entry)
	if errors.Is(err, errBufferFull) {
		b.list = append(b.list, newCacheBlock(entry.Timestamp(), b.sizePerBlock, entry))
	}

	// update metrics
	b.rowNum += entry.EntryNum()
	b.size += entry.Size()
	b.updateMetrics()
}

func (b *listDeleteBuffer[T]) ListAfter(ts uint64) []T {
	b.mut.RLock()
	defer b.mut.RUnlock()

	var result []T
	for _, block := range b.list {
		result = append(result, block.ListAfter(ts)...)
	}
	return result
}

func (b *listDeleteBuffer[T]) SafeTs() uint64 {
	b.mut.RLock()
	defer b.mut.RUnlock()
	return b.safeTs
}

func (b *listDeleteBuffer[T]) TryDiscard(ts uint64) {
	b.mut.Lock()
	defer b.mut.Unlock()
	if b.isPinned(ts) {
		return
	}
	b.tryCleanDelete(ts)
}

func (b *listDeleteBuffer[T]) tryCleanDelete(ts uint64) {
	if len(b.list) == 1 {
		return
	}
	var nextHead int
	for idx := len(b.list) - 1; idx >= 0; idx-- {
		block := b.list[idx]
		if block.headTs <= ts {
			nextHead = idx
			break
		}
	}

	if nextHead > 0 {
		for idx := 0; idx < nextHead; idx++ {
			rowNum, memSize := b.list[idx].Size()
			b.rowNum -= rowNum
			b.size -= memSize
			b.list[idx] = nil
		}
		b.list = b.list[nextHead:]
		b.updateMetrics()
	}
}

// check if any records is pinned before the cleanTs
func (b *listDeleteBuffer[T]) isPinned(cleanTs uint64) bool {
	// Check if there are any pinned timestamps before the cleanTs
	// If there are pinned timestamps before cleanTs, we should skip cleanup
	// because pinning a timestamp protects all data after that timestamp
	var pinnedSegments []int64
	var pinnedTimestamp uint64
	for pinnedTs, segmentMap := range b.pinnedTimestamps {
		if pinnedTs < cleanTs && len(segmentMap) > 0 {
			// Found a pinned timestamp before cleanTs
			pinnedTimestamp = pinnedTs
			for segmentID := range segmentMap {
				pinnedSegments = append(pinnedSegments, segmentID)
			}
			break
		}
	}

	if len(pinnedSegments) > 0 {
		log.Info("skip cleanup due to pinned timestamp before cleanTs",
			zap.Time("pinnedPhysicalTime", tsoutil.PhysicalTime(pinnedTimestamp)),
			zap.Time("cleanPhysicalTime", tsoutil.PhysicalTime(cleanTs)),
			zap.Int64s("pinningSegmentIDs", pinnedSegments),
			zap.Int("segmentCount", len(pinnedSegments)),
		)
		return true
	}
	return false
}

func (b *listDeleteBuffer[T]) Size() (entryNum, memorySize int64) {
	b.mut.RLock()
	defer b.mut.RUnlock()

	return b.rowNum, b.size
}

// Pin protects a specific timestamp from being cleaned up by a specific segment
func (b *listDeleteBuffer[T]) Pin(ts uint64, segmentID int64) {
	b.mut.Lock()
	defer b.mut.Unlock()

	if b.pinnedTimestamps[ts] == nil {
		b.pinnedTimestamps[ts] = make(map[int64]struct{})
	}
	b.pinnedTimestamps[ts][segmentID] = struct{}{}

	log.Info("pin timestamp for segment",
		zap.Uint64("timestamp", ts),
		zap.Int64("segmentID", segmentID),
		zap.Time("physicalTime", tsoutil.PhysicalTime(ts)),
	)
}

// Unpin removes protection for a specific timestamp by a specific segment and triggers cleanup
func (b *listDeleteBuffer[T]) Unpin(ts uint64, segmentID int64) {
	b.mut.Lock()
	defer b.mut.Unlock()

	if segmentMap, exists := b.pinnedTimestamps[ts]; exists {
		delete(segmentMap, segmentID)
		if len(segmentMap) == 0 {
			delete(b.pinnedTimestamps, ts)
		}
	}

	log.Info("unpin timestamp for segment",
		zap.Uint64("timestamp", ts),
		zap.Int64("segmentID", segmentID),
		zap.Time("physicalTime", tsoutil.PhysicalTime(ts)),
	)
}
