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
	"cmp"
	"maps"
	"slices"
	"sync"

	"github.com/samber/lo"

	"github.com/milvus-io/milvus/internal/metacache"
	"github.com/milvus-io/milvus/internal/util/metrics"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/metricsinfo"
	"github.com/milvus-io/milvus/pkg/v3/util/tsoutil"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

type segDistCriterion struct {
	// Callers should not combine multiple node-scoped filters in one query.
	nodes          []int64
	collectionID   int64
	channel        string
	segmentID      int64
	hasSegmentID   bool
	hasOtherFilter bool
}

type SegmentDistFilter interface {
	Match(s *LoadedSegment) bool
	AddFilter(*segDistCriterion)
}

type SegmentDistFilterFunc func(s *LoadedSegment) bool

func (f SegmentDistFilterFunc) Match(s *LoadedSegment) bool {
	return f(s)
}

func (f SegmentDistFilterFunc) AddFilter(filter *segDistCriterion) {
	filter.hasOtherFilter = true
}

type ReplicaSegDistFilter struct {
	*Replica
}

func (f *ReplicaSegDistFilter) Match(s *LoadedSegment) bool {
	return f.GetCollectionID() == s.GetCollectionID() && f.Contains(s.Node)
}

func (f *ReplicaSegDistFilter) AddFilter(filter *segDistCriterion) {
	filter.nodes = f.GetNodes()
	filter.collectionID = f.GetCollectionID()
}

func WithReplica(replica *Replica) SegmentDistFilter {
	return &ReplicaSegDistFilter{
		Replica: replica,
	}
}

type NodeSegDistFilter int64

func (f NodeSegDistFilter) Match(s *LoadedSegment) bool {
	return s.Node == int64(f)
}

func (f NodeSegDistFilter) AddFilter(filter *segDistCriterion) {
	filter.nodes = []int64{int64(f)}
}

func WithNodeID(nodeID int64) SegmentDistFilter {
	return NodeSegDistFilter(nodeID)
}

type SegmentIDSegDistFilter int64

func (f SegmentIDSegDistFilter) Match(s *LoadedSegment) bool {
	return s.GetID() == int64(f)
}

func (f SegmentIDSegDistFilter) AddFilter(filter *segDistCriterion) {
	filter.segmentID = int64(f)
	filter.hasSegmentID = true
}

func WithSegmentID(segmentID int64) SegmentDistFilter {
	return SegmentIDSegDistFilter(segmentID)
}

type CollectionSegDistFilter int64

func (f CollectionSegDistFilter) Match(s *LoadedSegment) bool {
	return s.GetCollectionID() == int64(f)
}

func (f CollectionSegDistFilter) AddFilter(filter *segDistCriterion) {
	filter.collectionID = int64(f)
}

func WithCollectionID(collectionID typeutil.UniqueID) SegmentDistFilter {
	return CollectionSegDistFilter(collectionID)
}

type ChannelSegDistFilter string

func (f ChannelSegDistFilter) Match(s *LoadedSegment) bool {
	return s.GetInsertChannel() == string(f)
}

func (f ChannelSegDistFilter) AddFilter(filter *segDistCriterion) {
	filter.channel = string(f)
}

func WithChannel(channelName string) SegmentDistFilter {
	return ChannelSegDistFilter(channelName)
}

// LoadedSegment records what one QueryNode reports it loaded for a sealed
// segment. Every other segment field is read from the shared store by ID.
type LoadedSegment struct {
	ID int64
	// CollectionID and Channel are identity keys for filtering and for
	// releasing a segment the store no longer has: the store's values, or
	// the node's report when the store has no entry.
	CollectionID int64
	Channel      string
	Node         int64 // Node the segment is in

	Version            int64  // Version is the timestamp of loading segment
	LastDeltaTimestamp uint64 // The timestamp of the last delta record
	// IndexBuilds maps each loaded index ID to the build the node loaded.
	// Every other index field is read from the shared store.
	IndexBuilds map[int64]int64
	// JSONStatsFormats maps each field with loaded JSON key stats to the
	// data format version the node loaded.
	JSONStatsFormats map[int64]int64
	ManifestPath     string // manifest path of the loaded segment, as reported
	// DataVersion tracks the storage v2 data version of the loaded segment.
	// nil means the reporting QueryNode does not support DataVersion (mixed-version rollout),
	// so the field is not comparable.
	DataVersion *int32
}

// GetID, GetCollectionID and GetInsertChannel return the record's identity.
func (s *LoadedSegment) GetID() int64             { return s.ID }
func (s *LoadedSegment) GetCollectionID() int64   { return s.CollectionID }
func (s *LoadedSegment) GetInsertChannel() string { return s.Channel }

// LoadedFieldIndexInfos describes the indexes a node loaded for a segment,
// sorted by index ID. builds maps index ID to the loaded build; every other
// field comes from view: the definition for field and name, and the current
// segment index record for size and versions. A field the store no longer
// has is left zero. Params and file paths are not filled.
func LoadedFieldIndexInfos(view metacache.MetaView, collectionID, segmentID int64, builds map[int64]int64) []*querypb.FieldIndexInfo {
	infos := make([]*querypb.FieldIndexInfo, 0, len(builds))
	for indexID, buildID := range builds {
		info := &querypb.FieldIndexInfo{EnableIndex: true, IndexID: indexID, BuildID: buildID}
		if index, ok := view.GetIndex(collectionID, indexID); ok {
			info.FieldID = index.FieldID
			info.IndexName = index.IndexName
		}
		if record, ok := view.GetSegmentIndex(segmentID, indexID); ok {
			info.IndexSize = int64(record.IndexMemSize)
			info.NumRows = record.NumRows
			info.IndexVersion = record.IndexVersion
			info.CurrentIndexVersion = record.CurrentIndexVersion
			info.CurrentScalarIndexVersion = record.CurrentScalarIndexVersion
		}
		infos = append(infos, info)
	}
	slices.SortFunc(infos, func(a, b *querypb.FieldIndexInfo) int { return cmp.Compare(a.GetIndexID(), b.GetIndexID()) })
	return infos
}

// newSegmentMetricsFrom describes a loaded segment with the store's fields.
// On a store miss only the record's identity and node are filled.
func newSegmentMetricsFrom(segment *LoadedSegment, view metacache.MetaView) *metricsinfo.Segment {
	info, ok := view.GetSegment(segment.ID)
	if !ok {
		info = &datapb.SegmentInfo{ID: segment.ID, CollectionID: segment.CollectionID, InsertChannel: segment.Channel}
	}
	convertedSegment := metrics.NewSegmentFrom(info)
	convertedSegment.NodeID = segment.Node
	convertedSegment.LoadedTimestamp = tsoutil.PhysicalTimeFormat(segment.LastDeltaTimestamp)
	loaded := LoadedFieldIndexInfos(view, segment.GetCollectionID(), segment.GetID(), segment.IndexBuilds)
	convertedSegment.IndexedFields = lo.Map(loaded, func(e *querypb.FieldIndexInfo, i int) *metricsinfo.IndexedField {
		return &metricsinfo.IndexedField{
			IndexFieldID: e.FieldID,
			IndexID:      e.IndexID,
			BuildID:      e.BuildID,
			IndexSize:    e.IndexSize,
			IsLoaded:     true,
		}
	})
	return convertedSegment
}

type SegmentDistManagerInterface interface {
	Update(nodeID typeutil.UniqueID, segments ...*LoadedSegment)
	Patch(nodeID typeutil.UniqueID, upserts []*LoadedSegment, removedSegmentIDs []int64)
	GetByFilter(filters ...SegmentDistFilter) []*LoadedSegment
	GetSegmentDist(collectionID int64, view metacache.MetaView) []*metricsinfo.Segment
	GetVersion() int64
	// NodeRows and NodeCollectionRows return copies of the row totals.
	NodeRows() map[int64]int64
	NodeCollectionRows() map[int64]map[int64]int64
}

type SegmentDistManager struct {
	rwmutex sync.RWMutex

	// nodeID -> []*LoadedSegment
	segments map[typeutil.UniqueID]nodeSegments
	version  int64

	// view resolves the rows each entry contributes to the node totals. It
	// is read before rwmutex is taken, so the two locks are never held
	// together. A nil view counts every entry as 0 rows.
	view metacache.MetaView
}

func (m *SegmentDistManager) GetVersion() int64 {
	m.rwmutex.RLock()
	defer m.rwmutex.RUnlock()
	return m.version
}

type nodeSegments struct {
	segments        []*LoadedSegment
	collSegments    map[int64][]*LoadedSegment
	channelSegments map[string][]*LoadedSegment
	segmentSegments map[int64][]*LoadedSegment

	// rows is what each segment contributes to the totals, resolved from
	// the store when the entry was added; a store miss contributes 0.
	rows     map[int64]int64
	nodeRows int64
	collRows map[int64]int64
}

func (s nodeSegments) Filter(criterion *segDistCriterion, filter func(*LoadedSegment) bool) []*LoadedSegment {
	var segments []*LoadedSegment
	needFilter := criterion.hasOtherFilter
	switch {
	case criterion.hasSegmentID:
		segments = s.segmentSegments[criterion.segmentID]
		needFilter = needFilter || criterion.collectionID != 0 || criterion.channel != ""
	case criterion.channel != "":
		segments = s.channelSegments[criterion.channel]
	case criterion.collectionID != 0:
		segments = s.collSegments[criterion.collectionID]
	default:
		segments = s.segments
	}
	if needFilter {
		segments = lo.Filter(segments, func(segment *LoadedSegment, _ int) bool {
			return filter(segment)
		})
	}
	return segments
}

func composeNodeSegments(segments []*LoadedSegment, rows map[int64]int64) nodeSegments {
	ns := nodeSegments{
		segments:        segments,
		collSegments:    lo.GroupBy(segments, func(segment *LoadedSegment) int64 { return segment.GetCollectionID() }),
		channelSegments: lo.GroupBy(segments, func(segment *LoadedSegment) string { return segment.GetInsertChannel() }),
		segmentSegments: lo.GroupBy(segments, func(segment *LoadedSegment) int64 { return segment.GetID() }),
		rows:            rows,
		collRows:        make(map[int64]int64),
	}
	// Sum over entries, as a scan of the distribution would.
	for _, segment := range segments {
		n := rows[segment.GetID()]
		ns.nodeRows += n
		ns.collRows[segment.GetCollectionID()] += n
	}
	return ns
}

// resolveRows reads the row count of each segment from the view, in one
// batched read. It must be called without rwmutex held.
func (m *SegmentDistManager) resolveRows(segments []*LoadedSegment) map[int64]int64 {
	rows := make(map[int64]int64, len(segments))
	if m.view == nil || len(segments) == 0 {
		return rows
	}
	ids := make([]int64, len(segments))
	for i, segment := range segments {
		ids[i] = segment.GetID()
	}
	m.view.ForEachSegment(ids, func(id int64, info *datapb.SegmentInfo, ok bool) {
		if ok {
			rows[id] = info.GetNumOfRows()
		}
	})
	return rows
}

// NodeRows returns each node's total rows over its loaded sealed segments.
func (m *SegmentDistManager) NodeRows() map[int64]int64 {
	m.rwmutex.RLock()
	defer m.rwmutex.RUnlock()
	ret := make(map[int64]int64, len(m.segments))
	for node, ns := range m.segments {
		ret[node] = ns.nodeRows
	}
	return ret
}

// NodeCollectionRows returns each node's total rows per collection.
func (m *SegmentDistManager) NodeCollectionRows() map[int64]map[int64]int64 {
	m.rwmutex.RLock()
	defer m.rwmutex.RUnlock()
	ret := make(map[int64]map[int64]int64, len(m.segments))
	for node, ns := range m.segments {
		ret[node] = maps.Clone(ns.collRows)
	}
	return ret
}

func NewSegmentDistManager() *SegmentDistManager {
	return &SegmentDistManager{
		segments: make(map[typeutil.UniqueID]nodeSegments),
	}
}

func (m *SegmentDistManager) Update(nodeID typeutil.UniqueID, segments ...*LoadedSegment) {
	rows := m.resolveRows(segments)
	m.rwmutex.Lock()
	defer m.rwmutex.Unlock()

	if len(segments) == 0 {
		// Node offline, remove entry to avoid memory leak
		delete(m.segments, nodeID)
		m.version++
		return
	}

	for _, segment := range segments {
		segment.Node = nodeID
	}
	m.segments[nodeID] = composeNodeSegments(segments, rows)
	m.version++
}

func (m *SegmentDistManager) Patch(nodeID typeutil.UniqueID, upserts []*LoadedSegment, removedSegmentIDs []int64) {
	if len(upserts) == 0 && len(removedSegmentIDs) == 0 {
		return
	}

	upsertRows := m.resolveRows(upserts)
	m.rwmutex.Lock()
	defer m.rwmutex.Unlock()

	removedSegments := make(map[int64]struct{}, len(removedSegmentIDs))
	for _, segmentID := range removedSegmentIDs {
		removedSegments[segmentID] = struct{}{}
	}
	upsertSegments := make(map[int64]*LoadedSegment, len(upserts))
	for _, segment := range upserts {
		segment.Node = nodeID
		upsertSegments[segment.GetID()] = segment
	}

	existing := m.segments[nodeID]
	segments := make([]*LoadedSegment, 0, len(existing.segments)+len(upserts))
	rows := make(map[int64]int64, len(existing.rows)+len(upsertRows))
	for _, segment := range existing.segments {
		segmentID := segment.GetID()
		if _, ok := removedSegments[segmentID]; ok {
			continue
		}
		if _, ok := upsertSegments[segmentID]; ok {
			continue
		}
		segments = append(segments, segment)
		rows[segmentID] = existing.rows[segmentID]
	}
	for segmentID, segment := range upsertSegments {
		segments = append(segments, segment)
		rows[segmentID] = upsertRows[segmentID]
	}
	if len(segments) == 0 {
		delete(m.segments, nodeID)
	} else {
		m.segments[nodeID] = composeNodeSegments(segments, rows)
	}
	m.version++
}

// GetByFilter return segment list which match all given filters
func (m *SegmentDistManager) GetByFilter(filters ...SegmentDistFilter) []*LoadedSegment {
	m.rwmutex.RLock()
	defer m.rwmutex.RUnlock()

	criterion := &segDistCriterion{}
	for _, filter := range filters {
		filter.AddFilter(criterion)
	}

	mergedFilters := func(s *LoadedSegment) bool {
		for _, f := range filters {
			if f != nil && !f.Match(s) {
				return false
			}
		}
		return true
	}

	var ret []*LoadedSegment
	if criterion.nodes != nil {
		for _, nodeID := range criterion.nodes {
			ret = append(ret, m.segments[nodeID].Filter(criterion, mergedFilters)...)
		}
		return ret
	}

	for _, nodeSegments := range m.segments {
		ret = append(ret, nodeSegments.Filter(criterion, mergedFilters)...)
	}
	return ret
}

func (m *SegmentDistManager) GetSegmentDist(collectionID int64, view metacache.MetaView) []*metricsinfo.Segment {
	m.rwmutex.RLock()
	defer m.rwmutex.RUnlock()

	var segments []*metricsinfo.Segment
	for _, nodeSeg := range m.segments {
		for _, segment := range nodeSeg.segments {
			if collectionID > 0 && segment.GetCollectionID() != collectionID {
				continue
			}
			segments = append(segments, newSegmentMetricsFrom(segment, view))
		}
	}

	return segments
}
