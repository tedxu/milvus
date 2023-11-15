package writebuffer

import (
	"context"
	"sync"

	"github.com/samber/lo"
	"go.uber.org/atomic"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/msgpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/datanode/broker"
	"github.com/milvus-io/milvus/internal/datanode/metacache"
	"github.com/milvus-io/milvus/internal/datanode/syncmgr"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/mq/msgstream"
	"github.com/milvus-io/milvus/pkg/util/conc"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

const (
	nonFlushTS uint64 = 0
)

// WriteBuffer is the interface for channel write buffer.
// It provides abstraction for channel write buffer and pk bloom filter & L0 delta logic.
type WriteBuffer interface {
	// HasSegment checks whether certain segment exists in this buffer.
	HasSegment(segmentID int64) bool
	// BufferData is the method to buffer dml data msgs.
	BufferData(insertMsgs []*msgstream.InsertMsg, deleteMsgs []*msgstream.DeleteMsg, startPos, endPos *msgpb.MsgPosition) error
	// FlushTimestamp set flush timestamp for write buffer
	SetFlushTimestamp(flushTs uint64)
	// GetFlushTimestamp get current flush timestamp
	GetFlushTimestamp() uint64
	// FlushSegments is the method to perform `Sync` operation with provided options.
	FlushSegments(ctx context.Context, segmentIDs []int64) error
	// MinCheckpoint returns current channel checkpoint.
	// If there are any non-empty segment buffer, returns the earliest buffer start position.
	// Otherwise, returns latest buffered checkpoint.
	MinCheckpoint() *msgpb.MsgPosition
	// Close is the method to close and sink current buffer data.
	Close(drop bool)
}

func NewWriteBuffer(channel string, schema *schemapb.CollectionSchema, metacache metacache.MetaCache, syncMgr syncmgr.SyncManager, opts ...WriteBufferOption) (WriteBuffer, error) {
	option := defaultWBOption()
	option.syncPolicies = append(option.syncPolicies, GetFlushingSegmentsPolicy(metacache))
	for _, opt := range opts {
		opt(option)
	}

	switch option.deletePolicy {
	case DeletePolicyBFPkOracle:
		return NewBFWriteBuffer(channel, schema, metacache, syncMgr, option)
	case DeletePolicyL0Delta:
		return NewL0WriteBuffer(channel, schema, metacache, syncMgr, option)
	default:
		return nil, merr.WrapErrParameterInvalid("valid delete policy config", option.deletePolicy)
	}
}

// writeBufferBase is the common component for buffering data
type writeBufferBase struct {
	mut sync.RWMutex

	collectionID int64
	channelName  string

	metaWriter syncmgr.MetaWriter
	collSchema *schemapb.CollectionSchema
	metaCache  metacache.MetaCache
	syncMgr    syncmgr.SyncManager
	broker     broker.Broker
	buffers    map[int64]*segmentBuffer // segmentID => segmentBuffer

	syncPolicies   []SyncPolicy
	checkpoint     *msgpb.MsgPosition
	flushTimestamp *atomic.Uint64
}

func newWriteBufferBase(channel string, sch *schemapb.CollectionSchema, metacache metacache.MetaCache, syncMgr syncmgr.SyncManager, option *writeBufferOption) *writeBufferBase {
	flushTs := atomic.NewUint64(nonFlushTS)
	flushTsPolicy := GetFlushTsPolicy(flushTs, metacache)
	option.syncPolicies = append(option.syncPolicies, flushTsPolicy)

	return &writeBufferBase{
		channelName:    channel,
		collSchema:     sch,
		syncMgr:        syncMgr,
		metaWriter:     option.metaWriter,
		buffers:        make(map[int64]*segmentBuffer),
		metaCache:      metacache,
		syncPolicies:   option.syncPolicies,
		flushTimestamp: flushTs,
	}
}

func (wb *writeBufferBase) HasSegment(segmentID int64) bool {
	wb.mut.RLock()
	defer wb.mut.RUnlock()

	_, ok := wb.buffers[segmentID]
	return ok
}

func (wb *writeBufferBase) FlushSegments(ctx context.Context, segmentIDs []int64) error {
	wb.mut.RLock()
	defer wb.mut.RUnlock()

	return wb.flushSegments(ctx, segmentIDs)
}

func (wb *writeBufferBase) SetFlushTimestamp(flushTs uint64) {
	wb.flushTimestamp.Store(flushTs)
}

func (wb *writeBufferBase) GetFlushTimestamp() uint64 {
	return wb.flushTimestamp.Load()
}

func (wb *writeBufferBase) MinCheckpoint() *msgpb.MsgPosition {
	wb.mut.RLock()
	defer wb.mut.RUnlock()

	syncingPos := wb.syncMgr.GetMinCheckpoints(wb.channelName)

	positions := lo.MapToSlice(wb.buffers, func(_ int64, buf *segmentBuffer) *msgpb.MsgPosition {
		return buf.MinCheckpoint()
	})
	positions = append(positions, syncingPos)

	checkpoint := getEarliestCheckpoint(positions...)
	// all buffer are empty
	if checkpoint == nil {
		return wb.checkpoint
	}
	return checkpoint
}

func (wb *writeBufferBase) triggerAutoSync() error {
	segmentsToSync := wb.getSegmentsToSync(wb.checkpoint.GetTimestamp())
	if len(segmentsToSync) > 0 {
		log.Info("write buffer get segments to sync", zap.Int64s("segmentIDs", segmentsToSync))
		err := wb.syncSegments(context.Background(), segmentsToSync)
		if err != nil {
			log.Warn("segment segments failed", zap.Int64s("segmentIDs", segmentsToSync), zap.Error(err))
			return err
		}
	}

	return nil
}

func (wb *writeBufferBase) flushSegments(ctx context.Context, segmentIDs []int64) error {
	// mark segment flushing if segment was growing
	wb.metaCache.UpdateSegments(metacache.UpdateState(commonpb.SegmentState_Flushing),
		metacache.WithSegmentIDs(segmentIDs...),
		metacache.WithSegmentState(commonpb.SegmentState_Growing))
	// mark segment flushing if segment was importing
	wb.metaCache.UpdateSegments(metacache.UpdateState(commonpb.SegmentState_Flushing),
		metacache.WithSegmentIDs(segmentIDs...),
		metacache.WithImporting())
	return nil
}

func (wb *writeBufferBase) syncSegments(ctx context.Context, segmentIDs []int64) error {
	for _, segmentID := range segmentIDs {
		syncTask := wb.getSyncTask(ctx, segmentID)
		if syncTask == nil {
			// segment info not found
			continue
		}

		// discard Future here, handle error in callback
		_ = wb.syncMgr.SyncData(ctx, syncTask)
	}
	return nil
}

// getSegmentsToSync applies all policies to get segments list to sync.
// **NOTE** shall be invoked within mutex protection
func (wb *writeBufferBase) getSegmentsToSync(ts typeutil.Timestamp) []int64 {
	buffers := lo.Values(wb.buffers)
	segments := typeutil.NewSet[int64]()
	for _, policy := range wb.syncPolicies {
		segments.Insert(policy(buffers, ts)...)
	}

	return segments.Collect()
}

func (wb *writeBufferBase) getOrCreateBuffer(segmentID int64) *segmentBuffer {
	buffer, ok := wb.buffers[segmentID]
	if !ok {
		var err error
		buffer, err = newSegmentBuffer(segmentID, wb.collSchema)
		if err != nil {
			// TODO avoid panic here
			panic(err)
		}
		wb.buffers[segmentID] = buffer
	}

	return buffer
}

func (wb *writeBufferBase) yieldBuffer(segmentID int64) (*storage.InsertData, *storage.DeleteData, *msgpb.MsgPosition) {
	buffer, ok := wb.buffers[segmentID]
	if !ok {
		return nil, nil, nil
	}

	// remove buffer and move it to sync manager
	delete(wb.buffers, segmentID)
	start := buffer.MinCheckpoint()
	insert, delta := buffer.Yield()

	return insert, delta, start
}

// bufferInsert transform InsertMsg into bufferred InsertData and returns primary key field data for future usage.
func (wb *writeBufferBase) bufferInsert(insertMsgs []*msgstream.InsertMsg, startPos, endPos *msgpb.MsgPosition) (map[int64][]storage.FieldData, error) {
	insertGroups := lo.GroupBy(insertMsgs, func(msg *msgstream.InsertMsg) int64 { return msg.GetSegmentID() })
	segmentPKData := make(map[int64][]storage.FieldData)
	segmentPartition := lo.SliceToMap(insertMsgs, func(msg *msgstream.InsertMsg) (int64, int64) { return msg.GetSegmentID(), msg.GetPartitionID() })

	for segmentID, msgs := range insertGroups {
		_, ok := wb.metaCache.GetSegmentByID(segmentID)
		// new segment
		if !ok {
			wb.metaCache.AddSegment(&datapb.SegmentInfo{
				ID:            segmentID,
				PartitionID:   segmentPartition[segmentID],
				CollectionID:  wb.collectionID,
				InsertChannel: wb.channelName,
				StartPosition: startPos,
				State:         commonpb.SegmentState_Growing,
			}, func(_ *datapb.SegmentInfo) *metacache.BloomFilterSet { return metacache.NewBloomFilterSet() })
		}

		segBuf := wb.getOrCreateBuffer(segmentID)

		pkData, err := segBuf.insertBuffer.Buffer(msgs, startPos, endPos)
		if err != nil {
			log.Warn("failed to buffer insert data", zap.Int64("segmentID", segmentID), zap.Error(err))
			return nil, err
		}
		segmentPKData[segmentID] = pkData
		wb.metaCache.UpdateSegments(metacache.UpdateBufferedRows(segBuf.insertBuffer.rows),
			metacache.WithSegmentIDs(segmentID))
	}

	return segmentPKData, nil
}

// bufferDelete buffers DeleteMsg into DeleteData.
func (wb *writeBufferBase) bufferDelete(segmentID int64, pks []storage.PrimaryKey, tss []typeutil.Timestamp, startPos, endPos *msgpb.MsgPosition) error {
	segBuf := wb.getOrCreateBuffer(segmentID)
	segBuf.deltaBuffer.Buffer(pks, tss, startPos, endPos)
	return nil
}

func (wb *writeBufferBase) getSyncTask(ctx context.Context, segmentID int64) *syncmgr.SyncTask {
	segmentInfo, ok := wb.metaCache.GetSegmentByID(segmentID) // wb.metaCache.GetSegmentsBy(metacache.WithSegmentIDs(segmentID))
	if !ok {
		log.Ctx(ctx).Warn("segment info not found in meta cache", zap.Int64("segmentID", segmentID))
		return nil
	}
	var batchSize int64

	insert, delta, startPos := wb.yieldBuffer(segmentID)

	actions := []metacache.SegmentAction{metacache.RollStats()}
	if insert != nil {
		actions = append(actions, metacache.StartSyncing(int64(insert.GetRowNum())))
		batchSize = int64(insert.GetRowNum())
	}
	wb.metaCache.UpdateSegments(metacache.MergeSegmentAction(actions...), metacache.WithSegmentIDs(segmentID))

	syncTask := syncmgr.NewSyncTask().
		WithInsertData(insert).
		WithDeleteData(delta).
		WithCollectionID(wb.collectionID).
		WithPartitionID(segmentInfo.PartitionID()).
		WithChannelName(wb.channelName).
		WithSegmentID(segmentID).
		WithStartPosition(startPos).
		WithCheckpoint(wb.checkpoint).
		WithSchema(wb.collSchema).
		WithBatchSize(batchSize).
		WithMetaCache(wb.metaCache).
		WithMetaWriter(wb.metaWriter).
		WithFailureCallback(func(err error) {
			// TODO could change to unsub channel in the future
			panic(err)
		})
	if segmentInfo.State() == commonpb.SegmentState_Flushing {
		syncTask.WithFlush()
	}

	return syncTask
}

func (wb *writeBufferBase) Close(drop bool) {
	// sink all data and call Drop for meta writer
	wb.mut.Lock()
	defer wb.mut.Unlock()
	if !drop {
		return
	}

	var futures []*conc.Future[error]
	for id := range wb.buffers {
		syncTask := wb.getSyncTask(context.Background(), id)
		if syncTask == nil {
			continue
		}
		syncTask.WithDrop()
		f := wb.syncMgr.SyncData(context.Background(), syncTask)
		futures = append(futures, f)
	}

	err := conc.AwaitAll(futures...)
	if err != nil {
		log.Error("failed to sink write buffer data", zap.String("channel", wb.channelName), zap.Error(err))
		// TODO change to remove channel in the future
		panic(err)
	}
	err = wb.metaWriter.DropChannel(wb.channelName)
	if err != nil {
		log.Error("failed to drop channel", zap.String("channel", wb.channelName), zap.Error(err))
		// TODO change to remove channel in the future
		panic(err)
	}
}