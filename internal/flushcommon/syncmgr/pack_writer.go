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

package syncmgr

import (
	"context"
	"path"

	"github.com/milvus-io/milvus/internal/allocator"
	"github.com/milvus-io/milvus/internal/flushcommon/metacache"
	"github.com/milvus-io/milvus/internal/proto/datapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/pkg/common"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/metautil"
	"github.com/milvus-io/milvus/pkg/util/retry"
	"github.com/samber/lo"
	"go.uber.org/zap"
)

type PackWriter interface {
	Write(ctx context.Context, pack *SyncPack) (
		inserts []*datapb.Binlog, deletes *datapb.Binlog, stats *datapb.Binlog, bm25Stats *datapb.Binlog,
		size int64, err error)
}

type BulkPackWriter struct {
	metaCache      metacache.MetaCache
	chunkManager   storage.ChunkManager
	allocator      allocator.Interface
	writeRetryOpts []retry.Option

	// prefetched log ids
	ids         []int64
	sizeWritten int64
}

func NewBulkPackWriter(metaCache metacache.MetaCache, chunkManager storage.ChunkManager,
	allocator allocator.Interface, writeRetryOpts ...retry.Option,
) *BulkPackWriter {
	return &BulkPackWriter{
		metaCache:      metaCache,
		chunkManager:   chunkManager,
		allocator:      allocator,
		writeRetryOpts: writeRetryOpts,
	}
}

func (bw *BulkPackWriter) Write(ctx context.Context, pack *SyncPack) (
	inserts map[int64]*datapb.FieldBinlog,
	deltas *datapb.FieldBinlog,
	stats map[int64]*datapb.FieldBinlog,
	bm25Stats map[int64]*datapb.FieldBinlog,
	size int64,
	err error,
) {
	err = bw.prefetchIDs(pack)
	if err != nil {
		log.Warn("failed allocate ids for sync task", zap.Error(err))
		return
	}

	if inserts, err = bw.writeInserts(ctx, pack); err != nil {
		log.Error("failed to write insert data", zap.Error(err))
		return
	}
	if stats, err = bw.writeStats(ctx, pack); err != nil {
		log.Error("failed to process stats blob", zap.Error(err))
		return
	}
	if deltas, err = bw.writeDelta(ctx, pack); err != nil {
		log.Error("failed to process delta blob", zap.Error(err))
		return
	}
	if bm25Stats, err = bw.writeBM25Stasts(ctx, pack); err != nil {
		log.Error("failed to process bm25 stats blob", zap.Error(err))
		return
	}

	size = bw.sizeWritten

	return
}

// prefetchIDs pre-allcates ids depending on the number of blobs current task contains.
func (bw *BulkPackWriter) prefetchIDs(pack *SyncPack) error {
	totalIDCount := 0
	if len(pack.insertData) > 0 {
		totalIDCount += len(pack.insertData[0].Data) * 2 // binlogs and statslogs
	}
	if pack.isFlush {
		totalIDCount++ // merged stats log
	}
	if pack.deltaData != nil {
		totalIDCount++
	}
	if pack.bm25Stats != nil {
		totalIDCount += len(pack.bm25Stats)
		if pack.isFlush {
			totalIDCount++ // merged bm25 stats
		}
	}

	start, _, err := bw.allocator.Alloc(uint32(totalIDCount))
	if err != nil {
		return err
	}
	bw.ids = lo.RangeFrom(start, totalIDCount)
	return nil
}

func (bw *BulkPackWriter) nextID() int64 {
	if len(bw.ids) == 0 {
		panic("pre-fetched ids exhausted")
	}
	r := bw.ids[0]
	bw.ids = bw.ids[1:]
	return r
}

func (bw *BulkPackWriter) writeLog(ctx context.Context, blob *storage.Blob,
	root string, pack *SyncPack, field, log int64,
) (*datapb.Binlog, error) {
	k := metautil.JoinIDPath(pack.collectionID, pack.partitionID, pack.segmentID, field, log)
	key := path.Join(bw.chunkManager.RootPath(), root, k)
	err := retry.Do(ctx, func() error {
		return bw.chunkManager.Write(ctx, key, blob.Value)
	}, bw.writeRetryOpts...)
	if err != nil {
		return nil, err
	}
	size := int64(len(blob.GetValue()))
	bw.sizeWritten += size
	return &datapb.Binlog{
		EntriesNum:    blob.RowNum,
		TimestampFrom: pack.tsFrom,
		TimestampTo:   pack.tsTo,
		LogPath:       key,
		LogSize:       size,
		MemorySize:    blob.MemorySize,
	}, nil
}

func (bw *BulkPackWriter) writeInserts(ctx context.Context, pack *SyncPack) (map[int64]*datapb.FieldBinlog, error) {
	if len(pack.insertData) == 0 {
		return nil, nil
	}

	serializer, err := NewStorageSerializer(bw.metaCache)
	if err != nil {
		return nil, err
	}

	binlogBlobs, err := serializer.serializeBinlog(ctx, pack)
	if err != nil {
		return nil, err
	}

	logs := make(map[int64]*datapb.FieldBinlog)
	for fieldID, blob := range binlogBlobs {
		if binlog, err := bw.writeLog(ctx, blob, common.SegmentInsertLogPath, pack, fieldID, bw.nextID()); err != nil {
			return nil, err
		} else {
			logs[fieldID] = &datapb.FieldBinlog{
				FieldID: fieldID,
				Binlogs: []*datapb.Binlog{binlog},
			}
		}
	}
	return logs, nil
}

func (bw *BulkPackWriter) writeStats(ctx context.Context, pack *SyncPack) (map[int64]*datapb.FieldBinlog, error) {
	if len(pack.insertData) == 0 {
		return nil, nil
	}
	serializer, err := NewStorageSerializer(bw.metaCache)
	if err != nil {
		return nil, err
	}
	singlePKStats, batchStatsBlob, err := serializer.serializeStatslog(pack)
	if err != nil {
		return nil, err
	}

	actions := []metacache.SegmentAction{metacache.RollStats(singlePKStats)}
	bw.metaCache.UpdateSegments(metacache.MergeSegmentAction(actions...), metacache.WithSegmentIDs(pack.segmentID))

	pkFieldID := serializer.pkField.GetFieldID()
	binlogs := make([]*datapb.Binlog, 0)
	if batchStatsBlob, err := bw.writeLog(ctx, batchStatsBlob, common.SegmentStatslogPath, pack, pkFieldID, bw.nextID()); err != nil {
		return nil, err
	} else {
		binlogs = append(binlogs, batchStatsBlob)
	}

	if pack.isFlush {
		if pack.level != datapb.SegmentLevel_L0 {
			mergedStatsBlob, err := serializer.serializeMergedPkStats(pack)
			if err != nil {
				return nil, err
			}

			if mergedStatsBlob, err := bw.writeLog(ctx, mergedStatsBlob, common.SegmentStatslogPath, pack, pkFieldID, int64(storage.CompoundStatsType)); err != nil {
				return nil, err
			} else {
				binlogs = append(binlogs, mergedStatsBlob)
			}
		}
	}

	logs := make(map[int64]*datapb.FieldBinlog)
	logs[pkFieldID] = &datapb.FieldBinlog{
		FieldID: pkFieldID,
		Binlogs: binlogs,
	}
	return logs, nil
}

func (bw *BulkPackWriter) writeBM25Stasts(ctx context.Context, pack *SyncPack) (map[int64]*datapb.FieldBinlog, error) {
	if len(pack.bm25Stats) == 0 {
		return nil, nil
	}

	serializer, err := NewStorageSerializer(bw.metaCache)
	if err != nil {
		return nil, err
	}
	bm25Blobs, err := serializer.serializeBM25Stats(pack)
	if err != nil {
		return nil, err
	}

	logs := make(map[int64]*datapb.FieldBinlog)
	for fieldID, blob := range bm25Blobs {
		if binlog, err := bw.writeLog(ctx, blob, common.SegmentBm25LogPath, pack, fieldID, bw.nextID()); err != nil {
			return nil, err
		} else {
			logs[fieldID] = &datapb.FieldBinlog{
				FieldID: fieldID,
				Binlogs: []*datapb.Binlog{binlog},
			}
		}
	}

	actions := []metacache.SegmentAction{metacache.MergeBm25Stats(pack.bm25Stats)}
	bw.metaCache.UpdateSegments(metacache.MergeSegmentAction(actions...), metacache.WithSegmentIDs(pack.segmentID))

	if pack.isFlush {
		if pack.level != datapb.SegmentLevel_L0 {
			if hasBM25Function(bw.metaCache.Schema()) {
				mergedBM25Blob, err := serializer.serializeMergedBM25Stats(pack)
				if err != nil {
					return nil, err
				}
				for fieldID, blob := range mergedBM25Blob {
					if binlog, err := bw.writeLog(ctx, blob, common.SegmentBm25LogPath, pack, fieldID, int64(storage.CompoundStatsType)); err != nil {
						return nil, err
					} else {
						fieldBinlog, ok := logs[fieldID]
						if !ok {
							fieldBinlog = &datapb.FieldBinlog{
								FieldID: fieldID,
							}
							logs[fieldID] = fieldBinlog
						}
						fieldBinlog.Binlogs = append(fieldBinlog.Binlogs, binlog)
					}
				}
			}
		}
	}
	return logs, nil
}

func (bw *BulkPackWriter) writeDelta(ctx context.Context, pack *SyncPack) (*datapb.FieldBinlog, error) {
	if pack.deltaData == nil {
		return nil, nil
	}
	s, err := NewStorageSerializer(bw.metaCache)
	if err != nil {
		return nil, err
	}
	deltaBlob, err := s.serializeDeltalog(pack)
	if err != nil {
		return nil, err
	}

	if deltalog, err := bw.writeLog(ctx, deltaBlob, common.SegmentDeltaLogPath, pack, s.pkField.GetFieldID(), bw.nextID()); err != nil {
		return nil, err
	} else {
		return &datapb.FieldBinlog{
			FieldID: s.pkField.GetFieldID(),
			Binlogs: []*datapb.Binlog{deltalog},
		}, nil
	}
}