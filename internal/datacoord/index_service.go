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

package datacoord

import (
	"context"
	"fmt"
	"math"
	"strings"

	"github.com/samber/lo"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/internal/parser/planparserv2"
	"github.com/milvus-io/milvus/internal/util/indexparamcheck"
	"github.com/milvus-io/milvus/pkg/v2/common"
	pkgcommon "github.com/milvus-io/milvus/pkg/v2/common"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/metrics"
	"github.com/milvus-io/milvus/pkg/v2/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v2/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v2/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v2/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
	"github.com/milvus-io/milvus/pkg/v2/util/metautil"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v2/util/typeutil"
)

// serverID return the session serverID
func (s *Server) serverID() int64 {
	if s.session != nil {
		return s.session.GetServerID()
	}
	// return 0 if no session exist, only for UT
	return 0
}

func (s *Server) getFieldNameByID(schema *schemapb.CollectionSchema, fieldID int64) (string, error) {
	for _, field := range schema.GetFields() {
		if field.FieldID == fieldID {
			return field.Name, nil
		}
	}
	return "", nil
}

func (s *Server) getSchema(ctx context.Context, collID int64) (*schemapb.CollectionSchema, error) {
	resp, err := s.broker.DescribeCollectionInternal(ctx, collID)
	if err != nil {
		return nil, err
	}
	return resp.GetSchema(), nil
}

func isJsonField(schema *schemapb.CollectionSchema, fieldID int64) (bool, error) {
	for _, f := range schema.Fields {
		if f.FieldID == fieldID {
			return typeutil.IsJSONType(f.DataType), nil
		}
	}
	return false, merr.WrapErrFieldNotFound(fieldID)
}

func getIndexParam(indexParams []*commonpb.KeyValuePair, key string) (string, error) {
	for _, p := range indexParams {
		if p.Key == key {
			return p.Value, nil
		}
	}
	return "", merr.WrapErrParameterInvalidMsg("%s not found", key)
}

func setIndexParam(indexParams []*commonpb.KeyValuePair, key, value string) {
	for _, p := range indexParams {
		if p.Key == key {
			p.Value = value
		}
	}
}

func (s *Server) parseAndVerifyNestedPath(identifier string, schema *schemapb.CollectionSchema, fieldID int64) (string, error) {
	helper, err := typeutil.CreateSchemaHelper(schema)
	if err != nil {
		return "", err
	}

	var identifierExpr *planpb.Expr
	err = planparserv2.ParseIdentifier(helper, identifier, func(expr *planpb.Expr) error {
		identifierExpr = expr
		return nil
	})
	if err != nil {
		return "", err
	}
	if identifierExpr.GetColumnExpr().GetInfo().GetFieldId() != fieldID {
		return "", fmt.Errorf("field parsed from json path (%v) not match with field specified in request (%v)", identifierExpr.GetColumnExpr().GetInfo().GetFieldId(), fieldID)
	}

	nestedPath := identifierExpr.GetColumnExpr().GetInfo().GetNestedPath()
	// escape the nested path to avoid the path being interpreted as a JSON Pointer
	nestedPath = lo.Map(nestedPath, func(path string, _ int) string {
		s := strings.ReplaceAll(path, "~", "~0")
		s = strings.ReplaceAll(s, "/", "~1")
		return s
	})
	if len(nestedPath) == 0 {
		// if nested path is empty, it means the json path is the field name.
		// Dont return "/" here, it not a valid json path for simdjson.
		return "", nil
	}
	return "/" + strings.Join(nestedPath, "/"), nil
}

// CreateIndex create an index on collection.
// Index building is asynchronous, so when an index building request comes, an IndexID is assigned to the task and
// will get all flushed segments from DataCoord and record tasks with these segments. The background process
// indexBuilder will find this task and assign it to DataNode for execution.
func (s *Server) CreateIndex(ctx context.Context, req *indexpb.CreateIndexRequest) (*commonpb.Status, error) {
	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", req.GetCollectionID()),
	)
	log.Info("receive CreateIndex request",
		zap.String("IndexName", req.GetIndexName()), zap.Int64("fieldID", req.GetFieldID()),
		zap.Any("TypeParams", req.GetTypeParams()),
		zap.Any("IndexParams", req.GetIndexParams()),
		zap.Any("UserIndexParams", req.GetUserIndexParams()),
	)

	if err := merr.CheckHealthy(s.GetStateCode()); err != nil {
		log.Warn(msgDataCoordIsUnhealthy(paramtable.GetNodeID()), zap.Error(err))
		return merr.Status(err), nil
	}
	metrics.IndexRequestCounter.WithLabelValues(metrics.TotalLabel).Inc()

	schema, err := s.getSchema(ctx, req.GetCollectionID())
	if err != nil {
		return merr.Status(err), nil
	}
	isJson, err := isJsonField(schema, req.GetFieldID())
	if err != nil {
		return merr.Status(err), nil
	}

	if isJson {
		// check json_path and json_cast_type exist
		jsonPath, err := getIndexParam(req.GetIndexParams(), common.JSONPathKey)
		if err != nil {
			log.Warn("get json path failed", zap.Error(err))
			return merr.Status(err), nil
		}
		_, err = getIndexParam(req.GetIndexParams(), common.JSONCastTypeKey)
		if err != nil {
			log.Warn("get json cast type failed", zap.Error(err))
			return merr.Status(err), nil
		}

		nestedPath, err := s.parseAndVerifyNestedPath(jsonPath, schema, req.GetFieldID())
		if err != nil {
			log.Error("parse nested path failed", zap.Error(err))
			return merr.Status(err), nil
		}
		// set nested path as json path
		setIndexParam(req.GetIndexParams(), common.JSONPathKey, nestedPath)
	}

	if req.GetIndexName() == "" {
		indexes := s.meta.indexMeta.GetFieldIndexes(req.GetCollectionID(), req.GetFieldID(), req.GetIndexName())
		fieldName, err := s.getFieldNameByID(schema, req.GetFieldID())
		if err != nil {
			log.Warn("get field name from schema failed", zap.Int64("fieldID", req.GetFieldID()))
			return merr.Status(err), nil
		}
		defaultIndexName := fieldName
		if isJson {
			// ignore error, because it's already checked in getIndexParam before
			jsonPath, _ := getIndexParam(req.GetIndexParams(), common.JSONPathKey)
			// filter indexes by json path, the length of indexes should not be larger than 1
			// this is guaranteed by CanCreateIndex
			indexes = lo.Filter(indexes, func(index *model.Index, i int) bool {
				path, _ := getIndexParam(index.IndexParams, common.JSONPathKey)
				return path == jsonPath
			})

			defaultIndexName += jsonPath
		}
		if len(indexes) == 0 {
			req.IndexName = defaultIndexName
		} else if len(indexes) == 1 {
			req.IndexName = indexes[0].IndexName
		}
	}

	allocateIndexID, err := s.allocator.AllocID(ctx)
	if err != nil {
		log.Warn("failed to alloc indexID", zap.Error(err))
		metrics.IndexRequestCounter.WithLabelValues(metrics.FailLabel).Inc()
		return merr.Status(err), nil
	}

	// Get flushed segments and create index
	indexID, err := s.meta.indexMeta.CreateIndex(ctx, req, allocateIndexID, isJson)
	if err != nil {
		log.Error("CreateIndex fail",
			zap.Int64("fieldID", req.GetFieldID()), zap.String("indexName", req.GetIndexName()), zap.Error(err))
		metrics.IndexRequestCounter.WithLabelValues(metrics.FailLabel).Inc()
		return merr.Status(err), nil
	}

	select {
	case s.notifyIndexChan <- req.GetCollectionID():
	default:
	}

	log.Info("CreateIndex successfully",
		zap.String("IndexName", req.GetIndexName()), zap.Int64("fieldID", req.GetFieldID()),
		zap.Int64("IndexID", indexID))
	metrics.IndexRequestCounter.WithLabelValues(metrics.SuccessLabel).Inc()
	return merr.Success(), nil
}

func ValidateIndexParams(index *model.Index) error {
	if err := CheckDuplidateKey(index.IndexParams, "indexParams"); err != nil {
		return err
	}
	if err := CheckDuplidateKey(index.UserIndexParams, "userIndexParams"); err != nil {
		return err
	}
	if err := CheckDuplidateKey(index.TypeParams, "typeParams"); err != nil {
		return err
	}
	indexType := GetIndexType(index.IndexParams)
	indexParams := funcutil.KeyValuePair2Map(index.IndexParams)
	userIndexParams := funcutil.KeyValuePair2Map(index.UserIndexParams)
	if err := indexparamcheck.ValidateMmapIndexParams(indexType, indexParams); err != nil {
		return merr.WrapErrParameterInvalidMsg("invalid mmap index params: %s", err.Error())
	}
	if err := indexparamcheck.ValidateMmapIndexParams(indexType, userIndexParams); err != nil {
		return merr.WrapErrParameterInvalidMsg("invalid mmap user index params: %s", err.Error())
	}
	if err := indexparamcheck.ValidateOffsetCacheIndexParams(indexType, indexParams); err != nil {
		return merr.WrapErrParameterInvalidMsg("invalid offset cache index params: %s", err.Error())
	}
	if err := indexparamcheck.ValidateOffsetCacheIndexParams(indexType, userIndexParams); err != nil {
		return merr.WrapErrParameterInvalidMsg("invalid offset cache index params: %s", err.Error())
	}
	return nil
}

func CheckDuplidateKey(kvs []*commonpb.KeyValuePair, tag string) error {
	keySet := typeutil.NewSet[string]()
	for _, kv := range kvs {
		if keySet.Contain(kv.GetKey()) {
			return merr.WrapErrParameterInvalidMsg("duplicate %s key in %s params", kv.GetKey(), tag)
		}
		keySet.Insert(kv.GetKey())
	}
	return nil
}

func UpdateParams(index *model.Index, from []*commonpb.KeyValuePair, updates []*commonpb.KeyValuePair) []*commonpb.KeyValuePair {
	params := make(map[string]string)
	for _, param := range from {
		params[param.GetKey()] = param.GetValue()
	}

	// update the params
	for _, param := range updates {
		params[param.GetKey()] = param.GetValue()
	}

	return lo.MapToSlice(params, func(k string, v string) *commonpb.KeyValuePair {
		return &commonpb.KeyValuePair{
			Key:   k,
			Value: v,
		}
	})
}

func DeleteParams(from []*commonpb.KeyValuePair, deletes []string) []*commonpb.KeyValuePair {
	params := make(map[string]string)
	for _, param := range from {
		params[param.GetKey()] = param.GetValue()
	}

	// delete the params
	for _, key := range deletes {
		delete(params, key)
	}

	return lo.MapToSlice(params, func(k string, v string) *commonpb.KeyValuePair {
		return &commonpb.KeyValuePair{
			Key:   k,
			Value: v,
		}
	})
}

func (s *Server) AlterIndex(ctx context.Context, req *indexpb.AlterIndexRequest) (*commonpb.Status, error) {
	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", req.GetCollectionID()),
		zap.String("indexName", req.GetIndexName()),
	)
	log.Info("received AlterIndex request",
		zap.Any("params", req.GetParams()),
		zap.Any("deletekeys", req.GetDeleteKeys()))

	if req.IndexName == "" {
		return merr.Status(merr.WrapErrParameterInvalidMsg("index name is empty")), nil
	}

	if err := merr.CheckHealthy(s.GetStateCode()); err != nil {
		log.Warn(msgDataCoordIsUnhealthy(paramtable.GetNodeID()), zap.Error(err))
		return merr.Status(err), nil
	}

	indexes := s.meta.indexMeta.GetIndexesForCollection(req.GetCollectionID(), req.GetIndexName())
	if len(indexes) == 0 {
		err := merr.WrapErrIndexNotFound(req.GetIndexName())
		return merr.Status(err), nil
	}

	if len(req.GetDeleteKeys()) > 0 && len(req.GetParams()) > 0 {
		return merr.Status(merr.WrapErrParameterInvalidMsg("cannot provide both DeleteKeys and ExtraParams")), nil
	}

	collInfo, err := s.handler.GetCollection(ctx, req.GetCollectionID())
	if err != nil {
		log.Warn("failed to get collection", zap.Error(err))
		return merr.Status(err), nil
	}
	schemaHelper, err := typeutil.CreateSchemaHelper(collInfo.Schema)
	if err != nil {
		log.Warn("failed to create schema helper", zap.Error(err))
		return merr.Status(err), nil
	}

	reqIndexParamMap := funcutil.KeyValuePair2Map(req.GetParams())

	for _, index := range indexes {
		if len(req.GetParams()) > 0 {
			fieldSchema, err := schemaHelper.GetFieldFromID(index.FieldID)
			if err != nil {
				log.Warn("failed to get field schema", zap.Error(err))
				return merr.Status(err), nil
			}
			isVecIndex := typeutil.IsVectorType(fieldSchema.DataType)
			err = pkgcommon.ValidateAutoIndexMmapConfig(Params.AutoIndexConfig.Enable.GetAsBool(), isVecIndex, reqIndexParamMap)
			if err != nil {
				log.Warn("failed to validate auto index mmap config", zap.Error(err))
				return merr.Status(err), nil
			}

			// update user index params
			newUserIndexParams := UpdateParams(index, index.UserIndexParams, req.GetParams())
			log.Info("alter index user index params",
				zap.String("indexName", index.IndexName),
				zap.Any("params", newUserIndexParams),
			)
			index.UserIndexParams = newUserIndexParams

			// update index params
			newIndexParams := UpdateParams(index, index.IndexParams, req.GetParams())
			log.Info("alter index index params",
				zap.String("indexName", index.IndexName),
				zap.Any("params", newIndexParams),
			)
			index.IndexParams = newIndexParams
		} else if len(req.GetDeleteKeys()) > 0 {
			// delete user index params
			newUserIndexParams := DeleteParams(index.UserIndexParams, req.GetDeleteKeys())
			log.Info("alter index user deletekeys",
				zap.String("indexName", index.IndexName),
				zap.Any("params", newUserIndexParams),
			)
			index.UserIndexParams = newUserIndexParams

			// delete index params
			newIndexParams := DeleteParams(index.IndexParams, req.GetDeleteKeys())
			log.Info("alter index index deletekeys",
				zap.String("indexName", index.IndexName),
				zap.Any("params", newIndexParams),
			)
			index.IndexParams = newIndexParams
		}

		if err := ValidateIndexParams(index); err != nil {
			return merr.Status(err), nil
		}
	}

	err = s.meta.indexMeta.AlterIndex(ctx, indexes...)
	if err != nil {
		log.Warn("failed to alter index", zap.Error(err))
		return merr.Status(err), nil
	}

	return merr.Success(), nil
}

// GetIndexState gets the index state of the index name in the request from Proxy.
// Deprecated
func (s *Server) GetIndexState(ctx context.Context, req *indexpb.GetIndexStateRequest) (*indexpb.GetIndexStateResponse, error) {
	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", req.GetCollectionID()),
		zap.String("indexName", req.GetIndexName()),
	)
	log.Info("receive GetIndexState request")

	if err := merr.CheckHealthy(s.GetStateCode()); err != nil {
		log.Warn(msgDataCoordIsUnhealthy(paramtable.GetNodeID()), zap.Error(err))
		return &indexpb.GetIndexStateResponse{
			Status: merr.Status(err),
		}, nil
	}

	indexes := s.meta.indexMeta.GetIndexesForCollection(req.GetCollectionID(), req.GetIndexName())
	if len(indexes) == 0 {
		err := merr.WrapErrIndexNotFound(req.GetIndexName())
		log.Warn("GetIndexState fail", zap.Error(err))
		return &indexpb.GetIndexStateResponse{
			Status: merr.Status(err),
		}, nil
	}
	if len(indexes) > 1 {
		log.Warn(msgAmbiguousIndexName())
		err := merr.WrapErrIndexDuplicate(req.GetIndexName())
		return &indexpb.GetIndexStateResponse{
			Status: merr.Status(err),
		}, nil
	}
	ret := &indexpb.GetIndexStateResponse{
		Status: merr.Success(),
		State:  commonpb.IndexState_Finished,
	}

	indexInfo := &indexpb.IndexInfo{}
	// The total rows of all indexes should be based on the current perspective
	segments := s.selectSegmentIndexesStats(ctx, WithCollection(req.GetCollectionID()), SegmentFilterFunc(func(info *SegmentInfo) bool {
		return info.GetLevel() != datapb.SegmentLevel_L0 && (isFlush(info) || info.GetState() == commonpb.SegmentState_Dropped)
	}))

	s.completeIndexInfo(indexInfo, indexes[0], segments, false, indexes[0].CreateTime)
	ret.State = indexInfo.State
	ret.FailReason = indexInfo.IndexStateFailReason

	log.Info("GetIndexState success",
		zap.String("state", ret.GetState().String()),
	)
	return ret, nil
}

func (s *Server) GetSegmentIndexState(ctx context.Context, req *indexpb.GetSegmentIndexStateRequest) (*indexpb.GetSegmentIndexStateResponse, error) {
	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", req.GetCollectionID()),
	)
	log.Info("receive GetSegmentIndexState",
		zap.String("IndexName", req.GetIndexName()),
		zap.Int64s("segmentIDs", req.GetSegmentIDs()),
	)

	if err := merr.CheckHealthy(s.GetStateCode()); err != nil {
		log.Warn(msgDataCoordIsUnhealthy(paramtable.GetNodeID()), zap.Error(err))
		return &indexpb.GetSegmentIndexStateResponse{
			Status: merr.Status(err),
		}, nil
	}

	ret := &indexpb.GetSegmentIndexStateResponse{
		Status: merr.Success(),
		States: make([]*indexpb.SegmentIndexState, 0),
	}
	indexID2CreateTs := s.meta.indexMeta.GetIndexIDByName(req.GetCollectionID(), req.GetIndexName())
	if len(indexID2CreateTs) == 0 {
		err := merr.WrapErrIndexNotFound(req.GetIndexName())
		log.Warn("GetSegmentIndexState fail", zap.String("indexName", req.GetIndexName()), zap.Error(err))
		return &indexpb.GetSegmentIndexStateResponse{
			Status: merr.Status(err),
		}, nil
	}
	for _, segID := range req.GetSegmentIDs() {
		for indexID := range indexID2CreateTs {
			state := s.meta.indexMeta.GetSegmentIndexState(req.GetCollectionID(), segID, indexID)
			ret.States = append(ret.States, state)
		}
	}
	log.Info("GetSegmentIndexState successfully", zap.String("indexName", req.GetIndexName()))
	return ret, nil
}

func (s *Server) selectSegmentIndexesStats(ctx context.Context, filters ...SegmentFilter) map[int64]*indexStats {
	ret := make(map[int64]*indexStats)

	segments := s.meta.SelectSegments(ctx, filters...)
	segmentIDs := lo.Map(segments, func(info *SegmentInfo, i int) int64 {
		return info.GetID()
	})
	if len(segments) == 0 {
		return ret
	}
	segmentsIndexes := s.meta.indexMeta.getSegmentsIndexStates(segments[0].CollectionID, segmentIDs)
	for _, info := range segments {
		is := &indexStats{
			ID:             info.GetID(),
			numRows:        info.GetNumOfRows(),
			compactionFrom: info.GetCompactionFrom(),
			indexStates:    segmentsIndexes[info.GetID()],
			state:          info.GetState(),
			lastExpireTime: info.GetLastExpireTime(),
		}
		ret[info.GetID()] = is
	}
	return ret
}

func (s *Server) countIndexedRows(indexInfo *indexpb.IndexInfo, segments map[int64]*indexStats) int64 {
	unIndexed, indexed := typeutil.NewSet[int64](), typeutil.NewSet[int64]()
	for segID, seg := range segments {
		if seg.state != commonpb.SegmentState_Flushed && seg.state != commonpb.SegmentState_Flushing {
			continue
		}
		segIdx, ok := seg.indexStates[indexInfo.IndexID]
		if !ok {
			unIndexed.Insert(segID)
			continue
		}
		switch segIdx.GetState() {
		case commonpb.IndexState_Finished:
			indexed.Insert(segID)
		default:
			unIndexed.Insert(segID)
		}
	}
	retrieveContinue := len(unIndexed) != 0
	for retrieveContinue {
		for segID := range unIndexed {
			unIndexed.Remove(segID)
			segment := segments[segID]
			if segment == nil || len(segment.compactionFrom) == 0 {
				continue
			}
			for _, fromID := range segment.compactionFrom {
				fromSeg := segments[fromID]
				if fromSeg == nil {
					continue
				}
				if segIndex, ok := fromSeg.indexStates[indexInfo.IndexID]; ok && segIndex.GetState() == commonpb.IndexState_Finished {
					indexed.Insert(fromID)
					continue
				}
				unIndexed.Insert(fromID)
			}
		}
		retrieveContinue = len(unIndexed) != 0
	}
	indexedRows := int64(0)
	for segID := range indexed {
		segment := segments[segID]
		if segment != nil {
			indexedRows += segment.numRows
		}
	}
	return indexedRows
}

// completeIndexInfo get the index row count and index task state
// if realTime, calculate current statistics
// if not realTime, which means get info of the prior `CreateIndex` action, skip segments created after index's create time
func (s *Server) completeIndexInfo(indexInfo *indexpb.IndexInfo, index *model.Index, segments map[int64]*indexStats, realTime bool, ts Timestamp) {
	var (
		cntNone          = 0
		cntUnissued      = 0
		cntInProgress    = 0
		cntFinished      = 0
		cntFailed        = 0
		failReason       string
		totalRows        = int64(0)
		indexedRows      = int64(0)
		pendingIndexRows = int64(0)
	)

	minIndexVersion := int32(math.MaxInt32)
	maxIndexVersion := int32(math.MinInt32)

	for segID, seg := range segments {
		if seg.state != commonpb.SegmentState_Flushed && seg.state != commonpb.SegmentState_Flushing {
			continue
		}
		totalRows += seg.numRows
		segIdx, ok := seg.indexStates[index.IndexID]

		if !ok {
			if seg.lastExpireTime <= ts {
				cntUnissued++
			}
			pendingIndexRows += seg.numRows
			continue
		}
		if segIdx.GetState() != commonpb.IndexState_Finished {
			pendingIndexRows += seg.numRows
		}

		// if realTime, calculate current statistics
		// if not realTime, skip segments created after index create
		if !realTime && seg.lastExpireTime > ts {
			continue
		}

		switch segIdx.GetState() {
		case commonpb.IndexState_IndexStateNone:
			// can't to here
			log.Warn("receive unexpected index state: IndexStateNone", zap.Int64("segmentID", segID))
			cntNone++
		case commonpb.IndexState_Unissued:
			cntUnissued++
		case commonpb.IndexState_InProgress:
			cntInProgress++
		case commonpb.IndexState_Finished:
			cntFinished++
			indexedRows += seg.numRows
			if segIdx.IndexVersion < minIndexVersion {
				minIndexVersion = segIdx.IndexVersion
			}
			if segIdx.IndexVersion > maxIndexVersion {
				maxIndexVersion = segIdx.IndexVersion
			}
		case commonpb.IndexState_Failed:
			cntFailed++
			failReason += fmt.Sprintf("%d: %s;", segID, segIdx.FailReason)
		}
	}

	if realTime {
		indexInfo.IndexedRows = indexedRows
	} else {
		indexInfo.IndexedRows = s.countIndexedRows(indexInfo, segments)
	}
	indexInfo.TotalRows = totalRows
	indexInfo.PendingIndexRows = pendingIndexRows
	indexInfo.MinIndexVersion = minIndexVersion
	indexInfo.MaxIndexVersion = maxIndexVersion
	switch {
	case cntFailed > 0:
		indexInfo.State = commonpb.IndexState_Failed
		indexInfo.IndexStateFailReason = failReason
	case cntInProgress > 0 || cntUnissued > 0:
		indexInfo.State = commonpb.IndexState_InProgress
	case cntNone > 0:
		indexInfo.State = commonpb.IndexState_IndexStateNone
	default:
		indexInfo.State = commonpb.IndexState_Finished
	}

	log.RatedInfo(60, "completeIndexInfo success", zap.Int64("collectionID", index.CollectionID), zap.Int64("indexID", index.IndexID),
		zap.Int64("totalRows", indexInfo.TotalRows), zap.Int64("indexRows", indexInfo.IndexedRows),
		zap.Int64("pendingIndexRows", indexInfo.PendingIndexRows),
		zap.String("state", indexInfo.State.String()), zap.String("failReason", indexInfo.IndexStateFailReason),
		zap.Int32("minIndexVersion", indexInfo.MinIndexVersion), zap.Int32("maxIndexVersion", indexInfo.MaxIndexVersion))
}

// GetIndexBuildProgress get the index building progress by num rows.
// Deprecated
func (s *Server) GetIndexBuildProgress(ctx context.Context, req *indexpb.GetIndexBuildProgressRequest) (*indexpb.GetIndexBuildProgressResponse, error) {
	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", req.GetCollectionID()),
	)
	log.Info("receive GetIndexBuildProgress request", zap.String("indexName", req.GetIndexName()))

	if err := merr.CheckHealthy(s.GetStateCode()); err != nil {
		log.Warn(msgDataCoordIsUnhealthy(paramtable.GetNodeID()), zap.Error(err))
		return &indexpb.GetIndexBuildProgressResponse{
			Status: merr.Status(err),
		}, nil
	}

	indexes := s.meta.indexMeta.GetIndexesForCollection(req.GetCollectionID(), req.GetIndexName())
	if len(indexes) == 0 {
		err := merr.WrapErrIndexNotFound(req.GetIndexName())
		log.Warn("GetIndexBuildProgress fail", zap.String("indexName", req.IndexName), zap.Error(err))
		return &indexpb.GetIndexBuildProgressResponse{
			Status: merr.Status(err),
		}, nil
	}

	if len(indexes) > 1 {
		log.Warn(msgAmbiguousIndexName())
		err := merr.WrapErrIndexDuplicate(req.GetIndexName())
		return &indexpb.GetIndexBuildProgressResponse{
			Status: merr.Status(err),
		}, nil
	}
	indexInfo := &indexpb.IndexInfo{
		CollectionID:     req.GetCollectionID(),
		IndexID:          indexes[0].IndexID,
		IndexedRows:      0,
		TotalRows:        0,
		PendingIndexRows: 0,
		State:            0,
	}

	// The total rows of all indexes should be based on the current perspective
	segments := s.selectSegmentIndexesStats(ctx, WithCollection(req.GetCollectionID()), SegmentFilterFunc(func(info *SegmentInfo) bool {
		return info.GetLevel() != datapb.SegmentLevel_L0 && (isFlush(info) || info.GetState() == commonpb.SegmentState_Dropped)
	}))

	s.completeIndexInfo(indexInfo, indexes[0], segments, false, indexes[0].CreateTime)
	log.Info("GetIndexBuildProgress success", zap.Int64("collectionID", req.GetCollectionID()),
		zap.String("indexName", req.GetIndexName()))
	return &indexpb.GetIndexBuildProgressResponse{
		Status:           merr.Success(),
		IndexedRows:      indexInfo.IndexedRows,
		TotalRows:        indexInfo.TotalRows,
		PendingIndexRows: indexInfo.PendingIndexRows,
	}, nil
}

// indexStats just for indexing statistics.
// Please use it judiciously.
type indexStats struct {
	ID             int64
	numRows        int64
	compactionFrom []int64
	indexStates    map[int64]*indexpb.SegmentIndexState
	state          commonpb.SegmentState
	lastExpireTime uint64
}

// DescribeIndex describe the index info of the collection.
func (s *Server) DescribeIndex(ctx context.Context, req *indexpb.DescribeIndexRequest) (*indexpb.DescribeIndexResponse, error) {
	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", req.GetCollectionID()),
		zap.String("indexName", req.GetIndexName()),
		zap.Uint64("timestamp", req.GetTimestamp()),
	)
	if err := merr.CheckHealthy(s.GetStateCode()); err != nil {
		log.Warn(msgDataCoordIsUnhealthy(paramtable.GetNodeID()), zap.Error(err))
		return &indexpb.DescribeIndexResponse{
			Status: merr.Status(err),
		}, nil
	}

	indexes := s.meta.indexMeta.GetIndexesForCollection(req.GetCollectionID(), req.GetIndexName())
	if len(indexes) == 0 {
		err := merr.WrapErrIndexNotFound(req.GetIndexName())
		log.RatedWarn(60, "DescribeIndex fail", zap.Error(err))
		return &indexpb.DescribeIndexResponse{
			Status: merr.Status(err),
		}, nil
	}

	// The total rows of all indexes should be based on the current perspective
	segments := s.selectSegmentIndexesStats(ctx, WithCollection(req.GetCollectionID()), SegmentFilterFunc(func(info *SegmentInfo) bool {
		return info.GetLevel() != datapb.SegmentLevel_L0 && (isFlush(info) || info.GetState() == commonpb.SegmentState_Dropped)
	}))

	indexInfos := make([]*indexpb.IndexInfo, 0)
	for _, index := range indexes {
		indexInfo := &indexpb.IndexInfo{
			CollectionID:         index.CollectionID,
			FieldID:              index.FieldID,
			IndexName:            index.IndexName,
			IndexID:              index.IndexID,
			TypeParams:           index.TypeParams,
			IndexParams:          index.IndexParams,
			IndexedRows:          0,
			TotalRows:            0,
			State:                0,
			IndexStateFailReason: "",
			IsAutoIndex:          index.IsAutoIndex,
			UserIndexParams:      index.UserIndexParams,
		}
		createTs := index.CreateTime
		if req.GetTimestamp() != 0 {
			createTs = req.GetTimestamp()
		}
		s.completeIndexInfo(indexInfo, index, segments, false, createTs)
		indexInfos = append(indexInfos, indexInfo)
	}
	return &indexpb.DescribeIndexResponse{
		Status:     merr.Success(),
		IndexInfos: indexInfos,
	}, nil
}

// GetIndexStatistics get the statistics of the index. DescribeIndex doesn't contain statistics.
func (s *Server) GetIndexStatistics(ctx context.Context, req *indexpb.GetIndexStatisticsRequest) (*indexpb.GetIndexStatisticsResponse, error) {
	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", req.GetCollectionID()),
	)
	log.Info("receive GetIndexStatistics request", zap.String("indexName", req.GetIndexName()))
	if err := merr.CheckHealthy(s.GetStateCode()); err != nil {
		log.Warn(msgDataCoordIsUnhealthy(paramtable.GetNodeID()), zap.Error(err))
		return &indexpb.GetIndexStatisticsResponse{
			Status: merr.Status(err),
		}, nil
	}

	indexes := s.meta.indexMeta.GetIndexesForCollection(req.GetCollectionID(), req.GetIndexName())
	if len(indexes) == 0 {
		err := merr.WrapErrIndexNotFound(req.GetIndexName())
		log.Warn("GetIndexStatistics fail",
			zap.String("indexName", req.GetIndexName()),
			zap.Error(err))
		return &indexpb.GetIndexStatisticsResponse{
			Status: merr.Status(err),
		}, nil
	}

	// The total rows of all indexes should be based on the current perspective
	segments := s.selectSegmentIndexesStats(ctx, WithCollection(req.GetCollectionID()), SegmentFilterFunc(func(info *SegmentInfo) bool {
		return info.GetLevel() != datapb.SegmentLevel_L0 && (isFlush(info) || info.GetState() == commonpb.SegmentState_Dropped)
	}))

	indexInfos := make([]*indexpb.IndexInfo, 0)
	for _, index := range indexes {
		indexInfo := &indexpb.IndexInfo{
			CollectionID:         index.CollectionID,
			FieldID:              index.FieldID,
			IndexName:            index.IndexName,
			IndexID:              index.IndexID,
			TypeParams:           index.TypeParams,
			IndexParams:          index.IndexParams,
			IndexedRows:          0,
			TotalRows:            0,
			State:                0,
			IndexStateFailReason: "",
			IsAutoIndex:          index.IsAutoIndex,
			UserIndexParams:      index.UserIndexParams,
		}
		s.completeIndexInfo(indexInfo, index, segments, true, index.CreateTime)
		indexInfos = append(indexInfos, indexInfo)
	}
	log.Debug("GetIndexStatisticsResponse success",
		zap.String("indexName", req.GetIndexName()))
	return &indexpb.GetIndexStatisticsResponse{
		Status:     merr.Success(),
		IndexInfos: indexInfos,
	}, nil
}

// DropIndex deletes indexes based on IndexName. One IndexName corresponds to the index of an entire column. A column is
// divided into many segments, and each segment corresponds to an IndexBuildID. DataCoord uses IndexBuildID to record
// index tasks.
func (s *Server) DropIndex(ctx context.Context, req *indexpb.DropIndexRequest) (*commonpb.Status, error) {
	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", req.GetCollectionID()),
	)
	log.Info("receive DropIndex request",
		zap.Int64s("partitionIDs", req.GetPartitionIDs()), zap.String("indexName", req.GetIndexName()),
		zap.Bool("drop all indexes", req.GetDropAll()))

	if err := merr.CheckHealthy(s.GetStateCode()); err != nil {
		log.Warn(msgDataCoordIsUnhealthy(paramtable.GetNodeID()), zap.Error(err))
		return merr.Status(err), nil
	}

	indexes := s.meta.indexMeta.GetIndexesForCollection(req.GetCollectionID(), req.GetIndexName())
	if len(indexes) == 0 {
		log.Info(fmt.Sprintf("there is no index on collection: %d with the index name: %s", req.CollectionID, req.IndexName))
		return merr.Success(), nil
	}

	if !req.GetDropAll() && len(indexes) > 1 {
		log.Warn(msgAmbiguousIndexName())
		err := merr.WrapErrIndexDuplicate(req.GetIndexName())
		return merr.Status(err), nil
	}
	indexIDs := make([]UniqueID, 0)
	for _, index := range indexes {
		indexIDs = append(indexIDs, index.IndexID)
	}
	// Compatibility logic. To prevent the index on the corresponding segments
	// from being dropped at the same time when dropping_partition in version 2.1
	if len(req.GetPartitionIDs()) == 0 {
		// drop collection index
		err := s.meta.indexMeta.MarkIndexAsDeleted(ctx, req.GetCollectionID(), indexIDs)
		if err != nil {
			log.Warn("DropIndex fail", zap.String("indexName", req.IndexName), zap.Error(err))
			return merr.Status(err), nil
		}
	}

	log.Debug("DropIndex success", zap.Int64s("partitionIDs", req.GetPartitionIDs()),
		zap.String("indexName", req.GetIndexName()), zap.Int64s("indexIDs", indexIDs))
	return merr.Success(), nil
}

// GetIndexInfos gets the index file paths for segment from DataCoord.
func (s *Server) GetIndexInfos(ctx context.Context, req *indexpb.GetIndexInfoRequest) (*indexpb.GetIndexInfoResponse, error) {
	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", req.GetCollectionID()),
	)

	if err := merr.CheckHealthy(s.GetStateCode()); err != nil {
		log.Warn(msgDataCoordIsUnhealthy(paramtable.GetNodeID()), zap.Error(err))
		return &indexpb.GetIndexInfoResponse{
			Status: merr.Status(err),
		}, nil
	}
	ret := &indexpb.GetIndexInfoResponse{
		Status:      merr.Success(),
		SegmentInfo: map[int64]*indexpb.SegmentInfo{},
	}

	segmentsIndexes := s.meta.indexMeta.GetSegmentsIndexes(req.GetCollectionID(), req.GetSegmentIDs())
	for _, segID := range req.GetSegmentIDs() {
		segIdxes := segmentsIndexes[segID]
		ret.SegmentInfo[segID] = &indexpb.SegmentInfo{
			CollectionID: req.GetCollectionID(),
			SegmentID:    segID,
			EnableIndex:  false,
			IndexInfos:   make([]*indexpb.IndexFilePathInfo, 0),
		}
		if len(segIdxes) != 0 {
			ret.SegmentInfo[segID].EnableIndex = true
			for _, segIdx := range segIdxes {
				if segIdx.IndexState == commonpb.IndexState_Finished {
					indexFilePaths := metautil.BuildSegmentIndexFilePaths(s.meta.chunkManager.RootPath(), segIdx.BuildID, segIdx.IndexVersion,
						segIdx.PartitionID, segIdx.SegmentID, segIdx.IndexFileKeys)
					indexParams := s.meta.indexMeta.GetIndexParams(segIdx.CollectionID, segIdx.IndexID)
					indexParams = append(indexParams, s.meta.indexMeta.GetTypeParams(segIdx.CollectionID, segIdx.IndexID)...)
					ret.SegmentInfo[segID].IndexInfos = append(ret.SegmentInfo[segID].IndexInfos,
						&indexpb.IndexFilePathInfo{
							SegmentID:           segID,
							FieldID:             s.meta.indexMeta.GetFieldIDByIndexID(segIdx.CollectionID, segIdx.IndexID),
							IndexID:             segIdx.IndexID,
							BuildID:             segIdx.BuildID,
							IndexName:           s.meta.indexMeta.GetIndexNameByID(segIdx.CollectionID, segIdx.IndexID),
							IndexParams:         indexParams,
							IndexFilePaths:      indexFilePaths,
							SerializedSize:      segIdx.IndexSerializedSize,
							MemSize:             segIdx.IndexMemSize,
							IndexVersion:        segIdx.IndexVersion,
							NumRows:             segIdx.NumRows,
							CurrentIndexVersion: segIdx.CurrentIndexVersion,
						})
				}
			}
		}
	}

	log.Debug("GetIndexInfos successfully", zap.String("indexName", req.GetIndexName()))

	return ret, nil
}

// ListIndexes returns all indexes created on provided collection.
func (s *Server) ListIndexes(ctx context.Context, req *indexpb.ListIndexesRequest) (*indexpb.ListIndexesResponse, error) {
	log := log.Ctx(ctx).With(
		zap.Int64("collectionID", req.GetCollectionID()),
	)

	if err := merr.CheckHealthy(s.GetStateCode()); err != nil {
		log.Warn(msgDataCoordIsUnhealthy(paramtable.GetNodeID()), zap.Error(err))
		return &indexpb.ListIndexesResponse{
			Status: merr.Status(err),
		}, nil
	}

	indexes := s.meta.indexMeta.GetIndexesForCollection(req.GetCollectionID(), "")

	indexInfos := lo.Map(indexes, func(index *model.Index, _ int) *indexpb.IndexInfo {
		return &indexpb.IndexInfo{
			CollectionID:    index.CollectionID,
			FieldID:         index.FieldID,
			IndexName:       index.IndexName,
			IndexID:         index.IndexID,
			TypeParams:      index.TypeParams,
			IndexParams:     index.IndexParams,
			IsAutoIndex:     index.IsAutoIndex,
			UserIndexParams: index.UserIndexParams,
		}
	})
	log.Debug("List index success")
	return &indexpb.ListIndexesResponse{
		Status:     merr.Success(),
		IndexInfos: indexInfos,
	}, nil
}
