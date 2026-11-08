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

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/internal/metastore/model"
)

func TestPackIndexInfos(t *testing.T) {
	live1 := &model.Index{CollectionID: 1, FieldID: 101, IndexID: 1, IndexName: "a", IsAutoIndex: true}
	live2 := &model.Index{CollectionID: 1, FieldID: 102, IndexID: 2, IndexName: "b"}
	live3 := &model.Index{CollectionID: 1, FieldID: 103, IndexID: 3, IndexName: "c"}
	deleted := &model.Index{CollectionID: 1, FieldID: 104, IndexID: 101, IndexName: "d", IsDeleted: true}

	cases := []struct {
		name     string
		indexes  []*model.Index
		indexIDs []int64
		wantIDs  []int64
	}{
		{name: "live only", indexes: []*model.Index{live1, deleted}, wantIDs: []int64{1}},
		{name: "exact refs include deleted", indexes: []*model.Index{live1, deleted}, indexIDs: []int64{101}, wantIDs: []int64{101}},
		{name: "exact refs skip absent", indexes: []*model.Index{live1}, indexIDs: []int64{1, 7}, wantIDs: []int64{1}},
		{name: "sorted", indexes: []*model.Index{live3, live1, live2}, wantIDs: []int64{1, 2, 3}},
		{name: "empty", wantIDs: []int64{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			infos := PackIndexInfos(tc.indexes, tc.indexIDs)
			ids := make([]int64, 0, len(infos))
			for _, info := range infos {
				ids = append(ids, info.GetIndexID())
			}
			assert.Equal(t, tc.wantIDs, ids)
		})
	}

	infos := PackIndexInfos([]*model.Index{live1}, nil)
	require.Len(t, infos, 1)
	assert.Equal(t, int64(1), infos[0].GetCollectionID())
	assert.Equal(t, int64(101), infos[0].GetFieldID())
	assert.Equal(t, "a", infos[0].GetIndexName())
	assert.True(t, infos[0].GetIsAutoIndex())
}

func TestPackIndexInfosAppendDoesNotAlias(t *testing.T) {
	params := make([]*commonpb.KeyValuePair, 1, 4)
	params[0] = &commonpb.KeyValuePair{Key: "index_type", Value: "HNSW"}
	stored := &model.Index{CollectionID: 1, IndexID: 1, IndexParams: params, TypeParams: params, UserIndexParams: params}

	info := PackIndexInfos([]*model.Index{stored}, nil)[0]
	assert.Same(t, params[0], info.IndexParams[0], "entries are shared")
	info.IndexParams = append(info.IndexParams, &commonpb.KeyValuePair{Key: "warmup", Value: "sync"})
	info.TypeParams = append(info.TypeParams, &commonpb.KeyValuePair{Key: "dim", Value: "8"})
	info.UserIndexParams = append(info.UserIndexParams, &commonpb.KeyValuePair{Key: "x", Value: "y"})

	assert.Len(t, stored.IndexParams, 1)
	assert.Nil(t, stored.IndexParams[:2][1], "an append by the caller must not write into the stored array")
}
