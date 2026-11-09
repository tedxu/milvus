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

package index

import (
	"context"
	"testing"

	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/metacache"
	catalogmocks "github.com/milvus-io/milvus/internal/metastore/mocks"
	"github.com/milvus-io/milvus/internal/metastore/model"
)

func TestIndexMetaPublishesToStore(t *testing.T) {
	ctx := context.Background()
	index := &model.Index{CollectionID: 1, FieldID: 101, IndexID: 100, IndexName: "idx"}
	record := &model.SegmentIndex{CollectionID: 1, PartitionID: 2, SegmentID: 10, IndexID: 100, BuildID: 1000}

	t.Run("published after the catalog write", func(t *testing.T) {
		catalog := catalogmocks.NewDataCoordCatalog(t)
		catalog.EXPECT().ListIndexes(mock.Anything).Return(nil, nil)
		catalog.EXPECT().CreateIndex(mock.Anything, mock.Anything).Return(nil)
		catalog.EXPECT().CreateSegmentIndex(mock.Anything, mock.Anything).Return(nil)
		store := metacache.NewMetaStore(catalog)
		require.NoError(t, store.LoadIndexesFromCatalog(ctx, nil))
		im := NewIndexMeta(ctx, catalog, store)
		require.Same(t, store, im.Store())

		require.NoError(t, im.CreateIndex(ctx, index))
		require.NoError(t, im.AddSegmentIndex(ctx, record))

		got, ok := store.GetIndex(1, 100)
		require.True(t, ok)
		assert.Same(t, index, got)
		current, ok := store.GetSegmentIndex(10, 100)
		require.True(t, ok)
		assert.Equal(t, int64(1000), current.BuildID)
		byBuild, ok := store.GetSegmentIndexByBuildID(1000)
		require.True(t, ok)
		assert.Same(t, current, byBuild)
	})

	t.Run("not published when the catalog write fails", func(t *testing.T) {
		catalog := catalogmocks.NewDataCoordCatalog(t)
		catalog.EXPECT().ListIndexes(mock.Anything).Return(nil, nil)
		catalog.EXPECT().CreateIndex(mock.Anything, mock.Anything).Return(errors.New("etcd down"))
		catalog.EXPECT().CreateSegmentIndex(mock.Anything, mock.Anything).Return(errors.New("etcd down"))
		store := metacache.NewMetaStore(catalog)
		require.NoError(t, store.LoadIndexesFromCatalog(ctx, nil))
		im := NewIndexMeta(ctx, catalog, store)

		assert.Error(t, im.CreateIndex(ctx, index))
		assert.Error(t, im.AddSegmentIndex(ctx, &model.SegmentIndex{CollectionID: 1, SegmentID: 10, IndexID: 100, BuildID: 1000}))

		_, ok := store.GetIndex(1, 100)
		assert.False(t, ok)
		_, ok = store.GetSegmentIndexByBuildID(1000)
		assert.False(t, ok)
	})

	t.Run("removing a superseded build keeps the current one", func(t *testing.T) {
		catalog := catalogmocks.NewDataCoordCatalog(t)
		catalog.EXPECT().ListIndexes(mock.Anything).Return(nil, nil)
		catalog.EXPECT().CreateSegmentIndex(mock.Anything, mock.Anything).Return(nil)
		catalog.EXPECT().DropSegmentIndex(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil)
		store := metacache.NewMetaStore(catalog)
		require.NoError(t, store.LoadIndexesFromCatalog(ctx, nil))
		im := NewIndexMeta(ctx, catalog, store)

		require.NoError(t, im.AddSegmentIndex(ctx, &model.SegmentIndex{CollectionID: 1, SegmentID: 10, IndexID: 100, BuildID: 1}))
		require.NoError(t, im.AddSegmentIndex(ctx, &model.SegmentIndex{CollectionID: 1, SegmentID: 10, IndexID: 100, BuildID: 2}))
		_, ok := im.GetIndexJob(1)
		require.True(t, ok, "GC still finds the superseded build")

		require.NoError(t, im.RemoveSegmentIndex(ctx, 1))
		current, ok := store.GetSegmentIndex(10, 100)
		require.True(t, ok)
		assert.Equal(t, int64(2), current.BuildID)
		_, ok = im.GetIndexJob(1)
		assert.False(t, ok)
	})
}
