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
	"cmp"
	"slices"

	"github.com/milvus-io/milvus/internal/metastore/model"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
)

// PackIndexInfos converts store definitions to IndexInfo, sorted by index ID.
// With no indexIDs it returns the live definitions only. With indexIDs it
// returns exactly those that exist, including logically deleted ones, which an
// older live QueryView may still reference.
//
// The infos share the definitions' param entries, so callers must not modify
// an entry. Each param slice is clipped, so an append by the caller
// reallocates instead of writing into the stored array.
func PackIndexInfos(indexes []*model.Index, indexIDs []int64) []*indexpb.IndexInfo {
	infos := make([]*indexpb.IndexInfo, 0, len(indexes))
	for _, index := range indexes {
		if len(indexIDs) == 0 {
			if index.IsDeleted {
				continue
			}
		} else if !slices.Contains(indexIDs, index.IndexID) {
			continue
		}
		infos = append(infos, &indexpb.IndexInfo{
			CollectionID:    index.CollectionID,
			FieldID:         index.FieldID,
			IndexName:       index.IndexName,
			IndexID:         index.IndexID,
			TypeParams:      slices.Clip(index.TypeParams),
			IndexParams:     slices.Clip(index.IndexParams),
			IsAutoIndex:     index.IsAutoIndex,
			UserIndexParams: slices.Clip(index.UserIndexParams),
		})
	}
	slices.SortFunc(infos, func(left, right *indexpb.IndexInfo) int {
		return cmp.Compare(left.GetIndexID(), right.GetIndexID())
	})
	return infos
}
