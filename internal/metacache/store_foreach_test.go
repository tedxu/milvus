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

	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
)

func TestForEachSegment(t *testing.T) {
	store := NewMetaStore(nil)
	one := &datapb.SegmentInfo{ID: 1, CollectionID: 10}
	two := &datapb.SegmentInfo{ID: 2, CollectionID: 10}
	store.PutSegment(one)
	store.PutSegment(two)
	stored := map[int64]*datapb.SegmentInfo{}
	for id := range map[int64]struct{}{1: {}, 2: {}} {
		stored[id], _ = store.GetSegment(id)
	}

	type call struct {
		id int64
		ok bool
	}
	cases := []struct {
		name string
		ids  []int64
		want []call
	}{
		{"empty", nil, nil},
		{"present and absent", []int64{2, 3, 1}, []call{{2, true}, {3, false}, {1, true}}},
		{"duplicates", []int64{1, 1}, []call{{1, true}, {1, true}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got []call
			store.ForEachSegment(tc.ids, func(id int64, seg *datapb.SegmentInfo, ok bool) {
				got = append(got, call{id, ok})
				if ok {
					assert.Same(t, stored[id], seg)
				} else {
					assert.Nil(t, seg)
				}
			})
			assert.Equal(t, tc.want, got)
		})
	}
}
