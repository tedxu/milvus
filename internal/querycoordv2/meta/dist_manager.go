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
	"context"

	"github.com/milvus-io/milvus/internal/json"
	"github.com/milvus-io/milvus/internal/metacache"
	"github.com/milvus-io/milvus/internal/querycoordv2/session"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/util/metricsinfo"
)

type DistributionManager struct {
	SegmentDistManager SegmentDistManagerInterface
	ChannelDistManager ChannelDistManagerInterface

	view metacache.MetaView
}

// DistributionOption configures a DistributionManager.
type DistributionOption func(*DistributionManager)

// WithMetaView sets the store that distribution entries are resolved in.
func WithMetaView(view metacache.MetaView) DistributionOption {
	return func(dm *DistributionManager) { dm.view = view }
}

func NewDistributionManager(nodeManager *session.NodeManager, opts ...DistributionOption) *DistributionManager {
	dm := &DistributionManager{
		SegmentDistManager: NewSegmentDistManager(),
		ChannelDistManager: NewChannelDistManager(nodeManager),
	}
	for _, opt := range opts {
		opt(dm)
	}
	return dm
}

// MetaView returns the store that distribution entries are resolved in:
// every segment field other than the node-reported ones is read there by
// segment ID.
func (dm *DistributionManager) MetaView() metacache.MetaView {
	return dm.view
}

// GetDistributionJSON returns a JSON representation of the current distribution state.
// It includes segments, DM channels, and leader views.
// If there are no segments, channels, or leader views, it returns an empty string.
// In case of an error during JSON marshaling, it returns the error.
func (dm *DistributionManager) GetDistributionJSON(collectionID int64, view metacache.MetaView) string {
	segments := dm.SegmentDistManager.GetSegmentDist(collectionID, view)
	channels := dm.ChannelDistManager.GetChannelDist(collectionID)
	leaderView := dm.ChannelDistManager.GetLeaderView(collectionID)

	dist := &metricsinfo.QueryCoordDist{
		Segments:    segments,
		DMChannels:  channels,
		LeaderViews: leaderView,
	}

	v, err := json.Marshal(dist)
	if err != nil {
		mlog.Warn(context.TODO(), "failed to marshal dist", mlog.Err(err))
		return ""
	}
	return string(v)
}
