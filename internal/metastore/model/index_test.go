package model

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
)

var (
	indexID     int64 = 1
	indexName         = "idx"
	indexParams       = []*commonpb.KeyValuePair{
		{
			Key:   "field110-i1",
			Value: "field110-v1",
		},
	}

	indexModel = &Index{
		IndexID:     indexID,
		IndexName:   indexName,
		IndexParams: indexParams,
		IsDeleted:   true,
		CreateTime:  1,
	}

	indexPb = &indexpb.FieldIndex{
		IndexInfo: &indexpb.IndexInfo{
			CollectionID: colID,
			FieldID:      fieldID,
			IndexName:    indexName,
			IndexID:      indexID,
			TypeParams:   typeParams,
			IndexParams:  indexParams,
		},
		Deleted:    true,
		CreateTime: 1,
	}
)

func TestMarshalIndexModel(t *testing.T) {
	ret := MarshalIndexModel(indexModel)
	assert.Equal(t, indexPb.IndexInfo.IndexID, ret.IndexInfo.IndexID)
	assert.Nil(t, MarshalIndexModel(nil))
}

func TestUnmarshalIndexModel(t *testing.T) {
	ret := UnmarshalIndexModel(indexPb)
	assert.Equal(t, indexModel.IndexID, ret.IndexID)
	assert.Nil(t, UnmarshalIndexModel(nil))
}

func TestShallowCloneIndex(t *testing.T) {
	newParams := func() []*commonpb.KeyValuePair {
		// Spare capacity, as a slice grown by proto unmarshal may have.
		params := make([]*commonpb.KeyValuePair, 1, 4)
		params[0] = &commonpb.KeyValuePair{Key: "k", Value: "v"}
		return params
	}
	fields := []struct {
		name string
		get  func(*Index) *[]*commonpb.KeyValuePair
	}{
		{"TypeParams", func(i *Index) *[]*commonpb.KeyValuePair { return &i.TypeParams }},
		{"IndexParams", func(i *Index) *[]*commonpb.KeyValuePair { return &i.IndexParams }},
		{"UserIndexParams", func(i *Index) *[]*commonpb.KeyValuePair { return &i.UserIndexParams }},
	}
	for _, field := range fields {
		t.Run(field.name, func(t *testing.T) {
			stored := &Index{CollectionID: 1, IndexID: 2, IndexName: "idx", TypeParams: newParams(), IndexParams: newParams(), UserIndexParams: newParams()}
			cloned := ShallowCloneIndex(stored)
			assert.NotSame(t, stored, cloned)
			assert.Equal(t, stored, cloned)
			// Entries are shared, not copied.
			assert.Same(t, (*field.get(stored))[0], (*field.get(cloned))[0])

			// Appending to the clone must not write into stored's backing array.
			*field.get(cloned) = append(*field.get(cloned), &commonpb.KeyValuePair{Key: "added"})
			assert.Len(t, *field.get(stored), 1)
			assert.Nil(t, (*field.get(stored))[:2][1])

			// Replacing the clone's slice must not change stored.
			*field.get(cloned) = nil
			assert.Len(t, *field.get(stored), 1)
		})
	}
	// Nil slices stay nil.
	cloned := ShallowCloneIndex(&Index{IndexID: 3})
	assert.Nil(t, cloned.TypeParams)
	assert.Nil(t, cloned.IndexParams)
	assert.Nil(t, cloned.UserIndexParams)
}
