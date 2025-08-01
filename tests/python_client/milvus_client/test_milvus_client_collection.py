import pytest

from base.client_v2_base import TestMilvusClientV2Base
from utils.util_log import test_log as log
from common import common_func as cf
from common import common_type as ct
from common.common_type import CaseLabel, CheckTasks
from utils.util_pymilvus import *

prefix = "client_collection"
epsilon = ct.epsilon
default_nb = ct.default_nb
default_nb_medium = ct.default_nb_medium
default_nq = ct.default_nq
default_dim = ct.default_dim
default_limit = ct.default_limit
default_search_exp = "id >= 0"
exp_res = "exp_res"
default_search_string_exp = "varchar >= \"0\""
default_search_mix_exp = "int64 >= 0 && varchar >= \"0\""
default_invaild_string_exp = "varchar >= 0"
default_json_search_exp = "json_field[\"number\"] >= 0"
perfix_expr = 'varchar like "0%"'
default_search_field = ct.default_float_vec_field_name
default_search_params = ct.default_search_params
default_primary_key_field_name = "id"
default_vector_field_name = "vector"
default_float_field_name = ct.default_float_field_name
default_bool_field_name = ct.default_bool_field_name
default_string_field_name = ct.default_string_field_name
default_int32_array_field_name = ct.default_int32_array_field_name
default_string_array_field_name = ct.default_string_array_field_name


class TestMilvusClientCollectionInvalid(TestMilvusClientV2Base):
    """ Test case of create collection interface """

    @pytest.fixture(scope="function", params=[False, True])
    def auto_id(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=["COSINE", "L2"])
    def metric_type(self, request):
        yield request.param

    """
    ******************************************************************
    #  The following are invalid base cases
    ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize("collection_name", ["12-s", "12 s", "(mn)", "中文", "%$#"])
    def test_milvus_client_collection_invalid_collection_name(self, collection_name):
        """
        target: test fast create collection with invalid collection name
        method: create collection with invalid collection
        expected: raise exception
        """
        client = self._client()
        # 1. create collection
        error = {ct.err_code: 1100, ct.err_msg: f"Invalid collection name: {collection_name}. the first character of a "
                                                f"collection name must be an underscore or letter: invalid parameter"}
        self.create_collection(client, collection_name, default_dim,
                               check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_collection_name_over_max_length(self):
        """
        target: test fast create collection with over max collection name length
        method: create collection with over max collection name length
        expected: raise exception
        """
        client = self._client()
        # 1. create collection
        collection_name = "a".join("a" for i in range(256))
        error = {ct.err_code: 1100, ct.err_msg: f"the length of a collection name must be less than 255 characters"}
        self.create_collection(client, collection_name, default_dim,
                               check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_collection_name_empty(self):
        """
        target: test fast create collection name with empty
        method: create collection name with empty
        expected: raise exception
        """
        client = self._client()
        # 1. create collection
        collection_name = "  "
        error = {ct.err_code: 1100, ct.err_msg: "Invalid collection name"}
        self.create_collection(client, collection_name, default_dim,
                               check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize("dim", [ct.min_dim - 1, ct.max_dim + 1])
    def test_milvus_client_collection_invalid_dim(self, dim):
        """
        target: test fast create collection name with invalid dim
        method: create collection name with invalid dim
        expected: raise exception
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        error = {ct.err_code: 65535, ct.err_msg: f"invalid dimension: {dim} of field {default_vector_field_name}. "
                                                 f"float vector dimension should be in range 2 ~ 32768"}
        if dim < ct.min_dim:
            error = {ct.err_code: 65535, ct.err_msg: f"invalid dimension: {dim}. "
                                                     f"should be in range 2 ~ 32768"}
        self.create_collection(client, collection_name, dim,
                               check_task=CheckTasks.err_res, check_items=error)
        self.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.skip(reason="pymilvus issue 1554")
    def test_milvus_client_collection_invalid_primary_field(self):
        """
        target: test fast create collection name with invalid primary field
        method: create collection name with invalid primary field
        expected: raise exception
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        error = {ct.err_code: 1, ct.err_msg: f"Param id_type must be int or string"}
        self.create_collection(client, collection_name, default_dim, id_type="invalid",
                               check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_collection_string_auto_id(self):
        """
        target: test fast create collection without max_length for string primary key
        method: create collection name with invalid primary field
        expected: raise exception
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        error = {ct.err_code: 65535, ct.err_msg: f"type param(max_length) should be specified for the field(id) "
                                                 f"of collection {collection_name}"}
        self.create_collection(client, collection_name, default_dim, id_type="string", auto_id=True,
                               check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_create_collection_dup_name_different_params(self):
        """
        target: test create same collection with different parameters
        method: create same collection with different dims, schemas, and primary fields
        expected: raise exception for all different parameter cases
        """
        client = self._client()
        collection_name = cf.gen_collection_name_by_testcase_name()
        self.create_collection(client, collection_name, default_dim)
        # Test 1: Different dimensions
        error = {ct.err_code: 1, ct.err_msg: f"create duplicate collection with different parameters, "
                                             f"collection: {collection_name}"}
        self.create_collection(client, collection_name, default_dim + 1,
                               check_task=CheckTasks.err_res, check_items=error)
        # Test 2: Different schemas  
        schema_diff = self.create_schema(client, enable_dynamic_field=False)[0]
        schema_diff.add_field("new_id", DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False)
        schema_diff.add_field("new_vector", DataType.FLOAT_VECTOR, dim=128)
        self.create_collection(client, collection_name, schema=schema_diff,
                               check_task=CheckTasks.err_res, check_items=error)
        # Test 3: Different primary fields
        schema2 = self.create_schema(client, enable_dynamic_field=False)[0]
        schema2.add_field("id_2", DataType.INT64, is_primary=True, auto_id=False)
        schema2.add_field("vector", DataType.FLOAT_VECTOR, dim=default_dim)
        self.create_collection(client, collection_name, schema=schema2,
                               check_task=CheckTasks.err_res, check_items=error)
        # Verify original collection's primary field is unchanged
        self.describe_collection(client, collection_name,
                                check_task=CheckTasks.check_describe_collection_property,
                                check_items={"collection_name": collection_name,
                                             "dim": default_dim,
                                             "id_name": "id"})
        self.drop_collection(client, collection_name)


    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("metric_type", [1, " ", "invalid"])
    def test_milvus_client_collection_invalid_metric_type(self, metric_type):
        """
        target: test create same collection with invalid metric type
        method: create same collection with invalid metric type
        expected: raise exception
        """
        client = self._client()
        collection_name = cf.gen_collection_name_by_testcase_name()
        # 1. create collection
        error = {ct.err_code: 1100, ct.err_msg: f"float vector index does not support metric type: {metric_type}: "
                                                f"invalid parameter[expected=valid index params][actual=invalid index params"}
        self.create_collection(client, collection_name, default_dim, metric_type=metric_type,
                               check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.skip(reason="pymilvus issue 1864")
    def test_milvus_client_collection_invalid_schema_field_name(self):
        """
        target: test create collection with invalid schema field name
        method: create collection with invalid schema field name
        expected: raise exception
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        schema = self.create_schema(client, enable_dynamic_field=False)[0]
        schema.add_field("%$#", DataType.VARCHAR, max_length=64,
                         is_primary=True, auto_id=False)
        schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=128)
        # 1. create collection
        error = {ct.err_code: 65535,
                 ct.err_msg: "metric type not found or not supported, supported: [L2 IP COSINE HAMMING JACCARD]"}
        self.create_collection(client, collection_name, schema=schema,
                               check_task=CheckTasks.err_res, check_items=error)
    
    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_collection_empty_fields(self):
        """
        target: test create collection with empty fields
        method: create collection with schema that has no fields
        expected: raise exception
        """
        client = self._client()
        collection_name = cf.gen_collection_name_by_testcase_name()
        # Create an empty schema (no fields added)
        schema = self.create_schema(client, enable_dynamic_field=False)[0]
        error = {ct.err_code: 1100, ct.err_msg: "Schema must have a primary key field"}
        self.create_collection(client, collection_name, schema=schema,
                               check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_collection_dup_field(self):
        """
        target: test create collection with duplicate field names
        method: create schema with two fields having the same name
        expected: raise exception
        """
        client = self._client()
        collection_name = cf.gen_collection_name_by_testcase_name()
        # Create schema with duplicate field names
        schema = self.create_schema(client, enable_dynamic_field=False)[0]
        schema.add_field("int64_field", DataType.INT64, is_primary=True, auto_id=False)
        schema.add_field("int64_field", DataType.INT64)
        schema.add_field("vector_field", DataType.FLOAT_VECTOR, dim=default_dim)

        error = {ct.err_code: 1100, ct.err_msg: "duplicated field name"}
        self.create_collection(client, collection_name, schema=schema,
                               check_task=CheckTasks.err_res, check_items=error)
        has_collection = self.has_collection(client, collection_name)[0]
        assert not has_collection

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_collection_add_field_as_primary(self):
        """
        target: test fast create collection with add new field as primary
        method: create collection name with add new field as primary
        expected: raise exception
        """
        client = self._client()
        collection_name = cf.gen_collection_name_by_testcase_name()
        # 1. create collection
        dim, field_name = 8, "field_new"
        error = {ct.err_code: 1100, ct.err_msg: f"not support to add pk field, "
                                                f"field name = {field_name}: invalid parameter"}
        self.create_collection(client, collection_name, dim)
        collections = self.list_collections(client)[0]
        assert collection_name in collections
        self.add_collection_field(client, collection_name, field_name=field_name, data_type=DataType.INT64,
                                  nullable=True, is_primary=True, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_collection_none_desc(self):
        """
        target: test create collection with none description
        method: create collection with none description in schema
        expected: raise exception due to invalid description type
        """
        client = self._client()
        collection_name = cf.gen_collection_name_by_testcase_name()
        
        # Try to create schema with None description
        schema = self.create_schema(client, enable_dynamic_field=False, description=None)[0]
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
        schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=default_dim)
        
        error = {ct.err_code: 1100, ct.err_msg: "description [None] has type NoneType, but expected one of: bytes, str"}
        self.create_collection(client, collection_name, schema=schema,
                                   check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_collection_invalid_schema_multi_pk(self):
        """
        target: test create collection with schema containing multiple primary key fields
        method: create schema with two primary key fields and use it to create collection
        expected: raise exception due to multiple primary keys
        """
        client = self._client()
        collection_name = cf.gen_collection_name_by_testcase_name()
        # Create schema with multiple primary key fields
        schema = self.create_schema(client, enable_dynamic_field=False)[0]
        schema.add_field("field1", DataType.INT64, is_primary=True, auto_id=False)
        schema.add_field("field2", DataType.INT64, is_primary=True, auto_id=False)  # Second primary key
        schema.add_field("vector_field", DataType.FLOAT_VECTOR, dim=32)
        # Try to create collection with multiple primary keys
        error = {ct.err_code: 999, ct.err_msg: "Expected only one primary key field"}
        self.create_collection(client, collection_name, schema=schema,
                               check_task=CheckTasks.err_res, check_items=error)

class TestMilvusClientCollectionValid(TestMilvusClientV2Base):
    """ Test case of create collection interface """

    @pytest.fixture(scope="function", params=[False, True])
    def auto_id(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=["COSINE", "L2", "IP"])
    def metric_type(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=["int", "string"])
    def id_type(self, request):
        yield request.param

    """
    ******************************************************************
    #  The following are valid base cases
    ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.parametrize("dim", [ct.min_dim, default_dim, ct.max_dim])
    def test_milvus_client_collection_fast_creation_default(self, dim):
        """
        target: test fast create collection normal case
        method: create collection
        expected: create collection with default schema, index, and load successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        self.using_database(client, "default")
        # 1. create collection
        self.create_collection(client, collection_name, dim)
        collections = self.list_collections(client)[0]
        assert collection_name in collections
        self.describe_collection(client, collection_name,
                                 check_task=CheckTasks.check_describe_collection_property,
                                 check_items={"collection_name": collection_name,
                                              "dim": dim,
                                              "consistency_level": 0})
        index = self.list_indexes(client, collection_name)[0]
        assert index == ['vector']
        # load_state = self.get_load_state(collection_name)[0]
        self.load_partitions(client, collection_name, "_default")
        self.release_partitions(client, collection_name, "_default")
        if self.has_collection(client, collection_name)[0]:
            self.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize("dim", [ct.min_dim, default_dim, ct.max_dim])
    def test_milvus_client_collection_fast_creation_all_params(self, dim, metric_type, id_type, auto_id):
        """
        target: test fast create collection normal case
        method: create collection
        expected: create collection with default schema, index, and load successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        max_length = 100
        # 1. create collection
        self.create_collection(client, collection_name, dim, id_type=id_type, metric_type=metric_type,
                               auto_id=auto_id, max_length=max_length)
        collections = self.list_collections(client)[0]
        assert collection_name in collections
        self.describe_collection(client, collection_name,
                                 check_task=CheckTasks.check_describe_collection_property,
                                 check_items={"collection_name": collection_name,
                                              "dim": dim,
                                              "auto_id": auto_id,
                                              "consistency_level": 0})
        index = self.list_indexes(client, collection_name)[0]
        assert index == ['vector']
        # load_state = self.get_load_state(collection_name)[0]
        self.release_collection(client, collection_name)
        self.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L0)
    @pytest.mark.parametrize("nullable", [True, False])
    @pytest.mark.parametrize("vector_type", [DataType.FLOAT_VECTOR, DataType.INT8_VECTOR])
    @pytest.mark.parametrize("add_field", [True, False])
    def test_milvus_client_collection_self_creation_default(self, nullable, vector_type, add_field):
        """
        target: test self create collection normal case
        method: create collection
        expected: create collection with default schema, index, and load successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        dim = 128
        # 1. create collection
        schema = self.create_schema(client, enable_dynamic_field=False)[0]
        schema.add_field("id_string", DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False)
        schema.add_field("embeddings", vector_type, dim=dim)
        schema.add_field("title", DataType.VARCHAR, max_length=64, is_partition_key=True)
        schema.add_field("nullable_field", DataType.INT64, nullable=nullable, default_value=10)
        schema.add_field("array_field", DataType.ARRAY, element_type=DataType.INT64, max_capacity=12,
                         max_length=64, nullable=nullable)
        index_params = self.prepare_index_params(client)[0]
        index_params.add_index("embeddings", metric_type="COSINE")
        # index_params.add_index("title")
        self.create_collection(client, collection_name, dimension=dim, schema=schema, index_params=index_params)
        collections = self.list_collections(client)[0]
        assert collection_name in collections
        check_items = {"collection_name": collection_name,
                       "dim": dim,
                       "consistency_level": 0,
                       "enable_dynamic_field": False,
                       "num_partitions": 16,
                       "id_name": "id_string",
                       "vector_name": "embeddings"}
        if nullable:
            check_items["nullable_fields"] = ["nullable_field", "array_field"]
        if add_field:
            self.add_collection_field(client, collection_name, field_name="field_new_int64", data_type=DataType.INT64,
                                      nullable=True, is_cluster_key=True)
            self.add_collection_field(client, collection_name, field_name="field_new_var", data_type=DataType.VARCHAR,
                                      nullable=True, default_vaule="field_new_var", max_length=64)
            check_items["add_fields"] = ["field_new_int64", "field_new_var"]
        self.describe_collection(client, collection_name,
                                 check_task=CheckTasks.check_describe_collection_property,
                                 check_items=check_items)
        index = self.list_indexes(client, collection_name)[0]
        assert index == ['embeddings']
        if self.has_collection(client, collection_name)[0]:
            self.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_collection_self_creation_multiple_vectors(self):
        """
        target: test self create collection with multiple vectors
        method: create collection with multiple vectors
        expected: create collection with default schema, index, and load successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        dim = 128
        # 1. create collection
        schema = self.create_schema(client, enable_dynamic_field=False)[0]
        schema.add_field("id_int64", DataType.INT64, is_primary=True, auto_id=False)
        schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("embeddings_1", DataType.INT8_VECTOR, dim=dim * 2)
        schema.add_field("embeddings_2", DataType.FLOAT16_VECTOR, dim=int(dim / 2))
        schema.add_field("embeddings_3", DataType.BFLOAT16_VECTOR, dim=int(dim / 2))
        index_params = self.prepare_index_params(client)[0]
        index_params.add_index("embeddings", metric_type="COSINE")
        index_params.add_index("embeddings_1", metric_type="IP")
        index_params.add_index("embeddings_2", metric_type="L2")
        index_params.add_index("embeddings_3", metric_type="COSINE")
        # index_params.add_index("title")
        self.create_collection(client, collection_name, dimension=dim, schema=schema, index_params=index_params)
        collections = self.list_collections(client)[0]
        assert collection_name in collections
        check_items = {"collection_name": collection_name,
                       "dim": [dim, dim * 2, int(dim / 2), int(dim / 2)],
                       "consistency_level": 0,
                       "enable_dynamic_field": False,
                       "id_name": "id_int64",
                       "vector_name": ["embeddings", "embeddings_1", "embeddings_2", "embeddings_3"]}
        self.describe_collection(client, collection_name,
                                 check_task=CheckTasks.check_describe_collection_property,
                                 check_items=check_items)
        index = self.list_indexes(client, collection_name)[0]
        assert sorted(index) == sorted(['embeddings', 'embeddings_1', 'embeddings_2', 'embeddings_3'])
        if self.has_collection(client, collection_name)[0]:
            self.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_create_collection_dup_name(self):
        """
        target: test create collection with same name
        method: create collection with same name with same default params
        expected: collection properties consistent
        """
        client = self._client()
        collection_name = cf.gen_collection_name_by_testcase_name()
        # 1. create collection
        self.create_collection(client, collection_name, default_dim)
        # 2. create collection with same params
        self.create_collection(client, collection_name, default_dim)
        
        collections = self.list_collections(client)[0]
        collection_count = collections.count(collection_name)
        assert collection_name in collections
        assert collection_count == 1, f"Expected only 1 collection named '{collection_name}', but found {collection_count}"
        
        self.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_create_collection_dup_name_same_schema(self):
        """
        target: test create collection with dup name and same schema
        method: create collection with dup name and same schema
        expected: two collection object is available and properties consistent
        """
        client = self._client()
        collection_name = cf.gen_collection_name_by_testcase_name()
        dim = 128
        description = "test collection description"
        # Create schema
        schema = self.create_schema(client, enable_dynamic_field=False, description=description)[0]
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
        schema.add_field("float_field", DataType.FLOAT)
        schema.add_field("varchar_field", DataType.VARCHAR, max_length=100)
        schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=dim)
        # 1. Create collection with specific schema
        self.create_collection(client, collection_name, schema=schema)
        # Get first collection info
        collection_info_1 = self.describe_collection(client, collection_name)[0]
        description_1 = collection_info_1.get("description", "")
        # 2. Create collection again with same name and same schema
        self.create_collection(client, collection_name, schema=schema)
        # Verify collection still exists and properties are consistent
        collections = self.list_collections(client)[0]
        assert collection_name in collections
        # Get second collection info and compare
        collection_info_2 = self.describe_collection(client, collection_name)[0]
        description_2 = collection_info_2.get("description", "")
        # Verify collection properties are consistent
        assert collection_info_1["collection_name"] == collection_info_2["collection_name"]
        assert description_1 == description_2 == description
        assert len(collection_info_1["fields"]) == len(collection_info_2["fields"])
        # Verify field names and types are the same
        fields_1 = {field["name"]: field["type"] for field in collection_info_1["fields"]}
        fields_2 = {field["name"]: field["type"] for field in collection_info_2["fields"]}
        assert fields_1 == fields_2
        collection_count = collections.count(collection_name)
        assert collection_count == 1, f"Expected only 1 collection named '{collection_name}', but found {collection_count}"
        self.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_array_insert_search(self):
        """
        target: test search (high level api) normal case
        method: create connection, collection, insert and search
        expected: search/query successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        self.create_collection(client, collection_name, default_dim, consistency_level="Strong")
        collections = self.list_collections(client)[0]
        assert collection_name in collections
        # 2. insert
        rng = np.random.default_rng(seed=19530)
        rows = [{
            default_primary_key_field_name: i,
            default_vector_field_name: list(rng.random((1, default_dim))[0]),
            default_float_field_name: i * 1.0,
            default_int32_array_field_name: [i, i + 1, i + 2],
            default_string_array_field_name: [str(i), str(i + 1), str(i + 2)]
        } for i in range(default_nb)]
        self.insert(client, collection_name, rows)
        # 3. search
        vectors_to_search = rng.random((1, default_dim))
        insert_ids = [i for i in range(default_nb)]
        self.search(client, collection_name, vectors_to_search,
                    check_task=CheckTasks.check_search_results,
                    check_items={"enable_milvus_client_api": True,
                                 "nq": len(vectors_to_search),
                                 "ids": insert_ids,
                                 "limit": default_limit,
                                 "pk_name": default_primary_key_field_name})

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.skip(reason="issue 25110")
    def test_milvus_client_search_query_string(self):
        """
        target: test search (high level api) for string primary key
        method: create connection, collection, insert and search
        expected: search/query successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        self.create_collection(client, collection_name, default_dim, id_type="string", max_length=ct.default_length)
        self.describe_collection(client, collection_name,
                                 check_task=CheckTasks.check_describe_collection_property,
                                 check_items={"collection_name": collection_name,
                                              "dim": default_dim,
                                              "consistency_level": 0})
        # 2. insert
        rng = np.random.default_rng(seed=19530)
        rows = [
            {default_primary_key_field_name: str(i), default_vector_field_name: list(rng.random((1, default_dim))[0]),
             default_float_field_name: i * 1.0, default_string_field_name: str(i)} for i in range(default_nb)]
        self.insert(client, collection_name, rows)
        # self.flush(client, collection_name)
        # assert self.num_entities(client, collection_name)[0] == default_nb
        # 3. search
        vectors_to_search = rng.random((1, default_dim))
        self.search(client, collection_name, vectors_to_search,
                    check_task=CheckTasks.check_search_results,
                    check_items={"enable_milvus_client_api": True,
                                 "nq": len(vectors_to_search),
                                 "pk_name": default_primary_key_field_name,
                                 "limit": default_limit})
        # 4. query
        self.query(client, collection_name, filter="id in [0, 1]",
                   check_task=CheckTasks.check_query_results,
                   check_items={exp_res: rows,
                                "with_vec": True,
                                "pk_name": default_primary_key_field_name})
        self.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_search_different_metric_types_not_specifying_in_search_params(self, metric_type, auto_id):
        """
        target: test search (high level api) normal case
        method: create connection, collection, insert and search
        expected: search successfully with limit(topK)
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        self.create_collection(client, collection_name, default_dim, metric_type=metric_type, auto_id=auto_id,
                               consistency_level="Strong")
        # 2. insert
        rng = np.random.default_rng(seed=19530)
        rows = [{default_primary_key_field_name: i, default_vector_field_name: list(rng.random((1, default_dim))[0]),
                 default_float_field_name: i * 1.0, default_string_field_name: str(i)} for i in range(default_nb)]
        if auto_id:
            for row in rows:
                row.pop(default_primary_key_field_name)
        self.insert(client, collection_name, rows)
        # 3. search
        vectors_to_search = rng.random((1, default_dim))
        # search_params = {"metric_type": metric_type}
        self.search(client, collection_name, vectors_to_search, limit=default_limit,
                    output_fields=[default_primary_key_field_name],
                    check_task=CheckTasks.check_search_results,
                    check_items={"enable_milvus_client_api": True,
                                 "nq": len(vectors_to_search),
                                 "pk_name": default_primary_key_field_name,
                                 "limit": default_limit})
        self.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.skip("pymilvus issue #1866")
    def test_milvus_client_search_different_metric_types_specifying_in_search_params(self, metric_type, auto_id):
        """
        target: test search (high level api) normal case
        method: create connection, collection, insert and search
        expected: search successfully with limit(topK)
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        self.create_collection(client, collection_name, default_dim, metric_type=metric_type, auto_id=auto_id,
                               consistency_level="Strong")
        # 2. insert
        rng = np.random.default_rng(seed=19530)
        rows = [{default_primary_key_field_name: i, default_vector_field_name: list(rng.random((1, default_dim))[0]),
                 default_float_field_name: i * 1.0, default_string_field_name: str(i)} for i in range(default_nb)]
        if auto_id:
            for row in rows:
                row.pop(default_primary_key_field_name)
        self.insert(client, collection_name, rows)
        # 3. search
        vectors_to_search = rng.random((1, default_dim))
        search_params = {"metric_type": metric_type}
        self.search(client, collection_name, vectors_to_search, limit=default_limit,
                    search_params=search_params,
                    output_fields=[default_primary_key_field_name],
                    check_task=CheckTasks.check_search_results,
                    check_items={"enable_milvus_client_api": True,
                                 "nq": len(vectors_to_search),
                                 "pk_name": default_primary_key_field_name,
                                 "limit": default_limit})
        self.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_delete_with_ids(self):
        """
        target: test delete (high level api)
        method: create connection, collection, insert delete, and search
        expected: search/query successfully without deleted data
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        self.create_collection(client, collection_name, default_dim, consistency_level="Strong")
        # 2. insert
        default_nb = 1000
        rng = np.random.default_rng(seed=19530)
        rows = [{default_primary_key_field_name: i, default_vector_field_name: list(rng.random((1, default_dim))[0]),
                 default_float_field_name: i * 1.0, default_string_field_name: str(i)} for i in range(default_nb)]
        pks = self.insert(client, collection_name, rows)[0]
        # 3. delete
        delete_num = 3
        self.delete(client, collection_name, ids=[i for i in range(delete_num)])
        # 4. search
        vectors_to_search = rng.random((1, default_dim))
        insert_ids = [i for i in range(default_nb)]
        for insert_id in range(delete_num):
            if insert_id in insert_ids:
                insert_ids.remove(insert_id)
        limit = default_nb - delete_num
        self.search(client, collection_name, vectors_to_search, limit=default_nb,
                    check_task=CheckTasks.check_search_results,
                    check_items={"enable_milvus_client_api": True,
                                 "nq": len(vectors_to_search),
                                 "ids": insert_ids,
                                 "limit": limit,
                                 "pk_name": default_primary_key_field_name})
        # 5. query
        self.query(client, collection_name, filter=default_search_exp,
                   check_task=CheckTasks.check_query_results,
                   check_items={exp_res: rows[delete_num:],
                                "with_vec": True,
                                "pk_name": default_primary_key_field_name})
        self.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_delete_with_filters(self):
        """
        target: test delete (high level api)
        method: create connection, collection, insert delete, and search
        expected: search/query successfully without deleted data
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        self.create_collection(client, collection_name, default_dim, consistency_level="Strong")
        # 2. insert
        default_nb = 1000
        rng = np.random.default_rng(seed=19530)
        rows = [{default_primary_key_field_name: i, default_vector_field_name: list(rng.random((1, default_dim))[0]),
                 default_float_field_name: i * 1.0, default_string_field_name: str(i)} for i in range(default_nb)]
        pks = self.insert(client, collection_name, rows)[0]
        # 3. delete
        delete_num = 3
        self.delete(client, collection_name, filter=f"id < {delete_num}")
        # 4. search
        vectors_to_search = rng.random((1, default_dim))
        insert_ids = [i for i in range(default_nb)]
        for insert_id in range(delete_num):
            if insert_id in insert_ids:
                insert_ids.remove(insert_id)
        limit = default_nb - delete_num
        self.search(client, collection_name, vectors_to_search, limit=default_nb,
                    check_task=CheckTasks.check_search_results,
                    check_items={"enable_milvus_client_api": True,
                                 "nq": len(vectors_to_search),
                                 "ids": insert_ids,
                                 "limit": limit,
                                 "pk_name": default_primary_key_field_name})
        # 5. query
        self.query(client, collection_name, filter=default_search_exp,
                   check_task=CheckTasks.check_query_results,
                   check_items={exp_res: rows[delete_num:],
                                "with_vec": True,
                                "pk_name": default_primary_key_field_name})
        self.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_collection_rename_collection(self):
        """
        target: test fast create collection normal case
        method: create collection
        expected: create collection with default schema, index, and load successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        self.create_collection(client, collection_name, default_dim)
        collections = self.list_collections(client)[0]
        assert collection_name in collections
        old_name = collection_name
        new_name = collection_name + "new"
        self.rename_collection(client, old_name, new_name)
        collections = self.list_collections(client)[0]
        assert new_name in collections
        assert old_name not in collections
        self.describe_collection(client, new_name,
                                 check_task=CheckTasks.check_describe_collection_property,
                                 check_items={"collection_name": new_name,
                                              "dim": default_dim,
                                              "consistency_level": 0})
        index = self.list_indexes(client, new_name)[0]
        assert index == ['vector']
        # load_state = self.get_load_state(collection_name)[0]
        error = {ct.err_code: 100, ct.err_msg: f"collection not found"}
        self.load_partitions(client, old_name, "_default",
                             check_task=CheckTasks.err_res, check_items=error)
        self.load_partitions(client, new_name, "_default")
        self.release_partitions(client, new_name, "_default")
        if self.has_collection(client, collection_name)[0]:
            self.drop_collection(client, new_name)

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.skip(reason="db not ready")
    def test_milvus_client_collection_rename_collection_target_db(self):
        """
        target: test fast create collection normal case
        method: create collection
        expected: create collection with default schema, index, and load successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        self.create_collection(client, collection_name, default_dim)
        collections = self.list_collections(client)[0]
        assert collection_name in collections
        db_name = "new_db"
        self.using_database(client, db_name)
        old_name = collection_name
        new_name = collection_name + "new"
        self.rename_collection(client, old_name, new_name, target_db=db_name)
        collections = self.list_collections(client)[0]
        assert new_name in collections
        assert old_name not in collections
        self.describe_collection(client, new_name,
                                 check_task=CheckTasks.check_describe_collection_property,
                                 check_items={"collection_name": new_name,
                                              "dim": default_dim,
                                              "consistency_level": 0})
        index = self.list_indexes(client, new_name)[0]
        assert index == ['vector']
        # load_state = self.get_load_state(collection_name)[0]
        error = {ct.err_code: 100, ct.err_msg: f"collection not found"}
        self.load_partitions(client, old_name, "_default",
                             check_task=CheckTasks.err_res, check_items=error)
        self.load_partitions(client, new_name, "_default")
        self.release_partitions(client, new_name, "_default")
        if self.has_collection(client, collection_name)[0]:
            self.drop_collection(client, new_name)

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_collection_dup_name_drop(self):
        """
        target: test collection with dup name, and drop
        method: 1. create collection with client1
                2. create collection with client2 with same name
                3. use client2 to drop collection
                4. verify collection is dropped and client1 operations fail
        expected: collection is successfully dropped and subsequent operations from the first client should fail with collection not found error
        """
        client1 = self._client(alias="client1")
        client2 = self._client(alias="client2") 
        collection_name = cf.gen_collection_name_by_testcase_name()
        # 1. Create collection with client1
        self.create_collection(client1, collection_name, default_dim)
        # 2. Create collection with client2 using same name
        self.create_collection(client2, collection_name, default_dim)
        # 3. Use client2 to drop collection
        self.drop_collection(client2, collection_name)
        # 4. Verify collection is deleted
        has_collection = self.has_collection(client1, collection_name)[0]
        assert not has_collection
        error = {ct.err_code: 100, ct.err_msg:  f"can't find collection[database=default]"
                                                f"[collection={collection_name}]"}
        self.describe_collection(client1, collection_name, check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_collection_long_desc(self):
        """
        target: test create collection with long description
        method: create collection with description longer than 255 characters
        expected: collection created successfully with long description
        """
        client = self._client()
        collection_name = cf.gen_collection_name_by_testcase_name()
        # Create long description
        long_desc = "a".join("a" for _ in range(256))
        
        # Create schema with long description
        schema = self.create_schema(client, enable_dynamic_field=False, description=long_desc)[0]
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
        schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=default_dim)
        
        # Create collection with long description
        self.create_collection(client, collection_name, schema=schema)
        
        collection_info = self.describe_collection(client, collection_name)[0]
        actual_desc = collection_info.get("description", "")
        assert actual_desc == long_desc
        
        self.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("collection_name", ct.valid_resource_names)
    def test_milvus_client_collection_valid_naming_rules(self, collection_name):
        """
        target: test create collection with valid names following naming rules
        method: 1. create collection using names that follow all supported naming rule elements
                2. create fields with names that also use naming rule elements
                3. verify collection is created successfully with correct properties
        expected: collection created successfully for all valid names
        """
        client = self._client()
        
        # Create schema with fields that also use naming rule elements
        schema = self.create_schema(client, enable_dynamic_field=False)[0]
        schema.add_field(ct.default_int64_field_name, DataType.INT64, is_primary=True, auto_id=False)
        schema.add_field("_1nt", DataType.INT64)  # field name using naming rule elements
        schema.add_field("f10at_", DataType.FLOAT_VECTOR, dim=default_dim)  # vector field with naming elements
        
        # Create collection with valid name
        self.create_collection(client, collection_name, schema=schema)
        collections = self.list_collections(client)[0]
        assert collection_name in collections
        
        collection_info = self.describe_collection(client, collection_name)[0]
        assert collection_info["collection_name"] == collection_name
        
        field_names = [field["name"] for field in collection_info["fields"]]
        assert ct.default_int64_field_name in field_names
        assert "_1nt" in field_names
        assert "f10at_" in field_names
        
        self.drop_collection(client, collection_name)


class TestMilvusClientDropCollectionInvalid(TestMilvusClientV2Base):
    """ Test case of drop collection interface """
    """
    ******************************************************************
    #  The following are invalid base cases
    ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize("name", ["12-s", "12 s", "(mn)", "中文", "%$#"])
    @pytest.mark.skip(reason="https://github.com/milvus-io/milvus/pull/43064 change drop alias restraint")
    def test_milvus_client_drop_collection_invalid_collection_name(self, name):
        """
        target: Test drop collection with invalid params
        method: drop collection with invalid collection name
        expected: raise exception
        """
        client = self._client()
        error = {ct.err_code: 1100, ct.err_msg: f"Invalid collection name: {name}. "
                                                f"the first character of a collection name must be an underscore or letter"}
        self.drop_collection(client, name,
                             check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_drop_collection_not_existed(self):
        """
        target: test fast create collection normal case
        method: create collection
        expected: drop successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str("nonexisted")
        self.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("collection_name", ['', None])
    def test_milvus_client_drop_collection_with_empty_or_None_collection_name(self, collection_name):
        """
        target: test drop invalid collection
        method: drop collection with empty or None collection name
        expected: raise exception
        """
        client = self._client()
        # Set different error messages based on collection_name value
        error = {ct.err_code: 1, ct.err_msg: f"`collection_name` value {collection_name} is illegal"}
        self.drop_collection(client, collection_name, check_task=CheckTasks.err_res, check_items=error)

class TestMilvusClientReleaseCollectionInvalid(TestMilvusClientV2Base):
    """ Test case of release collection interface """
    """
    ******************************************************************
    #  The following are invalid base cases
    ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize("name", ["12-s", "12 s", "(mn)", "中文", "%$#"])
    def test_milvus_client_release_collection_invalid_collection_name(self, name):
        """
        target: test fast create collection normal case
        method: create collection
        expected: create collection with default schema, index, and load successfully
        """
        client = self._client()
        error = {ct.err_code: 1100,
                 ct.err_msg: f"Invalid collection name: {name}. "
                             f"the first character of a collection name must be an underscore or letter"}
        self.release_collection(client, name,
                                check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_release_collection_not_existed(self):
        """
        target: test fast create collection normal case
        method: create collection
        expected: drop successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str("nonexisted")
        error = {ct.err_code: 1100, ct.err_msg: f"collection not found[database=default]"
                                                f"[collection={collection_name}]"}
        self.release_collection(client, collection_name,
                                check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_release_collection_name_over_max_length(self):
        """
        target: test fast create collection normal case
        method: create collection
        expected: create collection with default schema, index, and load successfully
        """
        client = self._client()
        # 1. create collection
        collection_name = "a".join("a" for i in range(256))
        error = {ct.err_code: 1100, ct.err_msg: f"the length of a collection name must be less than 255 characters"}
        self.release_collection(client, collection_name, default_dim,
                                check_task=CheckTasks.err_res, check_items=error)


class TestMilvusClientReleaseCollectionValid(TestMilvusClientV2Base):
    """ Test case of release collection interface """

    @pytest.fixture(scope="function", params=[False, True])
    def auto_id(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=["COSINE", "L2", "IP"])
    def metric_type(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=["int", "string"])
    def id_type(self, request):
        yield request.param

    """
    ******************************************************************
    #  The following are valid base cases
    ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_release_unloaded_collection(self):
        """
        target: test fast create collection normal case
        method: create collection
        expected: create collection with default schema, index, and load successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        self.create_collection(client, collection_name, default_dim)
        self.release_collection(client, collection_name)
        self.release_collection(client, collection_name)
        if self.has_collection(client, collection_name)[0]:
            self.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_load_partially_loaded_collection(self):
        """
        target: test fast create collection normal case
        method: create collection
        expected: create collection with default schema, index, and load successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        partition_name = cf.gen_unique_str("partition")
        # 1. create collection
        self.create_collection(client, collection_name, default_dim)
        self.create_partition(client, collection_name, partition_name)
        self.release_partitions(client, collection_name, ["_default", partition_name])
        self.release_collection(client, collection_name)
        self.load_collection(client, collection_name)
        self.release_partitions(client, collection_name, [partition_name])
        self.release_collection(client, collection_name)
        if self.has_collection(client, collection_name)[0]:
            self.drop_collection(client, collection_name)


class TestMilvusClientLoadCollectionInvalid(TestMilvusClientV2Base):
    """ Test case of search interface """
    """
    ******************************************************************
    #  The following are invalid base cases
    ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize("name", ["12-s", "12 s", "(mn)", "中文", "%$#"])
    def test_milvus_client_load_collection_invalid_collection_name(self, name):
        """
        target: test fast create collection normal case
        method: create collection
        expected: create collection with default schema, index, and load successfully
        """
        client = self._client()
        error = {ct.err_code: 1100,
                 ct.err_msg: f"Invalid collection name: {name}. "
                             f"the first character of a collection name must be an underscore or letter"}
        self.load_collection(client, name,
                             check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_load_collection_not_existed(self):
        """
        target: test fast create collection normal case
        method: create collection
        expected: drop successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str("nonexisted")
        error = {ct.err_code: 1100, ct.err_msg: f"collection not found[database=default]"
                                                f"[collection={collection_name}]"}
        self.load_collection(client, collection_name,
                             check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_load_collection_over_max_length(self):
        """
        target: test fast create collection normal case
        method: create collection
        expected: drop successfully
        """
        client = self._client()
        collection_name = "a".join("a" for i in range(256))
        error = {ct.err_code: 1100, ct.err_msg: f"Invalid collection name: {collection_name}. "
                                                f"the length of a collection name must be less than 255 characters: "
                                                f"invalid parameter"}
        self.load_collection(client, collection_name,
                             check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_load_collection_without_index(self):
        """
        target: test fast create collection normal case
        method: create collection
        expected: create collection with default schema, index, and load successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        self.create_collection(client, collection_name, default_dim)
        self.release_collection(client, collection_name)
        self.drop_index(client, collection_name, "vector")
        error = {ct.err_code: 700, ct.err_msg: f"index not found[collection={collection_name}]"}
        self.load_collection(client, collection_name,
                             check_task=CheckTasks.err_res, check_items=error)
        if self.has_collection(client, collection_name)[0]:
            self.drop_collection(client, collection_name)


class TestMilvusClientLoadCollectionValid(TestMilvusClientV2Base):
    """ Test case of search interface """

    @pytest.fixture(scope="function", params=[False, True])
    def auto_id(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=["COSINE", "L2", "IP"])
    def metric_type(self, request):
        yield request.param

    @pytest.fixture(scope="function", params=["int", "string"])
    def id_type(self, request):
        yield request.param

    """
    ******************************************************************
    #  The following are valid base cases
    ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_load_loaded_collection(self):
        """
        target: test fast create collection normal case
        method: create collection
        expected: create collection with default schema, index, and load successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        self.create_collection(client, collection_name, default_dim)
        self.load_collection(client, collection_name)
        if self.has_collection(client, collection_name)[0]:
            self.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_load_partially_loaded_collection(self):
        """
        target: test fast create collection normal case
        method: create collection
        expected: create collection with default schema, index, and load successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        partition_name = cf.gen_unique_str("partition")
        # 1. create collection
        self.create_collection(client, collection_name, default_dim)
        self.create_partition(client, collection_name, partition_name)
        self.release_collection(client, collection_name)
        self.load_partitions(client, collection_name, [partition_name])
        self.load_collection(client, collection_name)
        self.release_collection(client, collection_name)
        self.load_partitions(client, collection_name, ["_default", partition_name])
        self.load_collection(client, collection_name)
        if self.has_collection(client, collection_name)[0]:
            self.drop_collection(client, collection_name)


class TestMilvusClientDescribeCollectionInvalid(TestMilvusClientV2Base):
    """ Test case of search interface """
    """
    ******************************************************************
    #  The following are invalid base cases
    ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize("name", ["12-s", "12 s", "(mn)", "中文", "%$#"])
    def test_milvus_client_describe_collection_invalid_collection_name(self, name):
        """
        target: test fast create collection normal case
        method: create collection
        expected: create collection with default schema, index, and load successfully
        """
        client = self._client()
        error = {ct.err_code: 1100,
                 ct.err_msg: f"Invalid collection name: {name}. "
                             f"the first character of a collection name must be an underscore or letter"}
        self.describe_collection(client, name,
                                 check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_describe_collection_not_existed(self):
        """
        target: test fast create collection normal case
        method: create collection
        expected: drop successfully
        """
        client = self._client()
        collection_name = "nonexisted"
        error = {ct.err_code: 100, ct.err_msg: "can't find collection[database=default][collection=nonexisted]"}
        self.describe_collection(client, collection_name,
                                 check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_describe_collection_deleted_collection(self):
        """
        target: test fast create collection normal case
        method: create collection
        expected: drop successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        self.create_collection(client, collection_name, default_dim)
        self.drop_collection(client, collection_name)
        error = {ct.err_code: 100, ct.err_msg: f"can't find collection[database=default][collection={collection_name}]"}
        self.describe_collection(client, collection_name,
                                 check_task=CheckTasks.err_res, check_items=error)


class TestMilvusClientHasCollectionInvalid(TestMilvusClientV2Base):
    """ Test case of search interface """
    """
    ******************************************************************
    #  The following are invalid base cases
    ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize("name", ["12-s", "12 s", "(mn)", "中文", "%$#"])
    def test_milvus_client_has_collection_invalid_collection_name(self, name):
        """
        target: test fast create collection normal case
        method: create collection
        expected: create collection with default schema, index, and load successfully
        """
        client = self._client()
        error = {ct.err_code: 1100,
                 ct.err_msg: f"Invalid collection name: {name}. "
                             f"the first character of a collection name must be an underscore or letter"}
        self.has_collection(client, name,
                            check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_has_collection_not_existed(self):
        """
        target: test fast create collection normal case
        method: create collection
        expected: drop successfully
        """
        client = self._client()
        collection_name = "nonexisted"
        result = self.has_collection(client, collection_name)[0]
        assert result == False

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_has_collection_deleted_collection(self):
        """
        target: test fast create collection normal case
        method: create collection
        expected: drop successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        self.create_collection(client, collection_name, default_dim)
        self.drop_collection(client, collection_name)
        result = self.has_collection(client, collection_name)[0]
        assert result == False


class TestMilvusClientRenameCollectionInValid(TestMilvusClientV2Base):
    """ Test case of rename collection interface """

    """
    ******************************************************************
    #  The following are valid base cases
    ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize("name", ["12-s", "12 s", "(mn)", "中文", "%$#"])
    def test_milvus_client_rename_collection_invalid_collection_name(self, name):
        """
        target: test fast create collection normal case
        method: create collection
        expected: create collection with default schema, index, and load successfully
        """
        client = self._client()
        error = {ct.err_code: 100, ct.err_msg: f"collection not found[database=1][collection={name}]"}
        self.rename_collection(client, name, "new_collection",
                               check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_rename_collection_not_existed_collection(self):
        """
        target: test fast create collection normal case
        method: create collection
        expected: drop successfully
        """
        client = self._client()
        collection_name = "nonexisted"
        error = {ct.err_code: 100, ct.err_msg: f"collection not found[database=1][collection={collection_name}]"}
        self.rename_collection(client, collection_name, "new_collection",
                               check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_rename_collection_duplicated_collection(self):
        """
        target: test fast create collection normal case
        method: create collection
        expected: drop successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        self.create_collection(client, collection_name, default_dim)
        error = {ct.err_code: 65535, ct.err_msg: f"duplicated new collection name default:{collection_name} "
                                                 f"with other collection name or alias"}
        self.rename_collection(client, collection_name, collection_name,
                               check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_rename_deleted_collection(self):
        """
        target: test fast create collection normal case
        method: create collection
        expected: drop successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        self.create_collection(client, collection_name, default_dim)
        self.drop_collection(client, collection_name)
        error = {ct.err_code: 100, ct.err_msg: f"{collection_name}: collection not found[collection=default]"}
        self.rename_collection(client, collection_name, "new_collection",
                               check_task=CheckTasks.err_res, check_items=error)


class TestMilvusClientRenameCollectionValid(TestMilvusClientV2Base):
    """ Test case of rename collection interface """

    """
    ******************************************************************
    #  The following are valid base cases
    ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_rename_collection_multiple_times(self):
        """
        target: test fast create collection normal case
        method: create collection
        expected: create collection with default schema, index, and load successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        # 2. rename with invalid new_name
        new_name = "new_name_rename"
        self.create_collection(client, collection_name, default_dim)
        times = 3
        for _ in range(times):
            self.rename_collection(client, collection_name, new_name)
            self.rename_collection(client, new_name, collection_name)

    @pytest.mark.tags(CaseLabel.L2)
    def test_milvus_client_rename_collection_deleted_collection(self):
        """
        target: test fast create collection normal case
        method: create collection
        expected: drop successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        another_collection_name = cf.gen_unique_str("another_collection")
        # 1. create 2 collections
        self.create_collection(client, collection_name, default_dim)
        self.create_collection(client, another_collection_name, default_dim)
        # 2. drop one collection
        self.drop_collection(client, another_collection_name)
        # 3. rename to dropped collection
        self.rename_collection(client, collection_name, another_collection_name)


class TestMilvusClientUsingDatabaseInvalid(TestMilvusClientV2Base):
    """ Test case of using database interface """

    """
    ******************************************************************
    #  The following are invalid base cases
    ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.skip(reason="pymilvus issue 1900")
    @pytest.mark.parametrize("db_name", ["12-s", "12 s", "(mn)", "中文", "%$#"])
    def test_milvus_client_using_database_not_exist_db_name(self, db_name):
        """
        target: test fast create collection normal case
        method: create collection
        expected: drop successfully
        """
        client = self._client()
        # db_name = cf.gen_unique_str("nonexisted")
        error = {ct.err_code: 999, ct.err_msg: f"database not found[database={db_name}]"}
        self.using_database(client, db_name,
                            check_task=CheckTasks.err_res, check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.skip(reason="# this case is dup to using a non exist db name, try to add one for create database")
    def test_milvus_client_using_database_db_name_over_max_length(self):
        """
        target: test fast create collection normal case
        method: create collection
        expected: drop successfully
        """
        pass

class TestMilvusClientCollectionPropertiesInvalid(TestMilvusClientV2Base):
    """ Test case of alter/drop collection properties """
    """
    ******************************************************************
    #  The following are invalid base cases
    ******************************************************************
    """

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("alter_name", ["%$#", "test", " "])
    def test_milvus_client_alter_collection_properties_invalid_collection_name(self, alter_name):
        """
        target: test alter collection properties with invalid collection name
        method: alter collection properties with non-existent collection name
        expected: raise exception
        """
        client = self._client()
        # alter collection properties
        properties = {'mmap.enabled': True}
        error = {ct.err_code: 100, ct.err_msg: f"collection not found[database=default][collection={alter_name}]"}
        self.alter_collection_properties(client, alter_name, properties,
                                     check_task=CheckTasks.err_res,
                                     check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("properties", [""])
    def test_milvus_client_alter_collection_properties_invalid_properties(self, properties):
        """
        target: test alter collection properties with invalid properties
        method: alter collection properties with invalid properties
        expected: raise exception
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        self.create_collection(client, collection_name, default_dim, id_type="string", max_length=ct.default_length)
        self.describe_collection(client, collection_name,
                                     check_task=CheckTasks.check_describe_collection_property,
                                     check_items={"collection_name": collection_name,
                                                  "dim": default_dim,
                                                  "consistency_level": 0})
        error = {ct.err_code: 1, ct.err_msg: f"`properties` value {properties} is illegal"}
        self.alter_collection_properties(client, collection_name, properties,
                                     check_task=CheckTasks.err_res,
                                     check_items=error)

        self.drop_collection(client, collection_name)

    #TODO properties with non-existent params

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("drop_name", ["%$#", "test", " "])
    def test_milvus_client_drop_collection_properties_invalid_collection_name(self, drop_name):
        """
        target: test drop collection properties with invalid collection name
        method: drop collection properties with non-existent collection name
        expected: raise exception
        """
        client = self._client()
        # drop collection properties
        properties = {'mmap.enabled': True}
        error = {ct.err_code: 100, ct.err_msg: f"collection not found[database=default][collection={drop_name}]"}
        self.drop_collection_properties(client, drop_name, properties,
                                        check_task=CheckTasks.err_res,
                                        check_items=error)

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("property_keys", ["", {}, []])
    def test_milvus_client_drop_collection_properties_invalid_properties(self, property_keys):
        """
        target: test drop collection properties with invalid properties
        method: drop collection properties with invalid properties
        expected: raise exception
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        # 1. create collection
        self.create_collection(client, collection_name, default_dim, id_type="string", max_length=ct.default_length)
        self.describe_collection(client, collection_name,
                                     check_task=CheckTasks.check_describe_collection_property,
                                     check_items={"collection_name": collection_name,
                                                  "dim": default_dim,
                                                  "consistency_level": 0})
        error = {ct.err_code: 65535, ct.err_msg: f"The collection properties to alter and keys to delete must not be empty at the same time"}
        self.drop_collection_properties(client, collection_name, property_keys,
                                     check_task=CheckTasks.err_res,
                                     check_items=error)

        self.drop_collection(client, collection_name)

    #TODO properties with non-existent params


class TestMilvusClientCollectionPropertiesValid(TestMilvusClientV2Base):
    """ Test case of alter/drop collection properties """

    """
    ******************************************************************
    #  The following are valid base cases
    ******************************************************************
    """
    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_collection_alter_collection_properties(self):
        """
        target: test alter collection
        method: alter collection
        expected: alter successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        self.using_database(client, "default")
        # 1. create collection
        self.create_collection(client, collection_name, default_dim)
        collections = self.list_collections(client)[0]
        assert collection_name in collections
        self.release_collection(client, collection_name)
        properties = {"mmap.enabled": True}
        self.alter_collection_properties(client, collection_name, properties)
        describe = self.describe_collection(client, collection_name)[0].get("properties")
        assert describe["mmap.enabled"] == 'True'
        self.release_collection(client, collection_name)
        properties = {"mmap.enabled": False}
        self.alter_collection_properties(client, collection_name, properties)
        describe = self.describe_collection(client, collection_name)[0].get("properties")
        assert describe["mmap.enabled"] == 'False'
        #TODO add case that confirm the parameter is actually valid
        self.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_client_collection_drop_collection_properties(self):
        """
        target: test drop collection
        method: drop collection
        expected: drop successfully
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)
        self.using_database(client, "default")
        # 1. create collection
        self.create_collection(client, collection_name, default_dim)
        collections = self.list_collections(client)[0]
        assert collection_name in collections
        self.release_collection(client, collection_name)
        properties = {"mmap.enabled": True}
        self.alter_collection_properties(client, collection_name, properties)
        describe = self.describe_collection(client, collection_name)[0].get("properties")
        assert describe["mmap.enabled"] == 'True'
        property_keys = ["mmap.enabled"]
        self.drop_collection_properties(client, collection_name, property_keys)
        describe = self.describe_collection(client, collection_name)[0].get("properties")
        assert "mmap.enabled" not in describe
        #TODO add case that confirm the parameter is actually invalid
        self.drop_collection(client, collection_name)