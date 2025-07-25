import logging
from utils.util_pymilvus import *
from common.common_type import CaseLabel, CheckTasks
from common import common_type as ct
from common import common_func as cf
from base.client_v2_base import TestMilvusClientV2Base
import pytest
from idx_hnsw import HNSW

index_type = "HNSW"
success = "success"
pk_field_name = 'id'
vector_field_name = 'vector'
dim = ct.default_dim
default_nb = 2000
default_build_params = {"M": 16, "efConstruction": 200}
default_search_params = {"ef": 64}


class TestHnswBuildParams(TestMilvusClientV2Base):
    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize("params", HNSW.build_params)
    def test_hnsw_build_params(self, params):
        """
        Test the build params of HNSW index
        """
        client = self._client()
        collection_name = cf.gen_collection_name_by_testcase_name()
        schema, _ = self.create_schema(client)
        schema.add_field(pk_field_name, datatype=DataType.INT64, is_primary=True, auto_id=False)
        schema.add_field(vector_field_name, datatype=DataType.FLOAT_VECTOR, dim=dim)
        self.create_collection(client, collection_name, schema=schema)

        # Insert data in 2 batches with unique primary keys
        insert_times = 2
        random_vectors = list(cf.gen_vectors(default_nb * insert_times, dim, vector_data_type=DataType.FLOAT_VECTOR))
        for j in range(insert_times):
            start_pk = j * default_nb
            rows = [{
                pk_field_name: i + start_pk,
                vector_field_name: random_vectors[i + start_pk]
            } for i in range(default_nb)]
            self.insert(client, collection_name, rows)
        self.flush(client, collection_name)

        # create index
        build_params = params.get("params", None)
        index_params = self.prepare_index_params(client)[0]
        index_params.add_index(field_name=vector_field_name,
                               metric_type=cf.get_default_metric_for_vector_type(vector_type=DataType.FLOAT_VECTOR),
                               index_type=index_type,
                               params=build_params)
        # build index
        if params.get("expected", None) != success:
            self.create_index(client, collection_name, index_params,
                              check_task=CheckTasks.err_res,
                              check_items=params.get("expected"))
        else:
            self.create_index(client, collection_name, index_params)
            self.wait_for_index_ready(client, collection_name, index_name=vector_field_name)

            # load collection
            self.load_collection(client, collection_name)

            # search
            nq = 2
            search_vectors = cf.gen_vectors(nq, dim=dim, vector_data_type=DataType.FLOAT_VECTOR)
            self.search(client, collection_name, search_vectors,
                        search_params=default_search_params,
                        limit=ct.default_limit,
                        check_task=CheckTasks.check_search_results,
                        check_items={"enable_milvus_client_api": True,
                                     "nq": nq,
                                     "limit": ct.default_limit,
                                     "pk_name": pk_field_name})

            # verify the index params are persisted
            idx_info = client.describe_index(collection_name, vector_field_name)
            if build_params is not None:
                for key, value in build_params.items():
                    if value is not None:
                        assert key in idx_info.keys()
                        assert str(value) in idx_info.values()

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("vector_data_type", ct.all_vector_types)
    def test_hnsw_on_all_vector_types(self, vector_data_type):
        """
        Test HNSW index on all the vector types and metrics
        """
        client = self._client()
        collection_name = cf.gen_collection_name_by_testcase_name()
        schema, _ = self.create_schema(client)
        schema.add_field(pk_field_name, datatype=DataType.INT64, is_primary=True, auto_id=False)
        if vector_data_type == DataType.SPARSE_FLOAT_VECTOR:
            schema.add_field(vector_field_name, datatype=vector_data_type)
        else:
            schema.add_field(vector_field_name, datatype=vector_data_type, dim=dim)
        self.create_collection(client, collection_name, schema=schema)

        # Insert data in 2 batches with unique primary keys
        insert_times = 2
        random_vectors = list(cf.gen_vectors(default_nb*insert_times, dim, vector_data_type=vector_data_type)) \
            if vector_data_type == DataType.FLOAT_VECTOR \
            else cf.gen_vectors(default_nb*insert_times, dim, vector_data_type=vector_data_type)
        for j in range(insert_times):
            start_pk = j * default_nb
            rows = [{
                pk_field_name: i + start_pk,
                vector_field_name: random_vectors[i + start_pk]
            } for i in range(default_nb)]
            self.insert(client, collection_name, rows)
        self.flush(client, collection_name)

        # create index
        index_params = self.prepare_index_params(client)[0]
        metric_type = cf.get_default_metric_for_vector_type(vector_data_type)
        index_params.add_index(field_name=vector_field_name,
                               metric_type=metric_type,
                               index_type=index_type,
                               M=16,
                               efConstruction=200)
        if vector_data_type not in HNSW.supported_vector_types:
            self.create_index(client, collection_name, index_params,
                              check_task=CheckTasks.err_res,
                              check_items={"err_code": 999,
                                           "err_msg": f"can't build with this index HNSW: invalid parameter"})
        else:
            self.create_index(client, collection_name, index_params)
            self.wait_for_index_ready(client, collection_name, index_name=vector_field_name)
            # load collection
            self.load_collection(client, collection_name)
            # search
            nq = 2
            search_vectors = cf.gen_vectors(nq, dim=dim, vector_data_type=vector_data_type)
            self.search(client, collection_name, search_vectors,
                        search_params=default_search_params,
                        limit=ct.default_limit,
                        check_task=CheckTasks.check_search_results,
                        check_items={"enable_milvus_client_api": True,
                                     "nq": nq,
                                     "limit": ct.default_limit,
                                     "pk_name": pk_field_name})

    @pytest.mark.tags(CaseLabel.L2)
    @pytest.mark.parametrize("metric", HNSW.supported_metrics)
    def test_hnsw_on_all_metrics(self, metric):
        """
        Test the search params of HNSW index
        """
        client = self._client()
        collection_name = cf.gen_collection_name_by_testcase_name()
        schema, _ = self.create_schema(client)
        schema.add_field(pk_field_name, datatype=DataType.INT64, is_primary=True, auto_id=False)
        schema.add_field(vector_field_name, datatype=DataType.FLOAT_VECTOR, dim=dim)
        self.create_collection(client, collection_name, schema=schema)

        # insert data
        insert_times = 2
        random_vectors = list(cf.gen_vectors(default_nb*insert_times, dim, vector_data_type=DataType.FLOAT_VECTOR))
        for j in range(insert_times):
            start_pk = j * default_nb
            rows = [{
                pk_field_name: i + start_pk,
                vector_field_name: random_vectors[i + start_pk]
            } for i in range(default_nb)]
            self.insert(client, collection_name, rows)
        self.flush(client, collection_name)

        # create index
        index_params = self.prepare_index_params(client)[0]
        index_params.add_index(field_name=vector_field_name,
                               metric_type=metric,
                               index_type=index_type,
                               M=16,
                               efConstruction=200)
        self.create_index(client, collection_name, index_params)
        self.wait_for_index_ready(client, collection_name, index_name=vector_field_name)
        # load collection
        self.load_collection(client, collection_name)
        # search
        nq = 2
        search_vectors = cf.gen_vectors(nq, dim=dim, vector_data_type=DataType.FLOAT_VECTOR)
        self.search(client, collection_name, search_vectors,
                    search_params=default_search_params,
                    limit=ct.default_limit,
                    check_task=CheckTasks.check_search_results,
                    check_items={"enable_milvus_client_api": True,
                                 "nq": nq,
                                 "limit": ct.default_limit,
                                 "pk_name": pk_field_name})


@pytest.mark.xdist_group("TestHnswSearchParams")
class TestHnswSearchParams(TestMilvusClientV2Base):
    """Test search with pagination functionality for HNSW index"""

    def setup_class(self):
        super().setup_class(self)
        self.collection_name = "TestHnswSearchParams" + cf.gen_unique_str("_")
        self.float_vector_field_name = vector_field_name
        self.float_vector_dim = dim
        self.primary_keys = []
        self.enable_dynamic_field = False
        self.datas = []

    @pytest.fixture(scope="class", autouse=True)
    def prepare_collection(self, request):
        """
        Initialize collection before test class runs
        """
        client = self._client()
        collection_schema = self.create_schema(client)[0]
        collection_schema.add_field(pk_field_name, DataType.INT64, is_primary=True, auto_id=False)
        collection_schema.add_field(self.float_vector_field_name, DataType.FLOAT_VECTOR, dim=128)
        self.create_collection(client, self.collection_name, schema=collection_schema,
                               enable_dynamic_field=self.enable_dynamic_field, force_teardown=False)
        insert_times = 2
        float_vectors = cf.gen_vectors(default_nb * insert_times, dim=self.float_vector_dim,
                                       vector_data_type=DataType.FLOAT_VECTOR)
        for j in range(insert_times):
            rows = []
            for i in range(default_nb):
                pk = i + j * default_nb
                row = {
                    pk_field_name: pk,
                    self.float_vector_field_name: list(float_vectors[pk])
                }
                self.datas.append(row)
                rows.append(row)
            self.insert(client, self.collection_name, data=rows)
            self.primary_keys.extend([i + j * default_nb for i in range(default_nb)])
        self.flush(client, self.collection_name)
        # Create HNSW index
        index_params = self.prepare_index_params(client)[0]
        index_params.add_index(field_name=self.float_vector_field_name,
                               metric_type="COSINE",
                               index_type=index_type,
                               params=default_build_params)
        self.create_index(client, self.collection_name, index_params=index_params)
        self.wait_for_index_ready(client, self.collection_name, index_name=self.float_vector_field_name)
        self.load_collection(client, self.collection_name)

        def teardown():
            self.drop_collection(self._client(), self.collection_name)
        request.addfinalizer(teardown)

    @pytest.mark.tags(CaseLabel.L1)
    @pytest.mark.parametrize("params", HNSW.search_params)
    def test_hnsw_search_params(self, params):
        """
        Test the search params of HNSW index
        """
        client = self._client()
        collection_name = self.collection_name
        nq = 2
        search_vectors = cf.gen_vectors(nq, dim=self.float_vector_dim, vector_data_type=DataType.FLOAT_VECTOR)
        search_params = params.get("params", None)
        if params.get("expected", None) != success:
            self.search(client, collection_name, search_vectors,
                        search_params=search_params,
                        limit=ct.default_limit,
                        check_task=CheckTasks.err_res,
                        check_items=params.get("expected"))
        else:
            self.search(client, collection_name, search_vectors,
                        search_params=search_params,
                        limit=ct.default_limit,
                        check_task=CheckTasks.check_search_results,
                        check_items={"enable_milvus_client_api": True,
                                     "nq": nq,
                                     "limit": ct.default_limit,
                                     "pk_name": pk_field_name}) 