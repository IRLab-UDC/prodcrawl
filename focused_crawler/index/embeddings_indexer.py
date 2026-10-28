import re
from typing import Dict, Any, List, Optional, Sequence, Tuple, Union

from elasticsearch import Elasticsearch
from elasticsearch import helpers
from focused_crawler.settings import ES_URL


class IndexManager(object):

    def __init__(self,
                 es_url=ES_URL,
                 es_user=None,
                 es_pass=None):

        self.es_url = es_url
        self.http_auth = (es_user, es_pass) if es_user and es_pass else None
        self.es = Elasticsearch(
            self.es_url,
            http_auth=self.http_auth,
            timeout=3000
        )

    def create_index(self,
                     mapping: Dict[str, Any],
                     index_name: str):

        self.es.indices.create(index=index_name, ignore=400, body=mapping)

    def index_document(self,
                       document: Dict[str, Any],
                       index_name: str):

        return self.es.index(
            index=index_name,
            body=document
        )

    def page_is_indexed(self,
                        url: str,
                        index_name: str) -> bool:

        query: Dict[str, Any] = {
            "query": {
                "term": {
                    "url": url
                }
            },
            "_source": False
        }

        search_results = self.es.search(index=index_name, body=query, size=1)

        return len(search_results['hits']['hits']) > 0

    def count_all(self, index_name: str):

        query: Dict[str, Any] = {
            "query": {
                "match_all": {}
            }
        }

        return self.es.count(
            index=index_name,
            body=query,
        )['count']

    def count_by_field(self,
                       field_name: str,
                       field_value: Any,
                       index_name: str):

        query: Dict[str, Any] = {
            "query": {
                "match": {
                    field_name: field_value
                }
            }
        }

        return self.es.count(
            index=index_name,
            body=query,
        )['count']

    def match_field(self,
                    field_name: str,
                    field_value: Any,
                    index_name: str,
                    k: int,
                    source: Optional[List[str]] = None):

        query: Dict[str, Any] = {
            "query": {
                "match": {
                    field_name: field_value
                }
            }
        }

        if source:
            query["fields"] = source
            query["_source"] = False

        return self.es.search(
            index=index_name,
            body=query,
            size=k
        )

    def term_field(self,
                   field_name: str,
                   field_value: Any,
                   index_name: str,
                   k: int,
                   source: Optional[List[str]] = None):

        query: Dict[str, Any] = {
            "query": {
                "term": {
                    field_name: field_value
                }
            }
        }

        if source:
            query["fields"] = source
            query["_source"] = False

        return self.es.search(
            index=index_name,
            body=query,
            size=k
        )

    def bm25_query_string(
        self,
        field_name: str,
        field_value: Any,
        index_name: Union[str, Sequence[str]],
        ranking: int,
        source: Optional[List[str]] = None,
    ):

        query: Dict[str, Any] = {
            "query": {
                "simple_query_string": {
                    "query": field_value,
                    "fields": [field_name]
                }
            },
            "size": ranking
        }

        if source:
            query["fields"] = source
            query["_source"] = False

        return self.es.search(
            index=index_name,
            body=query,
        )

    def search_by_field_with_cos_similarity_rescore(
        self,
        text_field_name: str,
        query_string: Any,
        embedding_field_name: str,
        query_embedding: List[float],
        index_name: Union[str, Sequence[str]],
        top_k: int,
        re_ranking_top: int,
        source: Optional[List[str]] = None,
    ):

        query: Dict[str, Any] = {
            "query": {
                "bool": {
                    # 1. Filter context: binary match, fast, does NOT affect score
                    "filter": [{"exists": {"field": embedding_field_name}}],
                    # 2. Must context: calculates BM25 score for initial retrieval
                    "must": [
                        {
                            "simple_query_string": {
                                "query": query_string,
                                "fields": [text_field_name],
                            }
                        }
                    ],
                }
            },
            "size": top_k,
            "rescore": {
                "window_size": re_ranking_top,
                "query": {
                    "rescore_query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": f'cosineSimilarity(params.query_vector, "{embedding_field_name}") + 1.0',
                                "params": {"query_vector": query_embedding},
                            },
                        }
                    },
                    "query_weight": 0,
                    "rescore_query_weight": 1,
                },
            },
        }

        if source:
            query["fields"] = source
            query["_source"] = False

        return self.es.search(index=index_name, body=query)

    def aknn_search(
        self,
        score_field_name: str,
        embedding: List[float],
        index_name: Union[str, Sequence[str]],
        ranking: int,
        num_candidates: int,
        source: Optional[List[str]] = None,
    ):

        query: Dict[str, Any] = {
            "knn": {
                "field": score_field_name,
                "query_vector": embedding,
                "k": ranking,
                "num_candidates": num_candidates,
            },
            "size": ranking
        }

        if source:
            query["fields"] = source
            query["_source"] = False

        return self.es.search(
            index=index_name,
            body=query,
        )

    def knn_search(
        self,
        score_field_name: str,
        embedding: List[float],
        index_name: Union[str, Sequence[str]],
        ranking: int,
        source: Optional[List[str]] = None,
    ):

        query: Dict[str, Any] = {
            "size": ranking,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": f'cosineSimilarity(params.query_vector, "{score_field_name}") + 1.0',
                        "params": {"query_vector": embedding},
                    },
                }
            },
            # First: we sort by the cosineSimilarity score
            # Second: to break ties, we sort by insert date
            "sort": [{"_score": "desc"}, {"insert_date": "asc"}],
        }

        if source:
            query["fields"] = source
            query["_source"] = False

        return self.es.search(
            index=index_name,
            body=query,
        )

    def scan(self,
             query,
             index_name,
             scroll='5m',
             size=1000,
             preserve_order=True):

        return helpers.scan(
            self.es,
            query=query,
            index=index_name,
            scroll=scroll,
            size=size,
            preserve_order=preserve_order
        )

    def bulk_update(self,
                    docs: List[Dict[str, Any]],
                    **kwargs) -> Tuple[int, Union[int, List[Dict[str, Any]]]]:

        success, errors = helpers.bulk(self.es,
                                       docs,
                                       refresh='wait_for',
                                       **kwargs)

        return success, errors

    def reindex(self,
                source_index_name: str,
                dest_index_name: str):

        helpers.reindex(self.es,
                        source_index=source_index_name,
                        target_index=dest_index_name)
