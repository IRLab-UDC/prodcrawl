from enum import Enum
import random
import logging

from abc import ABC
from typing import Any, Dict, List, Optional, Sequence, Union
from scrapy.http.request import Request as ScrapyRequest

from focused_crawler.common.db_utils import get_label
from focused_crawler.common.utils import (
    get_anchor_text as get_request_anchor_text,
    get_embedding_from_anchor_text,
)
from focused_crawler.common.exceptions import UnknownCalcScoreOption
from focused_crawler.index.embeddings_indexer import IndexManager
from focused_crawler.nlp.utils import get_cached_model
from focused_crawler.settings import PRIORITY_SCALE

index_manager: IndexManager = IndexManager()
logger = logging.getLogger(__name__)


class CalcScoreOptions(Enum):
    MAJORITY_VOTING = "majority"
    SCORES_RATIO = "ratio"


class ScoreStrategy(ABC):
    def get_anchor_text(self, request):
        return get_request_anchor_text(request)

    def get_score(self, request, *args, **kwargs):
        anchor_text: str = self.get_anchor_text(request)
        logger.info(f"Computing score for --> {anchor_text}")

    def get_priority(self, score: float) -> int:
        # Invert the score for the min-heap and scale before truncating, so the
        # integer priority keeps the ordering of the score rather than
        # collapsing every positive score onto 0: score 1.0 -> 0 (popped
        # first), score 0.0 -> PRIORITY_SCALE (popped last). Scrapy requires an
        # int priority and the indexed field is an integer, so the score cannot
        # be carried as a float.
        return int(round((1.0 - score) * PRIORITY_SCALE))

    def _calc_score_from_query_result(
        self,
        hits: List[Dict],
        option: CalcScoreOptions,
        query: str,
    ) -> float:
        
        if not hits:
            logger.info(f"Retrieved no hits for query --> {query}")
            return 0.0

        if option == CalcScoreOptions.MAJORITY_VOTING:
            positives: float = float(len([h for h in hits if h["fields"]["label"][0]]))
            total: float = float(len(hits))

        elif option == CalcScoreOptions.SCORES_RATIO:
            positives: float = float(sum([h["_score"] for h in hits if h["fields"]["label"][0]]))
            total: float = float(sum([h["_score"] for h in hits]))

        else:
            raise UnknownCalcScoreOption

        return positives / total


class BM25(ScoreStrategy):

    def get_score(
        self,
        request: ScrapyRequest,
        *args, 
        **kwargs,
    ) -> float:

        super().get_score(request, *args, **kwargs)

        index_name: Union[str, Sequence[str]] = kwargs.get("index_name")
        ranking: int = kwargs.get("ranking")
        option: CalcScoreOptions = CalcScoreOptions(kwargs.get("score_option"))

        anchor_text: str = self.get_anchor_text(request)
        hits = (
            index_manager.bm25_query_string(
                field_name="anchor_text",
                field_value=anchor_text,
                index_name=index_name,
                ranking=ranking,
                source=["label"],
            )
            .get("hits", {})
            .get("hits", [])
        )

        score: float = self._calc_score_from_query_result(
            hits,
            option,
            anchor_text,
        )

        return score

class EmbeddingCosSim(ScoreStrategy):

    def get_score(
        self,
        request: ScrapyRequest,
        *args, 
        **kwargs,
    ) -> float:

        super().get_score(
            request,
            *args,
            **kwargs,
        )

        index_name: Union[str, Sequence[str]] = kwargs.get("index_name")
        ranking: int = kwargs.get("ranking")
        embeddings_cache_dir: str = kwargs.get("embeddings_cache_dir")
        embeddings_model: str = kwargs.get("embeddings_model")
        clean_anchor: bool = kwargs.get("clean_anchor")
        nlp_api_url: str = kwargs.get("nlp_api_url")
        option: CalcScoreOptions = CalcScoreOptions(kwargs.get("score_option"))

        anchor_text: str = self.get_anchor_text(request)
        anchor_text_embedding: List[float] = get_embedding_from_anchor_text(
            anchor_text=anchor_text,
            cache_dir=embeddings_cache_dir,
            clean_anchor=clean_anchor,
            embeddings_model=embeddings_model,
            nlp_api_url=nlp_api_url,
        )

        hits = (
            index_manager.knn_search(
                score_field_name="anchor_text_embedding",
                embedding=anchor_text_embedding,
                index_name=index_name,
                ranking=ranking,
                source=["label"],
            )
            .get("hits", {})
            .get("hits", [])
        )

        score: float = self._calc_score_from_query_result(
            hits,
            option,
            anchor_text,
        )

        return score
    

class EmbeddingCosSimAproximated(ScoreStrategy):

    def get_score(
        self,
        request: ScrapyRequest,
        *args, 
        **kwargs,
    ) -> float:

        super().get_score(
            request,
            *args,
            **kwargs,
        )

        index_name: Union[str, Sequence[str]] = kwargs.get("index_name")
        ranking: int = kwargs.get("ranking")
        embeddings_cache_dir: str = kwargs.get("embeddings_cache_dir")
        embeddings_model: str = kwargs.get("embeddings_model")
        clean_anchor: bool = kwargs.get("clean_anchor")
        nlp_api_url: str = kwargs.get("nlp_api_url")
        option: CalcScoreOptions = CalcScoreOptions(kwargs.get("score_option"))

        anchor_text: str = self.get_anchor_text(request)
        anchor_text_embedding: List[float] = get_embedding_from_anchor_text(
            anchor_text=anchor_text,
            cache_dir=embeddings_cache_dir,
            clean_anchor=clean_anchor,
            embeddings_model=embeddings_model,
            nlp_api_url=nlp_api_url,
        )

        hits = (
            index_manager.aknn_search(
                score_field_name="anchor_text_embedding",
                embedding=anchor_text_embedding,
                index_name=index_name,
                ranking=ranking,
                num_candidates=1000,
                source=["label"],
            )
            .get("hits", {})
            .get("hits", [])
        )

        score: float = self._calc_score_from_query_result(
            hits,
            option,
            anchor_text,
        )

        return score


class BM25EmbeddingCosSim(ScoreStrategy):

    def get_score(
        self,
        request: ScrapyRequest,
        *args,
        **kwargs,
    ) -> float:

        super().get_score(
            request,
            *args,
            **kwargs,
        )

        index_name: Union[str, Sequence[str]] = kwargs.get("index_name")
        ranking: int = kwargs.get("ranking")
        re_ranking: int = kwargs.get("re_ranking")
        embeddings_cache_dir: str = kwargs.get("embeddings_cache_dir")
        embeddings_model: str = kwargs.get("embeddings_model")
        clean_anchor: bool = kwargs.get("clean_anchor")
        nlp_api_url: str = kwargs.get("nlp_api_url")
        option: CalcScoreOptions = CalcScoreOptions(kwargs.get("score_option"))

        anchor_text: str = self.get_anchor_text(request)
        anchor_text_embedding: List[float] = get_embedding_from_anchor_text(
            anchor_text=anchor_text,
            embeddings_model=embeddings_model,
            cache_dir=embeddings_cache_dir,
            clean_anchor=clean_anchor,
            nlp_api_url=nlp_api_url,
        )

        hits = (
            index_manager.search_by_field_with_cos_similarity_rescore(
                text_field_name="anchor_text",
                query_string=anchor_text,
                query_embedding=anchor_text_embedding,
                embedding_field_name="anchor_text_embedding",
                index_name=index_name,
                top_k=ranking,
                re_ranking_top=re_ranking,
                source=["label"],
            )
            .get("hits", {})
            .get("hits", [])
        )

        score: float = self._calc_score_from_query_result(
            hits,
            option,
            anchor_text,
        )

        return score


""" BASE LINES """


class RandomScore(ScoreStrategy):

    def get_score(
        self,
        request: ScrapyRequest,
        *args,
        **kwargs,
    ) -> float:
        
        ranking: int = kwargs.get("ranking")

        return random.randint(0, ranking)


class BreadthFirstSearchScore(ScoreStrategy):
    def __init__(self):
        self.counter: float = 0.0

    def get_score(
        self,
        request: ScrapyRequest,
        *args,
        **kwargs,
    ) -> float:

        self.counter += 1.0
        return self.counter

    def get_priority(self, score: float) -> int:
        return int(score)


class DepthFirstSearchScore(ScoreStrategy):

    def __init__(self):
        self.counter: float = 0.0

    def get_score(
        self,
        request: ScrapyRequest,
        *args,
        **kwargs,
    ) -> float:

        self.counter -= 1.0
        return self.counter

    def get_priority(self, score: float) -> int:
        return int(score)


class DefaultScore(ScoreStrategy):
    """Scrapy default"""

    def get_score(
        self,
        request: ScrapyRequest,
        *args,
        **kwargs,
    ) -> float:

        return 0


class ClassifierScore(ScoreStrategy):
    classifier_name: str = "No-classifier"
    vectorizer: Optional[Any]
    classifier_model: Optional[Any]

    def __init__(self):
        self.vectorizer = None
        self.classifier_model = None

    def get_score(
        self,
        request: ScrapyRequest,
        *args,
        **kwargs,
    ) -> float:

        anchor_text: str = self.get_anchor_text(request)
        anchor_text_tf_idf_vector: List[float] = self.vectorizer.transform(
            [anchor_text]
        )

        # predict -> binary label (0 or 1)
        # predict_proba -> probability of being in each class
        # our baseline are binary classifiers, so we use predict
        score = float(self.classifier_model.predict(
            anchor_text_tf_idf_vector)[0])

        return score


class SVMClassifierScore(ClassifierScore):
    classifier_name: str = "svm_linear_classifier_C_0.1_max_iter_1000"

    def __init__(self):
        super(SVMClassifierScore, self).__init__()
        self.vectorizer = get_cached_model("svm_tfidf_vectorizer.joblib")
        self.classifier_model = get_cached_model(
            f"{self.classifier_name}.joblib")


class NaiveBayesClassifierScore(ClassifierScore):
    classifier_name: str = "multinomial_naive_bayes_classifier"

    def __init__(self):
        super(NaiveBayesClassifierScore, self).__init__()
        self.vectorizer = get_cached_model("svm_tfidf_vectorizer.joblib")
        self.classifier_model = get_cached_model(
            f"{self.classifier_name}.joblib")


class OracleScore(ScoreStrategy):

    def get_score(
        self,
        request: ScrapyRequest,
        *args,
        **kwargs,
    ) -> float:

        original_url: str = request.meta["original_url"]
        db_dir: str = kwargs.get("db_dir", "")
        label: bool = get_label(original_url, db_dir)
        
        return float(label)
