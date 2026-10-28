import importlib
import re

import traceback
from typing import List, Sequence, Set, Union
from scrapy.spiders import Rule
from scrapy.spiders import CrawlSpider
from scrapy.linkextractors import LinkExtractor
from scrapy.http.response import Response as ScrapyResponse
from scrapy.http.request import Request as ScrapyRequest
from scrapy.http import HtmlResponse

from focused_crawler.common.link_extractors import SurroundingWordsLinkExtractor
from focused_crawler.common.db_utils import get_label, bulk_judgments
from focused_crawler.common.scoring_strategies import ScoreStrategy
from focused_crawler.common.utils import (
    extract_domain_from_url,
    get_anchor_text,
    get_embedding_from_anchor_text,
)
from focused_crawler.index.embeddings_indexer import IndexManager
from focused_crawler.items import FocusedCrawlerIndexItem
from focused_crawler.nlp.utils import get_page_raw_text_content
from focused_crawler.settings import CLASSIFICATION_LOG_DIR, ES_URL


class ProductsSpider(CrawlSpider):

    @staticmethod
    def _instantiate_strategy(name: str):
        strategy_class = getattr(
            importlib.import_module("focused_crawler.common.scoring_strategies"),
            name,
        )

        return strategy_class()

    def _parse_start_data_indices(self) -> Union[str, Sequence[str]]:
        indices: List[str] = []

        for start_data_index_name in self.start_data.split("+"):
            start_data_index_name_clean = start_data_index_name.strip()

            if not start_data_index_name_clean:
                continue 

            if start_data_index_name_clean == "coldstart":
                indices.append(self.index_name)
            else:
                indices.append(start_data_index_name_clean)

        return list(set(indices))

    def __init__(
        self,
        dataset_name: str,
        score_strategy_name: str,
        anchor_text_window: str,
        ranking: str,
        re_ranking: str,
        use_embeddings: str,
        start_data: str,
        commit_hash: str,
        embeddings_model: str,
        score_option: str,
        save_judgments: str,
        embeddings_cache: str,
        clean_anchors: str,
        db_dir: str,
        nlp_api_url: str = "",
        max_links_per_page: str = "100",
        *args,
        **kwargs,
    ):

        """############# spider properties #############"""
        self.dataset_name: str = dataset_name
        self.embeddings_model: str = embeddings_model
        self.embeddings_cache: str = embeddings_cache
        self.db_dir: str = db_dir
        self.nlp_api_url: str = nlp_api_url
        # Links scored per downloaded page, in document order (0 = no limit). The
        # experiments use 100; the collections were built without a limit.
        self.max_links_per_page: int | None = int(max_links_per_page) or None
        self.clean_anchors: bool = bool(clean_anchors == "yes")
        self.score_option: str = score_option
        self.start_urls: List[str] = [kwargs.get("seed", "")]
        self.allowed_domains: List[str] = [extract_domain_from_url(self.start_urls[0])]

        escaped_seed = self.start_urls[0].replace(".", "\\.")
        allowed_regex = re.compile(f"{escaped_seed}.*")

        self.anchor_text_window: int = int(anchor_text_window)

        if self.anchor_text_window == 0:
            self.rules = (
                Rule(
                    LinkExtractor(allow=allowed_regex),
                    callback="parse_item",
                    follow=True,
                ),
            )

        else:

            self.rules = (
                Rule(
                    SurroundingWordsLinkExtractor(allow=allowed_regex),
                    callback="parse_item",
                    follow=True,
                ),
            )

        super().__init__(*args, **kwargs)

        """############# ES setup #############"""
        es_url: str = kwargs.get("ES_URL", ES_URL)
        es_user: str | None = kwargs.get("user", None)
        es_pass: str | None = kwargs.get("pass", None)

        self.indexer: IndexManager = IndexManager(
            es_url=es_url,
            es_user=es_user,
            es_pass=es_pass,
        )

        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "index": {
                    "analysis": {
                        "analyzer": {
                            "simple_stem": {
                                "tokenizer": "standard",
                                "filter": ["lowercase", "stop", "simple_f_stemmer"],
                            },
                        },
                        "filter": {
                            "simple_f_stemmer": {
                                "type": "stemmer",
                                "name": "minimal_english",
                            }
                        },
                    }
                },
            },
            "mappings": {
                "properties": {
                    "anchor_text_embedding": {
                        "type": "dense_vector",
                        "dims": 768,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "anchor_text": {"type": "text", "analyzer": "simple_stem"},
                    "url": {"type": "keyword", "ignore_above": 1000000},
                    "label": {"type": "boolean"},
                    "insert_date": {"type": "date"},
                    "page_content": {"type": "text"},
                    "priority": {"type": "integer"},
                    "depth": {"type": "integer"},
                    "download_order": {"type": "integer"},
                    "domain": {"type": "keyword"},
                }
            },
        }

        """############# experiments meta-params #############"""
        self.score_strategy: ScoreStrategy = self._instantiate_strategy(
            score_strategy_name
        )
        self.ranking: int = int(ranking)
        self.re_ranking: int = int(re_ranking)
        self.use_embeddings: bool = use_embeddings == "yes"
        self.start_data: str = start_data
        self.save_judgments: bool = save_judgments == "yes"
        self.judged_urls: Set[str] = set()

        # If it is not "coldstart", the start string contains the naming of the training data index.
        indices: Union[str, Sequence[str]] = [s.strip() for s in self.start_data.split("+") if s.strip()]

        if len(indices) == 1:
            if indices[0] == "coldstart":
                start_str = "coldstart"
            else:
                start_str = "training_data"

        else:
            if "coldstart" in indices:
                start_str = "coldstart__and__training_data"
            else:
                start_str = "__".join(indices)

        self.index_name: str = (
            f"{score_strategy_name}-top{self.ranking}-re_ranking{self.re_ranking}-{start_str}-{self.score_option}-{commit_hash}-{self.dataset_name}".lower()
        )

        if self.embeddings_model:
            self.index_name = self.index_name + f"-{self.embeddings_model}"

        self.indexer.create_index(
            index_name=self.index_name,
            mapping=mapping,
        )

    name: str = "ProductsSpider"

    def start_requests(self):
        if not self.start_urls and hasattr(self, "start_url"):
            raise AttributeError(
                "Crawling could not start: 'start_urls' not found "
                "or empty (but found 'start_url' attribute instead, "
                "did you miss an 's'?)"
            )
        for url in self.start_urls:
            request = ScrapyRequest(url, dont_filter=True)
            # Default prediction for seeds
            request.meta["prediction"] = 1
            yield request

    def _get_original_url(self, url: str) -> str:
        return url.replace(self.allowed_domains[0]+'/', '')

    def _bulk_judgments_into_db(self, requests: list[ScrapyRequest]):
        judgments_tuples_to_bulk = []

        for r in requests:
            original_url: str = r.meta["original_url"]
            anchor_text: str = get_anchor_text(r)

            if not self._is_seed(original_url):
                try:
                    label: bool = get_label(original_url, self.db_dir)
                except IndexError:
                    continue

                judgments_tuples_to_bulk.append(
                    (
                        original_url,
                        anchor_text,
                        r.meta["score"],
                        label,
                    )
                )

        bulk_judgments(
            judgments_tuples_to_bulk,
            CLASSIFICATION_LOG_DIR,
            f"{self.index_name}.db",
        )

    def _is_seed(self, url: str) -> bool:
        seed = self._get_original_url(self.start_urls[0])
        return bool(url == seed or f"{url}/" == seed or url == f"{seed}/")

    def _get_score(self, request: ScrapyRequest):
        indices: Union[str, Sequence[str]] = self._parse_start_data_indices()

        return self.score_strategy.get_score(
            request,
            index_name=indices,
            ranking=self.ranking,
            re_ranking=self.re_ranking,
            db_dir=self.db_dir,
            embeddings_cache_dir=self.embeddings_cache,
            embeddings_model=self.embeddings_model,
            clean_anchor=self.clean_anchors,
            score_option=self.score_option,
            nlp_api_url=self.nlp_api_url,
        )

    def _requests_to_follow(self, response: HtmlResponse):
        if not isinstance(response, HtmlResponse):
            return []

        # Safeguard: Ensure selector root has the lxml '.iter' method.
        # If Scrapy parsed JSON, response.selector.root will be a 'dict' or 'list'.
        if not hasattr(response.selector.root, "iter"):
            return []

        seen = set()
        links_to_process = []

        for rule_index, rule in enumerate(self._rules):
            links = [
                lnk
                for lnk in rule.link_extractor.extract_links(response)
                if lnk not in seen
            ][:self.max_links_per_page]

            for link in rule.process_links(links):
                seen.add(link)

                request = self._build_request(rule_index, link)

                # Depth + 1
                request.meta["depth"] = response.request.meta["depth"] + 1

                # Original URL -> without testing environtment domain
                original_url: str = self._get_original_url(request.url)
                request.meta["original_url"] = original_url

                # Compute score
                score: float = self._get_score(request)
                request.meta["score"] = score

                # Priority is derived from score by the strategy itself.
                # Probability-based strategies invert (min-heap); BFS/DFS use score directly.
                request.priority = self.score_strategy.get_priority(score)
                request.meta["prediction"] = round(score)

                links_to_process.append(
                    rule.process_request(
                        request,
                        response,
                    ),
                )

        # Scheduler is given the list of Request object for the found links
        # sorted by priority and it will filter duplicates
        # Criteria: FIRST BEST OCCURRENCE
        sorted_by_prio = sorted(
            links_to_process,
            key=lambda r: r.priority,
            reverse=False,
        )

        if self.save_judgments:
            try:
                judgments_to_bulk = []

                for r in sorted_by_prio:
                    original_url: str = r.meta["original_url"]
                    if original_url not in self.judged_urls:
                        self.judged_urls.add(original_url)
                        judgments_to_bulk.append(r)

                self._bulk_judgments_into_db(judgments_to_bulk)

            except Exception:
                print(
                    f"Error trying to bulk judgments for {original_url} -> ERROR: {traceback.format_exc()}"
                )

        return sorted_by_prio

    def parse_item(self, response: ScrapyResponse):
        page_raw_text_content: str = get_page_raw_text_content(response.text)
        anchor_text: str = get_anchor_text(response)

        if self.use_embeddings:
            if not self.embeddings_model:
                raise Exception("ERROR -> No embeddings model provided!")

            anchor_text_embedding: List[float] = get_embedding_from_anchor_text(
                anchor_text=anchor_text,
                cache_dir=self.embeddings_cache,
                clean_anchor=self.clean_anchors,
                embeddings_model=self.embeddings_model,
                nlp_api_url=self.nlp_api_url,
            )

        else:
            anchor_text_embedding: List[float] = []

        original_url: str = self._get_original_url(response.request.url)
        original_url_domain: str = original_url.replace("http://", "").replace("https://", "").split("/")[0]
        label: bool = get_label(original_url, self.db_dir)

        yield FocusedCrawlerIndexItem(
            url=original_url,
            label=label,
            anchor_text=anchor_text,
            anchor_text_embedding=anchor_text_embedding,
            page_content=page_raw_text_content,
            priority=response.request.priority,
            depth=response.request.meta["depth"],
            download_order=response.request.meta["download_order"],
            domain=original_url_domain,
        )
