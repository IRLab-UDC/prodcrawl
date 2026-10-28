import datetime
import json
import os
import csv

import re
from typing import Dict, Any, List, Optional, Tuple, Union
from urllib.parse import urlparse

from tldextract.tldextract import ExtractResult, extract

from focused_crawler.common.db_utils import get_embedding_from_cache, save_embedding_into_cache
from focused_crawler.common.exceptions import EmbeddingCalcError
from focused_crawler.index.embeddings_indexer import IndexManager
from focused_crawler.nlp.extraction import get_embedding_from_text

index_manager: IndexManager = IndexManager()


def save_into_json(
    json_dict: Dict[str, Any],
    dir_name: str,
    file_name: str,
) -> None:
    
    os.makedirs(dir_name, exist_ok=True)

    with open(os.path.join(dir_name, file_name), "a", encoding="utf-8") as f:
        json.dump(json_dict, f)


def save_into_jsonl(
    json_dicts: List[Dict[str, Any]],
    dir_name: str,
    file_name: str,
) -> None:
    
    os.makedirs(dir_name, exist_ok=True)

    with open(os.path.join(dir_name, file_name), "a", encoding="utf-8") as f:
        for json_entry in json_dicts:
            json.dump(json_entry, f)
            f.write("\n")


def save_into_html(
    bytes_source: bytes,
    dir_name: str,
    file_name: str,
) -> None:
    
    os.makedirs(dir_name, exist_ok=True)

    with open(os.path.join(dir_name, file_name), "ab") as f:
        f.write(bytes_source)


def bulk_lines_into_tsv(
    lines: List,
    dir_name: str,
    file_name: str,
) -> None:
    
    os.makedirs(dir_name, exist_ok=True)

    with open(os.path.join(dir_name, file_name), "a") as t:
        writer = csv.writer(t, delimiter="\t", lineterminator="\n")

        for line in lines:
            writer.writerow(line)


def save_into_tsv(
    lines: List[str],
    dir_name: str,
    file_name: str,
) -> None:

    os.makedirs(dir_name, exist_ok=True)

    with open(os.path.join(dir_name, file_name), "a") as t:
        writer = csv.writer(t, delimiter="\t", lineterminator="\n")

        writer.writerow(lines)


def read_from_tsv(
    dir_name: str,
    file_name: str,
) -> List[List[str]]:
    
    try:
        with open(os.path.join(dir_name, file_name)) as t:
            reader = csv.reader(t, delimiter="\t", lineterminator="\n")

            return [row for row in reader]

    except:
        return []


def get_organization_slug_from_url(url: str) -> str:
    parts = urlparse(url)

    return parts.path.replace("/", "")


def extract_domain_from_url(url: str) -> str:
    # ExtractResult is a NamedTuple
    domain_extracted_info: ExtractResult = extract(url)
    domain = domain_extracted_info.domain

    if domain_extracted_info.suffix:
        domain = f"{domain}.{domain_extracted_info.suffix}"
    if domain_extracted_info.subdomain:
        domain = f"{domain_extracted_info.subdomain}.{domain}"

    return domain


def remove_extra_whitespaces(input_string: str) -> str:
    """Normalize the input string by removing extra whitespaces."""
    normalized = re.sub(r"\s+", " ", input_string)

    # Strip leading and trailing spaces
    normalized = normalized.strip()
    return normalized


def normalize_anchor_text(input_string: str) -> str:
    """Normalize the input string by removing extra whitespaces and converting to lowercase."""
    normalized = remove_extra_whitespaces(input_string)
    return normalized.lower()


def calculate_embedding_from_anchor_text(
    anchor_text: str,
    embeddings_model: str,
    nlp_api_url: str,
) -> List[float]:

    anchor_text_embedding = list(
        get_embedding_from_text(
            [anchor_text],
            embeddings_model,
            nlp_api_url=nlp_api_url,
        ).values()
    )[0]

    return anchor_text_embedding


def get_embedding_from_anchor_text(
    anchor_text: str,
    cache_dir: Optional[str] = None,
    clean_anchor: bool = False,
    embeddings_model: str = "all-mpnet-base-v2",
    nlp_api_url: str = "",
) -> List[float]:

    anchor_text_embedding: List[float] = []

    if clean_anchor:
        anchor_text = normalize_anchor_text(anchor_text)

    if cache_dir:
        anchor_text_embedding = get_embedding_from_cache(anchor_text, cache_dir)

    if not anchor_text_embedding:
        anchor_text_embedding = calculate_embedding_from_anchor_text(
            anchor_text,
            embeddings_model,
            nlp_api_url=nlp_api_url,
        )

        if not anchor_text_embedding:
            raise EmbeddingCalcError

        if cache_dir:
            save_embedding_into_cache(
                anchor_text,
                anchor_text_embedding,
                cache_dir,
            )

            print(f"Saving embedding for anchor -> {anchor_text}")

    return anchor_text_embedding


def get_anchor_text(
    request: Any,
    meta_key: str = "link_text",
) -> str:
    anchor_text: str = str(request.meta.get(meta_key, "")).strip()
    return remove_extra_whitespaces(anchor_text)
