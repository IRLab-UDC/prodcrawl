import json
import os.path
from typing import List

import requests
from requests import Response

from focused_crawler.nlp.settings import HEADERS


def get_embedding_from_text(
    text_list: List[str],
    embeddings_model: str,
    nlp_api_url: str,
):
    get_embedding_endpoint: str = os.path.join(nlp_api_url, "get-embedding")

    response: Response = requests.post(
        url=get_embedding_endpoint,
        json={"text_list": text_list, "model": embeddings_model},
        headers=HEADERS,
        timeout=30,
    )

    return json.loads(response.content)['embeddings']
