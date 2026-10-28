from enum import Enum
import logging
import os
from typing import List

import uvicorn
from fastapi import FastAPI
from langdetect import DetectorFactory, detect
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# Load models in memory
EMBEDDINGS_BASE_MODEL = SentenceTransformer('all-mpnet-base-v2')
# Path or Hugging Face id of the fine-tuned anchor-text encoder released with ProdCrawl.
FINE_TUNED_MODEL_PATH = os.environ.get("FINE_TUNED_MODEL", "irlab-udc/anchor-product-page-mpnet-base")
EMBEDDINGS_FINE_TUNED_MODEL = SentenceTransformer(FINE_TUNED_MODEL_PATH)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


EMBEDDINGS_NAMES_TO_MODELS = {
    "all-mpnet-base-v2": EMBEDDINGS_BASE_MODEL,
    "anchor-product-page-mpnet-base": EMBEDDINGS_FINE_TUNED_MODEL,
}


class TextItem(BaseModel):
    text: str

class TextListItem(BaseModel):
    text_list: List[str]
    model: str

class TextTupleItem(BaseModel):
    text: str
    text_list: List[str]


DetectorFactory.seed = 0

app = FastAPI()


def get_language(text: str) -> str:
    return detect(text)


def _get_embeddings_model_from_name(name: str):
    return EMBEDDINGS_NAMES_TO_MODELS[name]


def _get_embedding_from_text(text: str, model_name: str):
    logger.info(f"-> Calculating embedding for: {text}")

    embeddings_model = _get_embeddings_model_from_name(model_name)
    embedding = embeddings_model.encode(text)

    logger.info(f"-> Resulting embedding: {embedding.tolist()}")

    return embedding.tolist()


@app.post("/get-embedding")
def get_embedding(item: TextListItem):
    embeddings = {}

    for text in item.text_list:
        embeddings[text] = _get_embedding_from_text(
            text,
            item.model,
        )

    return {"embeddings": embeddings}


class HealthStatus(BaseModel):
    """Response model to validate and return when performing a health check."""

    status: str = "OK"


@app.get("/status", response_model=HealthStatus)
def get_status() -> HealthStatus:
    return HealthStatus(status="OK")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8030,
    )
