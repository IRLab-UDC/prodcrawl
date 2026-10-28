#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import os.path
from collections import deque
from typing import Any, Deque, Dict
from multiprocessing import Pool, cpu_count

import yaml

from twisted.internet import epollreactor

epollreactor.install()

from scrapy.crawler import CrawlerProcess
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings

from focused_crawler.spiders.ProductsSpider import ProductsSpider


# Root URL of the web server that serves the ProdCrawl mirrors, one directory per site
# (http://<server>/<site>/...). It must be served on the default port: the spider
# removes "<server>/" from every URL to recover the URL of the judgments.
TEST_DOMAIN = os.environ.get("MIRROR_URL", "http://127.0.0.1/")

DEFAULTS = {
    "seeds_path": "datasets/testing.txt",
    "strategy": "BM25",
    "use_embeddings": "no",
    "test": 1,
    "anchor_window": 0,
    "ranking": 10,
    "re_ranking": 0,
    "start_data": "coldstart",
    "embeddings_model": "",
    "score_option": "ratio",
    "save_judgments": "no",
    "embeddings_cache": "",
    "db_dir": "",
    "clean_anchors": "no",
    "max_concurrent_spiders": 1,
    "nlp_api_url": "",
    "max_pages": 119,
    "max_links_per_page": 100,
}


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    return {**DEFAULTS, **config}


def normalize_seed(seed: str, test_mode: bool) -> str:
    clean_seed = seed.strip().lower()

    if not clean_seed:
        return ""

    if test_mode:
        return os.path.join(
            TEST_DOMAIN,
            clean_seed.replace("http://", "").replace("https://", ""),
        )

    if "http" not in clean_seed:
        return f"https://{clean_seed}"

    return clean_seed


def load_seed_queue(seeds_path: str, test_mode: bool) -> Deque[str]:
    with open(seeds_path, "r", encoding="utf-8") as seeds_file:
        seeds = [normalize_seed(seed, test_mode) for seed in seeds_file]

    return deque([seed for seed in seeds if seed])


def build_spider_kwargs(
    config: Dict[str, Any],
    commit_hash: str,
    seed: str,
    dataset_name: str,
) -> Dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "seed": seed,
        "score_strategy_name": str(config["strategy"]),
        "use_embeddings": str(config["use_embeddings"]),
        "anchor_text_window": str(config["anchor_window"]),
        "ranking": str(config["ranking"]),
        "re_ranking": str(config["re_ranking"]),
        "start_data": str(config["start_data"]),
        "commit_hash": commit_hash,
        "embeddings_model": str(config["embeddings_model"]),
        "score_option": str(config["score_option"]),
        "save_judgments": str(config["save_judgments"]),
        "embeddings_cache": str(config["embeddings_cache"]),
        "db_dir": str(config["db_dir"]),
        "clean_anchors": str(config["clean_anchors"]),
        "nlp_api_url": str(config["nlp_api_url"]),
        "max_links_per_page": str(config["max_links_per_page"]),
    }


def run_experiment(args: tuple) -> None:
    """
    Executed in its own separate OS Process on a dedicated CPU core.
    Creates a fresh CrawlerProcess and Twisted reactor for a single seed.
    """
    seed: str = args[0]
    config: Dict[str, Any] = args[1]
    commit_hash: str = args[2]
    dataset_name: str = args[3]

    # 1. Load fresh settings inside the isolated process
    settings = get_project_settings()
    settings.set("CLOSESPIDER_PAGECOUNT", int(config["max_pages"]))
    configure_logging(settings=settings)

    logging.info(
        "Process [PID %s] starting crawl for seed: %s",
        os.getpid(),
        seed,
    )

    # 2. Instantiate a fresh CrawlerProcess (owns its own clean Twisted reactor)
    process = CrawlerProcess(settings, install_root_handler=False)

    try:
        # Build spider kwargs and schedule
        spider_kwargs = build_spider_kwargs(
            config,
            commit_hash,
            seed,
            dataset_name,
        )
        process.crawl(ProductsSpider, **spider_kwargs)

        # 3. Block until THIS individual seed finishes crawling
        process.start()
        logging.info("Process [PID %s] finished seed: %s", os.getpid(), seed)
    except Exception:
        logging.exception("Process [PID %s] failed seed: %s", os.getpid(), seed)


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file with experiment parameters",
    )
    parser.add_argument(
        "--commit-hash",
        type=str,
        default="unknown",
        help="Git commit hash injected automatically by the launch script",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    num_cpus = cpu_count()

    test_mode = bool(int(config["test"]))
    seed_queue = load_seed_queue(str(config["seeds_path"]), test_mode)
    dataset_name = config["seeds_path"].split("/")[-1].split(".")[0]

    configured_workers = int(config.get("max_concurrent_spiders", num_cpus))
    worker_count = min(
        configured_workers,
        num_cpus,
        len(seed_queue),
    )

    tasks = [
        (
            seed,
            config,
            args.commit_hash,
            dataset_name,
        )
        for seed in seed_queue
    ]

    with Pool(processes=worker_count, maxtasksperchild=1) as pool:
        # maxtasksperchild=1 forces the OS process to terminate and recreate
        # after finishing 1 seed, guaranteeing 100% clean memory garbage collection!
        # This guarantees that every seed runs in a completely fresh OS process,
        # with a fresh Twisted reactor and a fresh CrawlerProcess, avoiding memory leaks.
        pool.starmap(run_experiment, [(t,) for t in tasks])


if __name__ == "__main__":
    run()
