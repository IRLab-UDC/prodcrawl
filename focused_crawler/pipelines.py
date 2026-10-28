# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

import pytz
import datetime

from focused_crawler.index.embeddings_indexer import IndexManager
from focused_crawler.items import FocusedCrawlerIndexItem

indexer: IndexManager = IndexManager()


class IndexFocusedCrawlerItem(object):

    def process_item(self, item, spider):
        if isinstance(item, FocusedCrawlerIndexItem):
            if not indexer.page_is_indexed(
                url=item["url"],
                index_name=spider.index_name,
            ):
                document = {
                    "url": item["url"],
                    "label": item["label"],
                    "anchor_text": item["anchor_text"],
                    "insert_date": pytz.utc.localize(datetime.datetime.utcnow()),
                    "page_content": item["page_content"],
                    "priority": item["priority"],
                    "depth": item["depth"],
                    "download_order": item["download_order"],
                    "domain": item["domain"],
                }

                if item["anchor_text_embedding"]:
                    document["anchor_text_embedding"] = item["anchor_text_embedding"]

                indexer.index_document(
                    document,
                    index_name=spider.index_name,
                )

        return item
