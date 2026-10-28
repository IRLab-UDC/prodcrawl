# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html
from typing import List

import scrapy


class FocusedCrawlerIndexItem(scrapy.Item):
    url: str = scrapy.Field()
    label: bool = scrapy.Field()
    anchor_text: str = scrapy.Field()
    anchor_text_embedding: List[float] = scrapy.Field()
    page_content: str = scrapy.Field()
    priority: int = scrapy.Field()
    depth: int = scrapy.Field()
    download_order: int = scrapy.Field()
    domain: str = scrapy.Field()
