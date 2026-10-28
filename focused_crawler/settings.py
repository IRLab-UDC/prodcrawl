import logging
import os

# Project directories and paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
URLS_DIR = os.path.join(BASE_DIR, "datasets")

# Directory for the per-run judgment logs (only written when save_judgments is "yes").
CLASSIFICATION_LOG_DIR = os.environ.get("JUDGMENTS_LOG_DIR", os.path.join(BASE_DIR, "judgments"))

# Frontier priority resolution.
# ScoreStrategy.get_priority maps a score in [0,1] to int(round((1-score)*PRIORITY_SCALE)),
# so the integer priority the min-heap sorts on retains the ordering of the score.
# Recover the score from a stored priority as 1 - priority / PRIORITY_SCALE.
PRIORITY_SCALE = 1000

# Bot and logging
BOT_NAME = 'focused_crawler'
LOG_LEVEL = 'DEBUG'
LOG_FORMATTER = 'focused_crawler.common.logs.QuietLogFormatter'
ES_URL = os.environ.get('ES_URL', 'http://localhost:9200')
logging.getLogger('elasticsearch').setLevel(logging.ERROR)
logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)

# Spider modules
SPIDER_MODULES = ['focused_crawler.spiders']
NEWSPIDER_MODULE = 'focused_crawler.spiders'

# Crawl responsibly by identifying yourself (and your website) on the user-agent
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:96.0) Gecko/20100101 Firefox/96.0'

# Obey robots.txt rules
ROBOTSTXT_OBEY = False

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 1

# To avoid interferences from the Scrapy framework
REDIRECT_PRIORITY_ADJUST = 0
RETRY_PRIORITY_ADJUST = 0

# Configure a delay for requests for the same website (default: 0)
DOWNLOAD_DELAY = 1
CONCURRENT_REQUESTS_PER_DOMAIN = CONCURRENT_REQUESTS
CONCURRENT_REQUESTS_PER_IP = CONCURRENT_REQUESTS

DOWNLOAD_TIMEOUT = 600 #seconds

# Disable cookies (enabled by default)
COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
TELNETCONSOLE_ENABLED = False

# Maximum number of items/pages/errors before closing spider
CLOSESPIDER_ITEMCOUNT = 0
CLOSESPIDER_PAGECOUNT = 1  # Provided through settings -> this value is just to end as soon as the first page is crawled, to indicate that the settings are not being correctly loaded and applied.
CLOSESPIDER_ERRORCOUNT = 0

# Override the default request headers
DEFAULT_REQUEST_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Language': 'en-US,en;q=0.8',
    'Connection': 'keep-alive',
}

# Spider middlewares
#SPIDER_MIDDLEWARES = {
#    'focused_crawler.middlewares.FocusedCrawlerSpiderMiddleware': 543,
#}

# Downloader middlewares
DOWNLOADER_MIDDLEWARES = {
   'focused_crawler.middlewares.DownloadOrderMiddleware': 543,
}

# Extensions
#EXTENSIONS = {
#    'scrapy.extensions.telnet.TelnetConsole': None,
#}

# Item pipelines
ITEM_PIPELINES = {
    'focused_crawler.pipelines.IndexFocusedCrawlerItem': 300,
}

# AutoThrottle extension
AUTOTHROTTLE_ENABLED = False
#AUTOTHROTTLE_START_DELAY = 5
#AUTOTHROTTLE_MAX_DELAY = 60
#AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
#AUTOTHROTTLE_DEBUG = False

# HTTP caching
#HTTPCACHE_ENABLED = True
#HTTPCACHE_EXPIRATION_SECS = 0
#HTTPCACHE_DIR = 'httpcache'
#HTTPCACHE_IGNORE_HTTP_CODES = []
#HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'

# Scheduler memory queues and priority queues
DEPTH_PRIORITY = 0
SCHEDULER_DISK_QUEUE = 'focused_crawler.common.queues.StableMinHeapQueue'
SCHEDULER_MEMORY_QUEUE = 'focused_crawler.common.queues.StableMinHeapQueue'
SCHEDULER_PRIORITY_QUEUE = 'focused_crawler.common.queues.FrontierPriorityQueue'

