from scrapy.logformatter import LogFormatter

class QuietLogFormatter(LogFormatter):
    def scraped(self, item, response, spider):
        # Returning None completely suppresses the giant item log prints!
        return None
