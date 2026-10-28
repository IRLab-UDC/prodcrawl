import itertools
from queue import PriorityQueue

from scrapy.utils.misc import create_instance


class FrontierPriorityQueue:
    """
    Wrapper for StableMinHeapQueue in order to be used as min-heap by Scrapy framework.

    It only uses one priority queue (min-heap). Each request is sorted by its priority attribute value.
    The element with the smallest priority value is always at the top.
    """

    @classmethod
    def from_crawler(cls, crawler, downstream_queue_cls, key, startprios=()):
        return cls(crawler, downstream_queue_cls, key, startprios)

    def __init__(self, crawler, downstream_queue_cls, key, startprios=()):
        self.crawler = crawler
        self.downstream_queue_cls = downstream_queue_cls
        self.key = key
        self.queues = {}
        self.curprio = None
        self.init_prios(startprios)

    def init_prios(self, startprios):
        if not startprios:
            return

        for priority in startprios:
            self.queues[priority] = self.qfactory(priority)

        self.curprio = min(startprios)

    def qfactory(self, key):
        return create_instance(
            self.downstream_queue_cls,
            None,
            self.crawler,
            self.key + '/' + str(key),
        )

    def priority(self, request):
        return 1 # The only queue that will be used.

    def push(self, request):
        priority = self.priority(request)
        if priority not in self.queues:
            self.queues[priority] = self.qfactory(priority)
        q = self.queues[priority]
        q.push(request)  # this may fail (eg. serialization error)
        if self.curprio is None or priority < self.curprio:
            self.curprio = priority

    def pop(self):
        if self.curprio is None:
            return
        q = self.queues[self.curprio]
        m = q.pop()
        if not q:
            del self.queues[self.curprio]
            q.close()
            prios = [p for p, q in self.queues.items() if q]
            self.curprio = min(prios) if prios else None
        return m

    def peek(self):
        """Returns the next object to be returned by :meth:`pop`,
        but without removing it from the queue.

        Raises :exc:`NotImplementedError` if the underlying queue class does
        not implement a ``peek`` method, which is optional for queues.
        """
        if self.curprio is None:
            return None
        queue = self.queues[self.curprio]
        return queue.peek()

    def close(self):
        active = []
        for p, q in self.queues.items():
            active.append(p)
            q.close()
        return active

    def __len__(self):
        return sum(len(x) for x in self.queues.values()) if self.queues else 0


class StableMinHeapQueue:
    """Wrapper for Python's built-in PriorityQueue."""

    def __init__(self, _) -> None:
        self.q = PriorityQueue()
        # Monotonic sequence number as the tie-break. A wall-clock timestamp
        # collides when two requests are pushed within the same microsecond,
        # and the heap then falls through to comparing Request objects, which
        # are not orderable; it also makes the order of tied requests depend on
        # timing rather than on discovery order.
        self._counter = itertools.count()

    def push(self, obj) -> None:
        self.q._put((obj.priority, next(self._counter), obj))

    def pop(self):
        if self.q:
            _, _, request = self.q._get()
            return request
        else:
            return None

    def peek(self):
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __len__(self):
        return self.q._qsize()
