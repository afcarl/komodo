from heapq import heappush, heappush, heappop, heapify, heapreplace
from collections import deque
import numpy as np
import time

class Queue():
    """Basic FIFO queue."""

    def __init__(self, size, init_values=[]):
        assert len(init_values) <= size, "Initial values exceed queue length."
        self.size = size
        self.queue = deque(init_values)

    def __len__(self):
        return len(self.queue)

    def push(self, value):
        """Add a memory to queue."""
        if len(self.queue) == self.size:
            self.queue.popleft()
            self.queue.append(value)
        else:
            self.queue.append(value)

    def fill(self):
        """Fill queue by copying the first element and pushing to left."""
        while len(self) < self.size:
            self.queue.appendleft(self.queue[0])
