from __future__ import print_function, absolute_import

import functools
import heapq
import math
from io import StringIO

__all__ = ["Heap"]


def _with_docstring(source):
    @functools.wraps(source)
    def _with_docstring_wrap(target):
        target.__doc__ = source.__doc__
        return target

    return _with_docstring_wrap


class Heap(object):
    """Transform list, in-place, into a heap in O(len(heap)) time."""

    def __init__(self, heap):
        self.heap = heap
        heapq.heapify(self.heap)

    def __len__(self):
        return len(self.heap)

    def _validate_push(self, item):
        try:
            if self.heap:
                item < self.heap[0] < item
        except TypeError:
            raise ValueError(
                "can't order new item type ({}) with existing type ({})".format(
                    type(item).__name__, type(self.heap[0]).__name__
                )
            )

    @_with_docstring(heapq.heappush)
    def push(self, item):
        self._validate_push(item)
        heapq.heappush(self.heap, item)

    @_with_docstring(heapq.heappop)
    def pop(self):
        return heapq.heappop(self.heap)

    @_with_docstring(heapq.heappushpop)
    def pushpop(self, item):
        self._validate_push(item)
        return heapq.heappushpop(self.heap, item)

    @_with_docstring(heapq.heapreplace)
    def replace(self, item):
        self._validate_push(item)
        return heapq.heapreplace(self.heap, item)

    def show_tree(self, total_width=36, fill=' '):
        """Pretty-print a tree."""
        output = StringIO()
        last_row = -1
        for i, n in enumerate(self.heap):
            if i:
                row = int(math.floor(math.log(i+1, 2)))
            else:
                row = 0
            if row != last_row:
                output.write('\n')
            columns = 2**row
            col_width = int(math.floor((total_width * 1.0) / columns))
            output.write(str(n).center(col_width, fill))
            last_row = row
        print(output.getvalue())
        print('-' * total_width)
        print()
        return
    
