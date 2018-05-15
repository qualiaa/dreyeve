import collections
import random
import sys

class ShuffleQueue(collections.UserList):
    def __init__(self,l,rand=random):
        self.container = list(l)
        self.counter = len(self.container)
        self.rand=rand
        self._epoch = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.container) - self.counter

    def __next__(self):
        if len(self) <= 0:
            next_epoch()
        
        result = self.container[self.counter]
        self.counter += 1
        return result

    def next_epoch():
        self.counter = 0
        self.rand.shuffle(self.container)
        self._epoch += 1

    @property
    def epoch(self):
        return self._epoch
