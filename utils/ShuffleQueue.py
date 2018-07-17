import collections
import random
import sys

class ShuffleQueue(collections.UserList):
    def __init__(self,l,rand=random):
        self.data = list(l)
        self.counter = len(self.data)
        self.rand=rand
        self._epoch = 0
        self.next_epoch()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data) - self.counter

    def __next__(self):
        if len(self) <= 0:
            next_epoch()
        
        result = self.data[self.counter]
        self.counter += 1
        return result

    def next_epoch(self):
        self.counter = 0
        self.rand.shuffle(self.data)
        self._epoch += 1

    @property
    def epoch(self):
        return self._epoch
