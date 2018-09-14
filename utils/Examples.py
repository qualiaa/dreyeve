#!/usr/bin/env python3

import collections
import random
import sys

from abc import ABC, abstractmethod
from itertools import islice
from sys import maxsize as MAX_INT
from warnings import warn

import numpy as np
from keras.utils import Sequence

from .ShuffleQueue import ShuffleQueue


class KerasSequenceWrapper(Sequence):
    def __init__(self,cls,batch_size,*args,**kargs):
        if not issubclass(cls,Examples):
            raise TypeError("Wrapped object must be subclass of Examples")
        self.batch_size = batch_size
        self.examples = cls(*args,**kargs)

    def __getitem__(self,batch_index):
        return self.examples.get_batch(self.batch_size,batch_index)

    def __len__(self):
        return self.examples.n_batches(self.batch_size)

    def on_epoch_end(self):
        self.examples.example_queue.next_epoch()


class Examples(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get_example():
        pass

    def __init__(self, seed=None):
        if seed is None:
            seed = random.randrange(MAX_INT)
        self.seed = seed
        self.example_queue = ShuffleQueue(range(len(self)),self._rand)

        self.exception_handlers = getattr(self,"exception_handlers",dict())


    def __getitem__(self,n):
        return self.get_example(n)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_example()

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self,seed):
        self._seed = seed
        self._rand = random.Random(self._seed)

    @property
    def epoch(self):
        return self.example_queue.epoch

    def get_batch(self, batch_size, batch_index):
        n_batches = self.n_batches(batch_size)
        n_examples = len(self)
        if batch_index >= n_batches:
            raise IndexError("Batch",batch_index,"exceeds number of batches",n_batches)
        batch = []
        example_index = batch_size * batch_index
        while len(batch) < batch_size:
            if example_index >= n_examples: 
                raise IndexError("Batch",batch_index,"of",n_batches,"exceeded number of examples")
            try:
                real_index=(self.example_queue[example_index])
                batch.append(self.get_example(real_index))
            except tuple(self.exception_handlers.keys()) as e:
                self.exception_handlers[type(e)](e,real_index)
            except Exception as e:
                print("Exception raised for example {:d}".format(real_index))
                e.args += (real_index,)
                raise e
            example_index += 1
    
        def stack(x):
            if isinstance(x[0],(list,tuple)):
                x = list(map(stack,zip(*x)))
            elif isinstance(x[0],np.ndarray):
                x = np.stack(x)
            return x

        batch = tuple(stack(batch))

        return batch

    def next_example(self):
        example = None
        while example is None:
            example_id = next(self.example_queue)
            try:
                example = self.get_example(example_id)
            except tuple(self.exception_handlers.keys()) as e:
                self.exception_handlers[type(e)](e,example_id)
            except Exception as e:
                print("Exception raised for example {:d}".format(example_id))
                e.args += (example_id,)
                raise e

        return example

    def n_batches(self,batch_size):
        return len(self)//batch_size

    def next_batch(self,batch_size):
        batch=[]

        if len(self.example_queue.data) < batch_size:
            raise ValueError("Batch size",batch_size,"exceeds number of examples")
        if len(self.example_queue) < batch_size:
            self.example_queue.next_epoch()
            return []

        batch.extend(islice(iter(self),batch_size))
        batch = map(np.stack,zip(*batch))

        return examples

class Batch:
    def __init__(self, examples, batch_size):
        self.examples = examples
        self.batch_size = batch_size

    def __len__(self):
        return len(self.examples) // self.batch_size

    def __getitem__(self, i):
        return self.examples[i]

    def __iter__(self):
        batch=[]

        if len(self.examples.example_queue.data) < self.batch_size:
            raise ValueError("Batch size",self.batch_size,"exceeds number of examples")

        for example in self.examples:
            batch.extend(islice(iter(examples),batch_size))
            batch = map(np.stack,zip(*batch))
            yield batch

class Shuffle:
    def __init__(self, examples, rand=random):
        self.examples = examples
        self.rand = random
        self.order = list(range(len(examples)))
        self.rand.shuffle(self.order)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self,i):
        return self.examples[self.order[i]]

    def __iter__(self):
        for i in self.order:
            yield self.examples[i]
        self.rand.shuffle(self.order)


# could alternatively do chain(repeat(examples, num_epochs))
class Epoch:
    def __init__(self, examples, epochs):
        self.examples = examples
        self.epochs = epochs

    def __len__(self):
        return len(self.examples) * self.epochs

    def __getitem__(self, i):
        return self.examples[i]

    def __iter__(self):
        for _ in range(epochs):
            for example in self.examples:
                yield example
