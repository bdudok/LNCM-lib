import numpy
import os
from Proc2P.Bruker.ConfigVars import CF
'''Class for loading synchronous datasets for 2P sessions'''

class Sync:
    def __init__(self, procpath, prefix):
        self.procpath = procpath
        self.prefix = prefix
        self.cache = {}

    def fn(self, suffix):
        return os.path.join(self.procpath, self.prefix + '/', self.prefix+suffix)

    def load(self, tag, suppress=False):
        '''check if tag cached, if not load it from file
        file suffixes stored in config.
        '''
        assert hasattr(CF, tag)
        if tag not in self.cache:
            fn = self.fn(getattr(CF, tag))
            if os.path.exists(fn):
                self.cache[tag] = numpy.load(fn)
            else:
                if not suppress:
                    print('Sync: File not found for', tag)
                self.cache[tag] = None
        return self.cache[tag]




