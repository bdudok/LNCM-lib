import os
import re

class AssetFinder:
    def __init__(self, path=None):
        self.suffix_list = ''
        # add separate locator for rois or handle outside
        self.contain_list = ['roi']
        self.sufdir = {}
        self.prefixes = []
        self.flist = []
        self.path = path
        if path is not None:
            self.update()

    def update(self, path=None):
        if path is not None:
            self.path = path
        self.prefixes = []
        self.flist = os.listdir(self.path)
        for f in self.flist:
            if os.path.isdir(f):
                if os.path.exists(os.path.join(f, f + '_SessionInfo.json')):
                    self.prefixes.append(f)
        self.prefixes.sort(reverse=True)

    def get_prefixes(self):
        return self.prefixes

def get_processed_tags(procpath, prefix):
    '''
    :param procpath: Parent Processed folder item['Processed.Path']
    :param prefix:
    :return: list of tuples (roi tag, channel) that exist
    '''
    spath = os.path.join(procpath, prefix + '/')
    flist = os.listdir(spath)
    #pattern is f'{prefix}_trace_{tag}-ch{ch}'
    dirlist = [d for d in flist if os.path.isdir(spath + d)]
    pattern = prefix + r'_trace_(\w+)-ch(.)'
    hits = []
    for d in dirlist:
        match = re.match(pattern, d)
        if match is not None:
            hits.append((match.group(1), match.group(2)))
    return hits



