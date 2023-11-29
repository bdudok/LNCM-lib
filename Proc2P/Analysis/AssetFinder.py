import os


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
