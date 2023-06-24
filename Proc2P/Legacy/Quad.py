import numpy
import pandas
from .sbxreader import *
from .Batch_Utils import gapless

class SplitQuad(object):
    '''Used for quadrature.mat files with my modified arduino quad encoder that updates position with ttl syncs
    very old sessions may require older versions of this, see Quad class commented out at the end of this file'''
    def __init__(self, prefix):
        '''
        Load data if prefix is specified. Or populate with zero if length is specified.
        '''
        if type(prefix) == str:
            self.load_rotary(prefix)
        elif type(prefix) == int:
            self.fill_rotary(prefix)

    def load_rotary(self, prefix):
        mod1 = 10000
        raw = spio.loadmat(prefix + '_quadrature.mat', struct_as_record=False, squeeze_me=True)['quad_data']
        # read array 0:ttl, 1:pos
        self.data = numpy.empty((2, len(raw)), dtype=numpy.int)
        self.data[:, :] = divmod(raw, mod1)
        # fix resets
        for i in range(1, len(raw)):
            if (self.data[1, i - 1] - self.data[1, i]) > (mod1 * 0.5):
                self.data[1, i:] += mod1
        # fix issues before first tick
        self.data[:, :5] = 0
        if self.data[1, 6] == mod1 - 1:
            self.data[1, :5] = mod1 - 1
            self.data -= mod1 - 1
        # calc speed
        pos = numpy.array(self.data[1], dtype='float')
        self.speed = numpy.append(numpy.array([0.]), numpy.diff(pos))
        #fix outlier speed points
        self.speed[numpy.where(abs((self.speed / mod1) - 1) < 0.5)] = 0
        #check if speed is positive
        posspd = self.speed > 0
        if self.speed[posspd].sum() < abs(self.speed[~posspd].sum()):
            print('Flipping direction')
            self.speed *= -1
            # pos = mod1 - pos
        pos /= pos.max()
        self.qepos = pos
        self.pos = pos
        self.compute_distances()

    def fill_rotary(self, prefix):
        #'just filling up stuff so code works for session if movement file is missing'
        n = prefix
        self.data = numpy.zeros((2, n), dtype='int')
        pos = self.data[1].astype('float')
        self.speed = pos
        self.qepos = pos
        self.pos = pos
        self.smspd = pos
        self.events = pos
        self.anyevent = pos.astype('bool')
        self.durations = []
        self.durations_times = []
        self.movement = self.anyevent
        self.gapless = self.anyevent
        self.relpos = pos
        self.laps = pos

    def startstop(self, duration=50, gap=150, ret_loc='actual', span=None):
        mov = self.gapless
        if gap is None:
            gap = 150
        if span is None:
            span = gap, len(self.pos) - gap
        # collect stops
        stops, starts = [], []
        t = span[0] + duration
        while t < span[1] - gap:
            if not numpy.any(mov[t:t + gap]) and numpy.all(mov[t - duration:t]):
                t0 = t
                if ret_loc == 'peak':
                    while self.smspd[t0] < self.smspd[t0 - 1]:
                        t0 -= 1
                    stops.append(t0)
                    while mov[t0]:
                        t0 -= 1
                    starts.append(t0)
                elif ret_loc == 'stopped':
                    stops.append(t)
                    while mov[t0 - 1]:
                        t0 -= 1
                    starts.append(t0)
                elif ret_loc == 'actual':
                    # go back while raw speed is zero
                    while self.speed[t - 1] == 0 and t > 100:
                        t -= 1
                    if t > 100:
                        stops.append(t)
                        while mov[t0 - 1] and t0 > 100:
                            t0 -= 1
                        while self.speed[t0] == 0:
                            t0 += 1
                        starts.append(t0)
                t += gap
            t += 1
        return starts, stops

    def compute_distances(self):
        self.smspd = numpy.asarray(pandas.DataFrame(self.speed).ewm(span=10).mean())[:, 0]
        self.events = numpy.zeros(len(self.speed))
        self.anyevent = numpy.zeros(len(self.speed), dtype='bool')
        self.durations = []
        self.durations_times = []
        eid, md = 0, 0
        for t in range(len(self.speed)):
            if self.speed[t] > 2:
                if md == 0:
                    md = 1
                    eid += 1
                    self.durations.append(1)
                    self.durations_times.append(t)
                    self.events[t] = eid
                    self.anyevent[t] = 1
                else:
                    self.events[t] = eid
                    self.durations[-1] += 1
                    self.anyevent[t] = 1
            elif self.speed[t] > 0.1 and md == 1:
                self.events[t] = eid
                self.durations[-1] += 1
                self.anyevent[t] = 1
            else:
                md = 0
        # expand movement events with one sec
        self.movement = numpy.zeros(self.anyevent.shape, dtype='bool')
        l = len(self.anyevent)
        for i in range(l):
            start, stop = max(0, i - 7), min(i + 8, l)
            if True in self.anyevent[start:stop]:
                self.movement[i] = 1
        # gapless_move
        self.gapless = gapless(self.anyevent, 15)


#if need to use it, copy compute distances function from SplitQuad
# class Quad(object):
#     def __init__(self, prefix, source='BehaviorMate'):
#         '''The only usage of this is when no quad.mat saved - what is the use of that in lack of sync?'''
#         # load text file (removing last item as for some reason quad is 1 item longer than trace)
#         if source == 'Rotary':
#             self.load_rotary(prefix)
#         elif source == 'BehaviorMate':
#             self.bm = Tdml(prefix + '.tdml')
#             self.data = self.bm.data
#         self.pos = self.data[1]
#         # calculate speed
#         self.speed = numpy.zeros(len(self.pos), dtype=numpy.int)
#         for i in range(1, len(self.pos)):
#             p2, p1 = self.data[1, i], self.data[1, i - 1]
#             self.speed[i] = min(abs(p2), abs(p1 - p2))
#         # detect lap ends
#         self.laptimes, self.lapends = [], []
#         for i in range(len(self.pos) - 1):
#             if self.data[0, i] < self.data[0, i + 1]:
#                 self.laptimes.append(i)
#                 self.lapends.append(self.data[1, i])
#         self.lapends = numpy.array(self.lapends)
#         self.laptimes = numpy.array(self.laptimes)
#         # calculate length of lap
#         nonoutliers = mad_based_outlier(self.lapends[1:]) + 1
#         self.laplength = numpy.mean(self.lapends[nonoutliers])
#         # fix position of first partial lap
#         if source == 'Rotary':
#             self.pos[:self.laptimes[0]] += int(self.laplength - self.pos[self.laptimes[0]])
#         # calc relative position
#         self.relpos = numpy.empty(self.pos.shape)
#         self.relpos = self.pos / self.laplength
#         # force lap reset values
#         if len(self.laptimes)>0:
#             self.relpos[self.laptimes + 1] = 0
#             self.relpos[self.laptimes] = 1
#         compute_distances(self)

    # def load_rotary(self, prefix):
    #     f = prefix + '_quadrature.txt'
    #     mod1, mod2 = 10, 10000
    #     with open(f, 'r') as f:
    #         raw = numpy.fromstring(f.readline(), sep=',')[:-1]
    #     # read multiplexed position and reward data
    #     # array 0:lap, 1:pos, 2:rewardcount
    #     self.data = numpy.empty((4, len(raw)), dtype=numpy.int)
    #     self.data[1:3, :] = divmod(raw, mod1)
    #     self.data[:2, :] = divmod(self.data[1], mod2)
    #     # fix that lap value is reduced with negative running direction
    #     negloc = numpy.where(self.data[1] > mod2 / 2)
    #     self.data[1, negloc] -= mod2
    #     self.data[0, negloc] += 1
    #     # invert if dir is neg
    #     if self.data[1].mean() < 0:
    #         self.data[1] *= -1


# def mad_based_outlier(points, thresh=3.5):
#     median = numpy.median(points)
#     diff = (points - median) ** 2
#     diff = numpy.sqrt(diff)
#     med_abs_deviation = numpy.median(diff)
#     modified_z_score = 0.6745 * diff / med_abs_deviation
#     return numpy.where(modified_z_score < thresh)[0]

# class Tdml(object):
#     '''Removing this, BehaviorSession should be used for treadmill data
#     Tghe key difference between this and BehaviorSession that it updates positions on TTL triggers
#      instead of behaviormate updates'''
#     def __init__(self, f):
#         with open(f, 'r') as df:
#             lines = df.readlines()
#         ft, pos, rc, laps = [], [], [], []
#         self.rewards, self.licks = [], []
#         self.comments = []
#         self.other_events = []
#         p, r, lap = -42, 0, 0
#         for l in lines[3:-1]:
#             # try:
#                 items = l.split(',')
#                 if '"time":' in l:
#                     for item in items:
#                         if '"time":' in item:
#                             tt = item
#                             t = float(tt[tt.find(':') + 2:-2])
#                             break
#                 # detect the start of each frame, save with current position
#                 if '{"pin": 7,"action": "start"}' in l:
#                     ft.append(t)
#                     pos.append(p)
#                     rc.append(r)
#                     laps.append(lap)
#                 # update position
#                 elif '{"position": {"dy":' in l:
#                     if '"y":' in l:
#                         p = float(items[2][5:])
#                 # save reward times, lick times
#                 elif '"valve": {"pin": 5,"action": "open"}' in l:
#                     self.rewards.append(t)
#                     r += 1
#                 elif '{"lick": {"pin": 2,"action": "start"' in l:
#                     self.licks.append(t)
#                 # update lap
#                 elif '{"lap":' in l:
#                     lap += 1
#                 elif '{"comments"' in l:
#                     self.comments.append(l[12:])
#                 elif '"pin":' in l:
#                     s1 = l[l.find('"pin":'):]
#                     pin = int(s1[s1.find(': ') + 1:s1.find(',')])
#                     if pin not in (5, 7, 2):
#                         if 'created' not in s1:
#                             self.other_events.append((pin, t, 'open' in s1))
#             # except:
#             #     pass
#                 # print(l)
#         # format data as in quad class
#         self.data = numpy.array([laps, pos, rc])
#         self.times = numpy.array(ft)
#         # fix positions before first update
#         i = numpy.where(self.data[1] == -42)[0]
#         if len(i) > 0:
#             i = i[-1]
#             self.data[1, :i + 1] = self.data[1, i + 1]
#         # detect gap in sync and truncate data