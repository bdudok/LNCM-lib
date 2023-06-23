class BehaviorSession(object):
    def __init__(self, f, silent=False, show=False):
        with open(f, 'r') as df:
            lines = df.readlines()
        pt, pos = [], []
        self.bm_config = lines[1]
        self.rawpos = []
        self.other_events = []
        scanning, reward, lick, rzone, entries = [], [], [], [], []
        lap, laps = 0, []
        zone = 0
        comment = ''
        for l in lines[3:]:
            items = l.split(',')
            if '"time":' in l:
                for item in items:
                    if '"time":' in item:
                        tt = item
                        t = float(tt[tt.find(':') + 2:-2])
                        break
                if '{"pin": 7,"action": "start"}' in l:
                    scanning.append(t)
                elif '{"position": {"dy":' in l:
                    pos.append(float(items[2][5:]))
                    rzone.append(zone)
                    pt.append(t)
                    laps.append(lap)
                elif '"valve": {"pin": 5,"action": "open"}' in l:
                    reward.append(t)
                elif '{"lick": {"pin":' in l and '"action": "start"' in l:
                    lick.append(t)
                elif '{"context": {"action": "start","id": "reward"}' in l:
                    entries.append(t)
                    zone = 1
                elif '{"context": {"action": "stop","id": "reward"}' in l:
                    zone = 0
                elif '{"lap":' in l:
                    lap += 1
                elif '"pin":' in l:
                    s1 = l[l.find('"pin":'):]
                    pin = int(s1[s1.find(': ') + 1:s1.find(',')])
                    if pin not in (5, 7, 2):
                        if 'created' not in s1:
                            self.other_events.append((pin, t, 'open' in s1))
            elif '{"comments"' in l:
                comment = l[13:-2]
                if 'no tag' not in comment:
                    print(comment)

        # find scan range, draw pos, draw reward zones, correct licks
        l = (int(t) + 1)
        scr = numpy.zeros(l, dtype='bool')
        for t in scanning:
            scr[int(t)] = 1
        fp = numpy.empty(l)
        secs = range(l)
        ta = numpy.array(pt)
        for t in secs:
            if t < pt[0]:
                fp[t] = pos[0]
            elif t < pt[-1]:
                fp[t] = pos[numpy.where(ta > t)[0][0]]
            else:
                fp[t] = pos[-1]
        rz = numpy.zeros(l, dtype='bool')
        for t in secs[int(pt[0]) + 1:]:
            if rzone[numpy.where(ta < t)[0][-1]]:
                rz[t] = 1
        cl, icl = [], []
        for l in lick:
            if rz[int(l)]:
                cl.append(l)
            else:
                icl.append(l)
        self.data = numpy.array([pt, pos, rzone])
        self.licks = lick
        self.choices = [cl, icl]
        self.frametimes = scanning
        self.rewards = reward
        self.laps = laps
        self.entries = entries

        if not silent:
            self.fig, (axp, axz, axr, axl, axs) = plt.subplots(5, 1,
                                                               gridspec_kw={'height_ratios': [1, 1, 0.5, 0.5, 0.5]},
                                                               sharex=True)
            axp.scatter(secs, fp, s=2)
            # axp.plot(pt,pos, color='blue')
            axp.fill_between(pt, 0, pos, color='#ddebf7')
            axs.plot(scr, color='black')
            axs.fill_between(secs, 0, scr, color='grey')
            axs.set_ylim((0.25, 0.75))
            axz.plot(rz, color='green')
            axz.fill_between(secs, 0, rz, color='#e2efda')
            axz.set_ylim((0.49, 0.51))
            axr.scatter(reward, [1] * len(reward), marker="|", s=50)
            axl.scatter(icl, [1] * len(icl), marker="|", color='red', s=50)
            axl.scatter(cl, [1] * len(cl), marker="o", color='green', s=20)
            axs.set_xlabel('Time (s)')
            for ax in self.fig.axes:
                ax.yaxis.set_ticklabels([])
            axp.set_ylabel('Position')
            axs.set_ylabel('Scanning')
            axz.set_ylabel('Zone')
            axr.set_ylabel('Reward')
            axl.set_ylabel('Lick')
            self.fig.suptitle(f + '\n' + comment.replace('/n', '\n'))
            if not show:
                self.fig.savefig(f + '.png')
                # plt.close()

    def get_events(self, pin=11,):
        '''
        Return times of messages specific to a pin.
        :param pin: pin number e.g. 11 for tone
        #:param state: not implemented
        :return: list of times
        '''
        times = []
        for p, t, trig in self.other_events:
            if p == pin and trig:
                times.append(t)
        return times

    def get_rewarded_licks(self, open_time=5, open_count=3):
        '''
        :param open_time: licks count as rewarded after this many secs after valve open
        :param open_count: this many licks count as rewarded after each valve open
        :return: rewarded licks
        '''
        licks = numpy.array(self.licks)
        rewards = numpy.array(self.rewards)

        # time eligibility
        r_licks = []
        for ti, t in enumerate(licks):
            incl = False
            # how long since last valve open?
            i = rewards.searchsorted(t)
            if i > 1:
                r0 = rewards[i - 1]
                if t - r0 < open_time:
                    incl = True
            # how many licks since last valve open?
            if incl:
                if ti > open_count:
                    i0 = licks.searchsorted(r0)
                    if ti - i0 >= open_count:
                        incl = False
            if incl:
                r_licks.append(t)
        return numpy.array(r_licks)

    def get_consumed_rewards(self, open_time=5):
        '''
        :param open_time: lick within this many secs after entering
        :return: reward zone entries followed by licks
        '''
        licks = numpy.array(self.licks)
        rewards = numpy.array(self.rewards)
        entries = numpy.array(self.entries)

        # time eligibility
        r_entries = []
        for ti, t in enumerate(entries):
            incl = False
            # is it followed by valve open?
            i = rewards.searchsorted(t)
            if i < len(rewards):
                # print(i, t,)
                r0 = rewards[i]
                if r0 - t < open_time:
                    incl = True
            # any licks since last valve open?
            if incl:
                i0 = licks.searchsorted(r0)
                if i0 < len(licks):
                    l0 = licks[i0]
                    if not l0 - t < open_time:
                        incl = False
            if incl:
                r_entries.append(t)
        return numpy.array(r_entries)