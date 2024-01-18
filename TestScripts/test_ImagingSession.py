import Proc2P
from Proc2P import *
from PlotTools import *
from Proc2P.Analysis.ImagingSession import ImagingSession
from Proc2P.Analysis.AnalysisClasses import PhotoStim, EventMasks, PSTH

# example session
path = 'D:/Shares/Data/_Processed/2P/PVTot/Opto/'
prefix = 'PVTot5_2023-09-04_opto_023'  # on
tag = 'IN'

session = ImagingSession(path, prefix, tag=tag, ch=0)

# get pulse trains
t = PhotoStim.PhotoStim(session)
train_starts, train_ints = t.get_trains(isi=10)

fps = session.fps
plot_dur = 10
w = int(fps * plot_dur)

# get response of each cell to each stim
event, mask = EventMasks.masks_from_list(session, w, train_starts)
resps = PSTH.pull_session_with_mask(session, mask)

# sort cells by mean response
pre_s = (-1, 0)
post_s = (0, 5)
pre_slice = slice(*[int(w + x * fps) for x in pre_s])
post_slice = slice(*[int(w + x * fps) for x in post_s])
pre = numpy.nanmean(resps[:, pre_slice], axis=1)
post = numpy.nanmean(resps[:, post_slice], axis=1)
resp_mag = post - pre
plt.imshow(resps[numpy.argsort(resp_mag)], aspect='auto')

# mean_resp = numpy.nanmean(resps[resp_mag > 0], axis=0) * 100
mean_resp = numpy.nanmean(resps, axis=0) * 100

fig, ax = plt.subplots(ncols=1, figsize=(3, 3))
seconds = numpy.arange(-plot_dur, plot_dur+ 1, 2)

# plot mean power
ca = ax

# add light
lcol = "#ff0000"

stimfreq = 2
stimnum = 10
step = fps / stimfreq
stimdur = 0.01  # 10 ms
stimwidth = fps * stimdur
stimtimes = numpy.arange(w, w + step * stimnum, step)
for t in stimtimes:
    ca.axvspan(t, t + stimwidth, color=lcol)

prep_ax_psth(ca, w, seconds=seconds, show_x0=False)
ca.plot(mean_resp, color='black')
ca.set_xlim(w - 2.5 * fps, w + 8 * fps)
ca.set_ylim(-30, 30)

plt.tight_layout()
