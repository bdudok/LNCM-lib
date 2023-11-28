import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from tkinter import *
from tkinter import filedialog, messagebox
import os
from datetime import datetime
import numpy
import pandas

# for mc:
from multiprocessing import Queue, Process, cpu_count, freeze_support, set_start_method
# from MotionCorrect import Worker as mc_Worker
# from MotionCorrect import CleanupWorker

# for roiedit
from Proc2P.Analysis.RoiEditor import RoiEditor, Translate
from Proc2P.Analysis.RoiEditor import Gui as roi_Gui
from Proc2P.Analysis.RoiEditor import Worker as seg_Worker
from subprocess import call, Popen

from Proc2P.Analysis.PullSignals import Worker as pull_Worker

# for traces
from Proc2P.Analysis.CaTrace import CaTrace
from Proc2P.Analysis.CaTrace import Worker as tr_Worker

#for ephys
# from Ripples import Ripples, export_SCA
# from Spike_Sz_detect import Worker as SzDet_Worker

# for views
from Proc2P.Analysis.SessionGui import Gui as session_Gui
# from Scores import Scores
from Proc2P.Analysis.LoadPolys import LoadImage
import cv2, h5py
from scipy.signal import decimate

# utility
from Proc2P.Analysis.AssetFinder import AssetFinder
# from RenameTDML import RenameTDML
# from BehaviorSession import BehaviorSession
from TkApps import PickFromList
from TimeProfile import TimeProfile
from Proc2P.Analysis.AnalysisClasses.ExportStop import exportstop
# from Batch_Utils import split_ephys, export_speed, export_speed_cms

# specific analysis
# from M2score import calc_m2_index


class App:
    def __init__(self, request_queue):
        self.root = Tk()
        self.root.title('Batch processing')
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.current_column = 0
        self.request_queue = request_queue
        self.filelist = FileList(self, self.root, self.column())
        # self.mc = Motion(self, self.root, self.column())
        self.util = Util(self, self.root, self.column())
        self.preview = Prev(self, self.root, self.column())
        self.roidetect = RoiDet(self, self.root, self.column())
        self.roiedit = RoiEd(self, self.root, self.column())
        self.roiconvert = Rois(self, self.root, self.column())
        self.traces = Traces(self, self.root, self.column())
        self.views = View(self, self.root, self.column())
        self.pltconfigs = Cfg(self, self.root, self.column())
        self.text_box = Text(self.root, state=DISABLED, height=15)
        self.text_box.grid(row=1, column=0, columnspan=self.column() + 1)
        sys.stdout = StdRedirector(self.text_box)
        sys.stderr = StdRedirector(self.text_box)
        self.assets = AssetFinder()
        self.last_used = []
        self.session_cache = {}

        self.root.mainloop()

    def get_session(self, prefix, tag):
        lu = self.last_used
        ls = self.session_cache
        ch = self.pltconfigs.config['ch'].get()
        norip = not self.pltconfigs.config['ShowRipples'].get()
        epc = self.preview.get_ephys_channels()
        hash = prefix + str(tag) + str(ch) + str(norip) + ''.join([str(x) for x in epc])
        if hash in ls:
            del lu[lu.index(hash)]
            lu.append(hash)
        else:
            os.chdir(self.filelist.wdir)
            if len(lu) > 5:
                i = lu.pop(0)
                if i in ls:
                    del ls[i]
            print('Fetching', prefix, tag, ch)
            # print('Implement session GUI')
            # assert False
            ls[hash] = session_Gui(self.filelist.wdir, prefix, tag=tag, norip=norip, ch=ch,)# ephys_channels=epc)
            lu.append(hash)
        return ls[hash]

    def on_closing(self):
        if messagebox.askokcancel("Quit", "OK to quit?"):
            self.request_queue.put(('exit', None))
            print('Quit.')
            self.root.destroy()

    def column(self, incr=1):
        self.current_column += incr
        return self.current_column - incr


class StdRedirector:
    def __init__(self, text_widget):
        self.text_space = text_widget

    def write(self, string):
        self.text_space.config(state=NORMAL)
        self.text_space.insert("end", string)
        self.text_space.see("end")
        self.text_space.config(state=DISABLED)


class Cfg:
    def __init__(self, parent, master, column):
        self.parent = parent
        self.master = master
        self.current_row = 0
        self.current_row2 = 0
        self.config = {}

        self.frame = Frame(master)
        self.frame.grid(row=0, column=column, sticky=N + W)

        # STUFF IN FIRST COLUMN

        Label(self.frame, text='Plot config').grid(row=self.row(), pady=10)
        Label(self.frame, text='Parameter').grid(row=self.row())
        MODES = [u"\N{GREEK CAPITAL LETTER DELTA}" + 'F/F', 'NND', 'EWMA',
                 u"\N{GREEK CAPITAL LETTER DELTA}" + 'F/F (z)', 'smtr', ]
        self.config['param'] = StringVar()
        self.config['param'].set(MODES[-1])
        for r, text in enumerate(MODES):
            Radiobutton(self.frame, text=text, variable=self.config['param'], value=text).grid(row=self.row(),
                                                                                               sticky=N + W)
        self.row()

        Label(self.frame, text='EWMA period').grid(row=self.row())
        self.period = Scale(self.frame, from_=3, to=100, orient=HORIZONTAL)
        self.period.set(15)
        self.period.grid(row=self.row(), sticky=N)
        self.config['period'] = self.period

        # Label(self.frame, text='Traces to display').grid(row=self.row())
        # Label(self.frame, text='.np folder').grid(row=self.row(), pady=10)
        # MODES = ['Use default', 'Specify Tag']
        # self.config['roi'] = StringVar()
        # self.config['roi'].set(MODES[1])
        # for r, text in enumerate(MODES):
        #     Radiobutton(self.frame, text=text, variable=self.config['roi'], value=text).grid(row=self.row(), sticky=NW)
        # self.config['roi_name'] = self.parent.roiconvert.config['roi_name']
        # self.config['roi_name'].set('1')
        # row = self.row()
        # Label(self.frame, text='Tag from Pull Traces').grid(row=row, column=0, sticky=NW)
        # Entry(self.frame, textvariable=self.config['roi_name'], width=6).grid(row=row, column=0, sticky=NE)

        Button(self.frame, text='Clear cache', command=self.cache_callback).grid(row=self.row(), sticky=N)

        # STUFF IN SECOND COLUMN

        Label(self.frame, text='Units plot').grid(row=self.second_row(), column=1, pady=10)
        text = 'ShowRipples'
        self.config[text] = IntVar()
        Checkbutton(self.frame, text=text, variable=self.config[text]).grid(row=self.second_row(), column=1, sticky=W)
        self.config[text].set(0)

        Label(self.frame, text='Skin').grid(row=self.second_row(), column=1)
        MODES = ['dark', 'light', 'GS-fig']
        self.config['skin'] = StringVar()
        var = self.config['skin']
        var.set(MODES[1])
        # for r, text in enumerate(MODES):
        #     Radiobutton(self.frame, text=text, variable=var, value=text).grid(row=self.second_row(),
        #                                                                       column=1, sticky=N + W)

        Label(self.frame, text='Channels').grid(row=self.second_row(), column=1)
        MODES = ['Both', 'First', 'Second']
        self.config['ch'] = StringVar()
        var = self.config['ch']
        var.set(MODES[0])
        for r, text in enumerate(MODES):
            Radiobutton(self.frame, text=text, variable=var, value=text).grid(row=self.second_row(),
                                                                              column=1, sticky=N + W)

        Label(self.frame, text='Session plot').grid(row=self.second_row(), column=1, pady=10)
        Label(self.frame, text='Colormap').grid(row=self.second_row(), column=1)
        self.config['cmap'] = StringVar()
        self.cm = Entry(self.frame, textvariable=self.config['cmap'])
        self.cm.insert(0, 'inferno_r')
        self.cm.grid(row=self.second_row(), column=1, sticky=N)

        self.config['unit'] = StringVar()
        row = self.second_row()
        Label(self.frame, text='Units').grid(row=row, column=1, sticky=N + W)
        Entry(self.frame, textvariable=self.config['unit'], width=6).grid(row=row, column=1, sticky=N + E)
        self.config['unit'].set('frames')

        for text, defval  in zip(['Scatter', 'Heatmap', 'RateFromSpikes'], (0, 1, 0)):
            self.config[text] = IntVar()
            Checkbutton(self.frame, text=text, variable=self.config[text]).grid(row=self.second_row(), column=1,
                                                                                sticky=W)
            self.config[text].set(defval)

    def cache_callback(self):
        self.parent.session_cache.clear()
        self.parent.last_used = []

    def row(self, incr=1):
        self.current_row += incr
        return self.current_row - incr

    def second_row(self, incr=1):
        self.current_row2 += incr
        return self.current_row2 - incr

    def get_tag(self):
        if self.config['roi'].get() == 'Use default':
            tag = None
        else:
            tag = self.config['roi_name'].get()
        return tag


class View:
    def __init__(self, parent, master, column):
        self.parent = parent
        self.master = master
        self.current_row = 0
        self.active_item_name = None
        self.active_tag = None
        self.config = {}

        self.frame = Frame(master, bd=2, relief=SUNKEN)
        self.frame.grid(row=0, column=column, sticky=N + W)

        Label(self.frame, text='Display plots').grid(row=self.row(), pady=10)
        # Button(self.frame, text="Highlight", command=self.autosel_callback).grid(row=self.row(), sticky=N)
        Button(self.frame, text="Unit plots", command=self.unit_callback).grid(row=self.row(), sticky=N)
        Button(self.frame, text="Session plot", command=self.session_callback).grid(row=self.row(), sticky=N)
        Button(self.frame, text="Place fields", command=self.pf_callback).grid(row=self.row(), sticky=N)
        Button(self.frame, text="Weighted Pfields", command=self.pfsmooth_callback).grid(row=self.row(), sticky=N)
        Label(self.frame, text='Sorted plots').grid(row=self.row(), pady=10)
        Button(self.frame, text="Run sequence", command=self.runseq_callback).grid(row=self.row(), sticky=N)
        Button(self.frame, text="Placefield seq.", command=self.pfseq_callback).grid(row=self.row(), sticky=N)
        Button(self.frame, text="Weighted Pf. seq.", command=self.pfseqsmooth_callback).grid(row=self.row(), sticky=N)
        Button(self.frame, text="Running Pf. seq.", command=self.runpfseq_callback).grid(row=self.row(), sticky=N)
        # Label(self.frame, text='Export').grid(row=self.row(), pady=10)
        # Button(self.frame, text="Save csv", command=self.csv_callback).grid(row=self.row(), sticky=N)
        # Button(self.frame, text="Save figure", command=self.figsave_callback).grid(row=self.row(), sticky=N)
        # Button(self.frame, text="Save raw F", command=self.rawsave_callback).grid(row=self.row(), sticky=N)
        # Label(self.frame, text='QA Analysis').grid(row=self.row(), pady=10)
        # Button(self.frame, text="Save synch events", command=self.QAsynch_callback).grid(row=self.row(), sticky=N)

        # Label(self.frame, text='JF ripples').grid(row=self.row(), pady=10)
        # self.config['Ripple-Secs'] = StringVar()
        # row = self.row()
        # Label(self.frame, text='Range(s)').grid(row=row, column=0, sticky=N + W)
        # Entry(self.frame, textvariable=self.config['Ripple-Secs'], width=5).grid(row=row, column=0, sticky=N + E)
        # self.config['Ripple-Secs'].set('5')

        # Button(self.frame, text="Show ripple plot", command=self.RTA_callback).grid(row=self.row(), sticky=N)
        # Button(self.frame, text="Save to excel", command=self.RTA_save_callback).grid(row=self.row(), sticky=N)

    def row(self, incr=1):
        self.current_row += incr
        return self.current_row - incr

    def load(self):
        cf = self.parent.pltconfigs.config
        prefix = self.parent.filelist.get_active()[1][0]
        # if cf['roi'].get() == 'Use default':
        #     tag = None
        # else:
        tag = self.parent.roiconvert.config['roi_name'].get()
        os.chdir(self.parent.filelist.wdir)
        self.active_item = self.parent.get_session(prefix, tag)
        self.active_item_name = prefix
        self.active_tag = tag

    def getparam(self):
        resolvedict = {u"\N{GREEK CAPITAL LETTER DELTA}" + 'F/F': 'rel',
                       u"\N{GREEK CAPITAL LETTER DELTA}" + 'F/F (z)': 'ntr',
                       'Inferred spikes': 'spikes',
                       'EWMA': 'ewma',
                       'smtr': 'smtr',
                       'Peak locations': 'peaks',
                       'AUPeaks': 'aupeaks',
                       'NND': 'nnd',
                       'ontr': 'ontr'}
        param = resolvedict[self.parent.pltconfigs.config['param'].get()]
        if param == 'ewma':
            param = self.parent.pltconfigs.config['period'].get()
        return param

    def autosel_callback(self):
        wdir = self.parent.filelist.wdir
        newsel = []
        nps = []
        for f in os.listdir(wdir):
            if '.np' in f:
                nps.append(f)
        for i, prefix in enumerate(self.parent.filelist.prefix_list):
            for f in nps:
                if prefix in f:
                    newsel.append(i)
                    break
        self.parent.filelist.set_active(newsel)

    def unit_callback(self):
        self.load()
        self.active_item.start(param=self.getparam(), skin=self.parent.pltconfigs.config['skin'].get())

    def session_callback(self):
        self.load()
        cf = self.parent.pltconfigs.config
        if cf['Heatmap'].get():
            cf['Scatter'].set(0)
        rate = ('mean', 'spikes')[cf['RateFromSpikes'].get()]
        self.fig = self.active_item.plot_session(param=self.getparam(), cmap=cf['cmap'].get(),
                                                 scatter=cf['Scatter'].get(),
                                                 hm=cf['Heatmap'].get(), rate=rate, riplines=cf['ShowRipples'].get(),
                                                 axtitle=cf['param'].get(), unit=cf['unit'].get())

    def pf_callback(self):
        self.load()
        self.active_item.placefields(param=self.getparam(), cmap=self.parent.pltconfigs.config['cmap'].get(),
                                     silent=False, show=True)

    def pfsmooth_callback(self):
        self.load()
        self.active_item.placefields_smooth(param=self.getparam(), cmap=self.parent.pltconfigs.config['cmap'].get(),
                                            silent=False, show=True)

    def runseq_callback(self):
        self.load()
        self.active_item.RUNseq(param=self.getparam(), cmap=self.parent.pltconfigs.config['cmap'].get())

    def pfseq_callback(self):
        self.load()
        cf = self.parent.pltconfigs.config
        if cf['Heatmap'].get():
            cf['Scatter'].set(0)
        rate = ('mean', 'spikes')[cf['RateFromSpikes'].get()]
        self.active_item.placefields(param=self.getparam(), silent=True)
        self.fig = self.active_item.plot_session(param=self.getparam(), corder=self.active_item.corder_pf,
                                                 scatter=cf['Scatter'].get(), hm=cf['Heatmap'].get(),
                                                 cmap=cf['cmap'].get(),
                                                 riplines=cf['ShowRipples'].get(), axtitle=cf['param'].get(), rate=rate,
                                                 unit=cf['unit'].get())

    def pfseqsmooth_callback(self):
        self.load()
        cf = self.parent.pltconfigs.config
        if cf['Heatmap'].get():
            cf['Scatter'].set(0)
        rate = ('mean', 'spikes')[cf['RateFromSpikes'].get()]
        self.active_item.placefields_smooth(param=self.getparam(), silent=True)
        self.fig = self.active_item.plot_session(param=self.getparam(), corder=self.active_item.corder_smoothpf,
                                                 scatter=cf['Scatter'].get(), hm=cf['Heatmap'].get(),
                                                 cmap=cf['cmap'].get(),
                                                 riplines=cf['ShowRipples'].get(), axtitle=cf['param'].get(), rate=rate,
                                                 unit=cf['unit'].get())

    def runpfseq_callback(self):
        self.load()
        cf = self.parent.pltconfigs.config
        self.fig = self.active_item.running_placeseq(display_param=self.getparam(), cmap=cf['cmap'].get())

    def RTA_callback(self):
        self.load()
        self.active_item.ripple_triggered_nonoverlap_mean(self.getparam(), int(self.config['Ripple-Secs'].get()))

    def RTA_save_callback(self):
        self.load()
        self.active_item.ripple_triggered_nonoverlap_mean(self.getparam(), int(self.config['Ripple-Secs'].get()),
                                                          save=True)

    def csv_callback(self):
        self.load()
        self.active_item.export_timeseries()

    def figsave_callback(self):
        if not self.fig is None:
            fig = self.fig
            ts = str(datetime.now())
            ts = '-' + ts[:ts.find(' ')] + '-' + ts[ts.find(' ') + 1:ts.find('.')].replace(':', '-') + '.png'
            fig.savefig(self.active_item_name + ts, dpi=240)

    def rawsave_callback(self):
        self.load()
        self.active_item.export_raw()

    def QAsynch_callback(self):
        pflist = self.parent.filelist.get_active()[1]
        wdir = self.parent.filelist.wdir
        tag = self.parent.pltconfigs.get_tag()
        os.chdir(wdir)
        for prefix in pflist:
            Scores(prefix, tag=tag, norip=True).sync_events()
            print(prefix, 'events saved')


class Traces:
    def __init__(self, parent, master, column):
        self.parent = parent
        self.master = master
        self.current_row = 0
        self.config = {}

        self.frame = Frame(master)
        self.frame.grid(row=0, column=column, sticky=N + W)

        Label(self.frame, text='Process traces').grid(row=self.row(), pady=10)
        Button(self.frame, text="Auto select", command=self.autosel_callback).grid(row=self.row(), sticky=N)
        # self.version = IntVar()
        # Checkbutton(self.frame, text='Old baseline', variable=self.version).grid(row=self.row(), sticky=N)
        # self.version.set(0)

        # self.peakdet = IntVar()
        # Checkbutton(self.frame, text='Detect peaks', variable=self.peakdet).grid(row=self.row(), sticky='N')
        # self.peakdet.set(1)

        Label(self.frame, text='Exclude from baseline').grid(row=self.row())
        defs = 0, 0
        self.tracefields = ['Start', 'Stop']
        for ti, text in enumerate(self.tracefields):
            self.config[text] = StringVar()
            row = self.row()
            Label(self.frame, text=text).grid(row=row, column=0, sticky=N + W)
            Entry(self.frame, textvariable=self.config[text], width=5).grid(row=row, column=0, sticky=N + E)
            self.config[text].set(str(defs[ti]))

        self.specify = IntVar()
        Checkbutton(self.frame, text='Use ROI ID', variable=self.specify).grid(row=self.row(), sticky=N)
        self.specify.set(1)

        self.detect_signals = IntVar()
        Checkbutton(self.frame, text='Detect all ROIs', variable=self.detect_signals).grid(row=self.row(), sticky=N)
        self.detect_signals.set(0)

        Button(self.frame, text="Run", command=self.execute_callback).grid(row=self.row(), sticky=N)

        # Label(self.frame, text='Ripples').grid(row=self.row(), pady=10)
        # Button(self.frame, text="Highlight", command=self.autosel_rip_callback).grid(row=self.row(), sticky=N)

    #     Label(self.frame, text='Config').grid(row=self.row(), pady=10)
    #     defs = (5, 3, 1)
    #     self.cfgfields = ['tr1', 'tr2', 'y_scale']
    #     for i, text in enumerate(self.cfgfields):
    #         self.config[text] = StringVar()
    #         row = self.row()
    #         Label(self.frame, text=text).grid(row=row, column=0, sticky=N + W)
    #         Entry(self.frame, textvariable=self.config[text], width=3).grid(row=row, column=0, sticky=N + E)
    #         self.config[text].set(str(defs[i]))
    #     self.force = IntVar()
    #     Checkbutton(self.frame, text='Force recompute', variable=self.force).grid(row=self.row(), sticky='N')
    #     self.force.set(0)
    #
    #     Button(self.frame, text="Open Ripples", command=self.ripples_callback).grid(row=self.row(), sticky=N)
    #     Button(self.frame, text="Display", command=self.ripples_enum_callback).grid(row=self.row(), sticky=N)
    #
    #     self.excl_spikes = IntVar()
    #     Checkbutton(self.frame, text='Exclude spikes', variable=self.excl_spikes).grid(row=self.row(), sticky='N')
    #     self.excl_spikes.set(0)
    #     Button(self.frame, text="Detect recursive", command=self.ripples_rec_callback).grid(row=self.row(), sticky=N)
    #     Button(self.frame, text="Load ripp. set", command=self.ripples_load_callback).grid(row=self.row(), sticky=N)
    #     Button(self.frame, text="Save ripp. set", command=self.ripples_save_callback).grid(row=self.row(), sticky=N)
    #     Button(self.frame, text="export to excel", command=self.ripples_export_callback).grid(row=self.row(), sticky=N)
    #
    #     Button(self.frame, text="M2 Score", command=self.m2score_callback).grid(row=self.row(), sticky=N, pady=10)
    #     Button(self.frame, text="OROT Score", command=self.orotscore_callback).grid(row=self.row(), sticky=N, pady=10)
    #
    # def m2score_callback(self):
    #     for prefix in self.parent.filelist.get_active()[1]:
    #         print('Export M2 score:', prefix, 'queued.')
    #         self.parent.request_queue.put(('exportstop', (self.parent.filelist.wdir, prefix,
    #                                                       self.parent.mc.channels.get(),
    #                                                       self.parent.preview.get_ephys_channels(),
    #                                                       'm2')))
    #
    # def orotscore_callback(self):
    #     for prefix in self.parent.filelist.get_active()[1]:
    #         print('Export 1/M2 score:', prefix, 'queued.')
    #         self.parent.request_queue.put(('exportstop', (self.parent.filelist.wdir, prefix,
    #                                                       self.parent.mc.channels.get(),
    #                                                       self.parent.preview.get_ephys_channels(),
    #                                                       True,
    #                                                       'm2')))

    def row(self, incr=1):
        self.current_row += incr
        return self.current_row - incr

    def autosel_callback(self):
        wdir = self.parent.filelist.wdir
        newsel = []
        tag = ''
        if self.specify.get():
            tag = '-' + self.parent.roiconvert.config['roi_name'].get()
        for i, prefix in enumerate(self.parent.filelist.prefix_list):
            if not os.path.exists(wdir + prefix + tag + '.np'):
                if tag == '':
                    if os.path.exists(wdir + prefix + '_nonrigid.signals'):
                        newsel.append(i)
                elif os.path.exists(wdir + prefix + '_trace_' + tag[1:] + '.npy'):
                    newsel.append(i)
        self.parent.filelist.set_active(newsel)

    def autosel_rip_callback(self):
        wdir = self.parent.filelist.wdir
        newsel = []
        for i, prefix in enumerate(self.parent.filelist.prefix_list):
            if os.path.exists(wdir + prefix + '.ephys'):
                newsel.append(i)
        self.parent.filelist.set_active(newsel)

    def execute_callback(self):
        for prefix in self.parent.filelist.get_active()[1]:
            # if self.version.get():
            #     bsltype = 'original'
            # else:
            bsltype = 'poly'
            tags = [None]
            if self.specify.get():
                tags = [self.parent.roiconvert.config['roi_name'].get()]
            if self.detect_signals.get():  # queue all rois that were extracted
                tags = []
                match_text = '_trace_'
                for f in os.listdir(self.parent.filelist.wdir):
                    if prefix in f and match_text in f and f.endswith('.npy'):
                        tags.append(f[f.find(match_text) + len(match_text):-4])
            peakdet = False#self.peakdet.get()
            excl = (int(self.config[self.tracefields[0]].get()), int(self.config[self.tracefields[1]].get()))
            sz_mode = False#self.parent.mc.ignore_sat.get()
            for tag in tags:
                print(f'{prefix}: tag {tag} queued for processing traces.')
                self.parent.request_queue.put(('firing',
                                               (self.parent.filelist.wdir, prefix, bsltype, excl, sz_mode, peakdet,
                                                tag)))

    def ripples_callback(self):
        prefix = self.parent.filelist.get_active()[1][0]
        os.chdir(self.parent.filelist.wdir)
        cfg = {}
        for text in self.cfgfields:
            try:
                cfg[text] = int(self.config[text].get())
            except:
                print(text + ' is not a number')
                return -1
        self.ripples = Ripples(prefix, config=cfg, force=self.force.get(),
                               ephys_channels=self.parent.preview.get_ephys_channels())
        print(prefix, 'ripples loaded.')

    def ripples_enum_callback(self):
        self.ripples.enum_ripples(no_save=True)

    def ripples_rec_callback(self):
        self.ripples.rec_enum_ripples(exclude_spikes=self.excl_spikes.get())

    def ripples_save_callback(self):
        self.ripples.save_ripples()

    def ripples_export_callback(self):
        self.ripples.export_ripple_times()

    def ripples_load_callback(self):
        oloc = self.parent.filelist.wdir + self.ripples.prefix + '_ripples//'
        fn = filedialog.askopenfilename(initialdir=oloc)
        self.ripples.load_ripples(fn.split('/')[-1])
        self.ripples.enum_ripples(no_save=True)


class RoiDet:
    def __init__(self, parent, master, column):
        self.parent = parent
        self.master = master
        self.current_row = 0
        self.config = {}

        self.frame = Frame(master, bd=2, relief=SUNKEN)
        self.frame.grid(row=0, column=column, sticky=N + W)

        Label(self.frame, text='Segmentation').grid(row=self.row(), pady=10)
        # Button(self.frame, text="Auto select", command=self.autosel_callback).grid(row=self.row(), sticky=N)
        #
        Label(self.frame, text='Methods').grid(row=self.row())
        approaches = ['iPC', 'PC', 'STICA', 'iPC-1', 'PC-1', 'STICA-1']
        self.rd_apps = {}
        for i, app in enumerate(approaches):
            self.rd_apps[app] = IntVar()
            Checkbutton(self.frame, text=app, variable=self.rd_apps[app]).grid(row=self.row(), column=0, sticky='W')
            self.rd_apps[app].set(i < 3)

        Label(self.frame, text='Span').grid(row=self.row())
        defs = 0, 'end', 20, 100, 800
        for ti, text in enumerate(['Start', 'Stop', 'Diameter', 'MinSize', 'MaxSize']):
            self.config[text] = StringVar()
            row = self.row()
            Label(self.frame, text=text).grid(row=row, column=0, sticky=N + W)
            Entry(self.frame, textvariable=self.config[text], width=5).grid(row=row, column=0, sticky=N + E)
            self.config[text].set(str(defs[ti]))

        Button(self.frame, text="Run", command=self.execute_callback).grid(row=self.row(), sticky=N)

        # Label(self.frame, text='Sublayers').grid(row=self.row(), pady=10)
        # Button(self.frame, text="Compute", command=self.sublayer_callback).grid(row=self.row(), sticky=N)

    def autosel_callback(self):
        wdir = self.parent.filelist.wdir
        newsel = []
        for i, prefix in enumerate(self.parent.filelist.prefix_list):
            if os.path.exists(wdir + prefix + '_nonrigid.sbx'):
                if not os.path.exists(wdir + prefix + '_nonrigid.segment'):
                    if not any([os.path.exists(wdir + prefix + '_saved_roi_' + s + '.npy') for s in
                                ['iPC', 'PC', 'STICA', 'iPC-1', 'PC-1', 'STICA-1']]):
                        newsel.append(i)
        self.parent.filelist.set_active(newsel)

    def execute_callback(self):
        for prefix in self.parent.filelist.get_active()[1]:
            print('Segmentation:', prefix, 'queued.')
            apps = []
            for app in self.rd_apps:
                if self.rd_apps[app].get():
                    apps.append(app)
            config = {}
            for x in ['Start', 'Stop', 'Diameter', 'MinSize', 'MaxSize']:
                config[x] = self.config[x].get()
            self.parent.request_queue.put(('segment', (self.parent.filelist.wdir, prefix, apps, config)))

    def sublayer_callback(self):
        roi_prefix = self.parent.filelist.get_active()[1][0]
        roi_path = self.parent.filelist.wdir
        r = LoadImage(roi_prefix)
        roi_pxs = r.pixelsize
        fn = filedialog.askopenfilename(title='Select stack tif file', initialdir=roi_path)
        suffix = '_stack.tif'
        if suffix not in fn:
            raise ValueError(f'Select a {suffix} file')
        stack_path, stack_fn = os.path.split(fn)
        stack_path += os.path.sep
        stack_prefix = stack_fn[:stack_fn.find(suffix)]
        os.chdir(stack_path)
        s = LoadImage(stack_prefix)
        stack_pxs = s.pixelsize
        stack_zstep = int(self.parent.preview.config['Step'].get())
        roi_tag = self.parent.roiconvert.config['roi_name'].get()
        compute_laminar(stack_path, stack_prefix, stack_pxs, stack_zstep, roi_path, roi_prefix, roi_pxs, roi_tag)
        print('Output saved.')
        os.chdir(roi_path)

    def row(self, incr=1):
        self.current_row += incr
        return self.current_row - incr


class RoiEd:
    def __init__(self, parent, master, column):
        self.parent = parent
        self.master = master
        self.current_row = 0
        self.config = {}

        self.frame = Frame(master)
        self.frame.grid(row=0, column=column, sticky=N + W)

        Label(self.frame, text='Filter & Edit').grid(row=self.row(), pady=10)
        # Button(self.frame, text="Highlight", command=self.highlight_callback).grid(row=self.row(), sticky=N)
        # Button(self.frame, text="Translate", command=self.translate_callback).grid(row=self.row(), sticky=N)
        Button(self.frame, text="Start", command=self.roied_callback).grid(row=self.row(), sticky=N)
        # Button(self.frame, text="Clear", command=self.clear_callback).grid(row=self.row(), sticky=N)
        # Button(self.frame, text="Dilate", command=self.dilate_callback).grid(row=self.row(), sticky=N)
        # Button(self.frame, text="Save", command=self.save_callback).grid(row=self.row(), sticky=N)
        # Button(self.frame, text="Export to sbx", command=self.sbx_callback).grid(row=self.row(), sticky=N)
        # Button(self.frame, text="Close", command=self.close_callback).grid(row=self.row(), sticky=N)

    def highlight_callback(self):
        wdir = self.parent.filelist.wdir
        newsel = []
        for i, prefix in enumerate(self.parent.filelist.prefix_list):
            if os.path.exists(wdir + prefix + '_nonrigid.sbx'):
                if not os.path.exists(wdir + prefix + '_nonrigid.segment'):
                    if any([os.path.exists(wdir + prefix + '_saved_roi_' + s + '.npy') for s in
                            ['iPC', 'PC', 'STICA']]):
                        newsel.append(i)
        self.parent.filelist.set_active(newsel)

    def translate_callback(self):
        tt = Process(target=Translate, args=(self.parent.filelist.wdir,))
        tt.start()

    def roied_callback(self):
        prefix = self.parent.filelist.get_active()[1][0]
        self.gui = roi_Gui(self.parent.filelist.wdir, prefix)
        if self.gui.loop(client='gui'):
            self.save_callback()
            self.sbx_callback()
            self.close_callback()
        # else:
        #     self.close_callback()

    def clear_callback(self):
        self.gui.clear_callback()

    def dilate_callback(self):
        self.gui.dilate_callback()

    def save_callback(self):
        self.gui.save()

    def close_callback(self):
        self.gui.close_callback()

    def sbx_callback(self):
        self.gui.save_sbx()

    def row(self, incr=1):
        self.current_row += incr
        return self.current_row - incr


class Util:
    def __init__(self, parent, master, column):
        self.parent = parent
        self.master = master
        self.current_row = 0
        self.config = {}

        self.frame = Frame(master, bd=2, relief=SUNKEN)
        self.frame.grid(row=0, column=column, sticky=N + W)

        Label(self.frame, text='Preview').grid(row=self.row(), pady=10)
        # Button(self.frame, text="Auto select", command=self.autosel_callback).grid(row=self.row(), sticky=N)

        # Button(self.frame, text="Run", command=self.execute_callback).grid(row=self.row(), sticky=N)
        Button(self.frame, text="Export Stop", command=self.exportstop_callback).grid(row=self.row(), sticky=N)
        Button(self.frame, text="Show", command=self.show_callback).grid(row=self.row(), sticky=N)

        Button(self.frame, text="Play movie", command=self.viewer_callback).grid(row=self.row(), sticky=N)

        # Label(self.frame, text='Rename Tdmls').grid(row=self.row(), pady=10)
        # Button(self.frame, text="Auto select", command=self.autosel_tdml_callback).grid(row=self.row(), sticky=N)
        # Button(self.frame, text="Rename", command=self.execute_tdml_callback).grid(row=self.row(), sticky=N)
        Button(self.frame, text="Export behavior plots", command=self.export_treadmill_callback).grid(row=self.row(),
                                                                                                 sticky=N)
        # Button(self.frame, text="Laps spreadsheet", command=self.export_list_callback).grid(row=self.row(),
        #                                                                                     sticky=N)

        Label(self.frame, text='Export Time profile').grid(row=self.row(), pady=10)
        defs = (0, 1000, 256, 500)
        self.cfgfields = ['Start', 'Stop', 'Line', 'Kernel']
        for i, text in enumerate(self.cfgfields):
            self.config[text] = StringVar()
            row = self.row()
            Label(self.frame, text=text).grid(row=row, column=0, sticky=N + W)
            Entry(self.frame, textvariable=self.config[text], width=4).grid(row=row, column=0, sticky=N + E)
            self.config[text].set(str(defs[i]))
        Button(self.frame, text="Export image", command=self.timeprofile_callback).grid(row=self.row(), sticky=N)

    def timeprofile_callback(self):
        pflist = self.parent.filelist.get_active()[1]
        wdir = self.parent.filelist.wdir
        cfg = {}
        for text in self.cfgfields:
            try:
                cfg[text] = int(self.config[text].get())
            except:
                print(text + ' is not a number')
                return -1
        for prefix in pflist:
            TimeProfile(prefix, cfg)

    def export_treadmill_callback(self):
        print('not implemented')

    def autosel_callback(self):
        wdir = self.parent.filelist.wdir
        newsel = []
        for i, prefix in enumerate(self.parent.filelist.prefix_list):
            if not os.path.exists(wdir + prefix + '_preview.tif'):
                if os.path.exists(wdir + prefix + '_nonrigid.sbx'):
                    newsel.append(i)
        self.parent.filelist.set_active(newsel)

    def autosel_tdml_callback(self):
        wdir = self.parent.filelist.wdir
        newsel = []
        for i, prefix in enumerate(self.parent.filelist.prefix_list):
            if not os.path.exists(wdir + prefix + '.tdml'):
                newsel.append(i)
        self.parent.filelist.set_active(newsel)

    def exportstop_callback(self):
        for prefix in self.parent.filelist.get_active()[1]:
            self.parent.request_queue.put(('exportstop', (self.parent.filelist.wdir, prefix, 'stop', 'Green')))
                                                          # self.parent.mc.channels.get())))


    def export_list_callback(self):
        pflist = self.parent.filelist.get_active()[1]
        wdir = self.parent.filelist.wdir
        os.chdir(wdir)
        s = 'Prefix\tLaps\n'
        for pf in pflist:
            bmname = pf + '.tdml'
            if os.path.exists(bmname):
                s += f'{pf}\t{len(numpy.unique(BehaviorSession(bmname).laps[0]))}\n'
            else:
                s += f'{pf}\tNot available\n'
        with open('_Lap_list.txt', 'a') as of:
            of.write(s)

    def show_callback(self):
        prefix = self.parent.filelist.get_active()[1][0]
        cstr = self.parent.filelist.wdir + prefix + '/' + prefix + '_preview.tif'
        print(cstr)
        Popen([cstr], shell=True)

    def viewer_callback(self):
        prefix = self.parent.filelist.get_active()[1][0]
        self.parent.request_queue.put(('movie', (self.parent.filelist.wdir, prefix)))

    def row(self, incr=1):
        self.current_row += incr
        return self.current_row - incr


class Prev:
    def __init__(self, parent, master, column):
        self.parent = parent
        self.master = master
        self.current_row = 0
        self.config = {}

        self.frame = Frame(master)
        self.frame.grid(row=0, column=column, sticky=N + W)

        # Label(self.frame, text='Cam View').grid(row=self.row(), pady=10)
        # Button(self.frame, text="Highlight", command=self.eyesel_callback).grid(row=self.row(), sticky=N)
        #
        # Button(self.frame, text="Play cam", command=self.eyeviewer_callback).grid(row=self.row(), sticky=N)
        #
        # Label(self.frame, text='Z stack').grid(row=self.row(), pady=10)
        # defs = (200, 5, 30)
        # for i, text in enumerate(['Range', 'Step', 'Frames']):
        #     self.config[text] = StringVar()
        #     row = self.row()
        #     Label(self.frame, text=text).grid(row=row, column=0, sticky=N + W)
        #     Entry(self.frame, textvariable=self.config[text], width=5).grid(row=row, column=0, sticky=N + E)
        #     self.config[text].set(str(defs[i]))
        #
        # Button(self.frame, text="Show stack", command=self.playstack_callback).grid(row=self.row(), sticky=N)
        #
        # Button(self.frame, text="Save stacks", command=self.savestack_callback).grid(row=self.row(), sticky=N)
        #
        # Button(self.frame, text="Show MIP", command=self.showmip_callback).grid(row=self.row(), sticky=N)

        Label(self.frame, text='Ephys channels').grid(row=self.row(), pady=10)
        subframe = Frame(self.frame)
        subframe.grid(row=self.row(), column=0, sticky=N + W)
        for i, text in enumerate(('Ch', 'of')):
            self.config[text] = StringVar()
            Label(subframe, text=text, width=2).grid(row=0, column=i * 2, sticky=N + W)
            Entry(subframe, textvariable=self.config[text], width=2).grid(row=0, column=i * 2 + 1, sticky=N + W)
            self.config[text].set('1')

        Label(self.frame, text='Ephys trace').grid(row=self.row(), pady=10)
        Button(self.frame, text="Show", command=self.showtrace_callback).grid(row=self.row(), sticky=N)
        # Button(self.frame, text="Export csv", command=self.export_trace_callback).grid(row=self.row(), sticky=N)
        # Button(self.frame, text="Export SCA", command=self.export_SCA_callback).grid(row=self.row(), sticky=N)
        # Button(self.frame, text='LFP + Ca', command=self.show_overlay_callback).grid(row=self.row(), sticky=N)
        # Button(self.frame, text='LFP + Ca QA', command=self.show_overlay_callback_QA).grid(row=self.row(), sticky=N)

        # Label(self.frame, text='Batch').grid(row=self.row(), pady=10)
        # Button(self.frame, text='Split ephys', command=self.split_ephys_callback).grid(row=self.row(), sticky=N)
        # Button(self.frame, text='Export speed (ephys)', command=self.export_speed_callback).grid(row=self.row(),
        #                                                                                          sticky=N)
        # Button(self.frame, text='Export speed (imaging)', command=self.export_speed_cms_callback).grid(row=self.row(),
        #                                                                                                sticky=N)
        # Button(self.frame, text="Detect sz and spikes", command=self.detect_sz_callback).grid(row=self.row(), sticky=N)
        #
        # Label(self.frame, text='Thresholds').grid(row=self.row(), pady=10)
        # defs = (5, )
        # for i, text in enumerate(['IISthr', ]):
        #     self.config[text] = StringVar()
        #     row = self.row()
        #     Label(self.frame, text=text).grid(row=row, column=0, sticky=N + W)
        #     Entry(self.frame, textvariable=self.config[text], width=5).grid(row=row, column=0, sticky=N + E)
        #     self.config[text].set(str(defs[i]))

    def get_ephys_channels(self):
        return [int(self.config[text].get()) for text in ('Ch', 'of')]

    # def split_ephys_callback(self):
    #     split_ephys(filedialog.askdirectory(), self.get_ephys_channels()[1])

    # def export_speed_callback(self):
    #     export_speed(filedialog.askdirectory(), self.get_ephys_channels()[1])

    # def export_speed_cms_callback(self):
    #     path = filedialog.askdirectory()
    #     export_speed_cms(path)
    #     print('Export finished for all files in ', path)

    # def detect_sz_callback(self):
    #     path = self.parent.filelist.wdir
    #     kwargs = {'IIS_threshold': float(self.config['IISthr'].get())}
    #     for prefix in self.parent.filelist.get_active()[1]:
    #         self.parent.request_queue.put(('SzDet', (path, prefix, kwargs)))
    #         print('Seizure detection queued:', prefix, )

    # def eyesel_callback(self):
    #     wdir = self.parent.filelist.wdir
    #     newsel = []
    #     for i, prefix in enumerate(self.parent.filelist.prefix_list):
    #         if os.path.exists(wdir + prefix + '_eye.mat'):
    #             newsel.append(i)
    #     self.parent.filelist.set_active(newsel)
    #
    # def eyeviewer_callback(self):
    #     prefix = self.parent.filelist.get_active()[1][0]
    #     self.parent.request_queue.put(('eye', (self.parent.filelist.wdir, prefix)))

    # def get_stack_cfg(self):
    #     cfg = []
    #     for text in ['Step', 'Range', 'Frames']:
    #         try:
    #             cfg.append(int(self.config[text].get()))
    #         except:
    #             print(text + ' is not a number')
    #             return -1
    #     return cfg
    #
    # def savestack_callback(self):
    #     prefix = self.parent.filelist.get_active()[1]
    #     cfg = self.get_stack_cfg()
    #     cfg.append('saving')
    #     self.parent.request_queue.put(('stack', (self.parent.filelist.wdir, prefix, cfg)))
    #
    # def playstack_callback(self):
    #     prefix = self.parent.filelist.get_active()[1][0]
    #     self.parent.request_queue.put(('stack', (self.parent.filelist.wdir, prefix, self.get_stack_cfg())))
    #
    # def showmip_callback(self):
    #     prefix = self.parent.filelist.get_active()[1][0]
    #     fn = self.parent.filelist.wdir + prefix + '_MIP.png'
    #     if not os.path.exists(fn):
    #         cfg = []
    #         for text in ['Step', 'Range', 'Frames']:
    #             try:
    #                 cfg.append(int(self.config[text].get()))
    #             except:
    #                 print(text + ' is not a number')
    #                 return -1
    #         os.chdir(self.parent.filelist.wdir)
    #         stack = LoadImage(prefix)
    #         stack.create_zstack(*cfg, save=True)
    #         if 'win' in sys.platform:
    #             Popen([fn], shell=True)
    #         else:
    #             print("File saved in .tif format, calling system viewer not implemented for platform", sys.platform)

    def showtrace_callback(self):
        prefix = self.parent.filelist.get_active()[1][0]
        self.parent.request_queue.put(('trace', (self.parent.filelist.wdir, prefix, self.get_ephys_channels())))

    # def export_trace_callback(self):
    #     for prefix in self.parent.filelist.get_active()[1]:
    #         play_ephys(self.parent.filelist.wdir, prefix, self.get_ephys_channels(), export=True)

    # def export_SCA_callback(self):
    #     for prefix in self.parent.filelist.get_active()[1]:
    #         export_SCA(self.parent.filelist.wdir, prefix, self.get_ephys_channels())
    #
    # def show_overlay_callback(self):
    #     prefix = self.parent.filelist.get_active()[1][0]
    #     print(prefix, 'LFP plot queued...')
    #     self.parent.request_queue.put(('lfp_overlay', (self.parent.filelist.wdir, prefix, self.get_ephys_channels())))
    #
    # def show_overlay_callback_QA(self):
    #     prefix = self.parent.filelist.get_active()[1][0]
    #     print(prefix, 'LFP plot queued...')
    #     self.parent.request_queue.put(('lfp_overlay_qa', (self.parent.filelist.wdir, prefix,
    #                                                       self.parent.pltconfigs.get_tag(), self.get_ephys_channels())))

    def row(self, incr=1):
        self.current_row += incr
        return self.current_row - incr


class Rois:
    def __init__(self, parent, master, column):
        self.parent = parent
        self.master = master
        self.current_row = 0
        self.config = {}

        self.frame = Frame(master, bd=2, relief=SUNKEN)
        self.frame.grid(row=0, column=column, sticky=N + W)

        Label(self.frame, text='Pull Traces').grid(row=self.row(), pady=10)
        # Button(self.frame, text="Autoselect IJ", command=self.autosel_callback).grid(row=self.row(), sticky=N)
        #
        # Button(self.frame, text="ImageJ -> Scanbox", command=self.execute_callback).grid(row=self.row(), sticky=N)
        # Button(self.frame, text="Scanbox -> ImageJ", command=self.execute2_callback).grid(row=self.row(), sticky=N)

        # Button(self.frame, text="Autoselect roi", command=self.autoroi_callback).grid(row=self.row(), sticky=N, pady=10)
        # Button(self.frame, text="Roieditor -> sbx", command=self.sbx_convert_callback).grid(row=self.row(), sticky=N)

        # Button(self.frame, text="Run sbxsegment", command=self.execute4_callback).grid(row=self.row(), sticky=N)

        # Label(self.frame, text='Extract signals').grid(row=self.row(), pady=10)
        # Button(self.frame, text="Run Matlab script", command=self.execute3_callback).grid(row=self.row(), sticky=N)
        #
        # Label(self.frame, text='------OR------').grid(row=self.row())
        Label(self.frame, text='Select channel').grid(row=self.row())
        MODES = ['All', 'First', 'Second']
        self.config['ch'] = StringVar()
        self.config['ch'].set('All')
        for r, text in enumerate(MODES):
            Radiobutton(self.frame, text=text, variable=self.config['ch'], value=text).grid(row=self.row(), sticky=NW)
        Label(self.frame, text='Select roi set').grid(row=self.row())
        MODES = ['Use latest', 'Specify Tag']
        self.config['roi'] = StringVar()
        self.config['roi'].set(MODES[1])
        # for r, text in enumerate(MODES):
        #     Radiobutton(self.frame, text=text, variable=self.config['roi'], value=text).grid(row=self.row(), sticky=NW)
        self.config['roi_name'] = StringVar()
        self.config['roi_name'].set('1')
        row = self.row()
        Label(self.frame, text='Roi ID').grid(row=row, column=0, sticky=NW)
        Entry(self.frame, textvariable=self.config['roi_name'], width=6).grid(row=row, column=0, sticky=NE)

        Button(self.frame, text='Add selection to queue', command=self.pull_callback).grid(row=self.row(), sticky=N)

        # self.config['conc_str'] = StringVar()
        # self.config['conc_str'].set('1+2')
        # Label(self.frame, text='Concatenate Sets').grid(row=self.row(), pady=10)
        # row = self.row()
        # Entry(self.frame, textvariable=self.config['conc_str'], width=6).grid(row=row, column=0, sticky=NE)
        # Button(self.frame, text='Join', command=self.conc_callback).grid(row=self.row(), sticky=N)

    # def conc_callback(self):
    #     os.chdir(self.parent.filelist.wdir)
    #     conc_str = self.config['conc_str'].get()
    #     tags = conc_str.split('+')
    #     if len(tags) != 2:
    #         print('Incorrect command. Type two sets separated with + (e.g. 1+2)')
    #         return -1
    #     ch = self.config['ch'].get()
    #     for prefix in self.parent.filelist.get_active()[1]:
    #         print(f'Joining {prefix} ROIs {conc_str}')
    #         da = Dual(prefix, tag=tags[0], ch=ch)
    #         da.concatenate(tags[1], ch=ch)

    def pull_callback(self):
        pflist = self.parent.filelist.get_active()[1]
        ch = self.config['ch'].get()
        # if self.config['roi'].get() == 'Use latest':
        #     roi = 'Auto'
        # else:
        roi = self.config['roi_name'].get()
        for prefix in pflist:
            self.parent.request_queue.put(('pull', (self.parent.filelist.wdir, prefix, roi, ch)))
            print(f'Pulling {prefix} ROI {roi} queued')

    def autosel_callback(self):
        wdir = self.parent.filelist.wdir
        newsel = []
        for i, prefix in enumerate(self.parent.filelist.prefix_list):
            if not os.path.exists(wdir + prefix + '_nonrigid.segment'):
                if os.path.exists(wdir + prefix + '_modrois.zip'):
                    newsel.append(i)
        self.parent.filelist.set_active(newsel)

    def autoroi_callback(self):
        p = self.parent
        p.assets.update()
        p.filelist.set_active(p.assets.get_list(p.filelist.prefix_list, 'roi-no-segment'))

    # def execute_callback(self):
    #     for prefix in self.parent.filelist.get_active()[1]:
    #         # print('Processing', prefix)
    #         RoiEditor(prefix).convert_roi_sbx()

    def sbx_convert_callback(self):
        wdir = self.parent.filelist.wdir
        for prefix in self.parent.filelist.get_active()[1]:
            print('Sbx export:', prefix, 'queued.')
            self.parent.request_queue.put(('sbxconvert', (wdir, prefix, True)))

    # def execute2_callback(self):
    #     for prefix in self.parent.filelist.get_active()[1]:
    #         # print('Processing', prefix)
    #         RoiEditor(prefix).save_roiset_ij_txt()

    def execute3_callback(self):
        commtxt = 'matlab -nosplash -nodesktop -r "extract_rois ' + self.parent.filelist.wdir[:-1] + '"'
        call(commtxt)

    def execute4_callback(self):
        commtxt = 'matlab -nosplash -nodesktop -r "sbxsegmenttool"'
        call(commtxt)

    def row(self, incr=1):
        self.current_row += incr
        return self.current_row - incr


# class Motion:
#     def __init__(self, parent, master, column):
#         self.parent = parent
#         self.master = master
#         self.current_row = 0
#
#         self.frame = Frame(master)
#         self.frame.grid(row=0, column=column, sticky=N + W)
#
#         Label(self.frame, text='Motion correction').grid(row=self.row(), pady=10)
#
#         Label(self.frame, text='Size limit (G)').grid(row=self.row())
#         self.slim = Scale(self.frame, from_=0, to=10, orient=HORIZONTAL)
#         self.slim.set(1)
#         self.slim.grid(row=self.row(), sticky=N)
#
#         self.autosel = Button(self.frame, text="Auto select", command=self.autosel_callback)
#         self.autosel.grid(row=self.row(), sticky=N)
#
#         Label(self.frame, text='Granularity').grid(row=self.row())
#         self.granularity = Scale(self.frame, from_=1, to=64, orient=HORIZONTAL)
#         self.granularity.set(8)
#         self.granularity.grid(row=self.row(), sticky=N)
#
#         Label(self.frame, text='Parallel files').grid(row=self.row())
#         self.cores = Scale(self.frame, from_=1, to=cpu_count(), orient=HORIZONTAL)
#         self.cores.set(2)
#         self.cores.grid(row=self.row(2), sticky=N)
#
#         Label(self.frame, text='Rigid steps').grid(row=self.row())
#         self.rigid_steps = Scale(self.frame, from_=0, to=3, orient=HORIZONTAL)
#         self.rigid_steps.set(3)
#         self.rigid_steps.grid(row=self.row(), sticky=N)
#
#         Label(self.frame, text='Max displacement').grid(row=self.row())
#         self.max_displacement = Scale(self.frame, from_=0, to=50, orient=HORIZONTAL)
#         self.max_displacement.set(40)
#         self.max_displacement.grid(row=self.row(), sticky=N)
#
#         self.opto_mode = IntVar()
#         Checkbutton(self.frame, text='Opto Mode', variable=self.opto_mode).grid(row=self.row(), sticky='W')
#         self.opto_mode.set(0)
#
#         Label(self.frame, text='Cutoff line').grid(row=self.row())
#         self.opto_cutoff = Scale(self.frame, from_=260, to=400, orient=HORIZONTAL)
#         self.opto_cutoff.set(260)
#         self.opto_cutoff.grid(row=self.row(), sticky=N)
#
#         self.ignore_sat = IntVar()
#         Checkbutton(self.frame, text='Ignore saturation', variable=self.ignore_sat).grid(row=self.row(), sticky='W')
#         self.ignore_sat.set(0)
#
#         Label(self.frame, text='Align based on:').grid(row=self.row())
#         MODES = ['Green', 'Red']
#         self.channels = StringVar()
#         self.channels.set('Green')
#         for r, text in enumerate(MODES):
#             Radiobutton(self.frame, text=text, variable=self.channels, value=text).grid(row=self.row(), sticky=N + W)
#
#         self.execute = Button(self.frame, text="Run", command=self.execute_callback)
#         self.execute.grid(row=self.row(), sticky=N)
#
#         Button(self.frame, text="Pull opto", command=self.opto_callback).grid(row=self.row(), sticky=N, pady=10)
#
#         Label(self.frame, text='Miniscope').grid(row=self.row(), pady=10)
#         Button(self.frame, text="Convert avi series", command=self.avi_callback).grid(row=self.row(), sticky=N)
#         Button(self.frame, text="Correct miniscope", command=self.miniscope_callback).grid(row=self.row(), sticky=N)
#
#     def avi_callback(self):
#         Process(target=convert_avi).start()
#
#     def miniscope_callback(self):
#         for prefix in self.parent.filelist.get_active()[1]:
#             print('Motion correction:', prefix, 'queued.')
#             self.parent.request_queue.put(('mini-mc', (self.parent.filelist.wdir, prefix, self.rigid_steps.get())))
#
#     def opto_callback(self):
#         self.parent.request_queue.put(('opto', (self.parent.filelist.wdir, self.parent.filelist.get_active()[1])))
#
#     def row(self, incr=1):
#         self.current_row += incr
#         return self.current_row - incr
#
#     def autosel_callback(self):
#         wdir = self.parent.filelist.wdir
#         sizelimit = self.slim.get() * (1024 ** 3)
#         newsel = []
#         for i, prefix in enumerate(self.parent.filelist.prefix_list):
#             if not os.path.exists(wdir + prefix + '_nonrigid.sbx'):
#                 if self.parent.filelist.sizevalues[i] > sizelimit:
#                     newsel.append(i)
#         self.parent.filelist.set_active(newsel)
#
#     def execute_callback(self):
#         g = int(self.granularity.get())
#         if g > 2:
#             while 512 % g > 0:
#                 g -= 1
#         if self.opto_mode.get():
#             optval = self.opto_cutoff.get()
#         else:
#             optval = 0
#         for prefix in self.parent.filelist.get_active()[1]:
#             print('Motion correction:', prefix, 'queued.')
#             self.parent.request_queue.put(('mc',
#                                            (self.parent.filelist.wdir, prefix, g, self.cores.get(),
#                                             self.channels.get(), self.rigid_steps.get(), self.max_displacement.get(),
#                                             optval, self.ignore_sat.get())))


class FileList:
    def __init__(self, parent, master, column):
        self.parent = parent
        self.master = master
        self.wdir = '/'
        self.wdir_text = StringVar()
        self.wdir_text.set('Current folder:' + self.wdir)

        self.frame = Frame(master)
        self.frame.grid(row=0, column=column, sticky=N + W)

        Label(self.frame, textvariable=self.wdir_text).grid(row=0, sticky=N + W)
        Button(self.frame, text='Select folder', command=self.getdir_callback).grid(row=1, sticky=N)

        # self.alphabet = IntVar()
        # Checkbutton(self.frame, text='Alphabetical', variable=self.alphabet).grid(row=1, column=1, sticky='E')
        # self.alphabet.set(1)

        self.listbox = Listbox(self.frame, selectmode=EXTENDED)
        self.listbox.grid(row=2)

        # self.dates = Listbox(self.frame)
        # self.dates.grid(column=1, row=2)

        # self.sizes = Listbox(self.frame)
        # self.sizes.grid(column=2, row=2)
        # self.sizevalues = []

    def getdir_callback(self):
        self.wdir = filedialog.askdirectory()
        self.wdir_text.set('Current folder:' + self.wdir)
        print(self.wdir_text.get())
        self.wdir += '//'
        os.chdir(self.wdir)
        self.parent.assets.update(self.wdir)
        self.prefix_list = []
        self.listbox.delete(0, END)
        pfs = self.parent.assets.get_prefixes()
        # times = []
        # sizes = []
        # for prefix in pfs:
        #     for suffix in ['.sbx', '_rigid.sbx', '_nonrigid.sbx', '.hdf5', '_rigid.hdf5', '.mat', '_nonrigid.mat',
        #                    '_rigid.mat']:
        #         fn = prefix + suffix
        #         if os.path.exists(fn):
        #             times.append(min(os.path.getctime(fn), os.path.getmtime(fn)))
        #             sizes.append(os.path.getsize(fn))
        #             break
        # if self.alphabet.get():
        #     ts = numpy.argsort(pfs)
        #     sorting = []
        #     for i in ts:
        #         sorting.insert(0, i)
        # else:
        #     sorting = numpy.argsort(times)
        #
        # for i in sorting:
        for i in range(len(pfs)):
            self.listbox.insert(0, pfs[i])
            self.prefix_list.insert(0, pfs[i])
            # self.dates.insert(0, time.ctime(times[i]))
            # self.sizes.insert(0, '%.2fG' % (sizes[i] / (1024 ** 3)))
            # self.sizevalues.insert(0, sizes[i])
        self.listbox.select_set(0, END)
        wlen = min(len(pfs), 30)
        self.listbox.config(height=wlen)
        maxlen = max([len(x) for x in pfs])
        self.listbox.config(width=maxlen+5)
        # self.dates.config(height=wlen)
        # self.dates.config(width=len(time.ctime(times[0])))
        # self.sizes.config(height=wlen)
        # self.sizes.config(width=8)

    def get_active(self):
        indices = self.listbox.curselection()
        return indices, [self.listbox.get(i) for i in indices]

    def set_active(self, newsel):
        self.listbox.select_clear(0, END)
        for i in newsel:
            self.listbox.select_set(i)


class play:
    def __init__(self, path, t):
        step = 2
        self.frame = 0
        self.zplane = 0
        tblen = 512
        f = LoadImage(path, t, explicit_need_data=False)
        single = (len(f.channels) == 1)
        nframes = f.nframes
        self.factor = tblen / nframes
        self.table = numpy.array([((i / 255.0) ** (1 / 1.8)) * 255 for i in numpy.arange(0, 256)]).astype('uint8')
        cv2.namedWindow('Movie')
        cv2.createTrackbar('Frame', 'Movie', 0, tblen, self.tbonChange)
        cv2.createTrackbar('Gamma', 'Movie', 33, 100, self.gammaonChange)
        if f.nplanes > 1:
            cv2.createTrackbar('ZPlane', 'Movie', 0, f.nplanes - 1, self.zonChange)
        while self.frame < f.nframes:
            if cv2.getWindowProperty('Movie', 0) < 0:
                break
            fr = numpy.zeros((*f.info['sz'], 3), dtype='uint8')
            d = f.get_frame(self.frame, zplane=self.zplane) / f.imdat.bitdepth * 256
            if single:
                fr[:, :, 1] = d.squeeze()
            else:
                fr[:, :, 1:] = d
            fr = cv2.LUT(fr, self.table)
            cv2.putText(fr, str(self.frame), (0, 40),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(128, 128, 128))
            cv2.imshow('Movie', fr)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            self.frame += step
            cv2.setTrackbarPos('Frame', 'Movie', int(self.frame * self.factor))
        cv2.destroyAllWindows()

    def tbonChange(self, v):
        self.frame = int(v / self.factor)

    def zonChange(self, v):
        self.zplane = v

    def gammaonChange(self, v):
        gamma = 1 + 3 * v / 100
        self.table = numpy.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in numpy.arange(0, 256)]).astype('uint8')


class play_eye:
    def __init__(self, path, t):
        os.chdir(path)
        step = 1
        self.frame = 0
        eye = h5py.File(t + '_eye.mat', 'r')['data']
        nframes = len(eye)
        cv2.namedWindow('Movie')
        tblen = 512
        self.factor = tblen / nframes
        cv2.createTrackbar('Frame', 'Movie', 0, tblen, self.tbonChange)
        while self.frame < nframes:
            if cv2.getWindowProperty('Movie', 0) < 0:
                break
            im = eye[self.frame, 0, :, :].transpose().astype('uint8').copy()
            cv2.putText(im, str(self.frame), (0, 40),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=128)
            cv2.imshow('Movie', im)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            self.frame += step
            cv2.setTrackbarPos('Frame', 'Movie', int(self.frame * self.factor))
        cv2.destroyAllWindows()

    def tbonChange(self, v):
        self.frame = int(v / self.factor)


class play_stack:
    def __init__(self, path, t, cfg):
        os.chdir(path)
        if cfg[-1] == 'saving':
            for prefix in t:
                LoadImage(prefix).create_zstack(*cfg[:-1], save=True)
        else:
            self.im = LoadImage(t)
            nframes = self.im.create_zstack(*cfg, save=True)
            self.frame = int(nframes / 2)
            cv2.namedWindow('Stack')
            cv2.createTrackbar('Slice', 'Stack', self.frame, nframes, self.tbonChange)
            cv2.imshow('Stack', self.im.get_slice(self.frame))
            while True:
                if cv2.getWindowProperty('Stack', 0) < 0:
                    break
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()

    def tbonChange(self, v):
        self.frame = cv2.getTrackbarPos('Slice', 'Stack')
        cv2.imshow('Stack', self.im.get_slice(self.frame))


class play_ephys:
    def __init__(self, path, prefix, channels=[1, 1], export=False):
        os.chdir(path)
        ch, n_channels = channels
        raw_shape = n_channels + 1
        ep_raw = numpy.fromfile(prefix + '.ephys', dtype='float32')
        n_samples = int(len(ep_raw) / raw_shape)
        ep_formatted = numpy.reshape(ep_raw, (n_samples, raw_shape))
        fps = 25
        self.fs = 10000
        self.trace = ep_formatted[:, ch]
        self.n = n_samples
        self.frame = 0
        self.sample = 0
        self.gain = 50
        self.speed = 10
        if export:
            pandas.DataFrame(self.trace).to_csv(prefix + '_ephys-trace.csv')
            print(prefix, 'ephys trace exported to csv')
        else:
            cv2.namedWindow('Trace')
            cv2.createTrackbar('Time', 'Trace', self.frame, int(self.n / self.fs), self.tbonChange)
            cv2.createTrackbar('Gain', 'Trace', self.gain, 100, self.gonChange)
            cv2.createTrackbar('Speed', 'Trace', self.speed, 100, self.sonChange)
            while True:
                key = cv2.waitKey(int(1000 / fps)) & 0xFF
                if cv2.getWindowProperty('Trace', 0) < 0:
                    break
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    self.frame += 1
                    self.sample += self.fs
                    cv2.setTrackbarPos('Time', 'Trace', self.frame)
                elif key == ord('a'):
                    self.frame -= 1
                    self.sample -= self.fs
                    if self.sample < 0:
                        self.sample = 0
                        self.frame = 0
                    cv2.setTrackbarPos('Time', 'Trace', self.frame)
                self.sample += int(self.speed / 10 * self.fs / fps)
                nf = int(self.sample / self.fs)
                if nf != self.frame:
                    self.frame = nf
                    cv2.setTrackbarPos('Time', 'Trace', self.frame)
                cv2.imshow('Trace', self.getgraph())
            cv2.destroyAllWindows()

    def tbonChange(self, v):
        self.frame = cv2.getTrackbarPos('Time', 'Trace')
        self.sample = self.frame * self.fs
        if self.sample + self.fs > self.n:
            self.frame = 0
            self.sample = 0
        cv2.imshow('Trace', self.getgraph())

    def gonChange(self, v):
        self.gain = cv2.getTrackbarPos('Gain', 'Trace')
        cv2.imshow('Trace', self.getgraph())

    def sonChange(self, v):
        self.speed = cv2.getTrackbarPos('Speed', 'Trace')

    def getgraph(self):
        h = 300
        l = 1000
        x = numpy.arange(l)
        frame = numpy.ones((2 * h, l, 3), dtype='uint8') * 255
        if not self.sample + self.fs < self.n:
            self.sample = 0
            self.frame = 0
            cv2.setTrackbarPos('Time', 'Trace', self.frame)
        y = self.trace[self.frame * self.fs:self.fs + self.frame * self.fs]
        y = decimate(y, 10)
        y = h - y * self.gain * h
        pts = numpy.vstack((x, y)).astype('int32').T
        cv2.polylines(frame, [pts], isClosed=False, color=(128, 128, 128))  # , thickness=1)#, lineType=cv2.LINE_AA)
        curr = int((self.sample % self.fs) / 10)
        pts = numpy.vstack((x[:curr], y[:curr])).astype('int32').T
        cv2.polylines(frame, [pts], isClosed=False, color=(255, 0, 0))  # , thickness=2, lineType=cv2.LINE_AA)
        return frame


if __name__ == '__main__':
    # sys.stdout = stdredir_main()
    scratch = 'C://MotionCorrect//'
    freeze_support()
    ncpu = 24  # cpu_count() #7 to be kind with i9
    try:
        set_start_method('spawn')
    except:
        pass
    if not os.path.exists(scratch):
        scratch_config = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '//scratch_path.txt'
        file_exists = os.path.exists(scratch_config)
        if file_exists:
            with open(scratch_config, 'r') as f:
                altp = f.read().strip()
                file_exists = os.path.exists(altp)
        if not file_exists:
            root = Tk()
            root.withdraw()
            messagebox.showinfo('Create a folder for temporary files (use an SSD)')
            altp = filedialog.askdirectory()
            with open(scratch_config, 'w') as f:
                f.write(altp)
                print(altp, 'was set for temporary files. You can change this later '
                            'by deleteing or editing scratch_path.txt in the repo.')
        scratch = altp
    request_queue = Queue()
    request_queue.cancel_join_thread()

    t1 = Process(target=App, args=(request_queue,))
    t1.start()
    print('Running.')
    mc_job_queue = Queue()
    cleanup_queue = Queue()
    result_queue = Queue()
    tr_job_queue = Queue()
    pull_queue = Queue()
    seg_job_queue = Queue()
    mini_request_queue = Queue()
    szdet_queue = Queue()
    mc_nworker = 0
    mini_mc_worker = 0
    cleanup_worker = False
    seg_nworker = 0
    pull_nworker = 0
    tr_nworker = 0
    szdet_nworker = 0
    for jobtype, job in iter(request_queue.get, None):
        if jobtype == 'mc':
            if not cleanup_worker:
                CleanupWorker(cleanup_queue).start()
                cleanup_worker = True
            path, prefix, g, cores, channels, rigid_steps, max_displacement, optomode, ignore_sat = job
            if mc_nworker < cores:
                mc_Worker(mc_job_queue, cleanup_queue).start()
                mc_nworker += 1
            if optomode:
                os.chdir(path)
                pullopto(prefix, path)
            mc_job_queue.put((path, scratch, prefix, g, channels, rigid_steps, max_displacement, optomode, ignore_sat))
        elif jobtype == 'mini-mc':
            path, prefix, passes = job
            if not os.path.exists(path + prefix + '_motion-crop.json'):
                os.chdir(path)
                CropCorrect(prefix).crop()
            if mini_mc_worker < 1:
                miniMcWorker(mini_request_queue).start()
                mini_mc_worker += 1
            mini_request_queue.put((path, prefix, passes))
        elif jobtype == 'pull':
            if pull_nworker < 1:
                pull_Worker(pull_queue).start()
                pull_nworker += 1
            pull_queue.put(job)
        elif jobtype == 'firing':
            path, prefix, bsltype, exclude, sz_mode, peakdet, tag = job
            os.chdir(path)
            run = False
            for ch in (0, ):
                a = CaTrace(path, prefix, bsltype=bsltype, exclude=exclude, peakdet=peakdet, ch=ch, tag=tag)
                print(a.pf)
                if a.open_raw() == -1:
                    continue
                if os.path.exists(a.pf):
                    print(f'{a.pf} folder exists, skipping')
                    continue
                run = True
                if sz_mode:
                    a.ol_index = []
                for c in range(a.cells):
                    if tr_nworker < ncpu:
                        tr_Worker(tr_job_queue, result_queue).start()
                        tr_nworker += 1
                    tr_job_queue.put(a.pack_data(c))
                for data in iter(result_queue.get, None):
                    finished = a.unpack_data(data)
                    if finished:
                        break
            # if a.channels == 2:
            #     if run or not os.path.exists(f'{prefix}-{tag}-dual.np'):
            #         Dual(prefix, tag).merge()
        elif jobtype == 'segment':
            if seg_nworker < 2:
                seg_Worker(seg_job_queue).start()
                seg_nworker += 1
            seg_job_queue.put(job)
        elif jobtype == 'movie':
            Process(target=play, args=job).start()
        elif jobtype == 'eye':
            Process(target=play_eye, args=job).start()
        elif jobtype == 'stack':
            Process(target=play_stack, args=job).start()
        elif jobtype == 'trace':
            Process(target=play_ephys, args=job).start()
        elif jobtype == 'lfp_overlay':
            Process(target=plot_overlay, args=job).start()
        elif jobtype == 'lfp_overlay_qa':
            Process(target=plot_overlay_qa, args=job).start()
        elif jobtype == 'SzDet':
            if szdet_nworker < 20:
                SzDet_Worker(szdet_queue).start()
                szdet_nworker += 1
            szdet_queue.put(job)
        elif jobtype == 'exportstop':
            print(job)
            # if job[-1] == 'm2':
            #     Process(target=calc_m2_index, args=job[:-1]).start()
            # else:
            Process(target=exportstop, args=job).start()
        elif jobtype == 'sbxconvert':
            Process(target=roi_Gui, args=job).start()
        elif jobtype == 'opto':
            path, pflist = job
            for prefix in pflist:
                os.chdir(path)
                pullopto(prefix, path)
        elif jobtype == 'exit':
            break
        else:
            print('Error in job type: ', jobtype)
