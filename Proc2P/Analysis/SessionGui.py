import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.widgets import Slider, Button, TextBox
from Proc2P.Bruker.ConfigVars import CF
from Proc2P.Analysis.ImagingSession import ImagingSession
import copy

import pandas
import numpy


class Gui(ImagingSession):

    def start(self, param=None, skin='light'):
        print('Called with param = ' + str(param))
        if skin == 'dark':
            lg = '#babaa3'
            self.colors = {'bgcolor': '#2b2b2b',
                           'slidercolor': '#434e60',
                           'wcolor': '#404040',
                           'axlabel': '790000',
                           'speed': lg,
                           'opto': 'dodgerblue',
                           'run_1': lg,
                           'event_2': '#ff553f',
                           'opto_1': 'lime',
                           'rewardzone': '#e2efda',
                           'rippletick': '#fffa48',
                           'legendalpha': 0.1,
                           'trace-g': '#436fb6'}
        if skin == 'light':
            self.colors = {'bgcolor': '#eef2f5',
                           'slidercolor': '#2dc2df',
                           'wcolor': '#ffffff',
                           'axlabel': '424c58',
                           'speed': '#cc6699',
                           'opto': 'dodgerblue',
                           'run_1': '#d671ad',
                           'event_2': '#aed361',
                           'opto_1': '#fdb64e',
                           'rewardzone': '#33c4b3',
                           'rippletick': 'black',
                           'legendalpha': 0.8,
                           'trace-g': '#436fb6'}
        param = numpy.nan_to_num(self.getparam(param))
        if self.dualch:
            self.colors['trace-g'] = '#71c055'
            self.colors['trace-r'] = '#ed1e24'
            self.colors['run_1'] = '#5b52a3'
            self.colors['speed'] = '#5b52a3'
            graph_param = param[..., 0]
        else:
            graph_param = param
        self.activecell = 0
        self.graph_param = graph_param
        if hasattr(self.pos, 'laps') and self.pos.laps[-1] > 2:
            self.using_laps = True
            self.placefields(param=graph_param, span=(100, self.ca.frames - 100), silent=True)
            self.calc_MI(param=graph_param, selection='movement')
        else:
            self.using_laps = False
        self.runspeed(param=graph_param, span=(100, self.ca.frames - 100))
        # if hasattr(self, 'bdat'):
        #     self.licktrigger(param=graph_param)
        if hasattr(self, 'ripples'):
            self.rippletrigger()
        self.time_graph(param=param)
        #TODO here
        self.overview()
        self.cell_update()

    def export_cells(self, cs):
        for c in cs:
            self.ncell.set_val(c)
            self.timefig.savefig(self.prefix + '_cell' + str(c) + '.png', dpi=600)

    def runspeed(self, param=None, bins=5, binsize=5, span=None):
        param = self.getparam(param)
        if span is None:
            span = (0, self.ca.frames)
        # find time points for each bin
        self.bin = numpy.empty((bins + 1, len(self.pos.pos)), dtype='bool')
        self.bin[0] = numpy.invert(self.pos.movement)
        for i in range(bins):
            self.bin[i + 1] = (self.pos.speed > i * binsize) * (self.pos.speed < (i + 1) * binsize) * self.pos.movement
        zscores = self.pull_means(param, span)
        self.speedrates = copy.copy(self.rates[self.pltnum])

    def overview(self):
        # show overview of field
        bgcolor = self.colors['bgcolor']
        slidercolor = self.colors['slidercolor']
        wcolor = self.colors['wcolor']
        im = self.get_preview()
        fig, ax = plt.subplots(facecolor=wcolor)
        '''rect = [left, bottom, width, height]'''
        plt.subplots_adjust(left=0.25, bottom=0.25)
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        plt.tick_params(axis='y', which='both', right='off', left='off', labelleft='off')
        axprev = plt.axes([0.025, 0.4, 0.2, 0.2], facecolor=bgcolor)
        axprev.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        axprev.tick_params(axis='y', which='both', right='off', left='off', labelleft='off')
        axn = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=slidercolor)
        axb_minus = plt.axes([0.2, 0.21, 0.1, 0.03], facecolor=slidercolor)
        axb_plus = plt.axes([0.4, 0.21, 0.1, 0.03], facecolor=slidercolor)
        axb_select = plt.axes([0.7, 0.21, 0.1, 0.03], facecolor=slidercolor)
        ax.imshow(im, cmap='magma')
        # draw rois
        self.ps = []
        for c in range(self.ca.cells):
            self.ps.append(Polygon(self.rois.polys[c]))
        self.pc = PatchCollection(self.ps, edgecolor='white', cmap='hot')
        if not self.using_laps:
            # add cells colored by snr
            if not self.dualch:
                self.pc.set_array(numpy.nanpercentile(self.ca.ntr, 99, axis=0))
            else:
                self.pc.set_array(numpy.arange(self.ca.cells))
        else:
            # add cells colored by mutual information
            self.pc.set_array(self.mi)
        ax.add_collection(self.pc)
        plt.colorbar(self.pc, ax=ax)
        # add cell selector slider
        self.ncell = Slider(axn, 'Cell', 0, self.ca.cells, valinit=0, valfmt='%1d')
        self.nextcell = Button(axb_plus, 'Next', color=slidercolor, hovercolor=bgcolor)

        def callback_plus(event):
            self.ncell.set_val(min(self.ca.cells, self.activecell + 1))

        self.nextcell.on_clicked(callback_plus)
        self.prevcell = Button(axb_minus, 'Previous', color=slidercolor, hovercolor=bgcolor)

        def callback_minus(event):
            self.ncell.set_val(max(0, self.activecell - 1))

        self.prevcell.on_clicked(callback_minus)
        self.pickcell = TextBox(axb_select, 'Set cell:', color=slidercolor, hovercolor=bgcolor, initial='   ')

        def callback_pick(text):
            self.ncell.set_val(int(text))

        self.pickcell.on_submit(callback_pick)
        # add selected cell view
        self.currpatch = copy.copy(self.ps[0])
        ax.add_patch(self.currpatch)
        self.cellprevs = {}

        def update(val):
            self.activecell = int(self.ncell.val)
            self.currpatch.remove()
            self.currpatch = copy.copy(self.ps[self.activecell])
            ax.add_patch(self.currpatch)
            if not self.activecell in self.cellprevs:
                self.add_preview()
            axprev.imshow(self.cellprevs[self.activecell], cmap='magma')
            fig.canvas.draw_idle()
            # if hasattr(self, 'timefig'):
            self.cell_update(0)
            self.pickcell.set_val(str(self.activecell))
            self.pickcell.stop_typing()

        def button_press_callback(event):
            if event.inaxes == ax:
                x, y = event.xdata, event.ydata
                for c in range(self.ca.cells):
                    if self.ps[c].contains_point([x, y]):
                        self.ncell.set_val(c)
                        break

        cid = fig.canvas.mpl_connect('button_press_event', button_press_callback)
        self.ncell.on_changed(update)
        fig.show()
        self.overviewfig = fig

    def add_preview(self):
        self.cellprevs[self.activecell] = self.rois.show_cell(self.activecell)

    def savefunction(self, event):
        c = self.activecell
        df = pandas.DataFrame()
        df['Frame'] = numpy.arange(0, self.ca.frames)
        df['DF/F (%)'] = self.ca.rel[c] * 100
        df['DF/F (z)'] = self.ca.ntr[c]
        df['Plotted values'] = self.graph_param[c]
        df['Speed'] = self.pos.speed
        df['Running.State'] = self.pos.gapless
        if hasattr(self, 'ripples'):
            r = numpy.zeros(self.ca.frames, dtype=bool)
            for t in self.ripple_frames:
                r[t] = 1
            df['Ripple'] = r
        df.to_excel(self.prefix + f'_trace_c{c}.xlsx')

    def time_graph(self, param=None):
        if param is None:
            param = self.ca.event
        bgcolor = self.colors['bgcolor']
        slidercolor = self.colors['slidercolor']
        wcolor = self.colors['wcolor']
        # initial values
        self.active_frame = int(self.ca.frames / 2)
        self.active_zoom = 1
        self.playing = False
        self.first_render = True
        # time plots:
        fig, axspike = plt.subplots(facecolor=wcolor)
        mng = plt.get_current_fig_manager()
        plt.subplots_adjust(left=0.05, bottom=0.8, right=0.75, top=0.95)
        axspike.tick_params(axis='y', which='both', right='off', left='off', labelleft='off')
        axspike.tick_params(axis='x', which='both', bottom='off', top='on', labelbottom='off', labeltop='on')
        axspike.set_facecolor(bgcolor)
        axspike.xaxis.label.set_color(self.colors['axlabel'])
        axtrace = plt.axes([0.05, 0.6, 0.7, 0.15], facecolor=bgcolor, sharex=axspike)
        axspeed = plt.axes([0.05, 0.35, 0.7, 0.15], facecolor=bgcolor, sharex=axspike)
        self.axspeed = axspeed
        axpos = plt.axes([0.05, 0.15, 0.7, 0.15], facecolor=bgcolor, sharex=axspike)
        self.axpos = axpos
        # cell plots:
        axrun = plt.axes([0.8, 0.8, 0.15, 0.15], facecolor=bgcolor, title='Rate vs speed')
        axlap = plt.axes([0.8, 0.6, 0.15, 0.15], facecolor=bgcolor, title='Rate by lap')
        axrpl = plt.axes([0.8, 0.4, 0.15, 0.15], facecolor=bgcolor, title='Rate vs ephys')
        # axlck = plt.axes([0.8, 0.4, 0.15, 0.15], facecolor=bgcolor, title='Rate vs lick')
        # axpol = plt.axes([0.8, 0.2, 0.15, 0.18], facecolor=bgcolor, projection='polar')
        # axpol.grid(False)

        # slider axes
        # axframe = plt.axes([0.1, 0.05, 0.6, 0.03], facecolor=slidercolor)
        # axzoom = plt.axes([0.8, 0.1, 0.15, 0.03], facecolor=slidercolor)
        # axbutton_play = plt.axes([0.8, 0.05, 0.15, 0.03], facecolor=bgcolor)
        axbutton_reset = plt.axes([0.8, 0.15, 0.15, 0.03], facecolor=bgcolor)
        axbutton_save = plt.axes([0.8, 0.05, 0.15, 0.03], facecolor=bgcolor)
        # remove ticks
        for ax in [axtrace, axspeed, axpos, axlap, axrun, axrpl,
                   axbutton_reset]:  # , axframe, axzoom, axbutton_play]:
            ax.tick_params(axis='y', which='both', right='off', left='off', labelleft='off')
            ax.tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off')

        # plot common data
        self.behavplot(axpos)
        rate_trim = numpy.nanmean(param, 0)
        rate_trim[:100] = 0
        rate_trim[-100:] = 0
        ripcol = self.colors['rippletick']
        rateval = numpy.array(pandas.DataFrame(rate_trim).ewm(span=15).mean())
        rateval -= rateval.min()
        rateval /= rateval.max()
        # axspeed.plot(rateval, color='#ff553f', label='rate')
        # # add eye plot
        # eyefn = self.prefix + '_eye.np'
        # if os.path.exists(eyefn):
        #     eye = numpy.fromfile(eyefn, dtype=numpy.float32)
        #     if round(len(eye) / len(self.pos.smspd)) == 2:
        #         l = int(len(eye) / 2)
        #         eye1 = numpy.empty(l)
        #         for i in range(l):
        #             eye1[i] = eye[i * 2] + eye[i * 2 + 1]
        #         eye = eye1 / 2
        #     eyesm = pandas.DataFrame(eye).ewm(span=15).mean()
        #     eyesm -= eyesm.min()
        #     eyesm /= eyesm.max()
        #
        #     self.eyesm = eyesm
        #     axspeed.plot(eyesm, label='eye')
        speed_trim = copy.copy(self.pos.speed)
        speed_trim[:10] = 0
        spdval = numpy.array(pandas.DataFrame(speed_trim).ewm(span=15).mean())
        spdval /= spdval.max()
        axspeed.plot(spdval, color=self.colors['speed'], label='speed', alpha=0.8)
        if self.opto is not None:
            axspeed.plot(self.opto, color=self.colors['opto'], label='light', alpha=0.8)
        if hasattr(self, 'ripples'):
            for t in self.ripple_frames:
                axspeed.axvline(t - 0.5, color=ripcol)
            if hasattr(self.ripples, 'theta'):
                rpsm = pandas.DataFrame(self.theta_power).ewm(span=3).mean()
                rpsm -= rpsm.min()
                rpsm /= rpsm.max()
                axspeed.plot(rpsm, label='ThetaPower', alpha=0.8)
            rpsm = pandas.DataFrame(self.ripple_power).ewm(span=3).mean()
            rpsm -= rpsm.min()
            rpsm /= rpsm.max()
            axspeed.plot(rpsm, label='RipplePower', alpha=0.8)
        if self.eye is not None:
            axspeed.plot(self.eye, color='blue', alpha=0.8, label='eye')
        # plot cell plots. instead of set_ydata use clear for update. drawing statements that would be repeated moved to update function
        # l_pol, = axpol.plot(numpy.arange(0, 6.28319, 6.28319 / 360), self.polar(self.graph_param))
        if not self.dualch:
            axrun.plot(numpy.nanmean(self.speedrates, axis=1), label='mean')
            l_run, = axrun.plot(self.speedrates[:, self.activecell], label='cell')
            if hasattr(self, 'ripples'):
                l_rpl, = axrpl.plot(self.ripplerates[:, self.activecell], label='Ripple')
            if self.using_laps:
                axlap.imshow(self.perlap_fields[:, self.activecell, :].transpose(), cmap='inferno_r', vmin=0)
            # for ax in [axspike, axtrace, axspeed, axpos]:
            #     ax.myvline = ax.axvline(self.active_frame, color='#790000')
            for ax in [axspeed, axrun, axrpl]:
                ax.legend(loc='upper right', framealpha=self.colors['legendalpha'])
            # axpol.myvline = axpol.axvline(-self.pos.relpos[self.active_frame] * 6.28319, color='#790000')
            axrpl.myvline = axrpl.axvline(CF.fps, color='black')

        # update active cell
        xvals = numpy.arange(self.ca.frames)
        wh_run = numpy.logical_not(self.pos.movement[:self.ca.frames])

        def replot(val=None):
            xlims = axspike.get_xlim()
            self.active_time_window = xlims
            ylims = axspike.get_ylim()
            axspike.clear()
            axtrace.clear()
            if not self.dualch:
                plot_y_val = param[self.activecell]
                axspike.plot(plot_y_val, label='rest', zorder=1)
                if not numpy.all(numpy.isnan(plot_y_val[wh_run])) and numpy.any(plot_y_val[wh_run] > 0):
                    axspike.plot(numpy.ma.masked_where(wh_run, plot_y_val), label='run',
                                 color=self.colors['run_1'], zorder=2)
                plot_y_val = self.ca.rel[self.activecell]
                # wh_active = numpy.logical_not(self.ca.event[self.activecell])
                # if numpy.nansum(wh_active) > 2:
                #     axtrace.plot(self.ca.rel[self.activecell], label='trace', zorder=1)
                #     if not numpy.all(numpy.isnan(plot_y_val[wh_active])) and numpy.any(plot_y_val[wh_active] > 0):
                #         axtrace.plot(numpy.ma.masked_where(wh_active, self.ca.event[self.activecell]), label='event',
                #                      color=self.colors['event_2'], zorder=2)
            else:
                for ch_index, (ch_color, color_tag) in enumerate(zip(('Green', 'Red'), ('trace-g', 'trace-r'))):
                    plot_y_val = param[self.activecell, :, ch_index]
                    axtrace.plot(plot_y_val, label=ch_color, color=self.colors[color_tag], zorder=1)
                    # axspike.plot(plot_y_val, label=ch_color, color=self.colors[color_tag], zorder=1)
                    # if not numpy.all(numpy.isnan(plot_y_val[wh_run])) and numpy.any(plot_y_val[wh_run] > 0):
                    #     axspike.plot(numpy.ma.masked_where(wh_run, plot_y_val), label='run',
                    #                  color=self.colors['run_1'], zorder=2)

            if self.opto is not None:
                plot_y_val = param[self.activecell]
                wh_opto = self.opto
                if numpy.nansum(wh_opto) > 2:
                    if not numpy.all(numpy.isnan(plot_y_val[wh_opto])) and numpy.any(plot_y_val[wh_opto] > 0):
                        axspike.fill_between(xvals, numpy.nanmin(plot_y_val), plot_y_val,
                                             where=wh_opto, facecolor=self.colors['opto'], interpolate=True, alpha=0.8,
                                             label='stim')
            if hasattr(self, 'ripples'):
                for t in self.ripple_frames:
                    axtrace.axvline(t - 0.5, color=ripcol)
                    axspike.axvline(t - 0.5, color=ripcol)
            # for ax in [axspike, axtrace]:
            #     ax.legend(loc='upper right', framealpha=self.colors['legendalpha'])
            # l_pol.set_ydata(self.polar(self.graph_param))
            if not self.dualch:
                l_run.set_ydata(self.speedrates[:, self.activecell])
                # if hasattr(self, 'bdat'):
                #     l_lck.set_ydata(self.lickrates['all'][:, self.activecell])
                if hasattr(self, 'ripples'):
                    l_rpl.set_ydata(self.ripplerates[:, self.activecell])
                    rrsp = int(numpy.nanmean(self.ripplerates[15:18, self.activecell]) * 100
                               / numpy.nanmean(self.ripplerates[0:15, self.activecell]))
                    axrpl.set_title(f'Ripple: {rrsp}%')
                if self.using_laps:
                    axlap.imshow(self.perlap_fields[:, self.activecell, :].transpose())
                if self.first_render:
                    self.first_render = False
                    autoscale()
                else:
                    axspike.set_xlim(xlims)
                    axspike.set_ylim(ylims)
            # fig.canvas.update()
            # fig.canvas.flush_events()
            # fig.canvas.draw_idle()
            fig.show()
            self.overviewfig.show()

        self.cell_update = replot

        # sll1 = axframe.axvline(0, color='black')
        # sll2 = axframe.axvline(self.ca.frames, color='black')

        # update pos with active conditions:
        def update(val):
            pass
            # self.active_frame = int(self.frame_slider.val)
            # # axroi.imshow(self.rois.get_cell(self.active_frame, self.activecell), cmap='magma')
            # self.active_zoom = int(self.zoom_slider.val)
            # try:
            #     l = int(min(self.ca.frames, max(100, self.ca.frames / self.active_zoom)))
            #     start = int(max(0, self.active_frame - l / 2))
            #     stop = int(min(self.ca.frames, self.active_frame + l / 2, self.ca.frames))
            #     axspike.set_xlim(start, stop)
            #     sll1.set_xdata(start)
            #     sll2.set_xdata(stop)
            #     # for ax in [axspike, axtrace, axspeed, axpos]:
            #     #     ax.myvline.set_xdata(self.active_frame)
            #     # ax.draw_artist(ax.myvline)
            #     axpol.myvline.set_xdata(-self.pos.relpos[self.active_frame] * 6.28319)
            #     # optimized redrawing:
            #     # for ax in [axspike, axtrace, axspeed, axpos]:
            #     #     for artist in ax.get_children():
            #     #         if type(artist).__name__=='Line2D':
            #     #             ax.draw_artist(artist)
            #     fig.canvas.update()
            #     fig.canvas.flush_events()
            #     fig.canvas.draw_idle()
            # except:
            #     pass


        def resetzoom(event):
            axspike.set_xlim(0, self.ca.frames)
            autoscale()

        def autoscale():
            for ax in [axspike, axtrace, axrun,]:
                ax.autoscale(True, axis='y')
                ax.relim()
                ax.autoscale_view(True, True, True)

        # def toggle_play(event):
        #     if stopFlag.is_set():
        #         stopFlag.clear()
        #         thread = PlayBack(stopFlag, self.frame_slider)
        #         thread.start()
        #     else:
        #         stopFlag.set()

        # add controls
        # self.frame_slider = Slider(axframe, 'Frame', 0, self.ca.frames, valinit=int(self.ca.frames / 2), valfmt='%1d')
        # self.frame_slider.on_changed(update)
        # self.frame_slider.vline.remove()
        # self.zoom_slider = Slider(axzoom, 'Zoom', 1, 100, valinit=self.ca.frames / (self.ca.frames - 200), valfmt='%1d')
        # self.zoom_slider.on_changed(update)
        # self.zoom_slider.vline.remove()
        self.reset_zoom_button = Button(axbutton_reset, 'Reset', color=slidercolor, hovercolor=bgcolor)
        self.reset_zoom_button.on_clicked(resetzoom)
        self.save_button = Button(axbutton_save, 'Save', color=slidercolor, hovercolor=bgcolor)
        self.save_button.on_clicked(self.savefunction)
        # self.play_button = Button(axbutton_play, 'Play', color=slidercolor, hovercolor=bgcolor)
        # stopFlag = threading.Event()
        # stopFlag.set()
        # self.play_button.on_clicked(toggle_play)

        # update lims from interactive pan and zoom:
        # def button_release_callback(event):
        #     if event.inaxes in [axspike, axtrace, axspeed, axpos]:
        #         x1, x2 = axspike.get_xlim()
        #         # z = self.ca.frames / (x2 - x1)
        #         self.frame_slider.set_val(0.5 * (x2 + x1))
        #         self.zoom_slider.set_val(self.ca.frames / (x2 - x1))

        # cid = fig.canvas.mpl_connect('button_release_event', button_release_callback)
        self.timefig = fig
        fig.show()
        mng.window.state('zoomed')

    # def polar(self, param):
    #     pol = numpy.zeros(360)
    #     for f in range(self.ca.frames):
    #         v = param[self.activecell, f]
    #         if v and self.pos.movement[f]:
    #             pol[int((1 - self.pos.relpos[f]) * 359)] += v
    #     # x = numpy.arange(0, 6.28319, 6.28319 / 360)
    #     return numpy.abs(scipy.signal.hilbert(pol))


    def behavplot(self, ax):
        ax.plot(self.pos.pos, color='grey')
        if hasattr(self, 'bdat'):
            # add reward zones to plot
            rz = self.bdat.data[2]
            td = self.bdat.data[0]
            l = len(self.pos.pos)
            m = max(self.pos.pos)
            x = numpy.empty(l)
            for f in range(l):
                bf = numpy.where(td <= self.frametotime(f))[0]
                if len(bf) > 0 and rz[bf[-1]]:
                    x[f] = m * 100
                else:
                    x[f] = -m * 100
            ax.fill_between(range(l), -m * 100, x, color=self.colors.get('rewardzone', '#e2efda'))
            ax.plot(self.pos.pos, color='grey')
            # add licks to plot
            # x, y = [], []
            # for l in self.bdat.choices[0]:
            #     if l < self.bmtime[-1] and l > self.bmtime[0]:
            #         t = self.timetoframe(l)
            #         x.append(t)
            #         y.append(self.pos.pos[t])
            # ax.scatter(x, y, marker="o", color='green', s=50)
            # x, y = [], []
            # for l in self.bdat.choices[1]:
            #     if l < self.bmtime[-1] and l > self.bmtime[0]:
            #         t = self.timetoframe(l)
            #         if t < self.ca.frames:
            #             x.append(t)
            #             y.append(self.pos.pos[t])
            # ax.scatter(x, y, marker="|", color='red', s=50)
            # x, y = [], []
            # for l in self.bdat.rewards:
            #     if l < self.bmtime[-1] and l > self.bmtime[0]:
            #         t = self.timetoframe(l)
            #         if t < self.ca.frames:
            #             x.append(t)
            #             y.append(self.pos.pos[t])
            # ax.scatter(x, y, marker="|", color='blue', s=25)
            # # plot extra pin info
            # x1, x2, y1, y2 = [], [], [], []
            # for pin, l, is_open in self.bdat.other_events:
            #     if l < self.bmtime[-1] and l > self.bmtime[0]:
            #         t = self.timetoframe(l)
            #         if t < self.ca.frames:
            #             if is_open:
            #                 x1.append(t)
            #                 y1.append(self.pos.pos[t])
            #             else:
            #                 x2.append(t)
            #                 y2.append(self.pos.pos[t])
            # ax.scatter(x1, y1, marker="<", color='black', s=50)
            # ax.scatter(x2, y2, marker=">", color='black', s=50)
            # add tones to plot
            # tones = self.get_tones()
            # for t in tones:
            #     ax.axvline(t, color='red')

            ax.set_ylim(-m * 0.1, m * 1.1)

    def plot_session(self, param=None, offset=None, scatter=False, spec=[], hm=True, corder=None, hlines=None,
                     riplines=False, cmap='hot', rate='mean', silent=False, axtitle=None, unit='frames', vmax=None):
        self.pltsessionplot = plt.subplots(4, 1, gridspec_kw={'height_ratios': [4, 1, 1, 1]}, sharex=True)
        fig, (axf, axr, axs, axp) = self.pltsessionplot
        param = self.getparam(param)
        if corder is None:
            corder = range(self.ca.cells)
        self.plotsession_corder = corder
        if hm:
            sortedhm = numpy.empty((len(corder), self.ca.frames))
            for i, c in enumerate(corder):
                if self.disc_param:
                    sortedhm[i] = param[c]
                else:
                    if numpy.nanmax(param[c]) > 0:
                        sortedhm[i] = param[c] / numpy.nanstd(param[c])
                    else:
                        sortedhm[i] = param[c]
            if not hasattr(self.ca, 'version_info'):
                sortedhm -= max(0, numpy.nanmin(sortedhm))
            elif self.ca.version_info['bsltype'] == 'original':
                sortedhm -= max(0, numpy.nanmin(sortedhm))
            if vmax is None:
                sortedhm /= numpy.nanmax(sortedhm)
                vmax = 0.9
            axf.imshow(sortedhm, aspect='auto', cmap=cmap, vmin=0, vmax=vmax)
            if hlines is not None:
                y = 0
                for i in hlines:
                    y += i
                    axf.axhline(y - 0.5, color='black', linewidth=0.5)
        else:
            if offset is None:
                offset = -numpy.nanstd(param)
            for i, c in enumerate(corder):
                if c in spec:
                    continue
                if scatter:
                    x = numpy.nonzero(param[c])
                    axf.scatter(x, [i * offset] * len(x[0]), marker="|")
                else:
                    axf.plot(param[c] + i * offset)
        if rate == 'spikes':
            r = numpy.copy(self.ca.rate)
        elif rate == 'mean':
            r = numpy.nanmean(param, axis=0)
        r[:5] = numpy.nanmean(r[5:10])
        axr.plot(r, color='black')
        for c in spec:
            axr.plot(param[c] / numpy.nanmax(param[c]), alpha=0.8 / len(spec))
        self.behavplot(axp)

        if unit == 'frames':
            axp.set_xlabel('Frame')
        else:
            try:
                unit = int(unit)
            except:
                unit = 60
            axp.set_xlabel('Seconds')
            t0 = int(self.frametotime(0))
            seconds = numpy.array(range(t0, int(self.frametotime(self.ca.frames - 1)), unit))
            axp.set_xticklabels(seconds - t0)
            axp.set_xticks([self.timetoframe(x) for x in seconds])

        # plot ripples if available:
        if hasattr(self, 'ripples'):
            if riplines:
                for t in self.ripple_frames:
                    axs.axvline(t - 0.5, color='black')
                    axr.axvline(t - 0.5, color='black')
                    axf.axvline(t - 0.5, color='black')
            if hasattr(self.ripples, 'theta'):
                rpsm = pandas.DataFrame(self.theta_power).ewm(span=3).mean()
                rpsm -= rpsm.min()
                rpsm /= rpsm.max()
                axs.plot(rpsm, label='ThetaPower', alpha=0.8)
            rpsm = pandas.DataFrame(self.ripple_power).ewm(span=3).mean()
            rpsm -= rpsm.min()
            rpsm /= rpsm.max()
            axs.plot(rpsm, label='RipplePower', alpha=0.8)

        # plot eye if available
        if self.eye is not None:
            axs.plot(self.eye, color='blue', alpha=0.8, label='Eye')

        # plot opto if available
        if self.opto is not None:
            axs.plot(self.opto, color='dodgerblue', label='light', alpha=0.8)


        # plot speed
        speed_trim = copy.copy(self.pos.speed)
        speed_trim[:10] = 0
        if speed_trim.max() > 0:
            spdval = numpy.array(pandas.DataFrame(speed_trim).ewm(span=15).mean())
            spdval /= spdval.max()
        else:
            spdval = speed_trim
        axs.plot(spdval, label='Speed')
        # for t in self.get_tones():
        #     axs.axvline(t, color='red')

        axs.legend(loc='upper right', framealpha=0.1)

        axf.yaxis.set_ticklabels([])
        axp.yaxis.set_ticklabels([])
        axs.yaxis.set_ticklabels([])
        axr.yaxis.set_ticklabels([])
        if axtitle is None:
            axtitle = u"\N{GREEK CAPITAL LETTER DELTA}" + 'F/F'
        axf.set_ylabel(axtitle)
        axp.set_ylabel('Position')
        axs.set_ylabel('Speed')
        axr.set_ylabel('Session Mean')
        if not silent:
            fig.show()

            def button_press_callback(event):
                if event.inaxes == axf:
                    cell_index = self.plotsession_corder[int(event.ydata + 0.5)]
                    stamp = f'Cell:{cell_index}, Frame:{int(event.xdata - 0.5)}'
                    if hasattr(self, 'sync_spline'):
                        stamp += f', Time:{self.frametotime(event.xdata) - int(self.frametotime(0)):.2f}'
                    if hasattr(self, 'ripples'):
                        stamp += f', EPhys:{self.frametosample(int(event.xdata - 0.5))[0] / self.ripples.fs:.2f}'
                    print(stamp)

            fig.canvas.mpl_connect('button_press_event', button_press_callback)
        return fig


if __name__ == '__main__':
    path = r'D:\Shares\Data\_Processed\2P\SncgDREADD/'
    prefix = 'SncgTot11_2023-11-20_Movie_002'
    tag = '1'
    a = Gui(path, prefix, tag=tag, ch=0)
    # plt.plot(a.pos.speed, color='black')
    # plt.plot(a.ca.smtr[0])
    # plt.imshow(a.ca.smtr, aspect='auto')
    # a.start()

    fig = a.plot_session(param='smtr', cmap='inferno_r',
                                         scatter=False,
                                         hm=True, rate='mean', riplines=False,
                                         axtitle='DF/F', unit='frames')


