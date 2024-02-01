def prep_ax_psth(ca, w, seconds=None, xlabel='Time (s)', ylabel='DF/F (%)', show_x0=True, fps=20):
    ca.axvline(w, color='black', alpha=0.8, linestyle=':')
    if show_x0:
        ca.axhline(0, color='black', alpha=0.8, linestyle=':')
    ca.spines['right'].set_visible(False)
    ca.spines['top'].set_visible(False)
    xtk = (seconds * fps + w)
    ca.set_xticks(xtk)
    ca.set_xticklabels(seconds)
    ca.set_xlabel(xlabel)
    ca.set_ylabel(ylabel)

def prep_ax_graph(ca, xlabel='Categories', ylabel='DF/F (%)', show_x0=True,):
    if show_x0:
        ca.axhline(0, color='black', alpha=0.8, linestyle=':')
    ca.spines['right'].set_visible(False)
    ca.spines['top'].set_visible(False)
    ca.set_xlabel(xlabel)
    ca.set_ylabel(ylabel)

def strip_ax(ca, full=True):
    ca.spines['right'].set_visible(False)
    ca.spines['top'].set_visible(False)
    if full:
        ca.spines['bottom'].set_visible(False)
        ca.spines['left'].set_visible(False)
        ca.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='off')
        ca.tick_params(axis='y', which='both', right='off', left='off', labelright='off')
        ca.xaxis.set_visible(False)
        ca.yaxis.set_visible(False)
