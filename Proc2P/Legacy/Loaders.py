from .Ripples import Ripples
def load_ephys(prefix, *args, **kwargs):
    return Ripples(prefix, load_minimal=True)