from matplotlib.colors import ListedColormap
import numpy

custom_cmaps = {}

bl = numpy.zeros(256)
wh = numpy.ones(256)
alpha = numpy.ones(256)
lin = numpy.linspace(0, 1, 256)
mid = numpy.zeros(256)
mid[:128] = numpy.linspace(0, 1, 128)
mid[128:] = numpy.linspace(1, 0, 128)
left = numpy.zeros(256)
left[:128] = numpy.linspace(1, 0, 128)
right = numpy.zeros(256)
right[128:] = numpy.linspace(0, 1, 128)



custom_cmaps['g'] = ListedColormap(numpy.array([bl, lin, bl, alpha]).transpose())
custom_cmaps['r'] = ListedColormap(numpy.array([lin, bl, bl, alpha]).transpose())

gamma = 1.0 / 2.2
custom_cmaps['ga'] = ListedColormap(numpy.array([bl, lin**gamma, bl, alpha * 0.8]).transpose())
custom_cmaps['ra'] = ListedColormap(numpy.array([lin**gamma, bl, bl, alpha * 0.8]).transpose())

gamma = 1.0 / 0.8
custom_cmaps['redblackblue'] = ListedColormap(numpy.array([left**gamma, bl, right**gamma, alpha]).transpose())