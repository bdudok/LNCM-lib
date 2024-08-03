def mpl_color(key):
    return color(key, 1)
def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def get_color(key, skin='plots'):
    return color(skin + ':' + key, 1)

def color(val, convert=None):
    if type(val) is tuple:
        if len(val) == 3:
            if convert is None:
                return val
            elif convert == 255:
                return tuple([int(i * 255) for i in val])
            elif convert == 1:
                return tuple([float(i) / 255 for i in val])
            elif convert == 'bgr':
                return val[::-1]
            else:
                raise ValueError(f'Conversion not understood: {convert}')
        else:
            raise TypeError('Define colors as 3-tuples of int 1-255. Got:', val)
    elif type(val) is str:
        if val[0] == '#':
            c = hex_to_rgb(val)
            if convert is None or convert == 255:
                return c
            elif convert in (1, 'bgr'):
                return color(c, convert=convert)
        else:
            return color(clib(val), convert=convert)


def clib(key, skin=None):
    '''get named colors. if skin not specified, default is retrieved. str: 'skin:key' format is also parsed'''
    if skin is None:
        if ':' in key:
            skin, key = key.split(':')
        else:
            skin = 'default'
    if skin == 'default':
        colors = {'g': '#71c055',
                  'r': '#ed1e24',
                  'v': '#5b52a3'}
    elif skin == 'dark':
        colors = {'bgcolor': '#2b2b2b',
                  'slidercolor': '#434e60',
                  'wcolor': '#404040',
                  'axlabel': '790000',
                  'speed': '#babaa3',
                  'opto': '#3e82fc',
                  'run_1': '#babaa3',
                  'event_2': '#ff553f',
                  'opto_1': 'lime',
                  'rewardzone': '#e2efda',
                  'rippletick': '#fffa48',
                  'legendalpha': 0.1,
                  'trace-g': '#436fb6'}
    elif skin == 'light':
        colors = {'bgcolor': '#eef2f5',
                  'slidercolor': '#2dc2df',
                  'wcolor': '#ffffff',
                  'axlabel': '424c58',
                  'speed': '#cc6699',
                  'opto': '#3e82fc',
                  'run_1': '#d671ad',
                  'event_2': '#aed361',
                  'opto_1': '#fdb64e',
                  'rewardzone': '#33c4b3',
                  'rippletick': '#000000',
                  'legendalpha': 0.8,
                  'trace-g': '#436fb6',
                  'mpl-b': '#1f77b4',
                  'mpl-o': '#ff7f0e',
                  'green': '#71c055'}
    elif skin == 'figures':
        colors = {'speed': '#21409A'}
    elif skin == 'plots':
        colors = {'orange': '#df5a49',
                  'brown': '#4c4747',
                  'navy': '#344d5c',
                  'blue': '#4ab5c1',
                  'yellow': '#efc94d',
                  'grey': '#808080',
                  'swr': '#003f1b',
                  'run': '#0d7f4e',
                  'rest': '#dae551',
                  'sunset1': '#824737',
                  'sunset2': '#b55232',
                  'sunset3': '#d6562b',
                  'sunset4': '#ed7c31',
                  'sunset5': '#f09b36',
                  'sunset6': '#f4bb3a',
                  'mountains1': '#577483',
                  'mountains2': '#8ea3ae',
                  'mountains3': '#e0e9ed',
                  'mountains4': '#b1c0c9',
                  'mountains5': '#abadb0',
                  'mountains6': '#c7c8ca',
                  'mountains7': '#e2e3e4',
                  'CMYKred': '#ed1c24',
                  'CMYKgreen': '#00a651',
                  'green': '#00FF00',
                  'RGBblue': '#0000FF',
                  'NegBlue': '#252783',
                  'PosRed': '#FF0000',
                  'axax_magenta': '#d734aa',
                  'magenta': '#ec008c',
                  '470nm': '#00a9ff',
                  'black': '#000000',
                  'swrgreen': '#67923d',
                  'thetablue': '#3d5591'}
    if key in colors:
        return colors[key]
    else:
        raise KeyError(f'Color {key} not defined in {skin} skin')
