import datetime

def lprint(obj, message):
    '''Add timestamp and object name to print calls'''
    ts = datetime.datetime.now().isoformat(timespec='seconds')
    print(f'{ts} - {obj.__name__}: {message}')