import pandas

def read_excel(*args, **kwargs):
   return pandas.read_excel(*args, **kwargs, engine='openpyxl')