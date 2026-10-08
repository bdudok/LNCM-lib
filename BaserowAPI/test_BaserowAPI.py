#use DBAPI env (>3.8)
import os
from baserow.client import BaserowClient
import json

with open('config.py', 'r') as f:
    config = json.load(f)

client = BaserowClient('http://10.18.4.112:5001', token=config['api_token'])
