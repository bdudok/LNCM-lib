#use DBAPI env (>3.8)
import os
from baserow.client import BaserowClient
import json

with open('config.py', 'r') as f:
    config = json.load(f)

client = BaserowClient('http://10.18.4.112:5001', token=config['api_token'])

#examples:
# for db in client.list_all_applications():
#   print(db, [t.name for t in db.tables]) #list apps requires password auth (jwt)
#
# for table in client.list_database_tables(13):
#   print(table)
#
# for field in client.list_database_table_fields(45):
#   print(field)