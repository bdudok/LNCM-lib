import pandas
import json

path = ''
fn = ''

with open(path + fn + '.json', 'r') as f:
    data = json.loads(f)

df = pandas.DataFrame(data)
df.to_csv(path + fn + '.csv')
df.to_excel(path + fn + '.xlsx')
