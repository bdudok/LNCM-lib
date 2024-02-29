import json

import numpy
import requests
import pandas
from BaserowAPI.config import config

'''download mouse, incjection, window, and immuno data from LG,
build a list of mice that contains the procedures done on them'''

# labguru API token. Expires in 30 days. Request a new one if necessary and paste here.
lg_api_token = "8a7e4cb392f54c19f159510279b513eab3e60e4b"
# lg_api_token = config["lg_token"]

# get list of mice from LG
pagesize = 100
mice_lg = []
next_page = 1
while next_page:
    print('Reading mice, page', next_page)
    resp = requests.get(f'{config["lg_mice_url"]}?token={lg_api_token}',
                        params={'page_size': pagesize, 'page': next_page})
    rjs = resp.json()
    mice_lg.append(pandas.DataFrame(rjs))
    if len(rjs) < pagesize:
        next_page = False
    else:
        next_page += 1

mice_lg = pandas.concat(mice_lg)

# get list of injections
resp = requests.get(f'{config["lg_protocol_url"]}?token={lg_api_token}', )
rjs = resp.json()
protocols = pandas.DataFrame(rjs)
injection_prot_names = ('AAV injection',)
inj_prot = protocols.loc[protocols['name'].isin(injection_prot_names)]

# get list of experiments associated with injections
experiments = {}
for _, inj in inj_prot.iterrows():
    resp = requests.get(f'{config["lg_api_root"]}{inj["api_url"]}?token={lg_api_token}', )
    rjs = resp.json()
    links = rjs['links']
    for uuid in links:
        # resp = requests.get(f'{config["lg_api_root"]}/api/v1/experiments?uuid={uuid}&token={lg_api_token})
        resp = requests.get(f'{config["lg_api_root"]}api/v1/experiments?uuid={uuid}',
                            params={'token': lg_api_token})
        for exp in resp.json():
            experiments[exp["id"]] = exp
print(f'Got {len(links)} injection experiments')

# get samples - these are entries that represent library elements (mouse, virus) linked to experiments
next_page = 1
samples_jsons = {}
exp_samples = {}
while next_page:
    print('Reading samples, page', next_page)
    resp = requests.get(f'{config["lg_api_root"]}api/v1/samples',
                        params={'token': lg_api_token, 'page_size': pagesize, 'page': next_page})
    rjs = resp.json()
    if len(rjs) < pagesize:
        next_page = False
    else:
        next_page += 1
    for sample in rjs:
        sid = sample["item_id"]
        if sid not in exp_samples:
            exp_samples[sid] = []
        if sid not in samples_jsons:
            samples_jsons[sid] = []
        exp_samples[sid].append(sample["experiment_id"])
        samples_jsons[sid].append(sample)

# create mouse output table
m_in = mice_lg.dropna(how='all', axis=1)
for _, m in m_in.iterrows():
    m['owner'] = m['owner']['name']
primary_columns = ['name', 'DOB', 'Strain', 'Sex', 'Status', 'End date', 'Note', 'owner', 'id', 'sys_id', 'SourceCage',
                   'SourceID', 'created_at', 'tags']
m_out = pandas.DataFrame()
additional_columns = []
for col in m_in.columns:
    if col.startswith('custom'):
        continue
    if not any(m_in[col].astype('bool')):
        continue
    if col not in primary_columns:
        additional_columns.append(col)
m_out = m_in[primary_columns]
m_out.columns = pandas.MultiIndex.from_product([["Mouse Data"], m_out.columns])
print('Formatted output spreadsheet')

# iterate through samples, and add experiment info to each included mouse
for sid, expids in exp_samples.items():
    # check if this sample is a mouse
    is_mouse = False
    for sample in samples_jsons[sid]:
        is_mouse = sample['container']['name'] == 'Mice'
    if is_mouse:
        this_mouse = m_out.loc[m_out[("Mouse Data", "id")] == sid].iloc[0]
        #make sure that the id links to the correct mouse
        assert this_mouse[("Mouse Data", "name")] == sample["name"]
    # check if the sample is an injection
    for expid in expids:
        if expid in experiments and is_mouse:
            this_exp = experiments[expid]
            #TODO get viruses that are linked, add date and name to output







#add remaining columns from labguru Mice collection
for col in additional_columns:
    m_out[pandas.MultiIndex.from_product([["Additional LabGuru Data"], additional_columns])] = m_in[additional_columns]


