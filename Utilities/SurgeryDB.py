import datetime
import json
import re

import numpy
import requests
import pandas
from BaserowAPI.config import config

'''download mouse, incjection, window, and immuno data from LG,
build a list of mice that contains the procedures done on them'''

from Utilities.LG_API_token import token
#labguru API token. Expires in 30 days. Request a new one if necessary (response:401) and paste here.
lg_api_token = token
# lg_api_token = config["lg_token"]
savepath = 'D:\Shares\Data\_Processed/MouseData/_InjectionList'
def pprint(x):
    print(json.dumps(x, indent=2))

date_regex = re.compile(r'^(-?(?:[1-9][0-9]*)?[0-9]{4})-(1[0-2]|0[1-9])-(3[01]|0[1-9]|[12][0-9])?$').match

# get list of mice from LG
pagesize = 200
mice_lg = []
next_page = 1
mouse_index = 0
while next_page:
    print('Reading mice, page', next_page)
    resp = requests.get(f'{config["lg_mice_url"]}?token={lg_api_token}',
                        params={'page_size': pagesize, 'page': next_page})
    if resp.status_code == 401:
        print('LG API unauthorized; get new token')
    assert resp.status_code == 200
    rjs = resp.json()
    mice_lg.append(pandas.DataFrame(rjs, index=numpy.arange(mouse_index, mouse_index+len(rjs))))
    if len(rjs) < pagesize:
        next_page = False
    else:
        next_page += 1
    mouse_index += len(rjs)

mice_lg = pandas.concat(mice_lg)

# get list of injections
resp = requests.get(f'{config["lg_protocol_url"]}?token={lg_api_token}', )
rjs = resp.json()
protocols = pandas.DataFrame(rjs)

injection_prot_names = ('AAV injection',)
more_prot_names = {
    'Window': ('Imaging window implant', ),
    'Electrode': ('Electrode implant', ),
    'Immuno': ('Fluorescent Immunostaining', )
}

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
    m_in.loc[m.name, 'owner'] = m['owner']['name']
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
v_dataframes = []
for sid, expids in exp_samples.items():
    # check if this sample is a mouse
    is_mouse = False
    for sample in samples_jsons[sid]:
        is_mouse = sample['container']['name'] == 'Mice'
        break
    if not is_mouse:
        continue
    this_mouse = m_out.loc[m_out[("Mouse Data", "id")] == sid].iloc[0]
    #make sure that the id links to the correct mouse
    assert this_mouse[("Mouse Data", "name")] == sample["name"]
    # check if the experiment is an injection
    this_exp = None
    for expid in expids:
        if expid in experiments and is_mouse:
            if this_exp is None:
                this_exp = experiments[expid]
            else:
                print('Multiple injections found', expid)
    if this_exp is None: #should not be possible
        print('No injection found for sample', sample["name"])
        continue
    # add experiment details to output
    expdat = pandas.DataFrame({'ExpName': this_exp['name']}, index=[this_mouse.name])
    #get date
    expdate = this_exp['name'].split(' ')[0]
    for date_field in ('start_date', 'created_at'):
        if date_regex(expdate) is None:
            expdate = this_exp[date_field].split(' ')[0]
    expdat['InjDate'] = expdate
    #get viruses
    # find all other samples linked to this exp:
    n_virus = 0
    for vid, _ in exp_samples.items():
        for vsample in samples_jsons[vid]:
            if vsample["experiment_id"] == this_exp['id']:
                if vsample["item_type"] == "Biocollections::Virus":
                    coltag = 'AAV'
                    if n_virus:
                        coltag += str(n_virus+1)
                    n_virus += 1
                    expdat[coltag] = vsample['name']
                    #If wanted, virus list can be looked up and virus type etc info can be added
    v_dataframes.append(expdat)

#add injection dat to output
# add_columns = pandas.MultiIndex.from_product([["Injection Data"], expdat.columns])
# for addcol, col in zip(add_columns, expdat.columns):
#     if addcol not in m_out.columns:
#         m_out[addcol] = pandas.Series()
# m_out.loc[this_mouse.name, add_columns] = expdat

v_df = pandas.concat(v_dataframes)
nan_df = v_df.copy()
add_cols = pandas.MultiIndex.from_product([["Injection Data"], v_df.columns])
v_df.columns = add_cols
for col in add_cols:
    m_out[col] = ""
for ii, item in v_df.iterrows():
    m_out.loc[item.name, add_cols] = item

#add more experiments
more_prot_names = {
    'Window': ('Imaging window implant', ),
    'Electrode': ('Electrode implant', ),
    'Immuno': ('Fluorescent Immunostaining', ),
    'Kainate': ('Intrahippocampal kainate injection', 'Intra-amygdala kainate injection', ),
}
kainate_handles = ('IHK', 'IAK') #match the prot list
match_kainate_type = {}
for prot_type, prot_names in more_prot_names.items():
    more_experiments = {}
    #get experiments for each protocol
    prots = protocols.loc[protocols['name'].isin(prot_names)]
    for _, expitem in prots.iterrows():
        this_prot = expitem['name']
        if prot_type == 'Kainate':
            kainate_type = kainate_handles[prot_names.index(this_prot)]
        resp = requests.get(f'{config["lg_api_root"]}{expitem["api_url"]}?token={lg_api_token}', )
        rjs = resp.json()
        links = rjs['links']
        for uuid in links:
            resp = requests.get(f'{config["lg_api_root"]}api/v1/experiments?uuid={uuid}',
                                params={'token': lg_api_token})
            for exp in resp.json():
                more_experiments[exp["id"]] = exp
                if prot_type == 'Kainate':
                    match_kainate_type[exp["id"]] = kainate_type
        print(f'Got {len(links)} {prot_type} experiments')
    #iterate samples to find linked exps
    more_dataframes = []
    for sid, expids in exp_samples.items():
        # check if this sample is a mouse
        is_mouse = False
        for sample in samples_jsons[sid]:
            is_mouse = sample['container']['name'] == 'Mice'
            break
        if not is_mouse:
            continue
        this_mouse = m_out.loc[m_out[("Mouse Data", "id")] == sid].iloc[0]
        # make sure that the id links to the correct mouse
        assert this_mouse[("Mouse Data", "name")] == sample["name"]
        # add experiment details to output
        if prot_type == 'Kainate':
            expdat = pandas.DataFrame({'ExpName': '', 'ExpDate': '', 'ExpType': ''}, index=[this_mouse.name])
        else:
            expdat = pandas.DataFrame({'ExpName': '', 'ExpDate': ''}, index=[this_mouse.name])
        # check if the experiment is of current type
        n_exps = 0
        for expid in expids:
            if expid in more_experiments and is_mouse:
                this_exp = more_experiments[expid]
                if expdat.iloc[0]['ExpName'] == '':
                    expdat.iloc[0]['ExpName'] += this_exp['name']
                else:
                    expdat.iloc[0]['ExpName'] += ', ' + this_exp['name']
                if prot_type == 'Kainate':
                    kainate_type = match_kainate_type[expid]
                    if expdat.iloc[0]['ExpType'] == '':
                        expdat.iloc[0]['ExpType'] += kainate_type
                    else:
                        expdat.iloc[0]['ExpType'] += ', ' + kainate_type
                # get date
                expdate = this_exp['name'].split(' ')[0]
                for date_field in ('start_date', 'created_at'):
                    if date_regex(expdate) is None:
                        expdate = this_exp[date_field].split(' ')[0]
                if expdat.iloc[0]['ExpDate'] == '':
                    expdat.iloc[0]['ExpDate'] += expdate
                else:
                    expdat.iloc[0]['ExpName'] += ', ' + expdate

        more_dataframes.append(expdat)

    m_df = pandas.concat(more_dataframes)
    add_cols = pandas.MultiIndex.from_product([[prot_type], m_df.columns])
    m_df.columns = add_cols
    for col in add_cols:
        m_out[col] = ""
    for ii, item in m_df.iterrows():
        m_out.loc[item.name, add_cols] = item

#add remaining columns from labguru Mice collection
m_out[pandas.MultiIndex.from_product([["Additional LabGuru Data"], additional_columns])] = m_in[additional_columns]
out_fn = savepath + f'_{datetime.date.today().isoformat()}.xlsx'
m_out.to_excel(out_fn)
print('Spreadsheet saved in', out_fn)

