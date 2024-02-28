import json
import requests
import pandas
from BaserowAPI.config import config

'''download mouse, incjection, window, and immuno data from LG,
build a list of mice that contains the procedures done on them'''

#labguru API token. Expires in 30 days. Request a new one if necessary and paste here.
lg_api_token = "8a7e4cb392f54c19f159510279b513eab3e60e4b"
# lg_api_token = config["lg_token"]

#get list of mice from LG
pagesize=100
mice_lg = []
next_page = 1
while next_page:
    resp = requests.get(f'{config["lg_mice_url"]}?token={lg_api_token}',
                        params={'page_size': pagesize, 'page': next_page})
    rjs = resp.json()
    mice_lg.append(pandas.DataFrame(rjs))
    if len(rjs) < pagesize:
        next_page = False
    else:
        next_page += 1

mice_lg = pandas.concat(mice_lg)
print('Got mouse list from LabGuru')

#get list of injections
resp = requests.get(f'{config["lg_protocol_url"]}?token={lg_api_token}',)
rjs = resp.json()
protocols = pandas.DataFrame(rjs)
injection_prot_names = ('AAV injection',)
inj_prot = protocols.loc[protocols['name'].isin(injection_prot_names)]

def json_find_keys(item, search_key):
    out_list = []
    jtext = json.dumps(item, indent=2)
    for line in jtext.split('\n'):
        if search_key in line:
            try:
                item = line.split('": ')[1]
                out_list.append(item.split('"')[1])
            except:
                print('Not parsed:', line)
    return out_list

#get list of experiments associated with injections
experiments = {}
exp_uuids = {}
for _, inj in inj_prot.iterrows():
    resp = requests.get(f'{config["lg_api_root"]}{inj["api_url"]}?token={lg_api_token}', )
    rjs = resp.json()
    links = rjs['links']
    for uuid in links:
        # resp = requests.get(f'{config["lg_api_root"]}/api/v1/experiments?uuid={uuid}&token={lg_api_token})
        resp = requests.get(f'{config["lg_api_root"]}api/v1/experiments?uuid={uuid}',
                            params={'token': lg_api_token})
        exp = resp.json()
        experiments[uuid] = exp
        linked_uuids = json_find_keys(exp, '"uuid"')
        exp_uuids[uuid] = linked_uuids


#mice with injection exps:
for uuid, links in exp_uuids.items():
    linked_mice = mice_lg.loc[mice_lg['uuid'].isin(links)]
    if not len(linked_mice):
        # alert for injections without linked mice:
        print(experiments[uuid][0]['title'], 'has no linked mice')
        # print(experiments[uuid]['name'], linked_mice['name'])



    # resp = requests.get(f'{config["lg_api_root"]}{"073274d3-4a29-4898-8f7b-9a75df7b7414"}?token={lg_api_token}', )

# # put new mouse in BR
# for mouse in diff_mouse:
#     m = mice_lg.loc[mice_lg['name'] == mouse].iloc[0]
#     j = {
#                 "Mouse.ID": m['name'],
#                 "System ID": m['sys_id'],
#                 "Owner": m['owner']['name'],
#                 "Created at": m['created_at'],
#                 "Tags": m['tags'],
#                 "Strain*": m['Strain'],
#                 "MouseSource*": m['MouseSource'],
#                 "SourceCage": m['SourceCage'],
#                 "SourceID": m['SourceID'],
#                 "DOB*": m['DOB'],
#                 "Note": m['Note'],
#                 "Sex*": m['Sex'],
#                 "Status*": m['Status'],
#                 "End date": m['End date'],
#                 "NextTask": m['NextTask'],
#         }
