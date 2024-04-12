import json
import requests
import pandas
from BaserowAPI.config import config
from Utilities.LG_API_token import token
#labguru API token. Expires in 30 days. Request a new one if necessary (response:401) and paste here.
lg_api_token = token
# lg_api_token = config["lg_token"]

#get list of mice from LG
pagesize=100
mice_lg = []
next_page = 1
while next_page:
    resp = requests.get(f'{config["lg_mice_url"]}?token={lg_api_token}',
                        params={'page_size': pagesize, 'page': next_page})
    if resp.status_code == 401:
        print('LG API unauthorized; get new token')
    assert resp.status_code == 200
    rjs = resp.json()
    mice_lg.append(pandas.DataFrame(rjs))
    if len(rjs) < pagesize:
        next_page = False
    else:
        next_page += 1

mice_lg = pandas.concat(mice_lg)
print('Got mouse list from LabGuru')

#get list of mice from BR (can be multipage)
auth_string = f"Token {config['api_token']}"
mice_br = []
next_page = 1
while next_page:
    resp = requests.get(config['mice_url'],
                        headers={"Authorization": auth_string},
                        params={'page_size': pagesize, 'page': next_page}
                        )
    if resp.status_code == 200:
        mice_br.append(pandas.DataFrame(resp.json()['results']))
        if resp.json()['next'] is None:
            next_page = False
        else:
            next_page += 1
print('Got mouse list from Baserow')

mice_br = pandas.concat(mice_br)
#find diff
lg_set = set(mice_lg['name'])
br_set = set(mice_br['Mouse.ID'])
print(f'{len(lg_set)} in LabGuru, {len(br_set)} in Baserow')
diff_mouse = list(lg_set.difference(br_set))
diff_mouse.sort()
if not len(diff_mouse):
    print(f'No new mouse')

# put new mouse in BR
for mouse in diff_mouse:
    m = mice_lg.loc[mice_lg['name'] == mouse].iloc[0]
    j = {
                "Mouse.ID": m['name'],
                "System ID": m['sys_id'],
                "Owner": m['owner']['name'],
                "Created at": m['created_at'],
                "Tags": m['tags'],
                "Strain*": m['Strain'],
                "MouseSource*": m['MouseSource'],
                "SourceCage": m['SourceCage'],
                "SourceID": m['SourceID'],
                "DOB*": m['DOB'],
                "Note": m['Note'],
                "Sex*": m['Sex'],
                "Status*": m['Status'],
                "End date": m['End date'],
                "NextTask": m['NextTask'],
        }
    for key, value in j.items():
        if type(j[key]) is not str:
            if j[key] == []:
                j[key] = ''
            else:
                j[key] = str(value)
        else:
            if value.startswith('<p') and value.endswith('</p>'):
                v1 = value[3:]
                v2 = v1[:v1.find('</p>')]
                j[key] = v2

    put = requests.post(config['mice_url'],
                        headers={"Authorization": auth_string,
                                 "Content-Type": "application/json"
                                 },
                        json=j
        )
    if put.status_code == 200:
        print(f'Mouse added: {mouse}')
    else:
        print(f'Error adding mouse: {mouse}')
