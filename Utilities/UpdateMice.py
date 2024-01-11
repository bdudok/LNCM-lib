import json
import requests
import pandas
from BaserowAPI.config import config

#labguru API token. Expires in 30 days. Request a new one if necessary and paste here.
lg_api_token = "a4d5d5e0104196c0378d238e06d1c2055038a369"
# lg_api_token = config["lg_token"]

#get list of mice from LG
resp = requests.get(f'{config["lg_mice_url"]}?token={lg_api_token}')
mice_lg = pandas.DataFrame(resp.json())

#get list of mice from BR (can be multipage)
auth_string = f"Token {config['api_token']}"
mice_br = []
for page in range(int(len(mice_lg)/100+10)):
    resp = requests.get(config['mice_url'],
                        headers={"Authorization": auth_string},
                        params={'page': page}
                        )
    if resp.status_code == 200:
        mice_br.append(pandas.DataFrame(resp.json()['results']))
        if resp.json()['next'] is None:
            break

mice_br = pandas.concat(mice_br)
#find diff
lg_set = set(mice_lg['name'])
br_set = set(mice_br['Mouse.ID'])
diff_mouse = list(lg_set.difference(br_set))
diff_mouse.sort()

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
