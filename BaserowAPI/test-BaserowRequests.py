import json
import requests
from BaserowAPI.config import config
import pandas

auth_string = f"Token {config['api_token']}"

project = "SncgTot"
task = "MotionCorr"

resp = requests.get(config['session_url'],
    headers={"Authorization": auth_string},
    params={
        f"search_Project_contains": "Voltage",
        f"search_Task_contains": "Motion"
    }
)

results = pandas.DataFrame(resp.json()['results'])
for ri, r in enumerate(results):
    print()