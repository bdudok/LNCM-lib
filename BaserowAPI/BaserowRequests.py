import json
import requests

with open('./config.json', 'r') as f:
    config = json.load(f)

auth_string = f"Token {config['api_token']}"

project = "SncgTot"
task = "MotionCorr"
filt_tree = {"filter_type": "AND", "filters": [{"field": "Project", "type": "equal", "value": project},
                                               {"field": "Task", "type": "equal", "value": task}]}

filt_tree = {{"filter_type":"AND","filters":[{"type":"contains","field":"Project","value":"SncgTot"},
                                             {"type":"single_select_equal","field":"Task","value":"2467"}],"groups":[]}}

tbl = requests.get(
    "http://10.18.4.112:5001/api/database/rows/table/490/?user_field_names=true",
    headers={"Authorization": auth_string},
    params={"filters": filt_tree}
)