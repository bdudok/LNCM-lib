import json
import requests
import pandas
from BaserowAPI.config import config

class GetSessions:
    def __init__(self):
        self.auth_string = f"Token {config['api_token']}"

    def get_field(self, key):
        return config['FieldID'].get(key, None)

    def search(self, project, task=None, incltag=None, more_filters=None):
        '''
        :param project: string to search the Project field
        :param task: string to search the Task field, None to ignore
        :param incltag: string to search the InclTag field, None to ignore
        :param more_filters: pass a list of 2-tuples for additional searches (key, value)
        :return: DataFrame of the search result
        '''
        params = {f"filter__field_{self.get_field('Project')}__contains": project}
        if task is not None:
            params[f"filter__field_{self.get_field('Task')}__contains"] = task
        if incltag is not None:
            params[f"filter__field_{self.get_field('InclTag')}__contains"] = incltag
        if more_filters is not None:
            for key, value in more_filters:
                params[key] = value
        print(params)
        resp = requests.get(config['session_url'],
            headers={"Authorization": self.auth_string},
            params=params
        )

        self.results = pandas.DataFrame(resp.json()['results'])
        return self.results


if __name__ == '__main__':
    project = 'Voltage'
    task = "MotionCorr"  # or None to ignore. default: "MotionCorr"
    incltag = 'voltage'  # or None to ignore. default: None
    db = GetSessions()
    session_df = db.search(project=project, task=task, incltag=incltag)
    print(session_df[['Image.ID', 'Processed.Path']])