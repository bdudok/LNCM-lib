import json, numpy
import os
import requests
import pandas
from BaserowAPI.config import config


class GetSessions:
    def __init__(self):
        self.auth_string = f"Token {config['api_token']}"
        self.sex_cache = {} #for memoizing fetching animal metadata

    def get_field(self, key):
        return config['FieldID'].get(key, None)

    def get_sex(self, prefix):
        a_tag = prefix.split('_')[0]
        if a_tag not in self.sex_cache:
            self.sex_cache[a_tag] = self.get_mouse(mtag=a_tag, ret_sex=True)
        return a_tag, self.sex_cache[a_tag]

    def search(self, project, task=None, incltag=None, more_filters=None):
        '''
        :param project: string to search the Project field
        :param task: string to search the Task field, None to ignore
        :param incltag: string to search the InclTag field, None to ignore
        :param more_filters: pass a list of 2-tuples for additional searches (key, value)
        :return: DataFrame of the search result
        '''
        params = {'page_size': 100}
        if project is not None:
            params[f"filter__field_{self.get_field('Project')}__contains"] = project
        if task is not None:
            params[f"filter__field_{self.get_field('Task')}__contains"] = task
        if incltag is not None:
            params[f"filter__field_{self.get_field('InclTag')}__contains"] = incltag
        if more_filters is not None:
            for key, value in more_filters:
                params[key] = value
        print(params)

        session_jsons = []
        next_page = 1
        session_index = 0
        while next_page:
            print('Reading db, page', next_page)
            params['page'] = next_page
            resp = requests.get(config['session_url'],
                                headers={"Authorization": self.auth_string},
                                params=params
                                )
            assert resp.status_code == 200
            rjs = resp.json()['results']
            session_jsons.append(pandas.DataFrame(rjs, index=numpy.arange(session_index, session_index + len(rjs))))
            if len(rjs) < params['page_size']:
                next_page = False
            else:
                next_page += 1
            session_index += len(rjs)

        self.results = pandas.concat(session_jsons)
        self.results.sort_values('Image.ID', inplace=True)
        return self.results

    def get_session(self, prefix, match='contains'):
        '''
        :param tag: string to search Image.ID
        :param match: 'equal' or 'contains'
        :return: DataFrame of the search result
        '''
        params = {f"filter__field_{self.get_field('Image.ID')}__{match}": prefix}
        resp = requests.get(config['session_url'],
            headers={"Authorization": self.auth_string},
            params=params
        )
        self.results = pandas.DataFrame(resp.json()['results'])
        if not len(self.results):
            print(f'{prefix} not found in DB')
        return self.results


    def get_mouse(self, item=None, mtag=None, ret_sex=False):
        '''
        get a mouse entry from the DB
        :param item: pass a session to get the animal it's linked to (or pass None and set mtag)
        :param mtag: if set, look up mouse based on name
        :param ret_sex: if True, only returns sex ('F' or 'M')
        :return: DataFrame of the search result
        '''
        if mtag is not None:
            params = {f"filter__field_{config['MouseID']['Mouse.ID']}__contains": mtag}
            resp = requests.get(config['mice_url'],
                                headers={"Authorization": self.auth_string},
                                params=params
                                )
            self.results = pandas.DataFrame(resp.json()['results'])
            if ret_sex:
                return self.results.iloc[0]['Sex*']
            return self.results
        else:
            mid = item['Mouse.ID'][0]['id']
            url = config['mice_url'].split('?')
            params = {f"filter__field_{self.get_field('Mouse.ID')}__equals": mtag}
            resp = requests.get(f'{url[0]}{mid}/?{url[1]}',
                headers={"Authorization": self.auth_string},
            )
            return resp.json()


class PutLogEntries:
    def __init__(self):
        self.auth_string = f"Token {config['logger_token']}"
        self.username = os.environ.get('USERNAME')

    def put(self, sessionID, imtag, message='', sourceclass='',):
        if sessionID in (None, 'None'):
            sessionID = 0
        put_json = {
            "Name": imtag,
            "Message": message,
            "Class": sourceclass,
            "Session": [
                int(sessionID)
            ],
            "User": self.username
        }

        return requests.post(
            config['log_url'],
            headers={
                "Authorization": self.auth_string,
                "Content-Type": "application/json"
            },
            json=put_json
        )

    def check(self, sessionID, source='ArchiveZip'):
        '''Return True if the sessionID is already registered in BR as archived'''
        params = {
            f"filter__field_{config['LogID']['Name']}__contains": sessionID,
            f"filter__field_{config['LogID']['Class']}__equal": source,
        }

        resp = requests.get(config['log_url'],
            headers={
                "Authorization": self.auth_string,
            },
            params=params
        )


        return pandas.DataFrame(resp.json())



if __name__ == '__main__':
    project = 'Voltage'
    task = "MotionCorr"  # or None to ignore. default: "MotionCorr"
    incltag = 'voltage'  # or None to ignore. default: None
    db = GetSessions()
    session_df = db.search(project=project, task=task, incltag=incltag)
    print(session_df[['Image.ID', 'Processed.Path']])