from pathlib import Path

from BaserowAPI.BaserowRequests import GetSessions

'''
Save the Baserow entry of each session in the test data _Processed folder,
as <prefix>_baserow_entry.pkl (pickled pandas Series of the DB row) next to the session folder.
Sessions that already have the file are skipped.
The paths in the saved entries refer to the real data.
'''

processed_path = Path(r'E:\test_data\_Processed')

db = GetSessions()
for session_folder in sorted(processed_path.iterdir()):
    if not session_folder.is_dir():
        continue
    prefix = session_folder.name
    entry_fn = processed_path / f'{prefix}_baserow_entry.pkl'
    if entry_fn.exists():
        print(prefix, 'already saved')
        continue
    results = db.get_session(prefix)
    if len(results) != 1:
        print(prefix, f'skipped: {len(results)} DB entries found')
        continue
    results.iloc[0].to_pickle(entry_fn)
    print(prefix, 'saved', entry_fn)
