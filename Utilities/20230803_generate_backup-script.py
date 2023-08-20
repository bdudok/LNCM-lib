import os

'''Back up raw data folders to onedrive'''
script_handle = 'C:/Users/u247640/OneDriveBackup.cmd'

dest_path = 'OneDrive:_RawData/'
source_path = 'D:\Shares\Data\_RawData/'

def get_cmd(folder):
    s = f'rclone copy {source_path+folder} {dest_path+folder} -P'
    s = os.sep.join(s.split('/'))
    return s+'\n'

script_s = ''

dlist = ['Confocal', 'Widefield']

for d in dlist:
    script_s += get_cmd(d)

with open(script_handle, 'w') as f:
    f.write(script_s)

