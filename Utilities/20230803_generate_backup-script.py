import os
import win32api, win32file

'''Back up raw data folders to onedrive
The script is scheduled to run on the server nightly.
update to include/exclude folders inthe backup.

for installing pywin32:

pip install pypiwin32
cd C:\ProgramData/anaconda3\envs\suite2p\Scripts
python pywin32_postinstall.py -install
'''
script_handle = 'C:/Users/u247640/OneDriveBackup.cmd'

dest_path = 'OneDrive:_RawData/'
source_path = 'D:\Shares\Data\_RawData/'

def get_cmd(folder):
    global source_path, dest_path
    s = f'rclone copy {source_path+folder} {dest_path+folder} -P'
    s = os.sep.join(s.split('/'))
    return s+'\n'

script_s = ''

#raw data folders to OneDrive
dlist = ['Confocal', 'Widefield', 'Bruker']

for d in dlist:
    script_s += get_cmd(d)

#get removable drive letters

drive_list = win32api.GetLogicalDriveStrings()
drive_list = drive_list.split("\x00")[0:-1]  # the last element is ""

drive_names = {}
for drive_letter in drive_list:
    drive_names[win32api.GetVolumeInformation(drive_letter)[0]] = drive_letter

#processed data forders to LNCM1
ext_drive_name = 'LNCM1'
if ext_drive_name in drive_names:
    dest_path = f'E:\ExtDrives/{ext_drive_name}/_Processed/'
    source_path = 'D:\Shares\Data\_Processed/'
    dlist = ['2P/PVTot']
    for d in dlist:
        script_s += get_cmd(d)


with open(script_handle, 'w') as f:
    f.write(script_s)

print(script_s)