import os
import win32api, win32file

'''Back up raw data folders to onedrive
The script is scheduled to run on the server nightly.
update to include/exclude folders in the backup.

for installing pywin32:

pip install pypiwin32
cd C:\ProgramData/anaconda3\envs\suite2p\Scripts
python pywin32_postinstall.py -install
'''
script_handle = 'C:/Users/u247640/OneDriveBackup.cmd'
ps_script_handle = 'C:/Users/u247640/DBBackup.ps1'

def get_cmd(folder):
    global source_path, dest_path
    s = f'rclone copy {source_path+folder} {dest_path+folder} -P'
    s = os.sep.join(s.split('/'))
    return s+'\n'

script_s = ''

#back up baserow db
dest_path = 'OneDrive:_baserow_backup/'
source_path = 'D:\Shares\Data\DB_Backups/'
dlist = ['30days']
for d in dlist:
    script_s += get_cmd(d)

#keep last 10 daily backups, then every 30 days
ps_script = r'''cd D:\Shares\Data\DB_Backups
Move-Item *.tar .\10days\
rm *.tar
docker stop baserow
docker run --rm -v baserow_data:/baserow/data -v ${PWD}:/backup ubuntu tar cvf /backup/baserow_$(get-date -f yyyy-MM-dd).tar /baserow/data
docker start baserow
$oldestFile = gci ./10days/ | select -first 1
$latestBu = gci ./30days/ | select -last 1
$timespanF = new-timespan -days 10
$timespanB = new-timespan -days 30
if (((get-date) - $oldestFile.LastWriteTime) -gt $timespanF) {
    if (((get-date) - $latestBu.LastWriteTime) -gt $timespanB) {
        mv ./10days/$oldestFile ./30days/
    }
    else {rm ./10days/$oldestFile}
}
'''

script_s += 'Powershell -NoProfile -ExecutionPolicy Bypass -File ' + ps_script_handle + '\n'

#raw data folders to OneDrive
dest_path = 'OneDrive:_RawData/'
source_path = 'D:\Shares\Data\_RawData/'
dlist = ['Confocal', 'Widefield', 'Bruker', 'Pinnacle']

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
    dlist = ['2P/PVTot', '2P/SncgTot', '2P/SncgDREADD',
             '2P/JEDI', '2P/VADER']
    for d in dlist:
        script_s += get_cmd(d)

#After OneDrive is expanded, also back up Processed there:
dest_path = 'OneDrive:_ProcessedData/'
# source_path = 'D:\Shares\Data\_Processed/'
# dlist = ['2P',]
for d in dlist:
    script_s += get_cmd(d)

#raw 2P data to LNCM2
ext_drive_name = 'LNCM2'
if ext_drive_name in drive_names:
    dest_path = f'E:\ExtDrives/{ext_drive_name}/_RawData/'
    source_path = 'D:\Shares\Data\_RawData/'
    dlist = ['Bruker']
    for d in dlist:
        script_s += get_cmd(d)

#raw EEG data to LNCM3
#... when have more recs

#DSI revision analysis to OneDrive
dest_path = 'OneDrive:Documents/_projects/2023-DSI/Revision_analysis/'
source_path = 'D:\Shares\Data\old_2P\DLX-ECB\PlaceFieldPlots/'
dlist = ['Revision']
for d in dlist:
    script_s += get_cmd(d)


with open(script_handle, 'w') as f:
    f.write(script_s)

with open(ps_script_handle, 'w') as f:
    f.write(ps_script)

print('--------------batch file for rclone backups----------------')
print(script_s)
print('--------------PS script for baserow backup-----------------')
print(ps_script)