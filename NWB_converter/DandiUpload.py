import json
import shutil
import subprocess
import sys
from pathlib import Path

'''
DANDI upload of NWB files exported with an external video (NWBExporter, video.storage: external).
The NWB file references the video by file name only. For each export, NWBExporter writes <nwb name>_dandi.json
next to the NWB file, with the paths and the equivalent PowerShell commands.
Upload steps, for each NWB file:
    1. copy the video from the raw data folder next to the NWB file, where dandi organize looks for it
    2. dandi organize: move the NWB file and the video into the dandiset folder, with DANDI names, and
       update the video path in the NWB file
then dandi upload from the dandiset folder.
Only uses the standard library, so it runs in any env with the dandi client (pip install dandi).
Set the DANDI_API_KEY environment variable before uploading.
The dandiset folder is made with: dandi download --download dandiset.yaml https://dandiarchive.org/dandiset/<id>/draft
Usage:
    python DandiUpload.py E:/NWB_export/<nwb name>_dandi.json [...]
    python DandiUpload.py E:/NWB_export/    (all *_dandi.json files in the folder)
'''

UPLOAD_FILE_SUFFIX = '_dandi.json'


def organize_args(nwb_path, dandiset_path):
    return ['dandi', 'organize', str(nwb_path), '-d', str(dandiset_path), '--files-mode', 'move',
            '--update-external-file-paths', '--media-files-mode', 'move']


def write_upload_file(nwb_path, video_source, dandiset_path):
    '''
    Save the paths and commands needed to upload an NWB file with its external video.
    :param nwb_path: exported NWB file
    :param video_source: the video file in the raw data folder
    :param dandiset_path: local dandiset folder (with dandiset.yaml), or None if not set yet
    :return: path of the saved json file
    '''
    nwb_path = Path(nwb_path)
    video_source = Path(video_source)
    staged_video = nwb_path.parent / video_source.name
    dandiset = dandiset_path if dandiset_path is not None else '<dandiset folder>'
    commands = [f'Copy-Item "{video_source}" "{staged_video}"',
                f'dandi organize "{nwb_path}" -d "{dandiset}" --files-mode move '
                f'--update-external-file-paths --media-files-mode move',
                f'cd "{dandiset}"',
                'dandi upload']
    upload_file = nwb_path.parent / (nwb_path.stem + UPLOAD_FILE_SUFFIX)
    with open(upload_file, 'w') as f:
        json.dump({'nwb_file': str(nwb_path),
                   'video_source': str(video_source),
                   'dandiset_path': None if dandiset_path is None else str(dandiset_path),
                   'powershell_commands': commands}, f, indent=1)
    return upload_file


def organize(upload_file):
    '''copy the video next to the NWB file and move both into the dandiset folder. Returns the dandiset path'''
    with open(upload_file, 'r') as f:
        info = json.load(f)
    nwb_path = Path(info['nwb_file'])
    video_source = Path(info['video_source'])
    if info['dandiset_path'] is None:
        raise ValueError(f'{upload_file}: dandiset_path is not set (dandi.dandiset_path in the export config)')
    dandiset_path = Path(info['dandiset_path'])
    for path in (nwb_path, video_source, dandiset_path / 'dandiset.yaml'):
        if not path.exists():
            raise FileNotFoundError(f'{upload_file}: {path} not found')
    staged_video = nwb_path.parent / video_source.name
    print(f'Copying {video_source} to {staged_video}...')
    shutil.copy2(video_source, staged_video)
    subprocess.run(organize_args(nwb_path, dandiset_path), check=True)
    return dandiset_path


def main(paths):
    upload_files = []
    for path in map(Path, paths):
        if path.is_dir():
            upload_files.extend(sorted(path.glob('*' + UPLOAD_FILE_SUFFIX)))
        else:
            upload_files.append(path)
    dandiset_paths = {organize(upload_file) for upload_file in upload_files}
    for dandiset_path in dandiset_paths:
        subprocess.run(['dandi', 'upload'], cwd=dandiset_path, check=True)


if __name__ == '__main__':
    main(sys.argv[1:])
