import json
import subprocess

from NWB_converter import DandiUpload


def test_upload_steps(tmp_path, monkeypatch):
    '''the video is copied next to the NWB file, then organize for each file and one upload per dandiset'''
    export_dir = tmp_path / 'NWB_export'
    raw_dir = tmp_path / 'RawData'
    dandiset = tmp_path / '000000'
    for folder in (export_dir, raw_dir, dandiset):
        folder.mkdir()
    (dandiset / 'dandiset.yaml').write_text('identifier: DANDI:000000\n')
    for name in ('session1', 'session2'):
        (export_dir / f'{name}-full.nwb').write_bytes(b'nwb')
        (raw_dir / f'{name}.avi').write_bytes(b'video')
        DandiUpload.write_upload_file(export_dir / f'{name}-full.nwb', raw_dir / f'{name}.avi', dandiset)

    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda args, **kwargs: calls.append((args, kwargs.get('cwd'))))
    DandiUpload.main([str(export_dir)])

    assert (export_dir / 'session1.avi').read_bytes() == b'video'
    assert (export_dir / 'session2.avi').read_bytes() == b'video'
    assert calls == [(DandiUpload.organize_args(export_dir / 'session1-full.nwb', dandiset), None),
                     (DandiUpload.organize_args(export_dir / 'session2-full.nwb', dandiset), None),
                     (['dandi', 'upload'], dandiset)]


def test_upload_file_commands(tmp_path):
    upload_file = DandiUpload.write_upload_file(tmp_path / 'a.nwb', tmp_path / 'raw' / 'a.avi', None)
    with open(upload_file, 'r') as f:
        info = json.load(f)
    assert info['dandiset_path'] is None
    assert info['powershell_commands'][0] == f'Copy-Item "{tmp_path / "raw" / "a.avi"}" "{tmp_path / "a.avi"}"'
    assert '-d "<dandiset folder>"' in info['powershell_commands'][1]
