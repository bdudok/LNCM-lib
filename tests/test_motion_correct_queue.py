import pytest

from JobQueue import common


def test_motion_correct_run_queues_sessions_and_raises_on_failure(monkeypatch):
    import pandas
    from Proc2P.Bruker import MotionCorrect

    class FakeClient:
        submitted = []

        def submit(self, job_type, data, label, key):
            self.submitted.append((job_type, data, label, key))
            return label

        def job_status(self, ids):
            return [{'job_id': i, 'status': 'failed' if i.endswith('2') else 'done', 'result': i, 'error': 'tb'}
                    for i in ids]

    monkeypatch.setattr(MotionCorrect, 'QueueClient', FakeClient)
    monkeypatch.setattr(MotionCorrect, 'POLL_S', 0)
    df = pandas.DataFrame([{'Image.ID': f'mouse_2026-01-0{i}_x_00{i}-000', 'Raw.Path': 'R', 'Processed.Path': 'P',
                            'Notes': None, 'Created': pandas.Timestamp('2026-01-01')} for i in (1, 2)])
    settings = {'registration': {'nonrigid': True}}
    with pytest.raises(RuntimeError, match='1 sessions: mouse_2026-01-02_x_002'):
        MotionCorrect.run(settings, df, False, True, False, 'Ch2')
    job_type, data, label, key = FakeClient.submitted[0]
    assert (job_type, label, key) == ('MotionCorrect', 'mouse_2026-01-01_x_001', 'mouse_2026-01-01_x_001')
    common.encode(data)  # JSON-compatible
    assert data[1]['Image.ID'] == 'mouse_2026-01-01_x_001-000' and data[1]['Notes'] is None
    assert data[2:] == (False, True, False, 'Ch2', None)


def test_preset_settings():
    from Proc2P.Bruker import MotionCorrect
    s = MotionCorrect.preset_settings('Voltage', mc_only=True)
    reg = s['registration']
    assert (reg['nonrigid'], reg['block_size'], reg['batch_size'], reg['snr_thresh'], reg['maxregshift']) == \
        (False, [512, 48], 5000, 1.5, 0.05)
    assert reg['do_bidiphase'] and reg['two_step_registration'] and not s['run']['do_detection']
    assert MotionCorrect.preset_settings('Calcium')['registration']['block_size'] == [511, 16]
    with pytest.raises(ValueError):
        MotionCorrect.preset_settings('GCaMP')


def test_gui_form_defaults_match_suite2p():
    '''The GUI's hard-coded presets must give the same registration settings as suite2p's defaults with the preset.
    If this fails after a suite2p update, update SUITE2P_REG_DEFAULTS in MotionCorrectGUI.'''
    from Proc2P.Bruker import MotionCorrect
    from Proc2P.Bruker.MotionCorrectGUI import FORM_PRESETS, REG_FIELDS
    for preset, values in FORM_PRESETS.items():
        reg = MotionCorrect.preset_settings(preset)['registration']
        assert values == {k: reg[k] for k in REG_FIELDS}, preset


def test_with_defaults_fills_in_and_rejects_unknown_settings():
    from Proc2P.Bruker import MotionCorrect
    partial = MotionCorrect.lab_settings(None, mc_only=False, registration={'batch_size': 7})
    full = MotionCorrect.with_defaults(partial)
    assert full['registration']['batch_size'] == 7 and full['io']['move_bin'] and 'run' in full
    assert MotionCorrect.with_defaults(full) == full  # full settings (the template) pass through
    with pytest.raises(ValueError, match='registration.no_such_setting'):
        MotionCorrect.with_defaults({'registration': {'no_such_setting': 1}})


def test_gui_does_not_load_suite2p():
    import subprocess
    import sys
    code = 'import sys; import Proc2P.Bruker.MotionCorrectGUI; print("torch" in sys.modules or "suite2p" in sys.modules)'
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().endswith('False')


def test_existing_registered_movie_fails_before_suite2p(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from Proc2P.Bruker import MotionCorrect
    prefix = 'mouse_2026-01-01_x_001'
    procpath = tmp_path / prefix
    procpath.mkdir()
    (procpath / f'{prefix}_registered_Ch2.bin').touch()
    monkeypatch.setattr('Proc2P.Bruker.PreProc.PreProc',
                        lambda *args, **kwargs: SimpleNamespace(procpath=procpath, channelnames=['Ch2']))
    monkeypatch.setattr('suite2p.run_s2p', lambda **kwargs: pytest.fail('suite2p should not run'))
    item = {'Raw.Path': str(tmp_path), 'Processed.Path': str(tmp_path), 'Image.ID': prefix + '-000'}
    with pytest.raises(FileExistsError, match='already motion corrected.*overwrite_previous'):
        MotionCorrect.process_session({}, item, pre_only=False, overwrite_previous=False, overwrite_preproc=False,
                                      ref_ch='Ch2', fast_disk=str(tmp_path / 'scratch'))
