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
