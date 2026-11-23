import os
import time
import socket
import threading
from enum import Enum
from pathlib import Path
from http.server import ThreadingHTTPServer

import psutil
import pytest

from JobQueue import common
from JobQueue.client import QueueClient
from JobQueue.manager import Manager, RequestError, make_api_handler, make_dashboard_handler
from Proc2P.Analysis.GEVIReg.Register import RegConfig


class Color(Enum):
    RED = 1


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def job(user, n=1, job_type='TEST'):
    return {'user': user, 'token': user + '-token', 'client_id': user + '-client', 'job_type': job_type,
            'label': f'{user}{n}', 'lib_root': '/lib', 'payload': [f'{user}{n}']}


def sync(manager, user, **kwargs):
    return manager.sync({'runner_id': user + '-runner', 'user': user, 'token': user + '-token', 'lib_root': '/lib',
                         **kwargs})


def test_codec_round_trip():
    data = (Path('a') / 'b', 'prefix', RegConfig(ref_size=10), {'k': (1, Color.RED)})
    out = common.decode(common.encode(data))
    assert out == [str(Path('a') / 'b'), 'prefix', RegConfig(ref_size=10), {'k': [1, Color.RED]}]


def test_codec_rejects_unknown_types():
    with pytest.raises(TypeError):
        common.encode(object())


def test_token_is_bound_to_first_use(tmp_path):
    m = Manager(tmp_path / 'jobs.sqlite')
    m.submit(job('ann'))
    with pytest.raises(RequestError):
        m.submit({**job('ann'), 'token': 'other'})


def test_free_slot_goes_to_user_with_fewer_running_jobs(tmp_path):
    m = Manager(tmp_path / 'jobs.sqlite')
    max_n = common.JOB_TYPES['TEST'].max_n
    for i in range(max_n + 2):
        m.submit(job('ann', i))
    assigned = sync(m, 'ann')['assign']
    assert len(assigned) == max_n
    m.submit(job('bob'))
    assert sync(m, 'bob')['assign'] == []  # no free slot
    # ann finishes one job: the slot goes to bob, although ann's jobs are older
    first = assigned[0]['job_id']
    resp = sync(m, 'ann', started=[a['job_id'] for a in assigned],
                finished=[{'job_id': first, 'ok': True, 'result': 'x', 'worker': 'TEST-0'}])
    assert resp['assign'] == []  # no new job for ann
    assert [a['payload'] for a in sync(m, 'bob')['assign']] == [['bob1']]
    results = m.results({'user': 'ann', 'token': 'ann-token', 'client_id': 'ann-client'})['results']
    assert [r['result'] for r in results] == ['x']
    assert m.results({'user': 'ann', 'token': 'ann-token', 'client_id': 'ann-client'})['results'] == []


def test_second_runner_of_same_user_and_checkout_exits(tmp_path):
    m = Manager(tmp_path / 'jobs.sqlite')
    sync(m, 'ann')
    resp = m.sync({'runner_id': 'another', 'user': 'ann', 'token': 'ann-token', 'lib_root': '/lib'})
    assert resp.get('exit')


def test_jobs_of_lost_runner_fail(tmp_path, monkeypatch):
    m = Manager(tmp_path / 'jobs.sqlite')
    m.submit(job('ann'))
    assigned = sync(m, 'ann')['assign']
    sync(m, 'ann', started=[assigned[0]['job_id']])
    later = time.time() + 2 * common.RUNNER_LOST_S
    monkeypatch.setattr(time, 'time', lambda: later)
    m.housekeeping()
    status = m.jobs({'user': 'ann', 'token': 'ann-token', 'job_ids': [assigned[0]['job_id']]})['jobs'][0]
    assert status['status'] == 'failed'


def test_end_to_end_with_runner_process(tmp_path, monkeypatch):
    '''Manager in this process, a real runner started by the client, TEST jobs that succeed and fail.'''
    api, dash = free_port(), free_port()
    monkeypatch.setenv('LNCM_JOBQUEUE_API_PORT', str(api))
    monkeypatch.setenv('LNCM_JOBQUEUE_DASHBOARD_PORT', str(dash))
    monkeypatch.setenv('LNCM_JOBQUEUE_STATE_DIR', str(tmp_path / 'state'))
    m = Manager(tmp_path / 'jobs.sqlite')
    servers = [ThreadingHTTPServer(('127.0.0.1', api), make_api_handler(m)),
               ThreadingHTTPServer(('127.0.0.1', dash), make_dashboard_handler(m))]
    for s in servers:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        client = QueueClient()
        assert client.is_up()
        ok_id = client.submit('TEST', ('hello',), label='ok job')
        client.submit('TEST', ('boom', 0, 'fail'), label='bad job')
        results = []
        deadline = time.time() + 60
        while len(results) < 2 and time.time() < deadline:
            results += client.results()
            time.sleep(0.5)
        by_label = {r['label']: r for r in results}
        assert by_label['ok job']['status'] == 'done' and by_label['ok job']['result'] == 'hello'
        assert by_label['bad job']['status'] == 'failed' and 'TEST job failed on request' in by_label['bad job']['error']
        assert client.job_status([ok_id])[0]['status'] == 'done'
        status = common.call('/api/status', port=dash)
        assert {w['job_type'] for w in status['workers']} == {'TEST'}
    finally:
        for r in list(m.runners.values()):
            parent = psutil.Process(r['pid'])
            for p in parent.children(recursive=True) + [parent]:
                p.kill()
        for s in servers:
            s.shutdown()
    assert os.path.exists(tmp_path / 'state' / 'token')


def test_job_with_same_type_and_key_is_not_queued_again(tmp_path):
    m = Manager(tmp_path / 'jobs.sqlite')
    first = m.submit({**job('ann'), 'key': 'out.npy'})
    again = m.submit({**job('bob'), 'key': 'out.npy'})
    assert again == {'job_id': first['job_id'], 'duplicate': True}
    assert not m.submit({**job('ann'), 'key': 'other.npy'})['duplicate']
    assert not m.submit({**job('ann', job_type='Ripples'), 'key': 'out.npy'})['duplicate']
    # once finished, the same key can be queued again
    assigned = sync(m, 'ann')['assign']
    sync(m, 'ann', started=[a['job_id'] for a in assigned],
         finished=[{'job_id': a['job_id'], 'ok': True, 'result': '', 'worker': 'w'} for a in assigned])
    assert not m.submit({**job('bob'), 'key': 'out.npy'})['duplicate']


def test_video_job_handlers_write_error_file(tmp_path, monkeypatch):
    from LFP.Setup import Setup
    from Video import MotionEnergy, RecoveryMotion
    prefix = 'mouse_2026-10-14_test_001'
    (tmp_path / prefix).mkdir()
    assert MotionEnergy.run_job([Setup.LNCM, tmp_path, prefix, None]) == f'{prefix}: no video'  # no camera file
    assert MotionEnergy.error_fn(Setup.LNCM, tmp_path, prefix).read_text() == 'No video matches this recording.'
    video = tmp_path / '_RawData' / 'MonitorCam' / 'missing.mkv'
    RecoveryMotion.lock_fn(video).parent.mkdir(parents=True)
    RecoveryMotion.lock_fn(video).touch()

    def fail(video):
        raise RuntimeError('cannot read')
    monkeypatch.setattr(RecoveryMotion, 'compute_motion', fail)
    with pytest.raises(RuntimeError):
        RecoveryMotion.run_job(str(video))
    assert 'cannot read' in RecoveryMotion.error_fn(video).read_text()
    assert not RecoveryMotion.lock_fn(video).exists()

