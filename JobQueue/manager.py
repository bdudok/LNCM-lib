'''
Job queue manager. One instance runs on the analysis server (scheduled task, see JobQueue/README.md).
- Clients queue jobs through the localhost API. The manager is the only process that writes the database.
- Each user's runner (JobQueue/runner.py, in that user's account) reports every second; the manager answers with
  the jobs it assigned to it. Jobs run in the submitting user's account, with their own LNCM-lib checkout.
- Free slots of a job type go to the user with the fewest running jobs of that type, so one user's batch doesn't
  hold up the others.
- The dashboard (read-only) is served on the lab network.
Run: python -m JobQueue.manager (from the LNCM-lib folder)
'''
import sys
import json
import time
import uuid
import sqlite3
import threading
import traceback
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

from JobQueue.common import (JOB_TYPES, RUNNER_LOST_S, ASSIGN_TIMEOUT_S, api_port, dashboard_port,
                             manager_data_dir, display_name)

ACTIVE = ('assigned', 'running')
FINISHED = ('done', 'failed')
N_RECENT = 50

SCHEMA = '''
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    user TEXT NOT NULL,
    client_id TEXT,
    job_type TEXT NOT NULL,
    label TEXT,
    key TEXT,
    lib_root TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL,
    created REAL NOT NULL,
    assigned REAL,
    started REAL,
    finished REAL,
    runner_id TEXT,
    worker TEXT,
    result TEXT,
    error TEXT,
    delivered INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status);
CREATE INDEX IF NOT EXISTS jobs_client ON jobs (client_id, delivered);
CREATE INDEX IF NOT EXISTS jobs_key ON jobs (job_type, key);
CREATE TABLE IF NOT EXISTS users (
    user TEXT PRIMARY KEY,
    token TEXT NOT NULL,
    registered REAL NOT NULL
);
'''


class RequestError(Exception):
    pass


class Manager:

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()
        self.runners = {}  # runner_id: dict(user, lib_root, pid, last_seen, workers)
        self.started = time.time()

    # --- identity
    def _check_token(self, user: str, token: str):
        if not user or not token:
            raise RequestError('user and token are required')
        row = self.db.execute('SELECT token FROM users WHERE user=?', (user,)).fetchone()
        if row is None:
            self.db.execute('INSERT INTO users VALUES (?, ?, ?)', (user, token, time.time()))
        elif row['token'] != token:
            raise RequestError(f'token of {user} does not match the registered one. '
                               f'Ask the admin to remove {user} from the users table of the job queue database.')

    # --- client requests
    def submit(self, req: dict) -> dict:
        with self.lock:
            self._check_token(req.get('user'), req.get('token'))
            if req.get('job_type') not in JOB_TYPES:
                raise RequestError(f'unknown job type {req.get("job_type")}. Known: {list(JOB_TYPES)}')
            # a job of the same type and key that is waiting or running (queued by anyone) is not queued again
            key = req.get('key')
            if key:
                row = self.db.execute(f"SELECT id FROM jobs WHERE job_type=? AND key=? AND status IN "
                                      f"{('pending', *ACTIVE)}", (req['job_type'], key)).fetchone()
                if row is not None:
                    return {'job_id': row['id'], 'duplicate': True}
            job_id = uuid.uuid4().hex
            self.db.execute('INSERT INTO jobs (id, user, client_id, job_type, label, key, lib_root, payload, status, '
                            'created) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                            (job_id, req['user'], req.get('client_id'), req['job_type'], req.get('label', ''), key,
                             req['lib_root'], json.dumps(req['payload']), 'pending', time.time()))
            return {'job_id': job_id, 'duplicate': False}

    def results(self, req: dict) -> dict:
        '''Finished jobs of a client that it hasn't received yet.'''
        with self.lock:
            self._check_token(req.get('user'), req.get('token'))
            rows = self.db.execute(f'SELECT * FROM jobs WHERE client_id=? AND user=? AND delivered=0 AND status IN '
                                   f'{FINISHED} ORDER BY finished', (req['client_id'], req['user'])).fetchall()
            ids = [r['id'] for r in rows]
            self.db.executemany('UPDATE jobs SET delivered=1 WHERE id=?', [(i,) for i in ids])
            return {'results': [self._job_summary(r) for r in rows]}

    def jobs(self, req: dict) -> dict:
        '''Status of the given jobs (for scripts that wait for jobs they queued, which may have been queued earlier
        by another user: see the duplicate check in submit). The dashboard shows the same to everyone.'''
        with self.lock:
            self._check_token(req.get('user'), req.get('token'))
            ids = req.get('job_ids', [])
            rows = self.db.execute(f'SELECT * FROM jobs WHERE id IN ({",".join("?" * len(ids))})', ids).fetchall()
            return {'jobs': [self._job_summary(r) for r in rows]}

    def health(self, query: dict) -> dict:
        out = {'ok': True, 'uptime': time.time() - self.started}
        if 'user' in query and 'lib_root' in query:
            with self.lock:
                out['runner_alive'] = self._live_runner(query['user'], query['lib_root']) is not None
                out['pending'] = self.db.execute(
                    f'SELECT COUNT(*) FROM jobs WHERE user=? AND lib_root=? AND status IN {("pending", *ACTIVE)}',
                    (query['user'], query['lib_root'])).fetchone()[0]
        return out

    # --- runner requests
    def sync(self, req: dict) -> dict:
        '''
        A runner reports its workers, the jobs it started and finished, and receives newly assigned jobs.
        Request: runner_id, user, token, lib_root, pid, workers [{name, job_type, pid, job_id}],
                 started [job_id], finished [{job_id, ok, result, error, worker}]
        '''
        now = time.time()
        with self.lock:
            self._check_token(req.get('user'), req.get('token'))
            rid = req['runner_id']
            other = self._live_runner(req['user'], req['lib_root'])
            if other is not None and other != rid:
                return {'exit': True, 'reason': f'runner {other} already serves this user and checkout'}
            self.runners[rid] = {'user': req['user'], 'lib_root': req['lib_root'], 'pid': req.get('pid'),
                                 'last_seen': now, 'workers': req.get('workers', [])}
            for job_id in req.get('started', []):
                self.db.execute("UPDATE jobs SET status='running', started=? WHERE id=? AND runner_id=?",
                                (now, job_id, rid))
            for f in req.get('finished', []):
                self.db.execute('UPDATE jobs SET status=?, finished=?, result=?, error=?, worker=? '
                                'WHERE id=? AND runner_id=?',
                                ('done' if f['ok'] else 'failed', now, f.get('result'), f.get('error'),
                                 f.get('worker'), f['job_id'], rid))
            self._expire(now)
            self._schedule(now)
            rows = self.db.execute("SELECT id, job_type, payload FROM jobs WHERE runner_id=? AND status='assigned'",
                                   (rid,)).fetchall()
            pending = self.db.execute("SELECT COUNT(*) FROM jobs WHERE user=? AND lib_root=? AND status='pending'",
                                      (req['user'], req['lib_root'])).fetchone()[0]
            return {'assign': [{'job_id': r['id'], 'job_type': r['job_type'], 'payload': json.loads(r['payload'])}
                               for r in rows],
                    'pending': pending}

    def _live_runner(self, user, lib_root):
        now = time.time()
        live = [(r['last_seen'], rid) for rid, r in self.runners.items()
                if r['user'] == user and r['lib_root'] == lib_root and now - r['last_seen'] < RUNNER_LOST_S]
        return max(live)[1] if live else None

    def _expire(self, now):
        '''Drop runners that stopped reporting, fail their running jobs, and requeue stale assignments.'''
        for rid in [rid for rid, r in self.runners.items() if now - r['last_seen'] > RUNNER_LOST_S]:
            del self.runners[rid]
        # after a manager restart, give runners time to report before failing their jobs
        if now - self.started > RUNNER_LOST_S:
            for row in self.db.execute(f'SELECT id, runner_id, status FROM jobs WHERE status IN {ACTIVE}').fetchall():
                if row['runner_id'] not in self.runners:
                    if row['status'] == 'running':
                        self.db.execute("UPDATE jobs SET status='failed', finished=?, error=? WHERE id=?",
                                        (now, 'The runner stopped while the job was running.', row['id']))
                    else:
                        self._unassign(row['id'])
        for row in self.db.execute("SELECT id FROM jobs WHERE status='assigned' AND assigned<?",
                                   (now - ASSIGN_TIMEOUT_S,)).fetchall():
            self._unassign(row['id'])

    def _unassign(self, job_id):
        self.db.execute("UPDATE jobs SET status='pending', runner_id=NULL, assigned=NULL WHERE id=?", (job_id,))

    def _schedule(self, now):
        '''Assign pending jobs to live runners, within each type's max_n, giving each free slot to the user with
        the fewest active jobs of that type (then fewest active jobs overall, then the oldest job).'''
        runner_of = {}
        for rid, r in self.runners.items():
            key = (r['user'], r['lib_root'])
            if key not in runner_of or r['last_seen'] > self.runners[runner_of[key]]['last_seen']:
                runner_of[key] = rid
        if not runner_of:
            return
        active = self.db.execute(f'SELECT user, job_type, COUNT(*) AS n FROM jobs WHERE status IN {ACTIVE} '
                                 f'GROUP BY user, job_type').fetchall()
        n_type_user, n_type, n_user = {}, {}, {}
        for r in active:
            n_type_user[(r['job_type'], r['user'])] = r['n']
            n_type[r['job_type']] = n_type.get(r['job_type'], 0) + r['n']
            n_user[r['user']] = n_user.get(r['user'], 0) + r['n']
        pending = self.db.execute("SELECT id, user, job_type, lib_root, created FROM jobs WHERE status='pending' "
                                  "ORDER BY created").fetchall()
        waiting = {}  # job_type: user: [jobs] with a live runner, oldest first
        for job in pending:
            if (job['user'], job['lib_root']) in runner_of and job['job_type'] in JOB_TYPES:
                waiting.setdefault(job['job_type'], {}).setdefault(job['user'], []).append(job)
        for job_type, by_user in waiting.items():
            free = JOB_TYPES[job_type].max_n - n_type.get(job_type, 0)
            while free > 0 and by_user:
                user = min(by_user, key=lambda u: (n_type_user.get((job_type, u), 0), n_user.get(u, 0),
                                                   by_user[u][0]['created']))
                job = by_user[user].pop(0)
                if not by_user[user]:
                    del by_user[user]
                self.db.execute("UPDATE jobs SET status='assigned', assigned=?, runner_id=? WHERE id=?",
                                (now, runner_of[(job['user'], job['lib_root'])], job['id']))
                n_type_user[(job_type, user)] = n_type_user.get((job_type, user), 0) + 1
                n_user[user] = n_user.get(user, 0) + 1
                free -= 1

    def housekeeping(self):
        '''Called periodically so jobs of stopped runners fail even if no runner reports.'''
        with self.lock:
            self._expire(time.time())

    # --- dashboard
    @staticmethod
    def _job_summary(r) -> dict:
        return {'job_id': r['id'], 'user': r['user'], 'name': display_name(r['user']), 'job_type': r['job_type'],
                'label': r['label'], 'status': r['status'], 'created': r['created'], 'started': r['started'],
                'finished': r['finished'], 'worker': r['worker'], 'result': r['result'], 'error': r['error']}

    def status(self) -> dict:
        now = time.time()
        with self.lock:
            workers = []
            for r in self.runners.values():
                for w in r['workers']:
                    workers.append({'name': display_name(r['user']), 'worker': w['name'], 'job_type': w['job_type'],
                                    'pid': w['pid'], 'busy': w['job_id'] is not None,
                                    'label': self._label(w['job_id'])})
            counts = self.db.execute("SELECT job_type, status, COUNT(*) AS n FROM jobs "
                                     "WHERE status IN ('pending', 'assigned', 'running') GROUP BY job_type, status"
                                     ).fetchall()
            queue = {t: {'pending': 0, 'running': 0, 'max_n': spec.max_n} for t, spec in JOB_TYPES.items()}
            for c in counts:
                q = queue.setdefault(c['job_type'], {'pending': 0, 'running': 0, 'max_n': 0})
                q['pending' if c['status'] == 'pending' else 'running'] += c['n']
            waiting_users = [display_name(r['user']) for r in self.db.execute(
                "SELECT DISTINCT user, lib_root FROM jobs WHERE status='pending'").fetchall()
                if self._live_runner(r['user'], r['lib_root']) is None]
            running = self.db.execute(f"SELECT * FROM jobs WHERE status IN {ACTIVE} ORDER BY created").fetchall()
            done = self.db.execute("SELECT * FROM jobs WHERE status='done' ORDER BY finished DESC LIMIT ?",
                                   (N_RECENT,)).fetchall()
            failed = self.db.execute("SELECT * FROM jobs WHERE status='failed' ORDER BY finished DESC LIMIT ?",
                                     (N_RECENT,)).fetchall()
            return {'now': now, 'uptime': now - self.started, 'workers': workers, 'queue': queue,
                    'no_runner': sorted(set(waiting_users)),
                    'running': [self._job_summary(r) for r in running],
                    'done': [self._job_summary(r) for r in done],
                    'failed': [self._job_summary(r) for r in failed]}

    def _label(self, job_id):
        if job_id is None:
            return None
        row = self.db.execute('SELECT label FROM jobs WHERE id=?', (job_id,)).fetchone()
        return None if row is None else row['label']


def make_api_handler(manager: Manager):
    posts = {'/submit': manager.submit, '/results': manager.results, '/jobs': manager.jobs, '/sync': manager.sync}

    class ApiHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == '/health':
                self._reply(200, manager.health({k: v[0] for k, v in parse_qs(url.query).items()}))
            else:
                self._reply(404, {'error': 'not found'})

        def do_POST(self):
            func = posts.get(self.path)
            if func is None:
                return self._reply(404, {'error': 'not found'})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
                self._reply(200, func(req))
            except (RequestError, KeyError, ValueError) as e:
                self._reply(400, {'error': str(e)})
            except Exception:
                traceback.print_exc()
                self._reply(500, {'error': 'manager error, see its log'})

    return ApiHandler


def make_dashboard_handler(manager: Manager):
    page = (Path(__file__).parent / 'dashboard.html').read_bytes()

    class DashboardHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == '/':
                self._send(page, 'text/html; charset=utf-8')
            elif path == '/api/status':
                self._send(json.dumps(manager.status()).encode(), 'application/json')
            else:
                self.send_error(404)

    return DashboardHandler


def serve(api_port_n: int, dashboard_port_n: int, data_dir: Path):
    manager = Manager(data_dir / 'jobs.sqlite')
    api = ThreadingHTTPServer(('127.0.0.1', api_port_n), make_api_handler(manager))
    dash = ThreadingHTTPServer(('0.0.0.0', dashboard_port_n), make_dashboard_handler(manager))
    for server in (api, dash):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f'Job queue manager: API on 127.0.0.1:{api_port_n}, dashboard on port {dashboard_port_n}, '
          f'database in {data_dir}', flush=True)
    try:
        while True:
            time.sleep(5)
            manager.housekeeping()
    finally:
        api.shutdown()
        dash.shutdown()


def main():
    data_dir = manager_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    log = open(data_dir / 'manager.log', 'a', buffering=1)
    sys.stdout = sys.stderr = log
    print(time.strftime('%Y-%m-%d %H:%M:%S'), 'starting', flush=True)
    try:
        serve(api_port(), dashboard_port(), data_dir)
    except OSError as e:
        print(f'Could not open the ports ({e}). Is another manager running?', flush=True)
        raise


if __name__ == '__main__':
    main()
