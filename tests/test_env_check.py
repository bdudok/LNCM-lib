from envs.env_check import SYNCED_LOCK, env_is_current


def test_env_check(tmp_path):
    repo_lock = tmp_path / 'uv.lock'
    repo_lock.write_text('version = 2\n')
    env_dir = tmp_path / 'envs'
    env = env_dir / 'user' / 'lncm'
    env.mkdir(parents=True)
    # never synced by update_env.bat
    assert not env_is_current(env, env_dir, repo_lock)
    (env / SYNCED_LOCK).write_text('version = 2\n')
    assert env_is_current(env, env_dir, repo_lock)
    # uv.lock changed after the sync
    repo_lock.write_text('version = 3\n')
    assert not env_is_current(env, env_dir, repo_lock)
    # envs elsewhere are not checked
    assert env_is_current(tmp_path / 'other_env', env_dir, repo_lock)
