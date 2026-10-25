# Install instructions

## lncm env
The main env, used by the GUIs, analysis scripts and motion correction (suite2p). It is defined in `pyproject.toml` and pinned in `uv.lock` (both in the repo root). Each user builds their own env.

1. Install uv (once), in PowerShell:
   ```
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
2. Build the env by running `envs\update_env.bat` (double-click, or from a new cmd/PowerShell window).
   The location is `uv_env_dir` in `envs/site_config.json` (`N:/envs` in our lab); each user gets a subfolder:
   - The env goes to `<uv_env_dir>\<username>\lncm`.
   - uv's Python 3.11 goes to `<uv_env_dir>\<username>\uv\python`. The env needs it, don't delete it.
   - The download cache goes to `<uv_env_dir>\<username>\uv\cache`. It can be deleted.
3. In PyCharm, add `<uv_env_dir>\<username>\lncm\Scripts\python.exe` as an existing interpreter (type Python, not uv) and select it for the projects. Motion correction scripts run in this env too.
   - See `envs/pycharm-workspace.md` for how to set up PyCharm projects in a clean way.
   - See `envs/Pycharm console starting script.py` for setting up the console in PyCharm (required for run configs with Python console).
4. Check the install by running the tests from the repo root (takes about 20 s):
   ```
   <uv_env_dir>\<username>\lncm\Scripts\python.exe -m pytest
   ```
5. After pulling changes to `uv.lock`, run `envs\update_env.bat` again. It does nothing if the env is up to date.

Qt: PySide6 is the only Qt binding in the env. Don't add another one (e.g. PyQt6 via `suite2p[gui]`): bindings ship Qt DLLs with the same names, and when one binding's DLLs are loaded first, importing the other fails with `DLL load failed ... The specified procedure could not be found`.

To change packages, edit `pyproject.toml` and run `uv lock`, then commit both files.

## Linting with ruff
ruff is installed in the env (`dev` group in `pyproject.toml`) as `<uv_env_dir>\<username>\lncm\Scripts\ruff.exe`. The config in `pyproject.toml` only reports likely bugs (unused or undefined names, syntax errors), not style. Run it on the files you changed, not on the whole repo.

To run it from PyCharm on the open file, add an external tool in Settings > Tools > External Tools:
- Program: `$PyInterpreterDirectory$\ruff.exe`
- Arguments: `check --config <path to LNCM-lib>\pyproject.toml "$FilePath$"`. The explicit config path makes the same rules apply to files in other repos (e.g. 2p_code) that have no ruff config.
- Working directory: `$FileDir$`

Then use Tools > External Tools > ruff (or assign a shortcut in Keymap). `--fix` removes unused imports and similar automatically; check the diff afterwards.

## Envs for other tools
SIMA (Python 3.6), DeepLabCut and PyControl need separate envs: see `envs/ENV_SETUP_README.txt`.

# Looking for the python 3.6 version?
The main branch has been updated (Python 3.11).
The old version (Python 3.6) is available here: https://github.com/bdudok/LNCM-lib.git
