# Install instructions

## lncm env
1. Install uv (once), in PowerShell:
   ```
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
2. Build the env by running `envs\update_env.bat` (double-click, or from a new cmd/PowerShell window).
   The env goes to `<uv_env_dir>\<username>\lncm`; `uv_env_dir` is set in `envs/site_config.json` (`N:/envs` in our lab).
3. In PyCharm, add `<uv_env_dir>\<username>\lncm\Scripts\python.exe` as an existing interpreter (type Python, not uv) and select it for the projects.
   - See `envs/pycharm-workspace.md` for how to set up PyCharm projects in a clean way.
   - See `envs/Pycharm console starting script.py` for setting up the console in PyCharm (required for run configs with Python console).
4. Check the install by running the tests from the repo root (takes about 20 s):
   ```
   <uv_env_dir>\<username>\lncm\Scripts\python.exe -m pytest
   ```
5. If importing LNCM-lib warns that the env is not up to date, run `envs\update_env.bat` again.

## Envs for other tools
SIMA (Python 3.6), DeepLabCut and PyControl need separate envs: see `envs/ENV_SETUP_README.txt`.
