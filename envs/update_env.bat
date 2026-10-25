@echo off
rem Builds or updates the lncm env in <uv_env_dir>\%USERNAME%\lncm from uv.lock (uv_env_dir is set in site_config.json).
rem Run it (double-click) after pulling changes to uv.lock. It does nothing if the env is up to date.
setlocal
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "(Get-Content -Raw '%~dp0site_config.json' | ConvertFrom-Json).uv_env_dir -replace '/', '\'"`) do set "ENV_DIR=%%i"
if not defined ENV_DIR (
    echo uv_env_dir is missing from envs\site_config.json.
    pause
    exit /b 1
)
set "BASE=%ENV_DIR%\%USERNAME%"

rem uv's Python installs and download cache go there too (stored for future sessions)
if not defined UV_PYTHON_INSTALL_DIR (
    set "UV_PYTHON_INSTALL_DIR=%BASE%\uv\python"
    setx UV_PYTHON_INSTALL_DIR "%BASE%\uv\python" >nul
)
if not defined UV_CACHE_DIR (
    set "UV_CACHE_DIR=%BASE%\uv\cache"
    setx UV_CACHE_DIR "%BASE%\uv\cache" >nul
)
rem the network share may not support the hardlinks uv uses by default
set "UV_LINK_MODE=copy"
rem not stored with setx: every uv project would use this env
set "UV_PROJECT_ENVIRONMENT=%BASE%\lncm"

cd /d "%~dp0.."
echo Syncing %UV_PROJECT_ENVIRONMENT%
uv sync --locked
if errorlevel 1 (
    echo uv sync failed, see the messages above.
) else (
    echo Done. PyCharm interpreter: %UV_PROJECT_ENVIRONMENT%\Scripts\python.exe
)
pause
