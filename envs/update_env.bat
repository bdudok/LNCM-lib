@echo off
rem Builds or updates the lncm env in N:\envs\%USERNAME%\lncm from uv.lock.
rem Run it (double-click) after pulling changes to uv.lock. It does nothing if the env is up to date.
rem First build with motion correction: update_env.bat suite2p
rem Later runs keep suite2p if the env already has it.
setlocal
set "BASE=N:\envs\%USERNAME%"

rem uv's Python installs and download cache go to N: too (stored for future sessions)
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

set "EXTRA="
if /i "%~1"=="suite2p" set "EXTRA=--extra suite2p"
if exist "%UV_PROJECT_ENVIRONMENT%\Lib\site-packages\suite2p" set "EXTRA=--extra suite2p"

cd /d "%~dp0.."
echo Syncing %UV_PROJECT_ENVIRONMENT% %EXTRA%
uv sync --locked %EXTRA%
if errorlevel 1 (
    echo uv sync failed, see the messages above.
) else (
    echo Done. PyCharm interpreter: %UV_PROJECT_ENVIRONMENT%\Scripts\python.exe
)
pause
