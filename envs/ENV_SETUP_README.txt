# Description of installation process

# lncm env with uv (being tested, replaces the conda lncm and suite2p envs below)
The env is defined in pyproject.toml and pinned in uv.lock (both in the repo root). Each user builds their own env.
In PowerShell:
1. Install uv (once):
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   then open a new PowerShell window.
2. Build the env outside the repo (uv also downloads Python 3.11). Leave out "--extra suite2p" if you don't run motion correction:
    cd $HOME\PycharmProjects\LNCM-lib
    $env:UV_PROJECT_ENVIRONMENT = "$HOME\envs\lncm"
    uv sync --locked --extra suite2p
   Run the same two lines (after cd) again to update the env after uv.lock changes.
3. In PyCharm, add $HOME\envs\lncm\Scripts\python.exe as an existing interpreter and select it for the projects.
   Motion correction scripts run in this env, no separate suite2p interpreter is needed.
Qt: LNCM-lib GUIs use PySide6. suite2p's GUI (python -m suite2p) uses PyQt6, which comes with suite2p[gui].
To change packages, edit pyproject.toml and run "uv lock", then commit both files.

# To create the main env used in most analysis GUIs and scripts ("lncm"), use Python 3.11 and stick to conda-forge channel.

conda create --name lncm python=3.11 -c conda-forge --strict-channel-priority
conda activate lncm
conda install -c conda-forge -y scipy pyqt matplotlib opencv scikit-learn openpyxl shapely pandas jinja2 tifffile h5py scikit-image xlrd statsmodels numba requests pyqtgraph pyedflib lsq-ellipse pandas-stubs pyarrow qt6-multimedia

# See pycharm-workspace.md for how to set up Pycharm in a clean way
# see Pycharm console starting script.py for setting up console in PyCharm (required for run configs with Python console)

# To use SIMA for ROI detection in 2P movies, we need a Python 3.6 env ("lncm36"), 
# Note that calls to the functions that use this are made from the main env, and the path to the python 3.6 executable needs to be specified in envs/site_config.json
# It is enough to set this env up for one user on the server, and have the config point to that executable for all users.

conda create --name lncm36 python=3.6 -c conda-forge --strict-channel-priority
conda activate lncm36
conda install -c conda-forge -y scipy pyqt matplotlib opencv scikit-learn openpyxl shapely pandas jinja2 tifffile h5py scikit-image xlrd statsmodels numba requests pyqtgraph
pip install sima

# To use Suite2P for motion correction, we need a Python 3.11 env ("suite2p") with suite2p 1.x (tested with 1.1.0):
if you have a previous suite2p env (0.x), delete it (conda env remove --name suite2p),
and delete the remaining empty folder in envs called suite2p
note: suite2p 0.x envs don't work with the current pipeline.
in conda powershell prompt:
    conda create --name suite2p python=3.11 -c conda-forge --strict-channel-priority
    conda activate suite2p
    python -m pip install suite2p
    python -m pip install pandas requests openpyxl pyarrow
our analysis server has no GPU, suite2p runs on CPU with the default torch install (it logs that cuda is not working and uses cpu).

open the job script in your project (no need to have a separate S2P project)
for example, Proc2P/Analysis/S2P_template/20261007_BatchRun_MotionCorrect_Template.py (in LNCM-lib)
In settings/Python/interpreter, add the suite2p env as system python (but don't select it for the project)
Select run/edit configurations for your job script
    select the suite2p interpreter to run this script with
run the script with this specific configuration.

# For deeplabcut, we need a separate env (on GPU system with NVIDIA CUDA installed):
-clone DLC repo
-cd to repo root
conda env create -f DEEPLABCUT.yaml	
conda activate DEEPLABCUT
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
python -m deeplabcut

# For treadmill (PyControl_LNCM)
the lncm env does not work for this (gui written for an older pyqtgraph binding),
 use the default PyControl installation and add zmq if necessary