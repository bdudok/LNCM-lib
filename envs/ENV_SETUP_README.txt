# Description of installation process

# To create the main env used in most analysis GUIs and scripts ("lncm"), use Python 3.11 and stick to conda-forge channel.

conda create --name lncm python=3.11 -c conda-forge -c defaults --strict-channel-priority
conda activate lncm
conda install -y scipy pyqt matplotlib opencv scikit-learn openpyxl shapely pandas jinja2 tifffile h5py scikit-image xlrd statsmodels numba requests pyqtgraph pyedflib lsq-ellipse pandas-stubs pyarrow qt6-multimedia


# To use SIMA for ROI detection in 2P movies, we need a Python 3.6 env ("lncm36"), 
# Note that calls to the functions that use this are made from the main env, and the path to the python 3.6 executable needs to be specified in envs/site_config.json

conda create --name lncm36 python=3.6 -c conda-forge -c defaults --strict-channel-priority
conda activate lncm36
conda install -y scipy pyqt matplotlib opencv scikit-learn openpyxl shapely pandas jinja2 tifffile h5py scikit-image xlrd statsmodels numba requests pyqtgraph
pip install sima

# To use Suite2P for motion correction, we need a Python 3.9 env ("suite2p"):
follow S2P installation instructions. 

# For deeplabcut, we need a separate env (on GPU system with NVIDIA CUDA installed):
-clone DLC repo
-cd to repo root
conda env create -f DEEPLABCUT.yaml	
conda activate DEEPLABCUT
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
python -m deeplabcut

