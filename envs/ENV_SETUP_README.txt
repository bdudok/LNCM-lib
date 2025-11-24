# Description of installation process

conda create --name lncm python=3.11 -c conda-forge -c defaults --strict-channel-priority
conda activate lncm
conda install -y scipy pyqt matplotlib opencv scikit-learn openpyxl shapely pandas jinja2 tifffile h5py xlrd statsmodels requests pyqtgraph pyedflib lsq-ellipse pandas-stubs pyarrow
