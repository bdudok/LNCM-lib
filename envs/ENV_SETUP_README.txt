Description of installation process

conda create --name lncm python=3.11 -c conda-forge -c defaults --strict-channel-priority
conda activate lncm
conda install -y pyqt matplotlib opencv scikit-learn openpyxl shapely pandas jinja2 tifffile h5py scipy xlrd statsmodels requests pyqtgraph pyedflib lsq-ellipse pandas-stubs pyarrow
