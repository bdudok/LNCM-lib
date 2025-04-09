# Install instructions
This is tested on Python 3.6
A yml file is included in the repo for creating a conda env

- conda env create -f PATH_TO_FILE\2pcode36.yml

this does not export everything, read error messages and install missing dependencies with pip install
- conda install numba
- pip install imagecodecs


# Start from scratch
If cloning the conda  env doesn't work, here are the steps to start from scratch:
<<<<<<< HEAD
- conda create -n 2pcode36 python=3.6
- conda activate 2pcode36
=======
- conda create -n py36 python=3.6
- conda activate py36
>>>>>>> origin/main

- conda install shapely
- pip install sima pandas
- conda install opencv
- conda install h5py
- pip install pystackreg openpyxl xlrd
- conda install numba
- pip install imagecodecs

then read error messages and install missing dependencies with pip install

# Why are we on Python 3.6
Due to some dependencies (oasis, sima use shapely and numba which require old numpy which only exists for old python)