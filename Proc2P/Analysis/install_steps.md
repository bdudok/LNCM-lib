conda create -n py36 python=3.6\
conda activate py36\
\
conda install shapely\
pip install sima pandas\ 
conda install opencv h5py\
pip install pystackreg openpyxl xlrd imagecodecs

mkdir LNCM-lib/_Dependencies/PyControl\
cd LNCM-lib/_Dependencies/PyControl\
git clone https://github.com/pyControl/code.git\



