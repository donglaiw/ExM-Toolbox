# ExM-Toolbox
Python Toolbox for Expansion Microscopy Volumes


## Installation

1. Create and activate a conda environment
```
conda create -n exm-toolbox python=3
conda activate exm-toolbox
```

2. Compile SimpleITK + SimpleElastix [official link](https://simpleelastix.readthedocs.io/GettingStarted.html) (Can take hours to compile)
```
# may need install the higher version of Cmake
git clone https://github.com/SuperElastix/SimpleElastix
mkdir SimpleElastix_build
cd SimpleElastix_build
cmake ../SimpleElastix/SuperBuild
# -j4: 4 threads. do as many threads as needed
make -j4
```

3. Build and install Python package (current dir: `SimpleElastix_build/`)
```
cd SimpleITK-build/Wrapping/Python
python Packaging/setup.py install
```

4. Install other python packages
```
conda install -c conda-forge nd2reader numpy h5py pynrrd
```

5. Install environment in Jupyter notebook
```
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=exm-toolbox
```
