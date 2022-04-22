import os, sys
from distutils.sysconfig import get_python_inc
from setuptools import setup, Extension, find_packages

def getInclude():
    dirName = get_python_inc()
    return [dirName, os.path.dirname(dirName)]

def setup_package():
    __version__ = '0.1'
    url = 'https://github.com/donglaiw/ExM-Toolbox'

    setup(name='exm',
        description='Expansion Microscopy Image Analysis Toolbox',
        version=__version__,
        url=url,
        license='MIT',
        author='Donglai Wei, Emma Besier, Ruihan Zhang, Zudi Lin',
        include_dirs=getInclude(), 
        packages=find_packages()
    )

if __name__=='__main__':
    # install main python functions
    # pip install -r requirements.txt --editable . 
    setup_package()
