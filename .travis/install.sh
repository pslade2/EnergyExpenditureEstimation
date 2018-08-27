#!/bin/bash

# matplot lib install
#python -mpip install -U pip
#python -mpip install -U matplotlib

# scipy install
#python -m pip install --upgrade pip
#pip install --upgrade pip setuptools wheel
#sudo apt-get build-dep python-scipy
#sudo apt-get install -qq python-scipy
#pip install scipy

python -m pip install --user numpy scipy matplotlib

# scikit-learn install
pip install -U scikit-learn

# tensorflow
pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.1-cp35-cp35m-linux_x86_64.whl

