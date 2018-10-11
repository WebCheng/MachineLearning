# MachineLearning
Using python3.7.0
libs numpy, pandas, math, datetime  


1. Miniconda setting for running Assignment1
# Download mini conda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# Execute Andaconda
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
# Change into bash for export file
bash
# Put the path to the Miniconda
export PATH="{PATH}/miniconda3/bin:$PATH"
# Check python from conda3
which python
# Install libs
python -m pip install numpy
python -m pip install pandas
# Python from conda env
python pa1.py

