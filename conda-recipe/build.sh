#!/bin/bash
set -e  # Exit on error

# CONDA_PREFIX=$(conda info --json | python -c "import sys, json; print(json.load(sys.stdin)['active_prefix'])")
# export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH

# Change to the cpp directory and build the C++ code
cd cpp
make