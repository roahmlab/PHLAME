# PHLAME: Pseudo-spectral Heat fLow using the Affine geoMetric heat flow Equation

## Installation instructions for development
1. Clone the repo
2. `cd PHLAME`
3. Create a conda environment using the provided yaml file by running `conda env create --name phlame -f phlame_env.yml`
4. Activate the conda environment `conda activate phlame`.
5. Compile the pybinding: `cd cpp/` and `make`. Note, This step will fail in Mac unfortunately.

## Running tests
1. `conda activate phlame`
2. Add the modules to your `PYTHONPATH` using the following. Note in place of `<your_phlame_directory>` below make sure to use the full aboslute path to your phlame directory during the export:
```
export PYTHONPATH="<your_phlame_directory>src/phlame":"${PYTHONPATH}"
export PYTHONPATH="<your_phlame_directory>src/":"${PYTHONPATH}"
```
3. Run tests to verify correct installation
    - `python3 installation_tests/scikits_odes_test.py`
    - `python3 tests/run_unit_tests.py` 
