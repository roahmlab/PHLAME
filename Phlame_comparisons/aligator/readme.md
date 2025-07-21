
# Generating Aligator Results for the Paper

## Prerequisites

1. Create the conda environment from `alig_env_sp.yml`:
   ```bash
   conda env create -f alig_env_sp.yml
   ```

2. Install the Phlame forked version of [example-robot-data](https://github.com/roahmlab/example-robot-data-roahmlab-phlame)

## Execution

1. Change to the `aligator` directory:
```
cd aligator
```
2. Run the Aligator reproduction script:
```
python3 -m phlame_reproduction.scr_kinova_digit_obstacles
```