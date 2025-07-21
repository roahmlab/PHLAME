# Generating Crocoddyl Results for the Paper

## Prerequisites

1. Create the conda environment from `croco_env_sp.yml`:
   ```bash
   conda env create -f croco_env_sp.yml
   ```

2. Install the Phlame forked version of [example-robot-data](https://github.com/roahmlab/example-robot-data-roahmlab-phlame).

## Execution

1. Change to the `Crocoddyl` directory:
```
cd Crocoddyl
```
2. Run the Crocoddyl reproduction script:
```
python3 -m phlame_reproduction.scr_digit_no_obstacles
python3 -m phlame_reproduction.scr_kinova_no_obstacles
```