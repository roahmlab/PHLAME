# Generating Tables from the Paper

## Prerequisites
0. Install the Phlame fork of [example-robot-data](https://github.com/roahmlab/example-robot-data-roahmlab-phlame).
1. Follow the steps in:
   - `Phlame_comparisons/aligator/readme.md`
   - `Phlame_comparisons/Crocoddyl/readme.md`
2. Activate the `phlame` conda environment:
   ```bash
   conda activate phlame
   ```

## Execution

1. Change to the repository root
```bash
cd ~/Phlame/
```

2. Run the reproduction scripts that generate the data for Phlame.

```bash
python3 -m phlame_reproduction.scr_digit_no_obs
python3 -m phlame_reproduction.scr_digit_obstacles
python3 -m phlame_reproduction.scr_kinova_no_obs
python3 -m phlame_reproduction.scr_kinova_obstacles
```

3. Generate the tables
```
python3 phlame_reproduction/generate_tables.py
```

## Output
The generated tables are located in: `Phlame/storage_tables_paper`.