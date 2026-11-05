# Exploration notebooks

Install `uv sync --extra notebooks` and launch from the repository root:
`uv run marimo edit notebooks/try_model.py`.
Use the project environment instead of a sandboxed notebook environment.

- `try_model.py`: inspect the composed feature extractor and prediction head.
- `proteinmpnn.py`: explore encoder features.
- `inspect_megascale_validation.py`: inspect measured single-mutant validation records.
- `ThermoMPNN_D_Data_Exploration.ipynb`: explore measured double mutants and structures.

Prepare data/checkpoints using the root README first. The Marimo model notebooks
retain the measured-ddG comparison configuration; the supported epsilon training
command is documented in the root README. None of these notebooks is a required
pipeline step.
