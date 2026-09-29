# Mutation Prediction

## Setup

### Dependencies

### Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python dependency management.

1. **Install uv** if you don't already have it:

   ```bash
   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for other platforms.

2. **Clone the repository and enter the project directory:**

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

3. **Install the project dependencies:**

   ```bash
   uv sync
   ```

   This creates a virtual environment in `.venv` and installs the dependencies specified in `uv.lock`.

4. **Run commands using the project environment:**

   ```bash
   uv run python <script.py>
   ```

   Or activate the environment manually:

   ```bash
   source .venv/bin/activate
   ```

   On Windows:

   ```powershell
   .venv\Scripts\activate
   ```

5. **Adding new dependencies**
   If you add new dependencies, use `uv` instead of `pip`. For example, to install numpy, use

   ```bash
   uv add numpy
   ```
   This will automatically add the dependency for in `pyproject.toml`, and other users can install
   via `uv sync`

