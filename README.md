# Evolutionary Programming System Identification (`ep_sysid`)

`ep_sysid` is a Python tool for discrete-time dynamical system identification using Evolutionary Programming (EP). It estimates parameters and hyperparameters for linear (**ARMAX**) and nonlinear (**MLP** neural network and **SVR**) models across **SISO**, **MISO**, **SIMO**, and **MIMO** input-output configurations.

The project includes both a desktop GUI (`CustomTkinter`) and a command-line interface (`main_console.py`) for training, $K$-fold cross-validation, free-run simulation, and model comparison.

## What It Does

1. **System Modes (`SISO`, `MISO`, `SIMO`, `MIMO`)**

   * Routes single-channel or multi-channel input ($u$) and output ($y$) signals from local CSV files or UCI Machine Learning Repository datasets.

   * Applies per-channel `StandardScaler` normalization strictly on training splits to prevent test-set leakage.

2. **Model Architectures**

   * **ARMAX (`armax_simulator.py`):** Simulates $A(q)y(t) = B(q)u(t - n_k) + C(q)e(t)$ in both 1-step-ahead prediction and recursive free-run simulation. Unstable candidate $A(q)$ and $C(q)$ polynomials are projected inside the unit circle ($\vert{}z\vert{} \le 0.98$).

   * **MLP NARX (`mlp_simulator.py`):** Feedforward neural network with configurable hidden layers, $\tanh$ activations, Xavier weight initialization bounds, and periodic Ridge regression refinement on the linear output layer.

   * **SVR NARX (`svr_simulator.py`):** Support Vector Regression with an RBF kernel. Optimizes $[C, \epsilon, \gamma]$ using a subsampled training cache and evaluates free-run simulation directly in dual space.

3. **Evolutionary Optimizer (`ep_algorithm_core.py`)**

   * Built on `DEAP` with `multiprocessing` worker pools.

   * Supports self-adaptive Gaussian, Cauchy, and Hybrid mutation operators, $\text{BLX-}\alpha$ crossover, and truncation or round-robin tournament selection.

   * Saves atomic `.pkl` checkpoints per fold so interrupted runs can be resumed.

4. **Validation & Diagnostics (`validation_metrics.py`, `diagnostic_plotter.py`)**

   * Computes $\text{MSE}$, $\text{RMSE}$, $\text{MAPE}$, $R^2$, $\text{NRMSE}$ ($\text{FIT}\%$), $\text{VAF}\%$, $\text{AIC}$, $\text{AIC}_c$, $\text{BIC}$, $\text{FPE}$, and residual whiteness tests ($\text{ACF}$ and input-residual $\text{CCF}$).

   * Exports time-series tracking plots, parity plots, extrema zoom plots, residual distributions, Bode diagrams, pole-zero maps, $\text{LaTeX}$ tables (`.tex`), and Excel/CSV summary sheets.

## Repository Structure

```
ep_sysid/
├── config.json                  # Default configuration parameters
├── config_handler.py            # Config loader, validator, and checkpoint saver
├── multivariable_data_parser.py # CSV loader, UCI dataset fetcher, and channel router
├── ep_algorithm_core.py         # DEAP evolutionary loop, mutations, crossover, and worker pool
├── armax_simulator.py           # ARMAX 1-step prediction and free-run simulator
├── mlp_simulator.py             # MLP NARX 1-step prediction and free-run simulator
├── svr_simulator.py             # SVR NARX 1-step prediction and free-run simulator
├── optimization_benchmarks.py   # Standard test functions (Sphere, Rosenbrock, Rastrigin, etc.)
├── training_pipeline.py         # K-fold cross-validation, inference, comparison, and exports
├── validation_metrics.py        # Error metrics, information criteria, and ACF/CCF tests
├── diagnostic_plotter.py        # Matplotlib plotting routines
├── system_id_gui.py             # CustomTkinter desktop GUI
├── main_console.py              # Command-line interface
└── requirements.txt             # Python dependencies

```

## Installation & Build Instructions

### 1. Prerequisites

* **Python:** 3.9 to 3.12 (with `tkinter` installed)

* **OS:** Windows 10/11, Linux, or macOS

### 2. Clone and Set Up a Virtual Environment

```
git clone https://github.com/XPONOKPATWP/ep_sysid.git
cd ep_sysid

```

**Windows (PowerShell / CMD):**

```
python -m venv .venv
.venv\Scripts\activate

```

**Linux / macOS:**

```
python3 -m venv .venv
source .venv/bin/activate

```

### 3. Install Dependencies

```
python -m pip install --upgrade pip
pip install -r requirements.txt

```

If you do not have a `requirements.txt` file yet, install the required packages directly:

```
pip install numpy scipy pandas matplotlib scikit-learn deap customtkinter requests ucimlrepo openpyxl pillow

```

### 4. Build Standalone Executables (Optional)

You can package the GUI or CLI into standalone executables using `PyInstaller`:

```bash
pip install pyinstaller
```

**Build the Desktop GUI executable (Windows, Linux, or macOS):**
  ```bash
  pyinstaller --noconfirm --onedir --windowed --collect-all customtkinter --copy-metadata moocore --copy-metadata deap --name "EP_SysID_GUI" system_id_gui.py
  ```

**Build the Command-Line CLI executable (Windows, Linux, or macOS):**
  ```bash
  pyinstaller --noconfirm --onefile --console --copy-metadata moocore --copy-metadata deap --name "ep_sysid_cli" main_console.py
  ```

> **Note:** If `config.json` is not present alongside the executable on its first run, `config_handler.py` automatically generates one with default hyperparameters.

The compiled binaries will be placed in the `dist/` folder.

## How to Use

### Option 1: Desktop GUI

Run the GUI:

```
python system_id_gui.py

```

1. **Dataset Configuration:** Select a cached UCI dataset (`UCI_Energy_Efficiency` or `UCI_Power_Plant`), download another UCI dataset by numeric ID, or click **Load Local CSV** (last column is treated as the output by default).

2. **Model Architecture & Settings:** Choose the **Model Type** (`ARMAX`, `MLP`, `SVR`, or `ALL` for a 3-model batch) and **System Mode** (`SISO`, `MISO`, `SIMO`, `MIMO`). Set the lag orders (`na`, `nb`, `nc`, `nk`) and MLP hidden layers (e.g., `12, 8, 4`).

3. **Evolutionary Algorithm Hyperparameters:** Set the selection strategy (`Truncation` or `Tournament`), mutation operator (`Gaussian`, `Cauchy`, `Hybrid`), population size, max generations, mutation step, crossover probability, and $K$-folds.

4. **Execution:**

   * Click **Run Optimization** to start training. Progress for each job is shown in the **Active Model Training Monitor** at the bottom.

   * Enable **Queue Training** if you want multiple jobs to run one after another instead of in parallel.

   * Click **Resume Previous Run** to continue an interrupted run from its folder in `results/`.

   * Click **Evaluate / Compare Models** to test one or more saved `*_model.pkl` or `*_model.json` files on any dataset.

   * Click **Export Summary Sheet** to compile all runs inside `results/` into a single `.xlsx` and `.csv` summary file.

### Option 2: Command-Line Interface (`main_console.py`)

* **Run training using the settings in `config.json`:**

  ```
  python main_console.py
  
  ```

* **Train a specific model with command-line overrides:**

  ```
  python main_console.py train --dataset UCI_Energy_Efficiency --model ARMAX --mode MIMO --folds 10 --pop 180 --gens 120
  
  ```

  *(You can also pass a local CSV path to `--dataset`, or use `--model ALL` to train ARMAX, MLP, and SVR consecutively and compare them.)*

* **Evaluate a trained model on a dataset:**

  ```
  python main_console.py evaluate --model results/<run_folder>/<model_file>.pkl --dataset UCI_Energy_Efficiency
  
  ```

* **Compare 2 or more trained models head-to-head:**

  ```
  python main_console.py compare --models results/<folder1>/<model1>.pkl results/<folder2>/<model2>.pkl --dataset UCI_Energy_Efficiency
  
  ```

* **Run standard optimization benchmark functions (Sphere, Rosenbrock, Rastrigin, Ackley, Griewank, Schwefel):**

  ```
  python main_console.py benchmark --dimensions 5 --trials 10
  
  ```

* **Compile all results into a master CSV and Excel workbook:**

  ```
  python main_console.py summary --dir results
  
  ```

## Output Files

Each training run creates a timestamped folder inside `results/` containing:

* `<Model>_<Dataset>_<ID>_model.pkl` & `.json` — Saved model weights, lag structure, and training scalers.

* `training_report.txt` — Per-fold metrics, identified parameters, and 10-fold cross-validation averages.

* `table_results.tex` & `model_equations.tex` — $\text{LaTeX}$ performance table and transfer function equations.

* Diagnostic plots (`.png`) — Optimization convergence, time-series tracking, parity scatter, extrema zoom, residual distribution, residual $\text{ACF}$, cross-validation variance, free-run simulation vs. prediction, and Bode / pole-zero plots (for ARMAX).

## License & Author

* **Author:** Georgios Fanourios Karachalios Stagkas ([@XPONOKPATWP](https://github.com/XPONOKPATWP))

* **License:** GNU General Public License v3.0 (GPLv3) — see [LICENSE](LICENSE) for details.
