# ARMAX System Identification

Python software developed for parameter estimation and system identification of discrete-time ARMAX models using a self-adaptive Evolutionary Programming algorithm.

## Features

- Self-adaptive Gaussian mutation for the estimation of ARMAX model parameters.
- Optimization benchmarks using the Sphere, Rosenbrock, Rastrigin, and Ackley functions.
- 10-fold cross-validation for model evaluation.
- Residual autocorrelation and whiteness analysis.
- Calculation of MSE, RMSE, MAPE, R², and Akaike Information Criterion (AIC).
- Support for datasets from the UCI Machine Learning Repository and local SISO CSV files.
- Graphical interface implemented with `customtkinter`.
- Checkpoint saving and restoration during optimization.

## Installation

Python 3.8 or newer is required.

```bash
git clone https://github.com/XPONOKPATWP/ep_sysid.git
cd ep_sysid
pip install -r requirements.txt
```

## Usage

Run the graphical interface with:

```bash
python system_id_gui.py
```

The application is used to configure the ARMAX model and Evolutionary Programming parameters, load the dataset, and run the identification and validation procedure.

## Project Structure

- **`ep_algorithm_core.py`**: Evolutionary Programming algorithm.
- **`armax_simulator.py`**: ARMAX model simulation and prediction.
- **`training_pipeline.py`**: Training and 10-fold cross-validation.
- **`validation_metrics.py`**: Calculation of MSE, RMSE, MAPE, R², and AIC.
- **`diagnostic_plotter.py`**: Generation of tracking, parity, and residual plots.
- **`optimization_benchmarks.py`**: Benchmark functions for the Evolutionary Programming algorithm.
- **`config_handler.py`**: Configuration parameters and checkpoint management.
- **`siso_data_parser.py`**: Dataset loading, preprocessing, and caching.
- **`system_id_gui.py`**: Graphical user interface.

## License

This project is distributed under the GNU General Public License v3.0 (GPLv3). See the `LICENSE` file for more information.
