"""
main_console.py

Console interface for running system identification tasks without the GUI:
  1. Train models from config.json
  2. Run standard optimization benchmarks (Sphere, Rastrigin, etc.)
  3. Single-model diagnostics and plots
  4. Compare multiple trained models
"""

import sys
import os
import glob
import time
import uuid
import argparse
import multiprocessing
from datetime import datetime

# Limit linear algebra threads per process so multiprocessing workers don't fight over CPU cores
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from config_handler import load_hyperparameters
from multivariable_data_parser import (
    load_cached_uci_benchmarks,
    get_dataset_summary_info,
    parse_local_csv_file
)
from training_pipeline import (
    execute_kfold_training_pipeline,
    execute_inference_pipeline,
    execute_model_comparison_pipeline,
    generate_results_summary_sheet
)
from optimization_benchmarks import run_full_benchmark_suite


def _print_banner(title):
    print("\n" + "=" * 76)
    print(f"  EVOLUTIONARY SYSTEM IDENTIFICATION (EP-SysId) -- {title}")
    print("=" * 76)


def _resolve_dataset(dataset_name, cached_datasets):
    # Support direct local CSV file path in CLI
    if dataset_name and os.path.isfile(dataset_name):
        ds_name, u_mat, y_mat, err = parse_local_csv_file(dataset_name)
        if err is None and u_mat is not None and y_mat is not None:
            return ds_name, (u_mat, y_mat)
        else:
            print(f"[!] Error parsing local CSV file '{dataset_name}': {err}")
            sys.exit(1)

    if dataset_name and dataset_name in cached_datasets:
        return dataset_name, cached_datasets[dataset_name]

    # Case-insensitive or partial match fallback
    if dataset_name:
        for name in cached_datasets:
            if dataset_name.lower() == name.lower():
                return name, cached_datasets[name]

    if not cached_datasets:
        print("[!] Error: No datasets available in cache or online. Please check network connection or provide a local CSV file.")
        sys.exit(1)

    # Default to first available dataset
    first_name = list(cached_datasets.keys())[0]
    return first_name, cached_datasets[first_name]


def run_training_command(args, config):
    _print_banner("TRAINING PIPELINE")

    cached_datasets = load_cached_uci_benchmarks()
    target_ds_name = args.dataset or config.get("dataset", "UCI_Energy_Efficiency")
    ds_name, (u_raw, y_raw) = _resolve_dataset(target_ds_name, cached_datasets)

    active_config = config.copy()
    if args.model:
        active_config["model_type"] = args.model
    if args.mode:
        active_config["system_mode"] = args.mode.upper()
    if args.folds:
        active_config["k_folds"] = max(2, int(args.folds))
    if args.pop:
        active_config["pop_size"] = max(10, int(args.pop))
    if args.gens:
        active_config["num_gen"] = max(1, int(args.gens))

    raw_model_type = str(active_config.get("model_type") or "MLP")
    system_mode = str(active_config.get("system_mode") or "MIMO").upper()
    k_folds = max(2, int(active_config.get("k_folds") or 10))
    pop_size = max(10, int(active_config.get("pop_size") or 180))
    num_gen = max(1, int(active_config.get("num_gen") or 120))

    # If 'ALL' is selected, run ARMAX, MLP, and SVR consecutively for comparison
    models_to_run = ["ARMAX", "MLP", "SVR"] if "ALL" in raw_model_type.upper() else [raw_model_type]
    completed_models = []
    timestamp_sig = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_id = uuid.uuid4().hex[:6]

    for m_type in models_to_run:
        run_cfg = active_config.copy()
        run_cfg["model_type"] = m_type
        output_dir = os.path.join("results", f"{m_type}_{system_mode}_{ds_name}_{timestamp_sig}_{batch_id}")
        os.makedirs(output_dir, exist_ok=True)

        print(f"  Dataset         : {ds_name} (Samples: {len(y_raw)})")
        print(f"  Model Type      : {m_type} ({system_mode})")
        print(f"  Lags & Structure: na={run_cfg.get('na')}, nb={run_cfg.get('nb')}, nc={run_cfg.get('nc', 1)}, nk={run_cfg.get('nk')}")
        if m_type == "MLP":
            print(f"  Hidden Layers   : {run_cfg.get('mlp_hidden_nodes')}")
        print(f"  Algorithm Config: Pop={pop_size}, Max Gens={num_gen}, Folds={k_folds}, Seed={run_cfg.get('random_seed', 42)}")
        print(f"  Output Directory: {output_dir}")
        print("-" * 76)

        def _console_progress(msg):
            if isinstance(msg, dict):
                if msg.get("type") == "fold_complete":
                    print(f"    [OK] Fold {msg['fold']}/{msg['k_folds']} Finished | Val MSE: {msg['val_mse']:.6f} | R2: {msg['val_r2']:.4f}")
            else:
                print(f"[*] {msg}")

        execute_kfold_training_pipeline(
            u_raw, y_raw,
            dataset_name=ds_name,
            hyperparameters=run_cfg,
            output_dir=output_dir,
            progress_callback=_console_progress
        )
        print(f"\n[OK] {m_type} training successfully completed! Artifacts saved to: {output_dir}")
        pkl_files = [f for f in glob.glob(os.path.join(output_dir, "*.pkl")) if not os.path.basename(f).startswith("checkpoint_")]
        if pkl_files:
            completed_models.append(pkl_files[0])

    # If multiple models were trained in this batch, run head-to-head comparison automatically
    if len(completed_models) >= 2:
        comp_dir = os.path.join("results", f"BatchComparison_{len(completed_models)}Models_{ds_name}_{timestamp_sig}")
        execute_model_comparison_pipeline(
            completed_models, u_raw, y_raw,
            target_dataset_name=ds_name,
            output_dir=comp_dir
        )


def run_benchmark_command(args, config):
    _print_banner("OPTIMIZATION BENCHMARKS SUITE")
    dims = max(1, int(args.dimensions or config.get("benchmark_dimensions") or 5))
    trials = max(1, int(args.trials or config.get("benchmark_trials") or 10))

    timestamp_sig = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"results_benchmarks_{timestamp_sig}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Testing Functions: Sphere, Rosenbrock, Rastrigin, Ackley, Griewank, Schwefel")
    print(f"  Dimensions (D)   : {dims}")
    print(f"  Trials/Function  : {trials}")
    print(f"  Population Size  : {config.get('pop_size') or 150} | Max Gens: {config.get('num_gen') or 100}")
    print(f"  Output Directory : {output_dir}")
    print("-" * 76)

    def _bench_progress(msg):
        print(f"[*] {msg}")

    run_full_benchmark_suite(
        hyperparameters=config,
        dimensions=dims,
        total_trials=trials,
        output_dir=output_dir,
        progress_callback=_bench_progress
    )
    print(f"\n[OK] Benchmarks finished! Report & boxplots saved to: {output_dir}")


def run_evaluate_command(args, config):
    _print_banner("SINGLE-MODEL DEEP DIAGNOSTICS & EVALUATION")
    model_path = args.model
    if not model_path or not os.path.exists(model_path):
        print(f"[!] Error: Model file not found at: '{model_path}'")
        sys.exit(1)
    if os.path.getsize(model_path) == 0:
        print(f"[!] Error: Model file is empty (0 bytes): '{model_path}'")
        sys.exit(1)

    cached_datasets = load_cached_uci_benchmarks()
    target_ds = args.dataset or config.get("dataset", "UCI_Energy_Efficiency")
    ds_name, (u_raw, y_raw) = _resolve_dataset(target_ds, cached_datasets)

    timestamp_sig = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = os.path.join("results", f"eval_{stem}_{timestamp_sig}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Model File      : {os.path.abspath(model_path)}")
    print(f"  Evaluated On    : {ds_name} (N = {len(y_raw)} samples)")
    print(f"  Output Directory: {output_dir}")
    print("-" * 76)

    def _eval_progress(msg):
        print(f"[*] {msg}")

    execute_inference_pipeline(
        model_path, u_raw, y_raw,
        target_dataset_name=ds_name,
        output_dir=output_dir,
        progress_callback=_eval_progress
    )
    print(f"\n[OK] Evaluation complete! 10 plots and inference report saved to: {output_dir}")


def run_compare_command(args, config):
    _print_banner("MULTI-MODEL HEAD-TO-HEAD COMPARISON")
    model_paths = args.models
    if not model_paths or len(model_paths) < 2:
        print("[!] Error: At least 2 model files are required for comparison (e.g. --models model1.pkl model2.pkl)")
        sys.exit(1)

    for p in model_paths:
        if not os.path.exists(p):
            print(f"[!] Error: Model file not found: '{p}'")
            sys.exit(1)
        if os.path.getsize(p) == 0:
            print(f"[!] Error: Model file is empty (0 bytes): '{p}'")
            sys.exit(1)

    cached_datasets = load_cached_uci_benchmarks()
    target_ds = args.dataset or config.get("dataset", "UCI_Energy_Efficiency")
    ds_name, (u_raw, y_raw) = _resolve_dataset(target_ds, cached_datasets)

    timestamp_sig = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"comparison_{ds_name}_{timestamp_sig}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Target Dataset  : {ds_name}")
    print(f"  Comparing Models: {len(model_paths)} files")
    for idx, p in enumerate(model_paths, start=1):
        print(f"    {idx}. {os.path.basename(p)}")
    print(f"  Output Directory: {output_dir}")
    print("-" * 76)

    def _comp_progress(msg):
        print(f"[*] {msg}")

    execute_model_comparison_pipeline(
        model_paths, u_raw, y_raw,
        target_dataset_name=ds_name,
        output_dir=output_dir,
        progress_callback=_comp_progress
    )
    print(f"\n[OK] Multi-model comparison complete! Plots & tables saved to: {output_dir}")


def run_summary_command(args):
    _print_banner("RESULTS SUMMARY SHEET EXPORT")
    results_dir = args.dir or "results"
    print(f"  Scanning Results Directory: {results_dir}")
    print("-" * 76)
    try:
        cached_datasets = load_cached_uci_benchmarks()
        exported_path = generate_results_summary_sheet(base_results_dir=results_dir, loaded_datasets=cached_datasets)
        print(f"\n[OK] Master summary exported successfully to:\n     {exported_path}")
    except Exception as e:
        print(f"\n[!] Failed to generate summary sheet: {e}")
        sys.exit(1)


def run_gui_command():
    print("[*] Launching CustomTkinter Graphical User Interface...")
    try:
        from system_id_gui import EvolutionarySystemIDApp
        app = EvolutionarySystemIDApp()
        app.mainloop()
    except Exception as e:
        print(f"[!] Could not launch GUI: {e}")
        sys.exit(1)


def main():
    config = load_hyperparameters("config.json")

    # If user ran `python main_console.py` with zero arguments -> Run training from config.json (Original Job)
    if len(sys.argv) == 1:
        args_dummy = argparse.Namespace(
            dataset=None, model=None, mode=None, folds=None, pop=None, gens=None
        )
        run_training_command(args_dummy, config)
        return

    parser = argparse.ArgumentParser(
        description="Evolutionary System Identification (ep_sysid) -- Console Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_console.py train --dataset UCI_Energy_Efficiency --model ARMAX
  python main_console.py train --dataset UCI_Energy_Efficiency --model ALL
  python main_console.py benchmark --dimensions 5 --trials 10
  python main_console.py evaluate --model results/.../identified_model.pkl
  python main_console.py compare --models m1.pkl m2.pkl --dataset UCI_Energy_Efficiency
  python main_console.py summary --dir results
  python main_console.py gui
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # Sub-command: train (convert --model and --mode to uppercase and allow 'ALL' for batch runs)
    p_train = subparsers.add_parser("train", help="Run K-Fold training using config.json (Default)")
    p_train.add_argument("--dataset", type=str, default=None, help="Dataset name override (e.g. UCI_Energy_Efficiency)")
    p_train.add_argument("--model", type=str.upper, choices=["ARMAX", "MLP", "SVR", "ALL"], default=None, help="Model type override (or ALL for batch)")
    p_train.add_argument("--mode", type=str.upper, choices=["SISO", "MISO", "SIMO", "MIMO"], default=None, help="System mode override")
    p_train.add_argument("--folds", type=int, default=None, help="K-Fold splits override")
    p_train.add_argument("--pop", type=int, default=None, help="Population size override")
    p_train.add_argument("--gens", type=int, default=None, help="Max generations override")

    # Sub-command: benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run standard optimization benchmark functions")
    p_bench.add_argument("--dimensions", "-d", type=int, default=None, help="Problem dimension D (default: 5)")
    p_bench.add_argument("--trials", "-t", type=int, default=None, help="Trials per function (default: 10)")

    # Sub-command: evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run deep single-model evaluation and generate diagnostic plots")
    p_eval.add_argument("--model", "-m", type=str, required=True, help="Path to identified_model.pkl file")
    p_eval.add_argument("--dataset", type=str, default=None, help="Target dataset name")

    # Sub-command: compare
    p_comp = subparsers.add_parser("compare", help="Compare multiple trained models head-to-head")
    p_comp.add_argument("--models", nargs="+", required=True, help="Paths to 2 or more .pkl model files")
    p_comp.add_argument("--dataset", type=str, default=None, help="Target dataset name for comparison")

    # Sub-command: summary
    p_sum = subparsers.add_parser("summary", help="Aggregate all training runs into master CSV and Excel summary sheets")
    p_sum.add_argument("--dir", "-d", type=str, default="results", help="Directory containing training run folders (default: results)")

    # Sub-command: gui
    subparsers.add_parser("gui", help="Launch CustomTkinter desktop GUI")

    args = parser.parse_args()

    if args.command == "train" or args.command is None:
        run_training_command(args, config)
    elif args.command == "benchmark":
        run_benchmark_command(args, config)
    elif args.command == "evaluate":
        run_evaluate_command(args, config)
    elif args.command == "compare":
        run_compare_command(args, config)
    elif args.command == "summary":
        run_summary_command(args)
    elif args.command == "gui":
        run_gui_command()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
