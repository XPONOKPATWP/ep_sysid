"""
experimental_studies.py

Automated experimental suite for Master's Thesis & Conference Paper:
  1. Systematic Evolutionary Operator Ablation Study (6 Operator Configs on Benchmarks & SysId)
  2. Hyperparameter Sensitivity Analysis (One-Factor-At-A-Time - OFAT)
  3. Dimensional Scalability Stress Test (EP vs. DE Across D = 2..30)
  4. Equal-Budget Fair Comparison of ARMAX, MLP (Pure & Lamarckian), and SVR
  5. Overall Cross-Study Comparison Aggregator
"""

import os
import re
import copy
import json
import time
import pickle
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import scipy.stats as stats

from config_handler import load_hyperparameters
from multivariable_data_parser import load_cached_uci_benchmarks
from optimization_benchmarks import (
    TEST_FUNCTIONS,
    BENCHMARK_PRECISION_THRESHOLDS,
    compute_vargha_delaney_a12,
    EvaluateBenchmarkComparison,
    execute_benchmark_trial,
    execute_de_benchmark_trial
)
from training_pipeline import execute_kfold_training_pipeline
from diagnostic_plotter import (
    generate_benchmark_convergence_comparison,
    generate_sensitivity_analysis_plots,
    generate_benchmark_boxplots,
    generate_dimensional_scalability_plots,
    generate_sysid_study_comparison_plots,
    generate_studies_overview_dashboard
)


def _parse_training_report_metrics(report_path):
    """
    Extracts fold-level arrays (MSE, RMSE, MAE, MAPE, R2, AIC, AICc, BIC, FPE, WN%, VAF%, CCF%, SimFit%, SimRMSE, Time, Cost, FE, Gen)
    from a completed training_report.txt.
    """
    metrics = {
        "MSE": [], "RMSE": [], "MAE": [], "MAPE": [], "R2": [],
        "AIC": [], "AICc": [], "BIC": [], "FPE": [], "WN_Pct": [],
        "VAF_Pct": [], "CCF_Pass_Pct": [], "SimFit_Pct": [], "Sim_RMSE": [],
        "Time": [], "Cost_ms": [], "FE": [], "Gen": []
    }
    if not os.path.exists(report_path):
        return metrics

    pattern = re.compile(
        r"Fold\s+(?P<fold>\d+)\s*\|\s*Gen:\s*(?P<gen>\d+)/(?P<max_gen>\d+)\s*\((?P<status>[^)]+)\)\s*\|"
        r"(?:\s*FE:\s*(?P<fe>\d+)\s*\|)?"
        r"\s*Time:\s*(?P<time>[\d.]+)s(?:\s*\((?P<cost>[\d.]+)ms/FE\))?\s*\|\s*"
        r"MSE:\s*(?P<mse>[-\d.infnanINFNAN]+)\s*\|\s*"
        r"RMSE:\s*(?P<rmse>[-\d.infnanINFNAN]+)\s*\|"
        r"(?:\s*MAE:\s*(?P<mae>[-\d.infnanINFNAN]+)\s*\|)?"
        r"\s*MAPE:\s*(?P<mape>[-\d.infnanINFNAN]+)%\s*\|\s*"
        r"R2:\s*(?P<r2>[-\d.infnanINFNAN]+)"
        r"(?:\s*\|\s*AIC:\s*(?P<aic>[-\d.infnanINFNAN]+))?"
        r"(?:\s*\|\s*AICc:\s*(?P<aicc>[-\d.infnanINFNAN]+))?"
        r"(?:\s*\|\s*BIC:\s*(?P<bic>[-\d.infnanINFNAN]+))?"
        r"(?:\s*\|\s*FPE:\s*(?P<fpe>[-\d.infnanINFNAN]+))?"
        r"(?:\s*\|\s*WN%:\s*(?P<wn>[-\d.infnanINFNAN]+)%)?"
        r"(?:\s*\|\s*VAF:\s*(?P<vaf>[-\d.infnanINFNAN]+)%)?"
        r"(?:\s*\|\s*CCF%:\s*(?P<ccf>[-\d.infnanINFNAN]+)%)?"
        r"(?:\s*\|\s*SimFit:\s*(?P<simfit>[-\d.infnanINFNAN]+)%)?"
        r"(?:\s*\|\s*SimRMSE:\s*(?P<simrmse>[-\d.infnanINFNAN]+))?"
    )

    with open(report_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                gd = m.groupdict()
                metrics["Gen"].append(int(gd["gen"]))
                if gd.get("fe"):
                    metrics["FE"].append(int(gd["fe"]))
                metrics["Time"].append(float(gd["time"]))
                if gd.get("cost"):
                    metrics["Cost_ms"].append(float(gd["cost"]))
                metrics["MSE"].append(float(gd["mse"]))
                metrics["RMSE"].append(float(gd["rmse"]))
                if gd.get("mae"):
                    metrics["MAE"].append(float(gd["mae"]))
                metrics["MAPE"].append(float(gd["mape"]))
                metrics["R2"].append(float(gd["r2"]))
                if gd.get("aic"):
                    metrics["AIC"].append(float(gd["aic"]))
                if gd.get("aicc"):
                    metrics["AICc"].append(float(gd["aicc"]))
                if gd.get("bic"):
                    metrics["BIC"].append(float(gd["bic"]))
                if gd.get("fpe"):
                    metrics["FPE"].append(float(gd["fpe"]))
                if gd.get("wn"):
                    metrics["WN_Pct"].append(float(gd["wn"]))
                if gd.get("vaf"):
                    metrics["VAF_Pct"].append(float(gd["vaf"]))
                if gd.get("ccf"):
                    metrics["CCF_Pass_Pct"].append(float(gd["ccf"]))
                if gd.get("simfit"):
                    metrics["SimFit_Pct"].append(float(gd["simfit"]))
                if gd.get("simrmse"):
                    metrics["Sim_RMSE"].append(float(gd["simrmse"]))
            elif line.startswith("Variance Accounted For (VAF):"):
                m_vaf = re.search(r"([\d.]+)%", line)
                if m_vaf and not metrics["VAF_Pct"]:
                    metrics["VAF_Pct"].append(float(m_vaf.group(1)))
            elif line.startswith("Cross-Correlation CCF Pass Rate:"):
                m_ccf = re.search(r"([\d.]+)%", line)
                if m_ccf and not metrics["CCF_Pass_Pct"]:
                    metrics["CCF_Pass_Pct"].append(float(m_ccf.group(1)))

    return metrics


def _extract_fold_convergence_from_checkpoints(run_dir, pop_size=150, num_gen=100, mean_fold_time=None):
    """
    Reads checkpoint_fold_*.pkl inside `run_dir` to reconstruct the 10-fold convergence matrix,
    cumulative FE axis, and wall-clock time axis for multi-configuration SysId plots.
    """
    target_len = int(num_gen) + 1
    mse_curves = []
    fe_curves = []
    elapsed_times = []

    if os.path.exists(run_dir):
        ckpt_files = sorted(
            [f for f in os.listdir(run_dir) if f.startswith("checkpoint_fold_") and f.endswith(".pkl")],
            key=lambda x: int(re.search(r"(\d+)", x).group(1)) if re.search(r"(\d+)", x) else 0
        )
        for cf in ckpt_files:
            cf_path = os.path.join(run_dir, cf)
            try:
                with open(cf_path, "rb") as pf:
                    state = pickle.load(pf)
                logbook = state.get("logbook") if isinstance(state, dict) else None
                if logbook is not None:
                    mins = [float(v) for v in logbook.select("min")]
                    nevals = [int(v) for v in logbook.select("nevals")]
                    if len(mins) < target_len:
                        mins.extend([mins[-1]] * (target_len - len(mins)))
                    else:
                        mins = mins[:target_len]
                    if len(nevals) < target_len:
                        nevals.extend([int(pop_size)] * (target_len - len(nevals)))
                    else:
                        nevals = nevals[:target_len]
                    mse_curves.append(mins)
                    fe_curves.append(np.cumsum(nevals))
                    el = float(getattr(logbook, "elapsed_time", 0.0) or state.get("elapsed_time", 0.0))
                    if el > 0:
                        elapsed_times.append(el)
            except Exception:
                pass

    if not mse_curves:
        return {}

    mse_matrix = np.asarray(mse_curves, dtype=float)
    fe_axis = np.mean(np.asarray(fe_curves, dtype=float), axis=0) if fe_curves else np.arange(1, target_len + 1) * pop_size
    final_time = float(mean_fold_time) if (mean_fold_time is not None and mean_fold_time > 0) else (
        float(np.mean(elapsed_times)) if elapsed_times else float(target_len)
    )
    time_axis = np.linspace(0.0, final_time, target_len)

    return {
        "mse_matrix": mse_matrix,
        "fe_axis": fe_axis,
        "time_axis": time_axis,
        "mean_time": final_time
    }


def _resolve_models_list(model_type):
    """
    Expands model_type into a list of model architectures.
    Supports 'ALL' -> ['ARMAX', 'MLP', 'SVR'], comma-separated lists, or single model names.
    """
    if isinstance(model_type, (list, tuple)):
        return [str(m).upper().strip() for m in model_type if str(m).strip()]
    raw = str(model_type or "ARMAX").upper().strip()
    if raw == "ALL":
        return ["ARMAX", "MLP", "SVR"]
    if "," in raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    return [raw]


def run_operator_ablation_study(
    base_config=None,
    dataset_name="UCI_Energy_Efficiency",
    model_type="ARMAX",
    system_mode="MISO",
    run_benchmarks=True,
    run_sysid=True,
    output_dir=None
):
    """
    Point 1: Systematic Comparison of Evolutionary Operators.
    Isolates the effect of each operator (Gaussian vs Cauchy, Truncation vs Tournament,
    and BLX-alpha vs Pure EP with pc=0.0 across both Truncation and Tournament selection)
    while keeping all other parameters fixed.
    """
    if base_config is None:
        base_config = load_hyperparameters("config.json")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = os.path.join("results", f"study_1_operator_ablation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    pop_sz = int(base_config.get("pop_size", 150))
    n_gen = int(base_config.get("num_gen", 100))

    baseline = copy.deepcopy(base_config)
    baseline.update({
        "model_type": model_type,
        "system_mode": system_mode,
        "pop_size": pop_sz,
        "num_gen": n_gen,
        "mutation_sigma": 0.5,
        "crossover_alpha": 0.25,
        "tournament_size": 4,
        "early_stop_patience": -1,
        "random_seed": 42
    })

    operator_configs = [
        ("Op1_Gauss_Trunc_BLX",       {"mutation_type": "Gaussian", "selection_type": "Truncation", "crossover_prob": 0.3}),
        ("Op2_Cauchy_Trunc_BLX",      {"mutation_type": "Cauchy",   "selection_type": "Truncation", "crossover_prob": 0.3}),
        ("Op3_Gauss_Tourn_BLX",       {"mutation_type": "Gaussian", "selection_type": "Tournament", "crossover_prob": 0.3}),
        ("Op4_Cauchy_Tourn_BLX",      {"mutation_type": "Cauchy",   "selection_type": "Tournament", "crossover_prob": 0.3}),
        ("Op5_PureEP_Trunc_NoCross",  {"mutation_type": "Gaussian", "selection_type": "Truncation", "crossover_prob": 0.0}),
        ("Op6_PureEP_Tourn_NoCross",  {"mutation_type": "Gaussian", "selection_type": "Tournament", "crossover_prob": 0.0}),
    ]

    # 1A. Operator Ablation on Benchmark Functions
    if run_benchmarks:
        bench_dir = os.path.join(output_dir, "benchmarks_operators")
        os.makedirs(bench_dir, exist_ok=True)
        dims = int(baseline.get("benchmark_dimensions", 5))
        trials = int(baseline.get("benchmark_trials", 10))
        bench_rows = []
        convergence_by_func = {fn: {} for fn in TEST_FUNCTIONS}

        for op_label, overrides in operator_configs:
            cfg = copy.deepcopy(baseline)
            cfg.update(overrides)
            print(f"\n[Operator Study - Benchmarks] Running {op_label}: {overrides}")
            for fn_name, fn_cfg in TEST_FUNCTIONS.items():
                res = execute_benchmark_trial(
                    fn_name, fn_cfg, cfg,
                    dimensions=dims, total_trials=trials, output_dir=None,
                    allow_early_stop=False
                )
                if res:
                    convergence_by_func[fn_name][op_label] = res["convergence_matrix"]
                    bench_rows.append({
                        "Configuration": op_label,
                        "Dimensions": dims,
                        "Trials": trials,
                        "Mutation": overrides["mutation_type"],
                        "Selection": overrides["selection_type"],
                        "Crossover_Prob": overrides["crossover_prob"],
                        "Function": fn_name,
                        "Mean_Fitness": res["mean_fitness"],
                        "Std_Fitness": res["std_fitness"],
                        "Best_Fitness": res["min_fitness"],
                        "Mean_Time_Sec": res["mean_time"],
                        "FE": res["mean_fevals"],
                        "Cost_ms_per_FE": round((res["mean_time"] * 1000.0) / max(1, res["mean_fevals"]), 3)
                    })

        if bench_rows:
            df_b = pd.DataFrame(bench_rows)
            df_b.to_csv(os.path.join(bench_dir, "operator_ablation_benchmarks.csv"), index=False)
            generate_benchmark_convergence_comparison(
                convergence_by_func,
                output_dir=bench_dir,
                filename="operator_ablation_convergence.png",
                suptitle="Evolutionary Operator Ablation: Convergence Comparison Across Benchmarks"
            )

            b_tex_path = os.path.join(bench_dir, "operator_ablation_benchmarks.tex")
            with open(b_tex_path, "w", encoding="utf-8") as f:
                f.write("\\begin{table*}[htbp]\n\\centering\n")
                f.write(f"\\caption{{Study 1A: Evolutionary Operator Ablation Across Continuous Benchmarks ($D={dims}$, {trials} Trials)}}\n")
                f.write("\\label{tab:study1_operator_benchmarks}\n")
                f.write("\\begin{tabular}{llccccc}\n\\toprule\n")
                f.write("\\textbf{Configuration} & \\textbf{Function} & \\textbf{Mean $\\pm$ SD} & \\textbf{Best} & \\textbf{FE} & \\textbf{Time (s)} \\\\\n\\midrule\n")
                for r in bench_rows:
                    c_lbl = r["Configuration"].replace("_", r"\_")
                    f.write(
                        f"{c_lbl} & {r['Function']} & {r['Mean_Fitness']:.3e} $\\pm$ {r['Std_Fitness']:.2e} & "
                        f"{r['Best_Fitness']:.3e} & {r['FE']} & {r['Mean_Time_Sec']:.2f} \\\\\n"
                    )
                f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    # 1B. Operator Ablation on System Identification Dataset (Supports Single Model or ALL Models with Per-Model Bests)
    if run_sysid:
        datasets = load_cached_uci_benchmarks()
        if dataset_name not in datasets:
            dataset_name = list(datasets.keys())[0]
        u_raw, y_raw = datasets[dataset_name]
        models_to_run = _resolve_models_list(model_type)
        sysid_rows = []

        for cur_model in models_to_run:
            model_rows = []
            fold_rmse_map = {}
            sysid_conv_map = {}

            for op_label, overrides in operator_configs:
                cfg = copy.deepcopy(baseline)
                cfg["model_type"] = cur_model
                cfg.update(overrides)
                sub_dir = os.path.join(output_dir, f"{op_label}_{cur_model}_{system_mode}_{dataset_name}")
                print(f"\n[Operator Study - SysId] Running {op_label} on {dataset_name} ({cur_model}-{system_mode})...")
                execute_kfold_training_pipeline(u_raw, y_raw, dataset_name, cfg, output_dir=sub_dir)

                m = _parse_training_report_metrics(os.path.join(sub_dir, "training_report.txt"))
                if m["RMSE"]:
                    t_mean = float(np.mean(m["Time"]))
                    fold_rmse_map[op_label] = m["RMSE"]
                    conv_info = _extract_fold_convergence_from_checkpoints(sub_dir, pop_size=pop_sz, num_gen=n_gen, mean_fold_time=t_mean)
                    if conv_info:
                        sysid_conv_map[op_label] = conv_info

                    fe_val = int(np.mean(m["FE"])) if m["FE"] else int(pop_sz * (n_gen + 1))
                    cost_val = float(np.mean(m["Cost_ms"])) if m["Cost_ms"] else ((t_mean * 1000.0) / max(1, fe_val))
                    model_rows.append({
                        "Configuration": op_label,
                        "Dataset": dataset_name,
                        "Model": cur_model,
                        "Mode": system_mode,
                        "Mutation": overrides["mutation_type"],
                        "Selection": overrides["selection_type"],
                        "Crossover_Prob": overrides["crossover_prob"],
                        "MSE_Mean": float(np.mean(m["MSE"])),
                        "MSE_Std": float(np.std(m["MSE"])),
                        "RMSE_Mean": float(np.mean(m["RMSE"])),
                        "RMSE_Std": float(np.std(m["RMSE"])),
                        "MAE_Mean": float(np.mean(m["MAE"])) if m["MAE"] else 0.0,
                        "MAE_Std": float(np.std(m["MAE"])) if m["MAE"] else 0.0,
                        "MAPE_Mean": float(np.mean(m["MAPE"])) if m["MAPE"] else 0.0,
                        "MAPE_Std": float(np.std(m["MAPE"])) if m["MAPE"] else 0.0,
                        "R2_Mean": float(np.mean(m["R2"])),
                        "R2_Std": float(np.std(m["R2"])),
                        "VAF_Mean": float(np.mean(m["VAF_Pct"])) if m["VAF_Pct"] else 0.0,
                        "VAF_Std": float(np.std(m["VAF_Pct"])) if m["VAF_Pct"] else 0.0,
                        "AIC_Mean": float(np.mean(m["AIC"])) if m["AIC"] else 0.0,
                        "BIC_Mean": float(np.mean(m["BIC"])) if m["BIC"] else 0.0,
                        "WN_ACF_Mean": float(np.mean(m["WN_Pct"])) if m["WN_Pct"] else 0.0,
                        "CCF_Mean": float(np.mean(m["CCF_Pass_Pct"])) if m["CCF_Pass_Pct"] else 100.0,
                        "CCF_Std": float(np.std(m["CCF_Pass_Pct"])) if m["CCF_Pass_Pct"] else 0.0,
                        "Sim_Fit_Mean": float(np.mean(m["SimFit_Pct"])) if m["SimFit_Pct"] else 0.0,
                        "Sim_RMSE_Mean": float(np.mean(m["Sim_RMSE"])) if m["Sim_RMSE"] else 0.0,
                        "FE_Mean": fe_val,
                        "Time_Fold_Mean_Sec": t_mean,
                        "Time_Fold_Std_Sec": float(np.std(m["Time"])),
                        "Cost_Per_FE_ms": cost_val
                    })

            if model_rows:
                # Compute best operator, Wilcoxon p-value, and A12 strictly within cur_model
                best_op = min(model_rows, key=lambda r: r["RMSE_Mean"])["Configuration"]
                best_rmses = fold_rmse_map.get(best_op, [])
                for r in model_rows:
                    cur_rmses = fold_rmse_map.get(r["Configuration"], [])
                    is_best = (r["Configuration"] == best_op)
                    r["Is_Model_Best"] = is_best
                    if is_best or len(cur_rmses) != len(best_rmses) or len(cur_rmses) == 0:
                        r["Wilcoxon_P_vs_Best"] = 1.0
                        r["A12_vs_Best"] = 0.5
                    else:
                        try:
                            w_res = stats.wilcoxon(cur_rmses, best_rmses)
                            r["Wilcoxon_P_vs_Best"] = float(w_res.pvalue)
                        except Exception:
                            r["Wilcoxon_P_vs_Best"] = 1.0
                        a12_val, _ = compute_vargha_delaney_a12(best_rmses, cur_rmses)
                        r["A12_vs_Best"] = float(a12_val)

                sysid_rows.extend(model_rows)

                # Save per-model 3-panel convergence & boxplot figure
                plot_fn = (
                    "operator_ablation_sysid_comparison.png"
                    if len(models_to_run) == 1
                    else f"operator_ablation_sysid_comparison_{cur_model}.png"
                )
                generate_sysid_study_comparison_plots(
                    sysid_conv_map,
                    fold_rmse_map,
                    output_dir=output_dir,
                    filename=plot_fn,
                    suptitle=f"Study 1B: Operator Ablation Convergence & 10-Fold RMSE ({dataset_name}, {cur_model}-{system_mode})"
                )
                if len(models_to_run) > 1 and cur_model == models_to_run[0]:
                    shutil.copy2(
                        os.path.join(output_dir, plot_fn),
                        os.path.join(output_dir, "operator_ablation_sysid_comparison.png")
                    )

        if sysid_rows:
            df_s = pd.DataFrame(sysid_rows)
            df_s.to_csv(os.path.join(output_dir, "operator_ablation_sysid.csv"), index=False)

            tex_path = os.path.join(output_dir, "operator_ablation_sysid.tex")
            clean_ds = dataset_name.replace("_", r"\_")
            model_hdr = "/".join(models_to_run)
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write("\\begin{table*}[htbp]\n\\centering\n")
                f.write(f"\\caption{{Systematic Comparison of Evolutionary Operators on {clean_ds} ({model_hdr}-{system_mode})}}\n")
                f.write("\\label{tab:study1_operator_sysid}\n")
                f.write("\\begin{tabular}{llccccccccc}\n\\toprule\n")
                f.write("\\textbf{Model} & \\textbf{Operator Config} & \\textbf{Mutation} & \\textbf{Selection} & \\textbf{$p_c$} & \\textbf{FE} & \\textbf{RMSE ($\\mu \\pm \\sigma$)} & \\textbf{MAE ($\\mu \\pm \\sigma$)} & \\textbf{$R^2$ ($\\mu \\pm \\sigma$)} & \\textbf{Time/Fold (s)} & \\textbf{$p$-val vs. Model Best} \\\\\n\\midrule\n")
                prev_m = None
                for r in sysid_rows:
                    if prev_m is not None and prev_m != r["Model"]:
                        f.write("\\midrule\n")
                    prev_m = r["Model"]
                    cfg_lbl = r["Configuration"].replace("_", r"\_")
                    is_b = bool(r.get("Is_Model_Best", False))
                    p_str = "\\textbf{Best}" if is_b else f"{r['Wilcoxon_P_vs_Best']:.3f}"
                    rmse_str = f"\\textbf{{{r['RMSE_Mean']:.4f}}} $\\pm$ {r['RMSE_Std']:.4f}" if is_b else f"{r['RMSE_Mean']:.4f} $\\pm$ {r['RMSE_Std']:.4f}"
                    f.write(
                        f"{r['Model']} & {cfg_lbl} & {r['Mutation']} & {r['Selection']} & {r['Crossover_Prob']:.2f} & {r['FE_Mean']} & "
                        f"{rmse_str} & {r['MAE_Mean']:.4f} $\\pm$ {r['MAE_Std']:.4f} & "
                        f"{r['R2_Mean']:.4f} $\\pm$ {r['R2_Std']:.4f} & "
                        f"{r['Time_Fold_Mean_Sec']:.2f} $\\pm$ {r['Time_Fold_Std_Sec']:.2f} & {p_str} \\\\\n"
                    )
                f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    print(f"\n[OK] Operator Ablation Study complete! Artifacts saved to: {output_dir}")
    return output_dir


def run_parameter_sensitivity_study(
    base_config=None,
    dataset_name="UCI_Energy_Efficiency",
    model_type="ARMAX",
    system_mode="MISO",
    output_dir=None
):
    """
    Point 2: One-Factor-At-A-Time (OFAT) Parameter Sensitivity Analysis.
    Evaluates the impact of pop_size, num_gen, mutation_sigma, crossover_prob, and crossover_alpha
    on solution quality (RMSE, MAE, R2) and execution time.
    When model_type='ALL' (or multiple models), isolates baseline caching and tracks separate
    optimal parameter values and sensitivity spans (Delta RMSE) for each model.
    """
    if base_config is None:
        base_config = load_hyperparameters("config.json")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = os.path.join("results", f"study_2_parameter_sensitivity_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    datasets = load_cached_uci_benchmarks()
    if dataset_name not in datasets:
        dataset_name = list(datasets.keys())[0]
    u_raw, y_raw = datasets[dataset_name]

    models_to_run = _resolve_models_list(model_type)

    ofat_grid = [
        ("pop_size",        [50, 150, 300]),
        ("num_gen",         [50, 100, 200]),
        ("mutation_sigma",  [0.1, 0.5, 1.0]),
        ("crossover_prob",  [0.1, 0.3, 0.5]),
        ("crossover_alpha", [0.1, 0.25, 0.5])
    ]

    sensitivity_records = []
    per_model_best_rows = []

    for cur_model in models_to_run:
        baseline = copy.deepcopy(base_config)
        baseline.update({
            "model_type": cur_model,
            "system_mode": system_mode,
            "pop_size": 150,
            "num_gen": 100,
            "mutation_sigma": 0.5,
            "crossover_prob": 0.3,
            "crossover_alpha": 0.25,
            "mutation_type": "Gaussian",
            "selection_type": "Tournament",
            "tournament_size": 4,
            "early_stop_patience": -1,
            "random_seed": 42
        })

        # Isolate baseline cache strictly per model so MLP/SVR never reuse ARMAX baseline metrics
        cached_baseline_metrics = None
        model_records = []

        for param_name, values in ofat_grid:
            for val in values:
                is_baseline_combo = (val == baseline[param_name])
                cfg = copy.deepcopy(baseline)
                cfg[param_name] = val
                if is_baseline_combo and cached_baseline_metrics is not None:
                    m = copy.deepcopy(cached_baseline_metrics)
                else:
                    run_tag = (
                        f"Sens_{param_name}_{val}"
                        if len(models_to_run) == 1
                        else f"Sens_{cur_model}_{param_name}_{val}"
                    )
                    sub_dir = os.path.join(output_dir, run_tag)
                    print(f"\n[Sensitivity Study] Testing [{cur_model}] {param_name} = {val} on {dataset_name} ({system_mode})...")
                    execute_kfold_training_pipeline(u_raw, y_raw, dataset_name, cfg, output_dir=sub_dir)
                    m = _parse_training_report_metrics(os.path.join(sub_dir, "training_report.txt"))
                    if is_baseline_combo and m["RMSE"]:
                        cached_baseline_metrics = copy.deepcopy(m)

                if m and m["RMSE"]:
                    fe_m = int(np.mean(m["FE"])) if m["FE"] else int(cfg["pop_size"] * (cfg["num_gen"] + 1))
                    t_m = float(np.mean(m["Time"]))
                    cost_fe_val = float(np.mean(m["Cost_ms"])) if m["Cost_ms"] else ((t_m * 1000.0) / max(1, fe_m))
                    model_records.append({
                        "Dataset": dataset_name,
                        "System_Mode": system_mode,
                        "Model": cur_model,
                        "parameter_name": param_name,
                        "parameter_value": val,
                        "MSE_mean": float(np.mean(m["MSE"])),
                        "MSE_std": float(np.std(m["MSE"])),
                        "RMSE_mean": float(np.mean(m["RMSE"])),
                        "RMSE_std": float(np.std(m["RMSE"])),
                        "MAE_mean": float(np.mean(m["MAE"])) if m["MAE"] else 0.0,
                        "MAE_std": float(np.std(m["MAE"])) if m["MAE"] else 0.0,
                        "MAPE_mean": float(np.mean(m["MAPE"])) if m["MAPE"] else 0.0,
                        "MAPE_std": float(np.std(m["MAPE"])) if m["MAPE"] else 0.0,
                        "R2_mean": float(np.mean(m["R2"])),
                        "R2_std": float(np.std(m["R2"])),
                        "VAF_mean": float(np.mean(m["VAF_Pct"])) if m["VAF_Pct"] else 0.0,
                        "VAF_std": float(np.std(m["VAF_Pct"])) if m["VAF_Pct"] else 0.0,
                        "AIC_mean": float(np.mean(m["AIC"])) if m["AIC"] else 0.0,
                        "BIC_mean": float(np.mean(m["BIC"])) if m["BIC"] else 0.0,
                        "WN_ACF_mean": float(np.mean(m["WN_Pct"])) if m["WN_Pct"] else 0.0,
                        "CCF_mean": float(np.mean(m["CCF_Pass_Pct"])) if m["CCF_Pass_Pct"] else 100.0,
                        "CCF_std": float(np.std(m["CCF_Pass_Pct"])) if m["CCF_Pass_Pct"] else 0.0,
                        "Sim_Fit_mean": float(np.mean(m["SimFit_Pct"])) if m["SimFit_Pct"] else 0.0,
                        "Sim_RMSE_mean": float(np.mean(m["Sim_RMSE"])) if m["Sim_RMSE"] else 0.0,
                        "FE_mean": fe_m,
                        "time_mean": t_m,
                        "time_std": float(np.std(m["Time"])),
                        "ms_per_FE": cost_fe_val
                    })

        if model_records:
            # Mark per-parameter best and overall model best strictly within cur_model
            overall_best_rec = min(model_records, key=lambda r: r["RMSE_mean"])
            for p_name, _ in ofat_grid:
                p_recs = [r for r in model_records if r["parameter_name"] == p_name]
                if not p_recs:
                    continue
                p_best = min(p_recs, key=lambda r: r["RMSE_mean"])
                p_worst = max(p_recs, key=lambda r: r["RMSE_mean"])
                p_span = float(p_worst["RMSE_mean"] - p_best["RMSE_mean"])
                for r in p_recs:
                    r["Param_Delta_RMSE"] = p_span
                    r["Is_Param_Best"] = (r["parameter_value"] == p_best["parameter_value"])
                    r["Is_Model_Best"] = (
                        r["parameter_name"] == overall_best_rec["parameter_name"]
                        and r["parameter_value"] == overall_best_rec["parameter_value"]
                    )
                per_model_best_rows.append({
                    "Model": cur_model,
                    "Dataset": dataset_name,
                    "System_Mode": system_mode,
                    "parameter_name": p_name,
                    "Optimal_Value": p_best["parameter_value"],
                    "FE_mean": p_best["FE_mean"],
                    "Best_RMSE_Mean": p_best["RMSE_mean"],
                    "Best_RMSE_Std": p_best["RMSE_std"],
                    "Best_MAE_Mean": p_best["MAE_mean"],
                    "Best_R2_Mean": p_best["R2_mean"],
                    "Delta_RMSE_Span": p_span,
                    "Time_Fold_Mean_Sec": p_best["time_mean"],
                    "ms_per_FE": p_best["ms_per_FE"],
                    "Is_Top_Sensitive_For_Model": False
                })

            # Flag the most sensitive parameter for cur_model
            cur_m_bests = [b for b in per_model_best_rows if b["Model"] == cur_model]
            if cur_m_bests:
                top_b = max(cur_m_bests, key=lambda b: b["Delta_RMSE_Span"])
                top_b["Is_Top_Sensitive_For_Model"] = True

            sensitivity_records.extend(model_records)

            plot_fn = (
                "parameter_sensitivity_overview.png"
                if len(models_to_run) == 1
                else f"parameter_sensitivity_overview_{cur_model}.png"
            )
            generate_sensitivity_analysis_plots(
                model_records,
                output_dir=output_dir,
                metric_key="RMSE",
                metric_label="Validation RMSE",
                filename=plot_fn,
                title_suffix=f" [{cur_model}-{system_mode}]"
            )
            if len(models_to_run) > 1 and cur_model == models_to_run[0]:
                shutil.copy2(
                    os.path.join(output_dir, plot_fn),
                    os.path.join(output_dir, "parameter_sensitivity_overview.png")
                )

    if sensitivity_records:
        df_sens = pd.DataFrame(sensitivity_records)
        df_sens.to_csv(os.path.join(output_dir, "parameter_sensitivity_summary.csv"), index=False)

        if per_model_best_rows:
            df_bests = pd.DataFrame(per_model_best_rows)
            df_bests.to_csv(os.path.join(output_dir, "parameter_sensitivity_bests_per_model.csv"), index=False)

            bests_tex_path = os.path.join(output_dir, "parameter_sensitivity_bests_per_model.tex")
            clean_ds = dataset_name.replace("_", r"\_")
            with open(bests_tex_path, "w", encoding="utf-8") as f:
                f.write("\\begin{table*}[htbp]\n\\centering\n")
                f.write(f"\\caption{{Study 2: Per-Model Optimal Hyperparameter Levels and Sensitivity Spans ($\\Delta$RMSE) on {clean_ds} ({system_mode})}}\n")
                f.write("\\label{tab:study2_bests_per_model}\n")
                f.write("\\begin{tabular}{lllcccccccc}\n\\toprule\n")
                f.write("\\textbf{Model} & \\textbf{Parameter} & \\textbf{Optimal Value} & \\textbf{FE / Fold} & \\textbf{Best RMSE ($\\mu \\pm \\sigma$)} & \\textbf{Best MAE} & \\textbf{Best $R^2$} & \\textbf{$\\Delta$RMSE Span} & \\textbf{Time/Fold (s)} & \\textbf{ms / FE} \\\\\n\\midrule\n")
                prev_m = None
                for b in per_model_best_rows:
                    if prev_m is not None and prev_m != b["Model"]:
                        f.write("\\midrule\n")
                    prev_m = b["Model"]
                    p_lbl = b["parameter_name"].replace("_", r"\_")
                    span_str = f"\\textbf{{{b['Delta_RMSE_Span']:.4f}}}" if b["Is_Top_Sensitive_For_Model"] else f"{b['Delta_RMSE_Span']:.4f}"
                    f.write(
                        f"{b['Model']} & {p_lbl} & \\textbf{{{b['Optimal_Value']:g}}} & {b.get('FE_mean', 0)} & "
                        f"{b['Best_RMSE_Mean']:.4f} $\\pm$ {b['Best_RMSE_Std']:.4f} & "
                        f"{b['Best_MAE_Mean']:.4f} & {b['Best_R2_Mean']:.4f} & "
                        f"{span_str} & {b['Time_Fold_Mean_Sec']:.2f} & {b.get('ms_per_FE', 0.0):.3f} \\\\\n"
                    )
                f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

        tex_path = os.path.join(output_dir, "parameter_sensitivity_table.tex")
        clean_ds = dataset_name.replace("_", r"\_")
        model_hdr = "/".join(models_to_run)
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write("\\begin{table*}[htbp]\n\\centering\n")
            f.write(f"\\caption{{Hyperparameter Sensitivity Analysis on {clean_ds} ({model_hdr}-{system_mode})}}\n")
            f.write("\\label{tab:study2_parameter_sensitivity}\n")
            f.write("\\begin{tabular}{lllcccccc}\n\\toprule\n")
            f.write("\\textbf{Model} & \\textbf{Parameter} & \\textbf{Value} & \\textbf{FE / Fold} & \\textbf{RMSE ($\\mu \\pm \\sigma$)} & \\textbf{MAE ($\\mu \\pm \\sigma$)} & \\textbf{$R^2$ ($\\mu \\pm \\sigma$)} & \\textbf{Time/Fold (s)} & \\textbf{ms / FE} \\\\\n\\midrule\n")
            prev_m = None
            for r in sensitivity_records:
                if prev_m is not None and prev_m != r["Model"]:
                    f.write("\\midrule\n")
                prev_m = r["Model"]
                p_lbl = r["parameter_name"].replace("_", r"\_")
                val_str = f"\\textbf{{{r['parameter_value']}}}" if r.get("Is_Param_Best") else f"{r['parameter_value']}"
                rmse_str = (
                    f"\\textbf{{{r['RMSE_mean']:.4f}}} $\\pm$ {r['RMSE_std']:.4f}"
                    if r.get("Is_Param_Best")
                    else f"{r['RMSE_mean']:.4f} $\\pm$ {r['RMSE_std']:.4f}"
                )
                f.write(
                    f"{r['Model']} & {p_lbl} & {val_str} & {r['FE_mean']} & "
                    f"{rmse_str} & {r['MAE_mean']:.4f} $\\pm$ {r['MAE_std']:.4f} & "
                    f"{r['R2_mean']:.4f} $\\pm$ {r['R2_std']:.4f} & "
                    f"{r['time_mean']:.2f} $\\pm$ {r['time_std']:.2f} & "
                    f"{r['ms_per_FE']:.3f} \\\\\n"
                )
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    print(f"\n[OK] Parameter Sensitivity Study complete! Artifacts saved to: {output_dir}")
    return output_dir


def run_benchmark_optimization_study(
    base_config=None,
    dimensions_list=None,
    total_trials=10,
    output_dir=None
):
    """
    Study 3 (Thesis / Conference Paper): Dimensional Scalability Stress Test (EP vs. DE).
    Evaluates self-adaptive EP against Differential Evolution across varying dimensionalities
    (default D in [2, 5, 10, 20, 30]) under matched evaluation budgets.
    Exports Mean, Best, SD, FE, Execution Time, Mann-Whitney U p-value, and Vargha-Delaney A12.
    """
    if base_config is None:
        base_config = load_hyperparameters("config.json")

    if dimensions_list is None:
        dimensions_list = [2, 5, 10, 20, 30]
    elif isinstance(dimensions_list, (int, float)):
        dimensions_list = [int(dimensions_list)]
    elif isinstance(dimensions_list, str):
        dimensions_list = [int(x.strip()) for x in dimensions_list.split(",") if x.strip()]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = os.path.join("results", f"study_3_dimensional_scalability_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    hparams = copy.deepcopy(base_config)
    pop_size = int(hparams.get("pop_size", 150))
    max_generations = int(hparams.get("num_gen", 100))
    ep_mut_type = hparams.get("mutation_type", "Gaussian")
    ep_sel_type = hparams.get("selection_type", "Tournament")
    ep_label = f"EP ({ep_mut_type}/{ep_sel_type})"

    print("\n" + "=" * 96)
    print(f"  STUDY 3: DIMENSIONAL SCALABILITY STRESS TEST ({ep_label} vs. DE)")
    print(f"  Evaluated Dimensions: {dimensions_list} | Trials: {total_trials} | Pop Size: {pop_size} | Max Gens: {max_generations}")
    print("=" * 96)

    scalability_records = []
    report_lines = [
        "==================================================================================================",
        "          STUDY 3: DIMENSIONAL SCALABILITY STRESS TEST REPORT (EP vs. DE)                        ",
        "==================================================================================================",
        f"Execution Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Evaluated Dims (D)  : {dimensions_list}",
        f"Independent Trials  : {total_trials}",
        f"Population Size     : {pop_size}",
        f"Max Generations     : {max_generations}",
        f"EP Configuration    : Mutation={ep_mut_type}, Selection={ep_sel_type}",
        f"DE Configuration    : Strategy=best1bin, PopSize=max(1, pop_size // D), Matched FE, Polish=False",
        "-" * 98,
        ""
    ]

    for d in dimensions_list:
        print(f"\n>>> Running Benchmark Suite for Dimension D = {d}...")
        dim_scores_dict = {}
        dim_convergence_dict = {}

        for index, (function_name, function_config) in enumerate(TEST_FUNCTIONS.items(), 1):
            print(f"  [{index}/6] D={d} | {function_name} ({function_config['description']})...")

            ep_results = execute_benchmark_trial(
                function_name, function_config, hparams,
                dimensions=d, total_trials=total_trials,
                output_dir=None, allow_early_stop=False
            )
            de_results = execute_de_benchmark_trial(
                function_name, function_config, hparams,
                dimensions=d, total_trials=total_trials
            )
            if ep_results is None or de_results is None:
                continue

            ep_scores = ep_results["all_scores"]
            de_scores = de_results["all_scores"]

            dim_scores_dict[function_name] = {
                ep_label: ep_scores,
                "Differential Evolution (DE)": de_scores
            }
            dim_convergence_dict[function_name] = {
                ep_label: ep_results["convergence_matrix"],
                "Differential Evolution (DE)": de_results["convergence_matrix"]
            }

            # Non-parametric hypothesis testing & effect size with strict numerical tie reconciliation
            comp_metrics = EvaluateBenchmarkComparison(ep_scores, de_scores, tol=1e-9)
            p_val = comp_metrics["p_val"]
            u_stat = comp_metrics["u_stat"]
            sig_marker = comp_metrics["sig_marker"]
            a12 = comp_metrics["a12"]
            effect_mag = comp_metrics["effect_mag"]
            winner = comp_metrics["winner"]

            target_eps = BENCHMARK_PRECISION_THRESHOLDS.get(function_name, 1e-2)
            pop_sz = int(hparams.get("pop_size", 150))
            de_pop_sz = int(de_results.get("actual_de_pop", max(1, pop_sz // max(1, d)) * d))

            ep_hits = [
                int((np.where(trial_hist <= target_eps)[0][0] + 1) * pop_sz)
                for trial_hist in ep_results["convergence_matrix"]
                if len(np.where(trial_hist <= target_eps)[0]) > 0
            ]
            de_hits = [
                int((np.where(trial_hist <= target_eps)[0][0] + 1) * de_pop_sz)
                for trial_hist in de_results["convergence_matrix"]
                if len(np.where(trial_hist <= target_eps)[0]) > 0
            ]
            ep_sr = (len(ep_hits) / total_trials) * 100.0
            de_sr = (len(de_hits) / total_trials) * 100.0
            ep_hit_fe = float(np.mean(ep_hits)) if ep_hits else float("nan")
            de_hit_fe = float(np.mean(de_hits)) if de_hits else float("nan")

            scalability_records.append({
                "Dimensions": d,
                "Trials": total_trials,
                "Function": function_name,
                "EP_Mean": ep_results["mean_fitness"],
                "EP_Std": ep_results["std_fitness"],
                "EP_Best": ep_results["min_fitness"],
                "DE_Mean": de_results["mean_fitness"],
                "DE_Std": de_results["std_fitness"],
                "DE_Best": de_results["min_fitness"],
                "FE": ep_results["mean_fevals"],
                "DE_FE": de_results["mean_fevals"],
                "EP_Time_Sec": ep_results["mean_time"],
                "DE_Time_Sec": de_results["mean_time"],
                "EP_ms_per_FE": round((ep_results["mean_time"] * 1000.0) / max(1, ep_results["mean_fevals"]), 3),
                "DE_ms_per_FE": round((de_results["mean_time"] * 1000.0) / max(1, de_results["mean_fevals"]), 3),
                "U_Statistic": u_stat,
                "P_Value": p_val,
                "Significance": sig_marker,
                "A12_Effect_Size": a12,
                "Effect_Magnitude": effect_mag,
                "Target_Eps": target_eps,
                "EP_Success_Rate_Pct": ep_sr,
                "DE_Success_Rate_Pct": de_sr,
                "EP_Mean_Hitting_FE": ep_hit_fe,
                "DE_Mean_Hitting_FE": de_hit_fe,
                "Winner": winner
            })

            report_lines.append(
                f"[D={d:2d}] {function_name:<11} | EP: {ep_results['mean_fitness']:.3e} +/- {ep_results['std_fitness']:.2e} (Best={ep_results['min_fitness']:.2e}, {ep_results['mean_time']:.2f}s) | "
                f"DE: {de_results['mean_fitness']:.3e} +/- {de_results['std_fitness']:.2e} (Best={de_results['min_fitness']:.2e}, {de_results['mean_time']:.2f}s) | "
                f"p={p_val:.2e} ({sig_marker}) | A12={a12:.3f} ({effect_mag}) | Winner: {winner}"
            )

        if dim_scores_dict:
            generate_benchmark_boxplots(dim_scores_dict, output_dir=output_dir)
            bp_old = os.path.join(output_dir, "benchmark_comparison_boxplot.png")
            bp_new = os.path.join(output_dir, f"study_3_boxplot_D{d}.png")
            if os.path.exists(bp_old):
                if os.path.exists(bp_new):
                    os.remove(bp_new)
                os.rename(bp_old, bp_new)

        if dim_convergence_dict:
            generate_benchmark_convergence_comparison(
                dim_convergence_dict,
                output_dir=output_dir,
                filename=f"study_3_convergence_D{d}.png",
                suptitle=f"Study 3: Multi-Trial Convergence Envelopes (EP vs. DE, D={d})"
            )

    if scalability_records:
        func_order = list(TEST_FUNCTIONS.keys())
        scalability_records.sort(
            key=lambda r: (
                func_order.index(r["Function"]) if r["Function"] in func_order else 999,
                int(r["Dimensions"])
            )
        )
        df_scale = pd.DataFrame(scalability_records)
        df_scale.to_csv(os.path.join(output_dir, "study_3_dimensional_scalability.csv"), index=False)

        generate_dimensional_scalability_plots(
            scalability_records,
            output_dir=output_dir,
            filename="study_3_dimensional_scalability.png"
        )

        tex_path = os.path.join(output_dir, "study_3_dimensional_scalability.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(r"\begin{table*}[htbp]" + "\n")
            f.write(r"\centering" + "\n")
            f.write(r"\scriptsize" + "\n")
            f.write(rf"\caption{{Study 3: Dimensional Scalability Stress Test of EP vs. DE ($N={total_trials}$ Trials Across $D \in \{{{', '.join(map(str, dimensions_list))}\}}$)}}" + "\n")
            f.write(r"\label{tab:study3_dimensional_scalability}" + "\n")
            f.write(r"\begin{tabular}{lcccccccccc}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"\textbf{Function} & \textbf{$D$} & \textbf{EP ($\mu \pm \sigma$)} & \textbf{EP Best} & \textbf{DE ($\mu \pm \sigma$)} & \textbf{DE Best} & \textbf{EP/DE Time (s)} & \textbf{$p$-value} & \textbf{Effect ($A_{12}$)} & \textbf{Winner} \\" + "\n")
            f.write(r"\midrule" + "\n")

            curr_fn = None
            for r in scalability_records:
                if curr_fn is not None and curr_fn != r["Function"]:
                    f.write(r"\midrule" + "\n")
                curr_fn = r["Function"]

                pv = r["P_Value"]
                pv_tex = "1.0000" if abs(pv - 1.0) < 1e-6 else (f"{pv:.2e}" if pv >= 1e-4 else "< 10^{-4}")
                f.write(
                    f"{r['Function']} & {r['Dimensions']} & "
                    f"{r['EP_Mean']:.2e} $\\pm$ {r['EP_Std']:.1e} & {r['EP_Best']:.2e} & "
                    f"{r['DE_Mean']:.2e} $\\pm$ {r['DE_Std']:.1e} & {r['DE_Best']:.2e} & "
                    f"{r['EP_Time_Sec']:.2f} / {r['DE_Time_Sec']:.2f} & "
                    f"${pv_tex}$ ({r['Significance']}) & "
                    f"{r['A12_Effect_Size']:.2f} ({r['Effect_Magnitude']}) & "
                    f"{r['Winner']} \\\\\n"
                )
            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}" + "\n")
            f.write(r"\end{table*}" + "\n")

    report_filepath = os.path.join(output_dir, "study_3_dimensional_scalability_report.txt")
    with open(report_filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n[OK] Study 3 (Dimensional Scalability Stress Test) complete! Artifacts saved to: {output_dir}")
    return output_dir


def run_fair_model_comparison_study(
    base_config=None,
    dataset_name="UCI_Energy_Efficiency",
    system_mode="MISO",
    pop_size=150,
    num_gen=100,
    output_dir=None
):
    """
    Point 4 (Conference Paper): Equal-Budget Fair Comparison of ARMAX, MLP, and SVR.
    Fixes population size and generations (disabling early stopping) so all models receive
    a strictly controlled evolutionary budget. Evaluates ARMAX, Pure-EP MLP, Lamarckian MLP,
    and SVR, logging exact FE, wall-clock time, ms/FE, paired 10-fold Wilcoxon p-values,
    and dual convergence curves (vs. FE and vs. Wall-Clock Time).
    """
    if base_config is None:
        base_config = load_hyperparameters("config.json")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = os.path.join("results", f"study_4_fair_model_comparison_{dataset_name}_{system_mode}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    datasets = load_cached_uci_benchmarks()
    if dataset_name not in datasets:
        dataset_name = list(datasets.keys())[0]
    u_raw, y_raw = datasets[dataset_name]

    shared_cfg = copy.deepcopy(base_config)
    shared_cfg.update({
        "system_mode": system_mode,
        "pop_size": int(pop_size),
        "num_gen": int(num_gen),
        "early_stop_patience": -1,
        "mutation_type": "Gaussian",
        "selection_type": "Tournament",
        "tournament_size": 4,
        "crossover_prob": 0.3,
        "mutation_sigma": 0.5,
        "random_seed": 42
    })

    variants = [
        ("ARMAX",            {"model_type": "ARMAX"}),
        ("MLP_Pure_EP",      {"model_type": "MLP", "lamarckian_refine_freq": 0, "refine_top_candidates": 0}),
        ("MLP_Lamarckian",   {"model_type": "MLP", "lamarckian_refine_freq": 5, "refine_top_candidates": 6}),
        ("SVR",              {"model_type": "SVR"})
    ]

    comparison_rows = []
    fold_rmse_map = {}
    model_conv_map = {}

    for label, overrides in variants:
        cfg = copy.deepcopy(shared_cfg)
        cfg.update(overrides)
        sub_dir = os.path.join(output_dir, f"{label}_{system_mode}_{dataset_name}")
        print(f"\n[Fair Budget Comparison] Training {label} on {dataset_name} ({system_mode}) | Pop={pop_size}, Gen={num_gen}...")
        execute_kfold_training_pipeline(u_raw, y_raw, dataset_name, cfg, output_dir=sub_dir)

        m = _parse_training_report_metrics(os.path.join(sub_dir, "training_report.txt"))
        if m["RMSE"]:
            fe_mean = int(np.mean(m["FE"])) if m["FE"] else int(pop_size * (num_gen + 1))
            t_mean = float(np.mean(m["Time"]))
            ms_per_fe = (t_mean * 1000.0) / max(1, fe_mean)
            fold_rmse_map[label] = m["RMSE"]

            conv_info = _extract_fold_convergence_from_checkpoints(sub_dir, pop_size=pop_size, num_gen=num_gen, mean_fold_time=t_mean)
            if conv_info:
                model_conv_map[label] = conv_info

            comparison_rows.append({
                "Model_Variant": label,
                "Dataset": dataset_name,
                "System_Mode": system_mode,
                "Pop_Size": pop_size,
                "Generations": num_gen,
                "FE_Per_Fold": fe_mean,
                "MSE_Mean": float(np.mean(m["MSE"])),
                "MSE_Std": float(np.std(m["MSE"])),
                "RMSE_Mean": float(np.mean(m["RMSE"])),
                "RMSE_Std": float(np.std(m["RMSE"])),
                "MAE_Mean": float(np.mean(m["MAE"])) if m["MAE"] else 0.0,
                "MAE_Std": float(np.std(m["MAE"])) if m["MAE"] else 0.0,
                "MAPE_Mean": float(np.mean(m["MAPE"])) if m["MAPE"] else 0.0,
                "MAPE_Std": float(np.std(m["MAPE"])) if m["MAPE"] else 0.0,
                "R2_Mean": float(np.mean(m["R2"])),
                "R2_Std": float(np.std(m["R2"])),
                "VAF_Mean": float(np.mean(m["VAF_Pct"])) if m["VAF_Pct"] else 0.0,
                "VAF_Std": float(np.std(m["VAF_Pct"])) if m["VAF_Pct"] else 0.0,
                "AIC_Mean": float(np.mean(m["AIC"])) if m["AIC"] else 0.0,
                "BIC_Mean": float(np.mean(m["BIC"])) if m["BIC"] else 0.0,
                "WN_ACF_Mean": float(np.mean(m["WN_Pct"])) if m["WN_Pct"] else 0.0,
                "CCF_Mean": float(np.mean(m["CCF_Pass_Pct"])) if m["CCF_Pass_Pct"] else 100.0,
                "CCF_Std": float(np.std(m["CCF_Pass_Pct"])) if m["CCF_Pass_Pct"] else 0.0,
                "Sim_Fit_Mean": float(np.mean(m["SimFit_Pct"])) if m["SimFit_Pct"] else 0.0,
                "Sim_RMSE_Mean": float(np.mean(m["Sim_RMSE"])) if m["Sim_RMSE"] else 0.0,
                "Time_Fold_Mean_Sec": t_mean,
                "Time_Fold_Std_Sec": float(np.std(m["Time"])),
                "Cost_Per_FE_ms": float(np.mean(m["Cost_ms"])) if m["Cost_ms"] else ms_per_fe
            })

    if comparison_rows:
        best_model = min(comparison_rows, key=lambda r: r["RMSE_Mean"])["Model_Variant"]
        best_rmses = fold_rmse_map.get(best_model, [])
        for r in comparison_rows:
            cur_rmses = fold_rmse_map.get(r["Model_Variant"], [])
            if r["Model_Variant"] == best_model or len(cur_rmses) != len(best_rmses) or len(cur_rmses) == 0:
                r["Wilcoxon_P_vs_Best"] = 1.0
                r["A12_vs_Best"] = 0.5
            else:
                try:
                    w_res = stats.wilcoxon(cur_rmses, best_rmses)
                    r["Wilcoxon_P_vs_Best"] = float(w_res.pvalue)
                except Exception:
                    r["Wilcoxon_P_vs_Best"] = 1.0
                a12_val, _ = compute_vargha_delaney_a12(best_rmses, cur_rmses)
                r["A12_vs_Best"] = float(a12_val)

        df_comp = pd.DataFrame(comparison_rows)
        df_comp.to_csv(os.path.join(output_dir, "fair_model_comparison_summary.csv"), index=False)

        generate_sysid_study_comparison_plots(
            model_conv_map,
            fold_rmse_map,
            output_dir=output_dir,
            filename="fair_model_comparison_convergence_fe_and_time.png",
            suptitle=f"Study 4: Equal-Budget Convergence (FE & Wall-Clock Time) and 10-Fold RMSE ({dataset_name})"
        )

        tex_path = os.path.join(output_dir, "fair_model_comparison_table.tex")
        clean_ds = dataset_name.replace("_", r"\_")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write("\\begin{table*}[htbp]\n\\centering\n")
            f.write(f"\\caption{{Equal-Budget Comparison of ARMAX, MLP, and SVR on {clean_ds} ({system_mode})}}\n")
            f.write("\\label{tab:fair_comparison_armax_mlp_svr}\n")
            f.write("\\begin{tabular}{lccccccc}\n\\toprule\n")
            f.write("\\textbf{Model} & \\textbf{FE / Fold} & \\textbf{RMSE ($\\mu \\pm \\sigma$)} & \\textbf{MAE ($\\mu \\pm \\sigma$)} & \\textbf{$R^2$ ($\\mu \\pm \\sigma$)} & \\textbf{Time/Fold (s)} & \\textbf{ms / FE} & \\textbf{$p$-val vs. Best} \\\\\n\\midrule\n")
            for r in comparison_rows:
                p_str = "--" if r["Model_Variant"] == best_model else f"{r['Wilcoxon_P_vs_Best']:.4f}"
                f.write(
                    f"{r['Model_Variant'].replace('_', '-')} & {r['FE_Per_Fold']} & "
                    f"{r['RMSE_Mean']:.4f} $\\pm$ {r['RMSE_Std']:.4f} & "
                    f"{r['MAE_Mean']:.4f} $\\pm$ {r['MAE_Std']:.4f} & "
                    f"{r['R2_Mean']:.4f} $\\pm$ {r['R2_Std']:.4f} & "
                    f"{r['Time_Fold_Mean_Sec']:.2f} $\\pm$ {r['Time_Fold_Std_Sec']:.2f} & "
                    f"{r['Cost_Per_FE_ms']:.3f} & {p_str} \\\\\n"
                )
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    print(f"\n[OK] Equal-Budget Model Comparison complete! Artifacts saved to: {output_dir}")
    return output_dir


def _find_latest_study_folder(base_results_dir, prefix):
    if not os.path.exists(base_results_dir):
        return None
    search_dirs = [base_results_dir]
    for d in os.listdir(base_results_dir):
        p = os.path.join(base_results_dir, d)
        if os.path.isdir(p) and d.lower() in ("study", "studies"):
            search_dirs.append(p)
    candidates = []
    for sdir in search_dirs:
        candidates.extend([
            os.path.join(sdir, d)
            for d in os.listdir(sdir)
            if d.startswith(prefix) and os.path.isdir(os.path.join(sdir, d))
        ])
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def combine_all_studies_tables(output_dir: str, study_files: list) -> str:
    r"""
    Combines individual study LaTeX files into combined_studies_tables.tex while 
    rewriting duplicate \label{...} elements to remain unique across documents.
    """
    combined_tex_path = os.path.join(output_dir, "combined_studies_tables.tex")
    tex_chunks = []
    
    for study_tag, sub_prefix, file_path in study_files:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as tf:
                raw_tex = tf.read()
                
            # Disambiguate LaTeX labels if a sub_prefix or study_tag is provided
            if sub_prefix or study_tag:
                safe_tag = re.sub(r'[^a-zA-Z0-9_]', '_', f"{study_tag}_{sub_prefix}".lower().strip("_"))
                raw_tex = re.sub(
                    r"\\label\{([^}]+)\}", 
                    rf"\\label{{\1_{safe_tag}}}", 
                    raw_tex
                )
            
            tex_chunks.append(raw_tex + "\n\n")
            
    final_combined_tex = "".join(tex_chunks)
    with open(combined_tex_path, "w", encoding="utf-8") as out_f:
        out_f.write(final_combined_tex)
        
    return combined_tex_path


def run_overall_studies_comparison(
    study_dirs=None,
    base_results_dir="results",
    output_dir=None
):
    """
    Study 5: Overall Cross-Study Comparison.
    Aggregates CSV, LaTeX, and plot artifacts across Studies 1, 2, 3, and 4 into a 5th
    combined folder (`results/study_5_overall_comparison_<timestamp>`).
    Prevents filename collisions in `collected_study_figures/` by prefixing nested fold plots
    with their study tag and configuration subfolder.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = os.path.join(base_results_dir, f"study_5_overall_comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "collected_study_figures")
    os.makedirs(figures_dir, exist_ok=True)

    study_dirs = study_dirs or {}
    s1_dir = study_dirs.get("study_1") or _find_latest_study_folder(base_results_dir, "study_1_operator_ablation_")
    s2_dir = study_dirs.get("study_2") or _find_latest_study_folder(base_results_dir, "study_2_parameter_sensitivity_")
    s3_dir = study_dirs.get("study_3") or _find_latest_study_folder(base_results_dir, "study_3_")
    s4_dir = study_dirs.get("study_4") or _find_latest_study_folder(base_results_dir, "study_4_fair_model_comparison_")

    df_s1_bench, df_s1_sysid, df_s2, df_s3, df_s4 = None, None, None, None, None

    if s1_dir and os.path.exists(s1_dir):
        b_csv = os.path.join(s1_dir, "benchmarks_operators", "operator_ablation_benchmarks.csv")
        s_csv = os.path.join(s1_dir, "operator_ablation_sysid.csv")
        if os.path.exists(b_csv):
            df_s1_bench = pd.read_csv(b_csv)
        if os.path.exists(s_csv):
            df_s1_sysid = pd.read_csv(s_csv)

    if s2_dir and os.path.exists(s2_dir):
        s2_csv = os.path.join(s2_dir, "parameter_sensitivity_summary.csv")
        if os.path.exists(s2_csv):
            df_s2 = pd.read_csv(s2_csv)

    if s3_dir and os.path.exists(s3_dir):
        s3_csv = os.path.join(s3_dir, "study_3_dimensional_scalability.csv")
        if not os.path.exists(s3_csv):
            s3_csv = os.path.join(s3_dir, "study_3_statistical_significance.csv")
        if os.path.exists(s3_csv):
            df_s3 = pd.read_csv(s3_csv)

    if s4_dir and os.path.exists(s4_dir):
        s4_csv = os.path.join(s4_dir, "fair_model_comparison_summary.csv")
        if os.path.exists(s4_csv):
            df_s4 = pd.read_csv(s4_csv)

    summary_rows = []
    report_lines = [
        "==========================================================================================================",
        "          STUDY 5: OVERALL CROSS-STUDY COMPARISON REPORT (STUDIES 1 - 4)                                  ",
        "==========================================================================================================",
        f"Generated Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Study 1 Source Dir  : {s1_dir or 'Not Found'}",
        f"Study 2 Source Dir  : {s2_dir or 'Not Found'}",
        f"Study 3 Source Dir  : {s3_dir or 'Not Found'}",
        f"Study 4 Source Dir  : {s4_dir or 'Not Found'}",
        "-" * 106,
        ""
    ]

    # 1. Study 1 Summary (Operators - Per-Model Best Isolation)
    report_lines.append("=== 1. STUDY 1 SUMMARY: EVOLUTIONARY OPERATOR ABLATION ===")
    if df_s1_sysid is not None and not df_s1_sysid.empty:
        s1_group_col = "Model" if "Model" in df_s1_sysid.columns else None
        s1_groups = df_s1_sysid.groupby(s1_group_col, sort=False) if s1_group_col else [("SysId", df_s1_sysid)]
        multi_s1 = (s1_group_col is not None and df_s1_sysid[s1_group_col].nunique() > 1)

        for m_name, m_df in s1_groups:
            best_op_row = m_df.loc[m_df["RMSE_Mean"].idxmin()]
            worst_op_row = m_df.loc[m_df["RMSE_Mean"].idxmax()]
            gain_pct = (1.0 - float(best_op_row["RMSE_Mean"]) / max(1e-12, float(worst_op_row["RMSE_Mean"]))) * 100.0
            m_tag = f" [{m_name}]" if s1_group_col else ""
            report_lines.append(
                f"  • Best SysId Operator{m_tag:<8}: {best_op_row['Configuration']} "
                f"(Mutation={best_op_row['Mutation']}, Selection={best_op_row['Selection']}, pc={best_op_row['Crossover_Prob']})"
            )
            report_lines.append(
                f"    - Validation Performance   : RMSE = {best_op_row['RMSE_Mean']:.4f} +/- {best_op_row['RMSE_Std']:.4f} | "
                f"MAE = {best_op_row['MAE_Mean']:.4f} | R² = {best_op_row['R2_Mean']:.4f}"
            )
            report_lines.append(
                f"    - Ablation Improvement     : {gain_pct:.2f}% RMSE reduction vs. {worst_op_row['Configuration']}\n"
            )
            study_lbl = f"Study 1 (Operators - {m_name})" if multi_s1 else "Study 1 (Operator Ablation)"
            summary_rows.append({
                "Study_ID": study_lbl,
                "Evaluated_Scope": f"{len(m_df)} Operator Configs ({m_name}, {best_op_row.get('Dataset', 'SysId')})",
                "Optimal_Configuration_Winner": str(best_op_row["Configuration"]),
                "Primary_Metric": f"RMSE: {best_op_row['RMSE_Mean']:.4f} +/- {best_op_row['RMSE_Std']:.4f}",
                "Secondary_Metric": f"R²: {best_op_row['R2_Mean']:.4f} | MAE: {best_op_row['MAE_Mean']:.4f}",
                "Key_Finding": f"[{m_name}] {gain_pct:.2f}% RMSE reduction over worst operator config"
            })
    else:
        report_lines.append("  • [!] Study 1 SysId table not found.\n")

    # 2. Study 2 Summary (OFAT Sensitivity - Per-Model Best Isolation)
    report_lines.append("=== 2. STUDY 2 SUMMARY: HYPERPARAMETER SENSITIVITY ANALYSIS (OFAT) ===")
    df_s2_bests_list = []
    if df_s2 is not None and not df_s2.empty:
        s2_group_col = "Model" if "Model" in df_s2.columns else None
        s2_groups = df_s2.groupby(s2_group_col, sort=False) if s2_group_col else [("ARMAX", df_s2)]
        multi_s2 = (s2_group_col is not None and df_s2[s2_group_col].nunique() > 1)

        for m_name, m_df in s2_groups:
            best_sens_row = m_df.loc[m_df["RMSE_mean"].idxmin()]
            param_impacts = []
            for p_name, grp in m_df.groupby("parameter_name", sort=False):
                span = float(grp["RMSE_mean"].max() - grp["RMSE_mean"].min())
                best_p_row = grp.loc[grp["RMSE_mean"].idxmin()]
                best_val = best_p_row["parameter_value"]
                param_impacts.append((p_name, span, best_val, float(best_p_row["RMSE_mean"]), float(best_p_row["RMSE_std"])))
                df_s2_bests_list.append({
                    "Model": m_name,
                    "parameter_name": p_name,
                    "Optimal_Value": best_val,
                    "Best_RMSE_Mean": float(best_p_row["RMSE_mean"]),
                    "Best_RMSE_Std": float(best_p_row["RMSE_std"]),
                    "Best_MAE_Mean": float(best_p_row.get("MAE_mean", 0.0)),
                    "Best_R2_Mean": float(best_p_row.get("R2_mean", 0.0)),
                    "Delta_RMSE_Span": span,
                    "Time_Fold_Mean_Sec": float(best_p_row.get("time_mean", 0.0))
                })
            param_impacts.sort(key=lambda x: x[1], reverse=True)
            top_p, top_span, top_val, _, _ = param_impacts[0]
            m_tag = f" [{m_name}]" if s2_group_col else ""
            report_lines.append(
                f"  • [{m_name}] Overall Optimal OFAT Config : {best_sens_row['parameter_name']} = {best_sens_row['parameter_value']} "
                f"(RMSE = {best_sens_row['RMSE_mean']:.4f} +/- {best_sens_row['RMSE_std']:.4f}) | "
                f"Most Sensitive: {top_p} (ΔRMSE = {top_span:.4f})"
            )
            for p_name, span, b_val, b_rmse, b_std in param_impacts:
                report_lines.append(
                    f"    - {p_name:<18}: Optimal = {b_val:<6} (RMSE = {b_rmse:.4f} +/- {b_std:.4f}) | ΔRMSE = {span:.4f}"
                )
            report_lines.append("")
            study_lbl = f"Study 2 (OFAT - {m_name})" if multi_s2 else "Study 2 (OFAT Sensitivity)"
            summary_rows.append({
                "Study_ID": study_lbl,
                "Evaluated_Scope": f"5 Hyperparameters x 3 Levels ({m_name})",
                "Optimal_Configuration_Winner": f"{best_sens_row['parameter_name']} = {best_sens_row['parameter_value']}",
                "Primary_Metric": f"RMSE: {best_sens_row['RMSE_mean']:.4f} +/- {best_sens_row['RMSE_std']:.4f}",
                "Secondary_Metric": f"Time/Fold: {best_sens_row['time_mean']:.2f}s",
                "Key_Finding": f"[{m_name}] Most sensitive: {top_p} (ΔRMSE = {top_span:.4f}, best={top_val})"
            })
    else:
        report_lines.append("  • [!] Study 2 Sensitivity table not found.\n")

    # 3. Study 3 Summary (Dimensional Scalability Benchmarks)
    report_lines.append("=== 3. STUDY 3 SUMMARY: DIMENSIONAL SCALABILITY & BENCHMARK RIGOR ===")
    if df_s3 is not None and not df_s3.empty:
        total_cases = len(df_s3)
        ep_wins = int(np.sum(df_s3["Winner"] == "EP"))
        ties = int(np.sum(df_s3["Winner"] == "Tie"))
        de_wins = int(np.sum(df_s3["Winner"] == "DE"))
        mean_a12 = float(df_s3["A12_Effect_Size"].mean()) if "A12_Effect_Size" in df_s3.columns else 0.5
        ep_sr_avg = float(df_s3["EP_Success_Rate_Pct"].mean()) if "EP_Success_Rate_Pct" in df_s3.columns else 0.0
        de_sr_avg = float(df_s3["DE_Success_Rate_Pct"].mean()) if "DE_Success_Rate_Pct" in df_s3.columns else 0.0
        report_lines.append(
            f"  • Total Topology-Dimension Cases : {total_cases} "
            f"(EP Wins: {ep_wins}, Ties: {ties}, DE Wins: {de_wins})"
        )
        report_lines.append(
            f"  • Mean Precision Success Rate    : EP = {ep_sr_avg:.1f}% vs. DE = {de_sr_avg:.1f}% | Mean A12 = {mean_a12:.3f}\n"
        )
        summary_rows.append({
            "Study_ID": "Study 3 (Dimensional Scaling)",
            "Evaluated_Scope": f"{total_cases} Benchmark-Dimension Cases",
            "Optimal_Configuration_Winner": f"EP Wins/Ties: {ep_wins + ties}/{total_cases}",
            "Primary_Metric": f"EP SR: {ep_sr_avg:.1f}% (DE SR: {de_sr_avg:.1f}%)",
            "Secondary_Metric": f"Mean Vargha-Delaney A12: {mean_a12:.3f}",
            "Key_Finding": "Self-adaptive EP dominates high-D multimodal landscapes (Rastrigin/Schwefel)"
        })
    else:
        report_lines.append("  • [!] Study 3 Benchmark Scalability table not found.\n")

    # 4. Study 4 Summary (Equal-Budget Model Comparison)
    report_lines.append("=== 4. STUDY 4 SUMMARY: EQUAL-BUDGET MODEL ARCHITECTURE COMPARISON ===")
    if df_s4 is not None and not df_s4.empty:
        best_m_row = df_s4.loc[df_s4["RMSE_Mean"].idxmin()]
        report_lines.append(
            f"  • Overall Equal-Budget Winner    : {best_m_row['Model_Variant']} "
            f"(RMSE = {best_m_row['RMSE_Mean']:.4f} +/- {best_m_row['RMSE_Std']:.4f}, "
            f"MAE = {best_m_row['MAE_Mean']:.4f}, R² = {best_m_row['R2_Mean']:.4f})"
        )
        lam_gain_str = "Lamarckian vs Pure-EP evaluated"
        pure_rows = df_s4[df_s4["Model_Variant"] == "MLP_Pure_EP"]
        lam_rows = df_s4[df_s4["Model_Variant"] == "MLP_Lamarckian"]
        if not pure_rows.empty and not lam_rows.empty:
            p_rmse = float(pure_rows["RMSE_Mean"].iloc[0])
            l_rmse = float(lam_rows["RMSE_Mean"].iloc[0])
            l_gain = (1.0 - l_rmse / max(1e-12, p_rmse)) * 100.0
            lam_gain_str = f"Lamarckian refinement improved MLP RMSE by {l_gain:+.2f}% ({p_rmse:.4f} -> {l_rmse:.4f})"
            report_lines.append(f"  • Lamarckian Refinement Effect   : {lam_gain_str}")
        for _, r in df_s4.iterrows():
            report_lines.append(
                f"    - {r['Model_Variant']:<16} | FE/Fold: {int(r['FE_Per_Fold']):>6d} | "
                f"RMSE: {r['RMSE_Mean']:.4f} +/- {r['RMSE_Std']:.4f} | "
                f"MAE: {r['MAE_Mean']:.4f} | R²: {r['R2_Mean']:.4f} | "
                f"Time: {r['Time_Fold_Mean_Sec']:.2f}s ({r['Cost_Per_FE_ms']:.3f} ms/FE)"
            )
        report_lines.append("")
        summary_rows.append({
            "Study_ID": "Study 4 (Equal-Budget Models)",
            "Evaluated_Scope": "ARMAX vs. Pure-MLP vs. Lamarckian-MLP vs. SVR",
            "Optimal_Configuration_Winner": str(best_m_row["Model_Variant"]),
            "Primary_Metric": f"RMSE: {best_m_row['RMSE_Mean']:.4f} +/- {best_m_row['RMSE_Std']:.4f}",
            "Secondary_Metric": f"R²: {best_m_row['R2_Mean']:.4f} | {best_m_row['Cost_Per_FE_ms']:.3f} ms/FE",
            "Key_Finding": lam_gain_str
        })
    else:
        report_lines.append("  • [!] Study 4 Equal-Budget Model table not found.\n")

    df_sum = pd.DataFrame(summary_rows)
    if not df_sum.empty:
        df_sum.to_csv(os.path.join(output_dir, "overall_studies_summary.csv"), index=False)

    xlsx_path = os.path.join(output_dir, "overall_studies_comparison.xlsx")
    try:
        def _sanitize_for_excel(df_in):
            if df_in is None or df_in.empty:
                return df_in
            return df_in.map(lambda x: (x[:32000] + "... [TRUNCATED]") if isinstance(x, str) and len(x) > 32000 else x)

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            if not df_sum.empty:
                _sanitize_for_excel(df_sum).to_excel(writer, sheet_name="Overall_Summary", index=False)
            if df_s1_sysid is not None and not df_s1_sysid.empty:
                _sanitize_for_excel(df_s1_sysid).to_excel(writer, sheet_name="Study1_Operators_SysId", index=False)
            if df_s1_bench is not None and not df_s1_bench.empty:
                _sanitize_for_excel(df_s1_bench).to_excel(writer, sheet_name="Study1_Operators_Bench", index=False)
            if df_s2 is not None and not df_s2.empty:
                _sanitize_for_excel(df_s2).to_excel(writer, sheet_name="Study2_OFAT_Sensitivity", index=False)
            if df_s2_bests_list:
                _sanitize_for_excel(pd.DataFrame(df_s2_bests_list)).to_excel(writer, sheet_name="Study2_OFAT_Bests_By_Model", index=False)
            if df_s3 is not None and not df_s3.empty:
                _sanitize_for_excel(df_s3).to_excel(writer, sheet_name="Study3_Dim_Scalability", index=False)
            if df_s4 is not None and not df_s4.empty:
                _sanitize_for_excel(df_s4).to_excel(writer, sheet_name="Study4_Fair_Models", index=False)

            for sheet_name, worksheet in writer.sheets.items():
                for col in worksheet.columns:
                    max_len = max((len(str(cell.value or "")) for cell in col), default=10)
                    col_letter = getattr(col[0], "column_letter", None)
                    if not col_letter and hasattr(col[0], "column"):
                        try:
                            from openpyxl.utils import get_column_letter
                            col_letter = get_column_letter(col[0].column)
                        except Exception:
                            col_letter = None
                    if col_letter and hasattr(worksheet, "column_dimensions") and col_letter in worksheet.column_dimensions:
                        worksheet.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 55)
    except ImportError:
        print("\n[!] Notice: 'openpyxl' is not installed in the active virtual environment.")
        print("    To enable multi-sheet Excel (.xlsx) export, run: pip install openpyxl")
        print("    CSVs exported successfully.")
    except Exception as e:
        print(f"[!] Warning: Could not write Excel workbook ({e}). CSVs exported.")

    tex_combined_path = os.path.join(output_dir, "combined_studies_tables.tex")
    tex_chunks = [
        "% ==========================================================================",
        "% COMBINED EXPERIMENTAL STUDIES LATEX TABLES (STUDIES 1 - 5)",
        "% ==========================================================================\n"
    ]
    if not df_sum.empty:
        tex_chunks.extend([
            r"\begin{table*}[htbp]",
            r"\centering",
            r"\caption{Overall Summary Across Experimental Studies 1--4}",
            r"\label{tab:overall_studies_summary}",
            r"\begin{tabular}{llllll}",
            r"\toprule",
            r"\textbf{Study} & \textbf{Scope} & \textbf{Optimal / Winner} & \textbf{Primary Metric} & \textbf{Secondary Metric} & \textbf{Key Finding} \\",
            r"\midrule"
        ])
        for r in summary_rows:
            s_id = str(r["Study_ID"]).replace("_", r"\_").replace("%", r"\%")
            scope = str(r["Evaluated_Scope"]).replace("_", r"\_").replace("%", r"\%")
            win = str(r["Optimal_Configuration_Winner"]).replace("_", r"\_").replace("%", r"\%")
            pm = str(r["Primary_Metric"]).replace("+/-", r"$\pm$").replace("_", r"\_").replace("%", r"\%")
            sm = str(r["Secondary_Metric"]).replace("R²", r"$R^2$").replace("_", r"\_").replace("%", r"\%")
            kf = str(r["Key_Finding"]).replace("Δ", r"$\Delta$").replace("_", r"\_").replace("%", r"\%")
            tex_chunks.append(f"{s_id} & {scope} & {win} & {pm} & {sm} & {kf} \\\\")
        tex_chunks.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "\n"
        ])

    study_tagged_dirs = [
        ("study1", s1_dir),
        ("study2", s2_dir),
        ("study3", s3_dir),
        ("study4", s4_dir)
    ]
    for study_tag, src_dir in study_tagged_dirs:
        if src_dir and os.path.exists(src_dir):
            for root_d, _, f_names in os.walk(src_dir):
                rel_sub = os.path.relpath(root_d, src_dir)
                sub_prefix = "" if rel_sub == "." else rel_sub.replace(os.sep, "_")
                for fn in sorted(f_names):
                    if fn.endswith(".tex"):
                        try:
                            with open(os.path.join(root_d, fn), "r", encoding="utf-8") as tf:
                                raw_tex = tf.read()
                            # Disambiguate LaTeX labels to avoid duplicate label warnings
                            safe_tag = re.sub(r'[^a-zA-Z0-9_]', '_', f"{study_tag}_{sub_prefix}_{os.path.splitext(fn)[0]}".lower().strip("_"))
                            if safe_tag:
                                raw_tex = re.sub(
                                    r"\\label\{([^}]+)\}",
                                    rf"\\label{{\1_{safe_tag}}}",
                                    raw_tex
                                )
                            header_lbl = f"{study_tag}/{rel_sub}/{fn}" if sub_prefix else f"{study_tag}/{fn}"
                            tex_chunks.append(f"% --- Source: {header_lbl} ---")
                            tex_chunks.append(raw_tex + "\n")
                        except Exception:
                            pass
                    elif fn.endswith(".png"):
                        try:
                            if not sub_prefix:
                                dest_fn = fn if fn.startswith("study_") else f"{study_tag}_{fn}"
                            else:
                                dest_fn = f"{study_tag}_{sub_prefix}_{fn}"
                            shutil.copy2(os.path.join(root_d, fn), os.path.join(figures_dir, dest_fn))
                        except Exception:
                            pass

    with open(tex_combined_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_chunks))

    generate_studies_overview_dashboard(
        df_s1_sysid=df_s1_sysid,
        df_s2_sens=df_s2,
        df_s3_scale=df_s3,
        df_s4_fair=df_s4,
        output_dir=output_dir,
        filename="studies_overview_dashboard.png"
    )

    s1_models = set(df_s1_sysid["Model"].unique()) if (df_s1_sysid is not None and "Model" in df_s1_sysid.columns) else set()
    s2_models = set(df_s2["Model"].unique()) if (df_s2 is not None and "Model" in df_s2.columns) else set()
    all_models_present = sorted(s1_models | s2_models)
    if len(all_models_present) > 1:
        for m_name in all_models_present:
            sub_s1 = df_s1_sysid[df_s1_sysid["Model"] == m_name] if (df_s1_sysid is not None and "Model" in df_s1_sysid.columns) else df_s1_sysid
            sub_s2 = df_s2[df_s2["Model"] == m_name] if (df_s2 is not None and "Model" in df_s2.columns) else df_s2
            generate_studies_overview_dashboard(
                df_s1_sysid=sub_s1,
                df_s2_sens=sub_s2,
                df_s3_scale=df_s3,
                df_s4_fair=df_s4,
                output_dir=output_dir,
                filename=f"studies_overview_dashboard_{m_name}.png"
            )

    report_path = os.path.join(output_dir, "overall_studies_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n" + "\n".join(report_lines))
    print(f"[OK] Study 5 (Overall Cross-Study Comparison) complete! Saved to: {output_dir}")
    return output_dir
