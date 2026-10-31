"""
system_id_gui.py

GUI for configuring ARMAX, MLP, and SVR parameters, selecting SISO/MISO/SIMO/MIMO
system architectures, managing datasets, and running optimization and evaluation tasks.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import queue
import time
import os
import glob
import json
import uuid
import copy
import traceback
import warnings
from datetime import datetime
import multiprocessing

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

from optimization_benchmarks import run_full_benchmark_suite
from multivariable_data_parser import (
    load_cached_uci_benchmarks,
    fetch_and_parse_uci_dataset,
    parse_local_csv_file
)
from training_pipeline import (
    execute_kfold_training_pipeline,
    execute_inference_pipeline,
    execute_model_comparison_pipeline,
    execute_source_dataset_check,
    generate_results_summary_sheet,
    _parse_serialized_model
)
from config_handler import load_hyperparameters, get_parameter_label, atomic_replace_retry

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class ModelSelectionDialog(ctk.CTkToplevel):
    def __init__(self, parent, action_label="Select Model(s) for Evaluation / Comparison"):
        super().__init__(parent)
        self.title(action_label)
        self.geometry("820x560")
        self.minsize(680, 420)
        self.transient(parent)
        self.grab_set()

        self.selected_files = []
        self.result = None

        self._build_interface()
        self.after(150, self._prompt_add_files)

    def _build_interface(self):
        # Top toolbar
        top_bar = ctk.CTkFrame(self, fg_color="transparent")
        top_bar.pack(fill="x", padx=20, pady=(15, 10))

        title_lbl = ctk.CTkLabel(
            top_bar, text="Selected Models for Evaluation / Comparison",
            font=("Arial", 16, "bold"), text_color="#60A5FA"
        )
        title_lbl.pack(side="left")

        add_btn = ctk.CTkButton(
            top_bar, text="+ Add Model Files...", width=150,
            fg_color="#2563EB", hover_color="#1D4ED8",
            font=("Arial", 12, "bold"),
            command=self._prompt_add_files
        )
        add_btn.pack(side="right", padx=(8, 0))

        clear_btn = ctk.CTkButton(
            top_bar, text="Clear All", width=85,
            fg_color="#4B5563", hover_color="#374151",
            font=("Arial", 12),
            command=self._clear_all
        )
        clear_btn.pack(side="right")

        # Scrollable list of models
        self.scroll_frame = ctk.CTkScrollableFrame(self, corner_radius=8)
        self.scroll_frame.pack(fill="both", expand=True, padx=20, pady=5)

        # Bottom action bar
        bottom_bar = ctk.CTkFrame(self, fg_color="transparent")
        bottom_bar.pack(fill="x", padx=20, pady=(10, 15))

        self.counter_label = ctk.CTkLabel(
            bottom_bar, text="0 models selected", font=("Arial", 13)
        )
        self.counter_label.pack(side="left")

        cancel_btn = ctk.CTkButton(
            bottom_bar, text="Cancel", width=100,
            fg_color="#374151", hover_color="#1F2937",
            font=("Arial", 12),
            command=self._on_cancel
        )
        cancel_btn.pack(side="right", padx=(10, 0))

        self.proceed_btn = ctk.CTkButton(
            bottom_bar, text="Proceed with Evaluation", width=200,
            fg_color="#059669", hover_color="#047857",
            font=("Arial", 13, "bold"),
            command=self._on_proceed
        )
        self.proceed_btn.pack(side="right")

        self._refresh_list()

    def _prompt_add_files(self):
        os.makedirs("results", exist_ok=True)
        files = filedialog.askopenfilenames(
            parent=self,
            title="Select Trained Model Files (*_model.pkl, *_model.json)",
            initialdir="results",
            filetypes=[
                ("Trained Model Files", "*_model.pkl *_model.json identified_armax_model.pkl identified_armax_model.json"),
                ("All Model & JSON Files", "*.pkl *.json"),
                ("All Files", "*.*")
            ]
        )
        if files:
            for f in files:
                norm_f = os.path.normpath(f)
                stem, ext = os.path.splitext(norm_f)
                ext = ext.lower()

                existing_match = None
                for idx, existing_path in enumerate(self.selected_files):
                    existing_stem, existing_ext = os.path.splitext(existing_path)
                    if existing_stem == stem:
                        existing_match = (idx, existing_path, existing_ext.lower())
                        break

                if existing_match is None:
                    self.selected_files.append(norm_f)
                else:
                    match_idx, match_path, match_ext = existing_match
                    if match_ext == ".json" and ext == ".pkl":
                        self.selected_files[match_idx] = norm_f
            self._refresh_list()

    def _remove_file(self, target_file):
        if target_file in self.selected_files:
            self.selected_files.remove(target_file)
            self._refresh_list()

    def _clear_all(self):
        self.selected_files.clear()
        self._refresh_list()

    def _refresh_list(self):
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        if not self.selected_files:
            placeholder = ctk.CTkLabel(
                self.scroll_frame,
                text="No models selected yet.\nClick '+ Add Model Files...' above to select model files from results/.",
                font=("Arial", 13), text_color="#9CA3AF"
            )
            placeholder.pack(pady=50)
            self.proceed_btn.configure(state="disabled", text="Proceed with Evaluation (0)")
            self.counter_label.configure(text="0 models selected")
            return

        self.proceed_btn.configure(
            state="normal",
            text=f"Proceed with Evaluation ({len(self.selected_files)})"
        )
        self.counter_label.configure(
            text=f"{len(self.selected_files)} model(s) selected"
        )

        for idx, file_path in enumerate(self.selected_files):
            card = ctk.CTkFrame(self.scroll_frame, corner_radius=6, fg_color="#1E293B")
            card.pack(fill="x", padx=5, pady=4)

            # Detect model type badge
            base_name = os.path.basename(file_path)
            model_type = "MODEL"
            badge_color = "#3B82F6"
            if "armax" in base_name.lower():
                model_type = "ARMAX"
                badge_color = "#10B981"
            elif "mlp" in base_name.lower():
                model_type = "MLP"
                badge_color = "#8B5CF6"
            elif "svr" in base_name.lower():
                model_type = "SVR"
                badge_color = "#F59E0B"

            badge = ctk.CTkLabel(
                card, text=f" {model_type} ", font=("Arial", 11, "bold"),
                fg_color=badge_color, text_color="#FFFFFF", corner_radius=4
            )
            badge.pack(side="left", padx=(10, 8), pady=8)

            # Details
            parent_dir = os.path.basename(os.path.dirname(file_path)) or "."
            text_frame = ctk.CTkFrame(card, fg_color="transparent")
            text_frame.pack(side="left", fill="x", expand=True, padx=4, pady=6)

            name_lbl = ctk.CTkLabel(
                text_frame, text=f"#{idx+1}: {base_name}",
                font=("Arial", 12, "bold"), anchor="w"
            )
            name_lbl.pack(fill="x")

            dir_lbl = ctk.CTkLabel(
                text_frame, text=f"Folder: {parent_dir}",
                font=("Arial", 10), text_color="#94A3B8", anchor="w"
            )
            dir_lbl.pack(fill="x")

            # Red "X" remove button
            remove_btn = ctk.CTkButton(
                card, text="✕", width=34, height=28,
                fg_color="#DC2626", hover_color="#B91C1C",
                font=("Arial", 13, "bold"),
                command=lambda p=file_path: self._remove_file(p)
            )
            remove_btn.pack(side="right", padx=10, pady=8)

    def _on_proceed(self):
        if self.selected_files:
            self.result = list(self.selected_files)
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()


class EvolutionarySystemIDApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Evolutionary System Identification Tool (ARMAX / MLP / SVR — SISO / MISO / SIMO / MIMO)")
        self.geometry("920x1000")
        self.minsize(880, 850)

        self.loaded_datasets = load_cached_uci_benchmarks()
        self.dataset_registry_names = list(self.loaded_datasets.keys())
        self.active_hyperparameters = load_hyperparameters("config.json")
        self.stop_event = threading.Event()
        self.active_stop_events = []
        self.active_runs_count = 0
        self.resume_dir = None

        self.is_queue_training_enabled = False
        self.training_queue = queue.Queue()
        self.queue_worker_thread = None
        self.queue_worker_busy = False
        self.queue_lock = threading.Lock()
        self.current_queue_job = None
        self.active_model_cards = {}
        self.auto_clear_cards_var = ctk.BooleanVar(value=bool(self.active_hyperparameters.get("auto_clear_monitor", False)))

        self.ui_update_queue = queue.Queue()
        self._construct_user_interface()
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)
        self._poll_ui_queue()

    def _trigger_stop_all(self):
        self.stop_event.set()
        with self.queue_lock:
            for ev in list(self.active_stop_events):
                ev.set()

        cleared_count = 0
        while not self.training_queue.empty():
            try:
                self.training_queue.get_nowait()
                self.training_queue.task_done()
                cleared_count += 1
            except queue.Empty:
                break

        self._update_queue_ui()

        # Update all active monitor cards to stopped or cancelled
        def _stop_cards():
            for key, card in list(self.active_model_cards.items()):
                if card.get("state") == "training":
                    card["state"] = "stopped"
                    try:
                        card["frame"].configure(border_color="#F59E0B")
                        card["status_lbl"].configure(text="■ Stopped", text_color="#F59E0B")
                        card["progress_bar"].configure(progress_color="#F59E0B")
                        card["telemetry_lbl"].configure(text="Training aborted by user", text_color="#F59E0B")
                        card["close_btn"].pack(side="left")
                    except Exception:
                        pass
                elif card.get("state") == "queued":
                    card["state"] = "stopped"
                    try:
                        card["frame"].configure(border_color="#F59E0B")
                        card["status_lbl"].configure(text="■ Cancelled", text_color="#F59E0B")
                        card["telemetry_lbl"].configure(text="Removed from queue", text_color="#F59E0B")
                        card["close_btn"].pack(side="left")
                    except Exception:
                        pass
            self._update_monitor_placeholder()

        self._safe_after(_stop_cards)

        if cleared_count > 0:
            self._set_status(f"Stop signal sent to active task(s). Cleared {cleared_count} pending job(s) from queue.", "orange")
        else:
            self._set_status("Stop signal sent to all active tasks.", "orange")

    def _on_window_close(self):
        self._trigger_stop_all()
        try:
            self.destroy()
        except Exception:
            pass
        os._exit(0)

    def _poll_ui_queue(self):
        try:
            while not self.ui_update_queue.empty():
                try:
                    callback = self.ui_update_queue.get_nowait()
                    callback()
                except queue.Empty:
                    break
        except Exception:
            pass
        finally:
            try:
                self.after(50, self._poll_ui_queue)
            except Exception:
                pass

    def _safe_after(self, callback):
        # Tkinter is not thread-safe; route worker thread calls through the queue to the main thread
        if threading.current_thread() is threading.main_thread():
            try:
                callback()
            except Exception:
                pass
        else:
            self.ui_update_queue.put(callback)

    def _set_buttons_state(self, state):
        def _apply():
            # Keep train_btn interactive so users can launch concurrent optimizations
            for btn in (self.benchmark_btn, self.resume_btn):
                try:
                    btn.configure(state=state)
                except Exception:
                    pass
        self._safe_after(_apply)

    def _set_status(self, text, color="cyan"):
        self._safe_after(lambda: self.status_display.configure(text=text, text_color=color))

    def _construct_user_interface(self):
        # 1. Dataset Selection & Ingestion panel (At the top: choose the dataset / task first)
        self.dataset_panel = ctk.CTkFrame(self, corner_radius=10)
        self.dataset_panel.pack(pady=(14, 6), padx=20, fill="x")
        self.dataset_panel.grid_columnconfigure((0, 2), weight=1)
        self.dataset_panel.grid_columnconfigure((1, 3), weight=1)

        ctk.CTkLabel(
            self.dataset_panel, text="Dataset Configuration",
            font=("Arial", 14, "bold"), text_color="#60A5FA"
        ).grid(row=0, column=0, columnspan=4, padx=15, pady=(8, 4), sticky="w")

        ctk.CTkLabel(self.dataset_panel, text="Dataset:", font=("Arial", 12, "bold")).grid(
            row=1, column=0, padx=10, pady=5, sticky="e"
        )
        self.dataset_selector = ctk.CTkOptionMenu(
            self.dataset_panel, values=self.dataset_registry_names, width=220
        )
        self.dataset_selector.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(self.dataset_panel, text="Or load custom file:", font=("Arial", 12)).grid(
            row=1, column=2, padx=10, pady=5, sticky="e"
        )
        self.load_csv_btn = ctk.CTkButton(
            self.dataset_panel, text="Load Local CSV", width=160, fg_color="#2563EB", hover_color="#1D4ED8",
            command=self._prompt_local_csv_load
        )
        self.load_csv_btn.grid(row=1, column=3, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(self.dataset_panel, text="Fetch UCI ID:", font=("Arial", 12)).grid(
            row=2, column=0, padx=10, pady=(5, 10), sticky="e"
        )
        uci_frame = ctk.CTkFrame(self.dataset_panel, fg_color="transparent")
        uci_frame.grid(row=2, column=1, padx=10, pady=(5, 10), sticky="w")
        self.uci_input_field = ctk.CTkEntry(uci_frame, width=90, placeholder_text="e.g. 294")
        self.uci_input_field.pack(side="left", padx=(0, 6))
        self.download_uci_btn = ctk.CTkButton(
            uci_frame, text="Download", width=100, fg_color="#4338CA", hover_color="#3730A3",
            command=self._dispatch_uci_download_thread
        )
        self.download_uci_btn.pack(side="left")

        # 2. Model Architecture & Settings panel (Scroll select at the top of model settings)
        self.model_panel = ctk.CTkFrame(self, corner_radius=10)
        self.model_panel.pack(pady=6, padx=20, fill="x")
        self.model_panel.grid_columnconfigure((0, 2), weight=1)
        self.model_panel.grid_columnconfigure((1, 3), weight=1)

        ctk.CTkLabel(
            self.model_panel, text="Model Architecture & Settings",
            font=("Arial", 14, "bold"), text_color="#60A5FA"
        ).grid(row=0, column=0, columnspan=4, padx=15, pady=(8, 4), sticky="w")

        # Top row of model panel: Scroll select menus for Model Type & System Mode
        ctk.CTkLabel(self.model_panel, text="Model Type:", font=("Arial", 12, "bold")).grid(
            row=1, column=0, padx=10, pady=5, sticky="e"
        )
        self.model_type_selector = ctk.CTkOptionMenu(
            self.model_panel, values=["ARMAX", "MLP", "SVR", "ALL (Batch: ARMAX, MLP, SVR)"],
            width=220, command=self._on_model_type_changed
        )
        self.model_type_selector.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.model_type_selector.set(self.active_hyperparameters.get("model_type", "ARMAX"))

        ctk.CTkLabel(self.model_panel, text="System Mode (IO):", font=("Arial", 12, "bold")).grid(
            row=1, column=2, padx=10, pady=5, sticky="e"
        )
        self.system_mode_selector = ctk.CTkOptionMenu(
            self.model_panel, values=["SISO", "MISO", "SIMO", "MIMO"], width=150, command=self._on_system_mode_changed
        )
        self.system_mode_selector.grid(row=1, column=3, padx=10, pady=5, sticky="w")
        self.system_mode_selector.set(str(self.active_hyperparameters.get("system_mode", "SISO")).upper())

        self.hyperparameter_inputs = {}
        self.hyperparameter_labels = {}

        # Model structural parameters
        for key in ("na", "nb", "nc", "nk", "mlp_hidden_nodes", "input_col_idx", "output_col_idx"):
            default_val = self.active_hyperparameters.get(key, 0)
            lbl = ctk.CTkLabel(self.model_panel, text=f"{get_parameter_label(key)}:")
            self.hyperparameter_labels[key] = lbl
            w = 150 if key == "mlp_hidden_nodes" else 85
            inp = ctk.CTkEntry(self.model_panel, width=w)
            inp.insert(0, str(default_val))
            self.hyperparameter_inputs[key] = inp

        # Initial grid placement for model parameters (Row 2: na, nb)
        self.hyperparameter_labels["na"].grid(row=2, column=0, padx=10, pady=5, sticky="e")
        self.hyperparameter_inputs["na"].grid(row=2, column=1, padx=10, pady=5, sticky="w")
        self.hyperparameter_labels["nb"].grid(row=2, column=2, padx=10, pady=5, sticky="e")
        self.hyperparameter_inputs["nb"].grid(row=2, column=3, padx=10, pady=5, sticky="w")

        # 3. Evolutionary Algorithm Hyperparameters panel
        self.ea_panel = ctk.CTkFrame(self, corner_radius=10)
        self.ea_panel.pack(pady=6, padx=20, fill="x")
        self.ea_panel.grid_columnconfigure((0, 2), weight=1)
        self.ea_panel.grid_columnconfigure((1, 3), weight=1)

        ctk.CTkLabel(
            self.ea_panel, text="Evolutionary Algorithm Hyperparameters",
            font=("Arial", 14, "bold"), text_color="#60A5FA"
        ).grid(row=0, column=0, columnspan=4, padx=15, pady=(8, 4), sticky="w")

        ea_fields = [
            ("selection_type", 1, 0),
            ("mutation_type", 1, 2),
            ("pop_size", 2, 0),
            ("num_gen", 2, 2),
            ("mutation_sigma", 3, 0),
            ("crossover_prob", 3, 2),
            ("tournament_size", 4, 0),
            ("early_stop_patience", 4, 2),
            ("k_folds", 5, 0),
            ("l2_reg", 5, 2),
            ("random_seed", 6, 0),
            ("checkpoint_freq", 6, 2)
        ]

        for key, r, c in ea_fields:
            default_val = self.active_hyperparameters.get(key)
            is_dropdown = key in ("mutation_type", "selection_type")
            lbl_text = "Selection Strategy:" if key == "selection_type" else ("Mutation Operator:" if key == "mutation_type" else f"{get_parameter_label(key)}:")
            lbl_font = ("Arial", 12, "bold") if is_dropdown else ("Arial", 12)
            lbl = ctk.CTkLabel(self.ea_panel, text=lbl_text, font=lbl_font)
            self.hyperparameter_labels[key] = lbl
            if key == "mutation_type":
                inp = ctk.CTkOptionMenu(self.ea_panel, values=["Gaussian", "Cauchy", "Hybrid"], width=140)
                inp.set(str(default_val))
            elif key == "selection_type":
                inp = ctk.CTkOptionMenu(
                    self.ea_panel, values=["Truncation", "Tournament"], width=140,
                    command=lambda _: self._refresh_visible_fields()
                )
                inp.set(str(default_val))
            else:
                inp = ctk.CTkEntry(self.ea_panel, width=85)
                inp.insert(0, str(default_val))
            self.hyperparameter_inputs[key] = inp

            if key != "tournament_size":
                row_pady = (5, 10) if r == 6 else 5
                lbl.grid(row=r, column=c, padx=10, pady=row_pady, sticky="e")
                inp.grid(row=r, column=c + 1, padx=10, pady=row_pady, sticky="w")

        # 4. Training Mode Selection (Queue Training vs Parallel Training)
        self.mode_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.mode_panel.pack(pady=(6, 2))

        self.queue_training_var = ctk.BooleanVar(value=False)
        self.queue_training_checkbox = ctk.CTkCheckBox(
            self.mode_panel,
            text="Queue Training (Sequential instead of Parallel)",
            variable=self.queue_training_var,
            font=("Arial", 12, "bold"),
            command=self._on_queue_mode_toggled,
            checkbox_width=20,
            checkbox_height=20
        )
        self.queue_training_checkbox.pack(side="left", padx=10)

        self.queue_status_label = ctk.CTkLabel(
            self.mode_panel,
            text="Mode: Parallel Training (Concurrent)",
            font=("Arial", 11),
            text_color="#94A3B8"
        )
        self.queue_status_label.pack(side="left", padx=8)

        self.clear_queue_btn = ctk.CTkButton(
            self.mode_panel,
            text="Clear Pending Queue",
            width=135,
            height=24,
            font=("Arial", 11, "bold"),
            fg_color="#DC2626",
            hover_color="#B91C1C",
            command=self._clear_pending_queue
        )
        # clear_queue_btn will be dynamically displayed when items are pending

        # 5. Action buttons (all buttons have different, distinct colors)
        self.execution_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.execution_panel.pack(pady=(4, 10))

        btn_w, btn_h = 200, 36
        btn_font = ("Arial", 12, "bold")

        # Row 0: Checkpoints (Amber), Benchmarks (Royal Blue), Resume (Sky Blue)
        ctk.CTkButton(
            self.execution_panel, text="Clear Checkpoints", font=btn_font,
            fg_color="#D97706", hover_color="#B45309",
            command=self._clear_checkpoints, width=btn_w, height=btn_h
        ).grid(row=0, column=0, padx=6, pady=5)

        self.benchmark_btn = ctk.CTkButton(
            self.execution_panel, text="Run Benchmarks", font=btn_font,
            fg_color="#1D4ED8", hover_color="#1E40AF",
            command=self._dispatch_benchmark_suite, width=btn_w, height=btn_h
        )
        self.benchmark_btn.grid(row=0, column=1, padx=6, pady=5)

        self.resume_btn = ctk.CTkButton(
            self.execution_panel, text="Resume Previous Run", font=btn_font,
            fg_color="#0284C7", hover_color="#0369A1",
            command=self._prompt_resume_folder, width=btn_w, height=btn_h
        )
        self.resume_btn.grid(row=0, column=2, padx=6, pady=5)

        # Row 1: Optimization (Green), Stop (Crimson Red), Evaluate (Violet Purple)
        self.train_btn = ctk.CTkButton(
            self.execution_panel, text="Run Optimization", font=btn_font,
            fg_color="#2E7D32", hover_color="#1B5E20",
            command=self._dispatch_training_pipeline, width=btn_w, height=btn_h
        )
        self.train_btn.grid(row=1, column=0, padx=6, pady=5)

        self.stop_btn = ctk.CTkButton(
            self.execution_panel, text="Stop All", font=btn_font,
            fg_color="#C62828", hover_color="#8E0000",
            command=self._trigger_stop_all, width=btn_w, height=btn_h
        )
        self.stop_btn.grid(row=1, column=1, padx=6, pady=5)

        self.eval_btn = ctk.CTkButton(
            self.execution_panel, text="Evaluate / Compare Models", font=btn_font,
            fg_color="#6D28D9", hover_color="#5B21B6",
            command=self._dispatch_model_evaluation, width=btn_w, height=btn_h
        )
        self.eval_btn.grid(row=1, column=2, padx=6, pady=5)

        # Row 2: Summary Export (Sea Teal), Open Folder (Steel Slate), Exit (Deep Wine Red)
        self.export_sheet_btn = ctk.CTkButton(
            self.execution_panel, text="Export Summary Sheet", font=btn_font,
            fg_color="#0F766E", hover_color="#115E59",
            command=self._dispatch_summary_export, width=btn_w, height=btn_h
        )
        self.export_sheet_btn.grid(row=2, column=0, padx=6, pady=8)

        ctk.CTkButton(
            self.execution_panel, text="Open Results Folder", font=btn_font,
            fg_color="#475569", hover_color="#334155",
            command=self._open_results_folder, width=btn_w, height=btn_h
        ).grid(row=2, column=1, padx=6, pady=8)

        ctk.CTkButton(
            self.execution_panel, text="Exit", font=btn_font,
            fg_color="#881337", hover_color="#4C0519",
            command=self._terminate_application, width=btn_w, height=btn_h
        ).grid(row=2, column=2, padx=6, pady=8)

        # 6. Active Model Training Monitor
        self.monitor_frame = ctk.CTkFrame(self, corner_radius=10)
        self.monitor_frame.pack(pady=(2, 6), padx=20, fill="x")

        monitor_header = ctk.CTkFrame(self.monitor_frame, fg_color="transparent")
        monitor_header.pack(fill="x", padx=10, pady=(6, 2))

        ctk.CTkLabel(
            monitor_header,
            text="Active Model Training Monitor",
            font=("Arial", 13, "bold"),
            text_color=("#2563EB", "#38BDF8")
        ).pack(side="left", padx=(5, 14))

        self.auto_clear_checkbox = ctk.CTkCheckBox(
            monitor_header,
            text="Auto-clear on completion",
            variable=self.auto_clear_cards_var,
            font=("Arial", 11),
            width=16, height=16,
            checkbox_width=18, checkbox_height=18,
            command=self._on_auto_clear_toggled
        )
        self.auto_clear_checkbox.pack(side="left", padx=(0, 10))

        self.clear_completed_cards_btn = ctk.CTkButton(
            monitor_header,
            text="Clear All Trained",
            font=("Arial", 11, "bold"),
            width=125,
            height=24,
            fg_color=("#CBD5E1", "#334155"),
            hover_color=("#94A3B8", "#475569"),
            text_color=("#1E293B", "#F1F5F9"),
            command=self._clear_finished_cards
        )
        self.clear_completed_cards_btn.pack(side="right", padx=(8, 5))

        self.monitor_cards_container = ctk.CTkScrollableFrame(
            self.monitor_frame,
            height=150,
            fg_color=("#F8FAFC", "#0F172A"),
            corner_radius=6
        )
        self.monitor_cards_container.pack(fill="both", expand=True, padx=8, pady=(2, 8))

        self.monitor_placeholder = ctk.CTkLabel(
            self.monitor_cards_container,
            text="No models currently training.",
            font=("Arial", 12, "italic"),
            text_color="#64748B"
        )
        self.monitor_placeholder.pack(pady=35)

        # 7. Status Display
        self.status_display = ctk.CTkLabel(
            self, text="Ready.", font=("Arial", 12, "bold"), text_color="#94A3B8", wraplength=780
        )
        self.status_display.pack(pady=4)
        self._refresh_visible_fields()

    def _open_results_folder(self):
        os.makedirs("results", exist_ok=True)
        abs_path = os.path.abspath("results")
        try:
            if hasattr(os, "startfile"):
                os.startfile(abs_path)
            else:
                import subprocess
                import sys
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.Popen([opener, abs_path])
        except Exception as error:
            self.status_display.configure(text=f"Could not open folder: {error}", text_color="red")

    def _on_system_mode_changed(self, _selected_mode=None):
        self._refresh_visible_fields()

    def _on_model_type_changed(self, _selected_model=None):
        self._refresh_visible_fields()

    def _refresh_visible_fields(self):
        selected_model = self.model_type_selector.get()
        selected_mode = self.system_mode_selector.get()
        sel_widget = self.hyperparameter_inputs.get("selection_type")
        is_tournament = sel_widget and "tourn" in str(sel_widget.get()).lower()

        # Hide conditional model fields first
        for key in ("nc", "nk", "mlp_hidden_nodes", "input_col_idx", "output_col_idx"):
            if key in self.hyperparameter_inputs:
                self.hyperparameter_labels[key].grid_remove()
                self.hyperparameter_inputs[key].grid_remove()

        # Row 3 / 4: nc on left (cols 0, 1) and nk on right (cols 2, 3) for ARMAX / ALL
        if selected_model == "ARMAX":
            if "nc" in self.hyperparameter_inputs:
                self.hyperparameter_labels["nc"].grid(row=3, column=0, padx=10, pady=5, sticky="e")
                self.hyperparameter_inputs["nc"].grid(row=3, column=1, padx=10, pady=5, sticky="w")
            if "nk" in self.hyperparameter_inputs:
                self.hyperparameter_labels["nk"].grid(row=3, column=2, padx=10, pady=5, sticky="e")
                self.hyperparameter_inputs["nk"].grid(row=3, column=3, padx=10, pady=5, sticky="w")
        elif selected_model == "MLP":
            if "nk" in self.hyperparameter_inputs:
                self.hyperparameter_labels["nk"].grid(row=3, column=0, padx=10, pady=5, sticky="e")
                self.hyperparameter_inputs["nk"].grid(row=3, column=1, padx=10, pady=5, sticky="w")
            if "mlp_hidden_nodes" in self.hyperparameter_inputs:
                self.hyperparameter_labels["mlp_hidden_nodes"].grid(row=3, column=2, padx=10, pady=5, sticky="e")
                self.hyperparameter_inputs["mlp_hidden_nodes"].grid(row=3, column=3, padx=10, pady=5, sticky="w")
        elif "ALL" in selected_model:
            if "nc" in self.hyperparameter_inputs:
                self.hyperparameter_labels["nc"].grid(row=3, column=0, padx=10, pady=5, sticky="e")
                self.hyperparameter_inputs["nc"].grid(row=3, column=1, padx=10, pady=5, sticky="w")
            if "nk" in self.hyperparameter_inputs:
                self.hyperparameter_labels["nk"].grid(row=3, column=2, padx=10, pady=5, sticky="e")
                self.hyperparameter_inputs["nk"].grid(row=3, column=3, padx=10, pady=5, sticky="w")
            if "mlp_hidden_nodes" in self.hyperparameter_inputs:
                self.hyperparameter_labels["mlp_hidden_nodes"].grid(row=4, column=2, padx=10, pady=5, sticky="e")
                self.hyperparameter_inputs["mlp_hidden_nodes"].grid(row=4, column=3, padx=10, pady=5, sticky="w")
        else:  # SVR
            if "nk" in self.hyperparameter_inputs:
                self.hyperparameter_labels["nk"].grid(row=3, column=0, padx=10, pady=5, sticky="e")
                self.hyperparameter_inputs["nk"].grid(row=3, column=1, padx=10, pady=5, sticky="w")

        # Channel indices based on system_mode
        channel_row = 5 if "ALL" in selected_model else 4
        pad_b = (5, 8)
        if selected_mode == "SISO":
            if "input_col_idx" in self.hyperparameter_inputs:
                self.hyperparameter_labels["input_col_idx"].grid(row=channel_row, column=0, padx=10, pady=pad_b, sticky="e")
                self.hyperparameter_inputs["input_col_idx"].grid(row=channel_row, column=1, padx=10, pady=pad_b, sticky="w")
            if "output_col_idx" in self.hyperparameter_inputs:
                self.hyperparameter_labels["output_col_idx"].grid(row=channel_row, column=2, padx=10, pady=pad_b, sticky="e")
                self.hyperparameter_inputs["output_col_idx"].grid(row=channel_row, column=3, padx=10, pady=pad_b, sticky="w")
        elif selected_mode == "SIMO":
            if "input_col_idx" in self.hyperparameter_inputs:
                self.hyperparameter_labels["input_col_idx"].grid(row=channel_row, column=0, padx=10, pady=pad_b, sticky="e")
                self.hyperparameter_inputs["input_col_idx"].grid(row=channel_row, column=1, padx=10, pady=pad_b, sticky="w")
        elif selected_mode == "MISO":
            if "output_col_idx" in self.hyperparameter_inputs:
                self.hyperparameter_labels["output_col_idx"].grid(row=channel_row, column=0, padx=10, pady=pad_b, sticky="e")
                self.hyperparameter_inputs["output_col_idx"].grid(row=channel_row, column=1, padx=10, pady=pad_b, sticky="w")

        # Evolutionary Algorithm: tournament_size (dynamically shown in Row 4 alongside early_stop_patience)
        if "tournament_size" in self.hyperparameter_inputs:
            if is_tournament:
                self.hyperparameter_labels["tournament_size"].grid(row=4, column=0, padx=10, pady=5, sticky="e")
                self.hyperparameter_inputs["tournament_size"].grid(row=4, column=1, padx=10, pady=5, sticky="w")
            else:
                self.hyperparameter_labels["tournament_size"].grid_remove()
                self.hyperparameter_inputs["tournament_size"].grid_remove()

    def _register_parsed_dataset(self, dataset_name, u_signal, y_signal, status_message):
        self.loaded_datasets[dataset_name] = (u_signal, y_signal)
        if dataset_name not in self.dataset_registry_names:
            self.dataset_registry_names.append(dataset_name)
            self.dataset_selector.configure(values=self.dataset_registry_names)
        self.dataset_selector.set(dataset_name)
        self.status_display.configure(text=status_message, text_color="green")

    def _dispatch_uci_download_thread(self):
        repository_id = self.uci_input_field.get().strip()
        if not repository_id.isdigit():
            self.status_display.configure(text="Error: UCI ID must be numeric.", text_color="red")
            return
        self.status_display.configure(text=f"Requesting dataset ID {repository_id}...", text_color="cyan")
        self.download_uci_btn.configure(state="disabled")
        threading.Thread(target=self._execute_uci_download, args=(repository_id,)).start()

    def _execute_uci_download(self, repository_id):
        dataset_name, u_signal, y_signal, error_log = fetch_and_parse_uci_dataset(repository_id)
        def _finish():
            if dataset_name:
                self._register_parsed_dataset(dataset_name, u_signal, y_signal, f"Loaded dataset: '{dataset_name}'.")
            else:
                self.status_display.configure(text=f"Download failed: {error_log}", text_color="red")
            self.download_uci_btn.configure(state="normal")
        self._safe_after(_finish)

    def _prompt_local_csv_load(self):
        target_filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if not target_filepath:
            return
        dataset_name, u_signal, y_signal, error_log = parse_local_csv_file(target_filepath)
        if dataset_name:
            self._register_parsed_dataset(dataset_name, u_signal, y_signal, f"Loaded CSV '{dataset_name}'.")
        else:
            self.status_display.configure(text=f"CSV Error: {error_log}", text_color="red")

    def _prompt_resume_folder(self):
        os.makedirs("results", exist_ok=True)
        folder_selected = filedialog.askdirectory(title="Select Run Directory to Resume", initialdir="results")
        if not folder_selected or not os.path.isdir(folder_selected):
            return

        existing_ckpts = glob.glob(os.path.join(folder_selected, "checkpoint_fold_*.pkl"))
        if not existing_ckpts:
            self.status_display.configure(
                text=f"No checkpoints found in {os.path.basename(folder_selected)}.",
                text_color="red"
            )
            return

        meta_loaded = False
        meta_path = os.path.join(folder_selected, "run_metadata.json")
        if os.path.exists(meta_path) and os.path.getsize(meta_path) > 0:
            try:
                with open(meta_path, "r", encoding="utf-8", errors="replace") as f:
                    meta = json.load(f)
                saved_dataset = meta.get("dataset_name")
                if saved_dataset and saved_dataset in self.loaded_datasets:
                    self.dataset_selector.set(saved_dataset)

                saved_hparams = meta.get("hyperparameters", {})
                saved_model = saved_hparams.get("model_type")
                if saved_model in ("ARMAX", "MLP", "SVR"):
                    self.model_type_selector.set(saved_model)

                saved_mode = meta.get("system_mode", saved_hparams.get("system_mode"))
                if saved_mode in ("SISO", "MISO", "SIMO", "MIMO"):
                    self.system_mode_selector.set(saved_mode)

                self._refresh_visible_fields()

                # Reset channel indices to 0 first in case they were excluded in MISO/SIMO/MIMO metadata
                for idx_key in ("input_col_idx", "output_col_idx"):
                    if idx_key in self.hyperparameter_inputs:
                        self.hyperparameter_inputs[idx_key].delete(0, "end")
                        self.hyperparameter_inputs[idx_key].insert(0, str(saved_hparams.get(idx_key, 0)))

                for k, v in saved_hparams.items():
                    if k in self.hyperparameter_inputs:
                        w = self.hyperparameter_inputs[k]
                        if isinstance(w, ctk.CTkOptionMenu):
                            w.set(str(v))
                        else:
                            w.delete(0, "end")
                            w.insert(0, str(v))
                meta_loaded = True
            except Exception as e:
                print(f"[!] Warning: Could not load run_metadata.json: {e}")

        if not meta_loaded:
            folder_base = os.path.basename(folder_selected)
            parts = folder_base.split("_")
            if len(parts) >= 2:
                candidate_model = parts[0]
                candidate_mode = parts[1]
                if candidate_model in ("ARMAX", "MLP", "SVR"):
                    self.model_type_selector.set(candidate_model)
                if candidate_mode in ("SISO", "MISO", "SIMO", "MIMO"):
                    self.system_mode_selector.set(candidate_mode)
                for ds_name in self.dataset_registry_names:
                    if ds_name in folder_base:
                        self.dataset_selector.set(ds_name)
                        break
            self._refresh_visible_fields()

        # Set resume_dir AFTER all UI updates so _refresh_visible_fields does not clear it
        self.resume_dir = folder_selected

        self.status_display.configure(
            text=f"Ready to resume '{os.path.basename(folder_selected)}' ({len(existing_ckpts)} fold checkpoints found). Click 'Run Optimization'.",
            text_color="cyan"
        )

    def _clear_checkpoints(self):
        # Don't let users delete checkpoint files while a background training job is writing to them
        with self.queue_lock:
            is_running = (self.active_runs_count > 0) or self.queue_worker_busy
        if is_running:
            messagebox.showwarning(
                "Training in Progress",
                "Cannot delete checkpoints while a training job is actively running.\nPlease wait or click 'Stop All' first."
            )
            return

        root_checkpoints = glob.glob("checkpoint_*.pkl")
        nested_checkpoints = glob.glob(os.path.join("results", "**", "checkpoint_*.pkl"), recursive=True)
        all_checkpoints = root_checkpoints + nested_checkpoints

        if not all_checkpoints:
            self.status_display.configure(
                text="No checkpoint files (*.pkl) found to clear.",
                text_color="orange"
            )
            messagebox.showinfo(
                "Clear Checkpoints",
                "No training checkpoint files (*.pkl) were found.\nNothing was deleted."
            )
            return

        confirm = messagebox.askyesno(
            "Confirm Checkpoint Deletion",
            f"Found {len(all_checkpoints)} training checkpoint file(s) across root and results folders.\n\n"
            f"What are checkpoints?\n"
            f"Checkpoints save intermediate evolutionary generations so interrupted training "
            f"runs can be resumed later with 'Resume Previous Run'.\n\n"
            f"Note: Completed model weights (*_model.pkl, *_model.json) and diagnostic plots will NOT be deleted.\n\n"
            f"Are you sure you want to permanently delete these {len(all_checkpoints)} checkpoint file(s)?"
        )

        if not confirm:
            self.status_display.configure(
                text="Checkpoint deletion cancelled.",
                text_color="#94A3B8"
            )
            return

        deleted_count = 0
        for file in all_checkpoints:
            try:
                os.remove(file)
                deleted_count += 1
            except OSError:
                pass
        self.status_display.configure(
            text=f"Deleted {deleted_count} checkpoint file(s).",
            text_color="orange"
        )

    def _update_status(self, message):
        self._safe_after(lambda: self.status_display.configure(text=message, text_color="yellow"))

    def _handle_pipeline_telemetry(self, payload):
        if isinstance(payload, dict):
            self._safe_after(lambda: self._apply_model_telemetry(payload))
        elif isinstance(payload, str):
            self._update_status(payload)

    def _apply_model_telemetry(self, payload):
        ptype = payload.get("type")
        job_key = payload.get("job_key")
        if not job_key:
            return

        card = self.active_model_cards.get(job_key)
        if not card:
            m_type = payload.get("model_type", "Model")
            sys_mode = payload.get("system_mode", "SISO")
            ds_name = payload.get("dataset_name", "Dataset")
            card = self._create_model_card(job_key, m_type, sys_mode, ds_name, initial_status="● Training")

        k_folds = payload.get("k_folds", card.get("k_folds", 10))
        card["k_folds"] = k_folds

        if ptype == "generation":
            if card.get("state") != "training":
                card["state"] = "training"
                card["frame"].configure(border_color="#38BDF8")
                card["status_lbl"].configure(text="● Training", text_color="#38BDF8")
                card["progress_bar"].configure(progress_color=("#2563EB", "#3B82F6"))
                card["close_btn"].pack_forget()

            fold = payload.get("fold", 1)
            gen = payload.get("generation", 0)
            max_gen = payload.get("max_gen", 100)
            best_score = payload.get("best_score", 0.0)
            elapsed = payload.get("elapsed", 0.0)

            score_val = f"{best_score:.4f}" if isinstance(best_score, (int, float)) else str(best_score)
            telem_msg = f"Fold {fold}/{k_folds}  |  Gen {gen}/{max_gen}  |  Best MSE: {score_val}  |  Time: {elapsed:.1f}s"
            card["telemetry_lbl"].configure(text=telem_msg, text_color=("#334155", "#CBD5E1"))

            if max_gen > 0 and k_folds > 0:
                prog = max(0.0, min(1.0, ((fold - 1) + (gen / max_gen)) / k_folds))
                card["progress_bar"].set(prog)

        elif ptype == "fold_complete":
            fold = payload.get("fold", 1)
            val_mse = payload.get("val_mse", 0.0)
            val_r2 = payload.get("val_r2", 0.0)
            telem_msg = f"Fold {fold}/{k_folds} Finished  |  Val MSE: {val_mse:.4f}  |  Val R²: {val_r2:.4f}"
            card["telemetry_lbl"].configure(text=telem_msg, text_color=("#1E3A8A", "#93C5FD"))
            if k_folds > 0:
                card["progress_bar"].set(fold / k_folds)

        elif ptype == "job_complete":
            card["state"] = "completed"
            card["frame"].configure(border_color="#10B981")
            card["status_lbl"].configure(text="✓ Completed", text_color="#10B981")
            card["progress_bar"].configure(progress_color="#10B981")
            card["progress_bar"].set(1.0)
            best_mse = payload.get("best_mse", 0.0)
            best_r2 = payload.get("best_r2", 0.0)
            total_time = payload.get("total_time", 0.0)
            card["telemetry_lbl"].configure(
                text=f"All {k_folds} Folds Complete  |  Best MSE: {best_mse:.4f}  |  R²: {best_r2:.4f}  |  Total: {total_time:.1f}s",
                text_color="#10B981"
            )
            card["close_btn"].pack(side="left")
            self._update_monitor_placeholder()

            if self.auto_clear_cards_var.get():
                self.after(2000, lambda jk=job_key: self._dismiss_card(jk))

        elif ptype == "job_stopped":
            card["state"] = "stopped"
            card["frame"].configure(border_color="#F59E0B")
            card["status_lbl"].configure(text="■ Stopped", text_color="#F59E0B")
            card["progress_bar"].configure(progress_color="#F59E0B")
            fold = payload.get("fold", 1)
            card["telemetry_lbl"].configure(
                text=f"Stopped at Fold {fold}/{k_folds}  |  Checkpoints saved",
                text_color="#F59E0B"
            )
            card["close_btn"].pack(side="left")
            self._update_monitor_placeholder()

        elif ptype == "job_error":
            card["state"] = "error"
            card["frame"].configure(border_color="#EF4444")
            card["status_lbl"].configure(text="✕ Error", text_color="#EF4444")
            card["progress_bar"].configure(progress_color="#EF4444")
            err_msg = str(payload.get("error_msg", "Error"))
            card["telemetry_lbl"].configure(text=f"Error: {err_msg[:65]}", text_color="#EF4444")
            card["close_btn"].pack(side="left")
            self._update_monitor_placeholder()

    def _create_model_card(self, job_key, model_type, system_mode, dataset_name, initial_status="● Training"):
        if job_key in self.active_model_cards:
            return self.active_model_cards[job_key]

        is_queued = "Queued" in initial_status or "Waiting" in initial_status
        card_border = "#94A3B8" if is_queued else "#38BDF8"

        card_frame = ctk.CTkFrame(
            self.monitor_cards_container,
            fg_color=("#FFFFFF", "#1E293B"),
            border_width=1,
            border_color=card_border,
            corner_radius=8
        )
        card_frame.pack(fill="x", padx=4, pady=3)

        header_row = ctk.CTkFrame(card_frame, fg_color="transparent")
        header_row.pack(fill="x", padx=8, pady=(4, 2))

        model_badge = ctk.CTkLabel(
            header_row,
            text=f"{model_type} [{system_mode}]",
            font=("Arial", 12, "bold"),
            text_color=("#2563EB", "#60A5FA")
        )
        model_badge.pack(side="left", padx=(0, 6))

        dataset_lbl = ctk.CTkLabel(
            header_row,
            text=f"•  {dataset_name}",
            font=("Arial", 11),
            text_color=("#475569", "#94A3B8")
        )
        dataset_lbl.pack(side="left")

        status_container = ctk.CTkFrame(header_row, fg_color="transparent")
        status_container.pack(side="right")

        status_color = "#94A3B8" if is_queued else "#38BDF8"
        status_lbl = ctk.CTkLabel(
            status_container,
            text=initial_status,
            font=("Arial", 11, "bold"),
            text_color=status_color
        )
        status_lbl.pack(side="left", padx=(0, 6))

        close_btn = ctk.CTkButton(
            status_container,
            text="✕",
            width=20,
            height=20,
            corner_radius=10,
            fg_color=("#E2E8F0", "#334155"),
            hover_color=("#CBD5E1", "#475569"),
            text_color=("#475569", "#CBD5E1"),
            font=("Arial", 10, "bold"),
            command=lambda k=job_key: self._dismiss_card(k)
        )
        if is_queued:
            close_btn.pack(side="left")

        progress_row = ctk.CTkFrame(card_frame, fg_color="transparent")
        progress_row.pack(fill="x", padx=8, pady=(1, 5))

        pbar = ctk.CTkProgressBar(
            progress_row,
            height=6,
            corner_radius=3,
            progress_color=("#2563EB", "#3B82F6"),
            fg_color=("#CBD5E1", "#334155")
        )
        pbar.pack(fill="x", pady=(1, 2))
        pbar.set(0.0)

        initial_telem = "Waiting in queue..." if is_queued else "Initializing optimization..."
        telem_lbl = ctk.CTkLabel(
            progress_row,
            text=initial_telem,
            font=("Arial", 10),
            text_color=("#334155", "#CBD5E1"),
            anchor="w"
        )
        telem_lbl.pack(fill="x")

        card_data = {
            "frame": card_frame,
            "model_lbl": model_badge,
            "dataset_lbl": dataset_lbl,
            "status_lbl": status_lbl,
            "progress_bar": pbar,
            "telemetry_lbl": telem_lbl,
            "close_btn": close_btn,
            "state": "queued" if is_queued else "training",
            "model_type": model_type,
            "system_mode": system_mode,
            "dataset_name": dataset_name,
            "k_folds": 10
        }
        self.active_model_cards[job_key] = card_data
        self._update_monitor_placeholder()
        return card_data

    def _dismiss_card(self, job_key):
        card = self.active_model_cards.pop(job_key, None)
        if card:
            try:
                card["frame"].destroy()
            except Exception:
                pass
            if card.get("state") == "queued":
                items = []
                while not self.training_queue.empty():
                    try:
                        item = self.training_queue.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        # batch_dirs is None for single-model jobs, so fall back to an empty dict before calling .values()
                        b_dirs = item.get("batch_dirs") or {}
                        if item.get("output_directory") == job_key:
                            match = True
                        elif job_key in b_dirs.values():
                            # If this is a 3-model batch and the user only closed one model card,
                            # remove just that model from batch_dirs and keep the batch job if others remain
                            for m_k, m_path in list(b_dirs.items()):
                                if m_path == job_key:
                                    del b_dirs[m_k]
                            match = (len(b_dirs) == 0)
                        else:
                            match = False

                        if not match:
                            items.append(item)
                    finally:
                        self.training_queue.task_done()
                for itm in items:
                    self.training_queue.put(itm)
                self._update_queue_ui()
        self._update_monitor_placeholder()

    def _clear_finished_cards(self):
        to_delete = [
            k for k, card in self.active_model_cards.items()
            if card.get("state") in ("completed", "stopped", "error")
        ]
        for k in to_delete:
            self._dismiss_card(k)

    def _update_monitor_placeholder(self):
        if not self.active_model_cards:
            self.monitor_placeholder.pack(pady=35)
        else:
            self.monitor_placeholder.pack_forget()

    def _on_auto_clear_toggled(self):
        try:
            is_enabled = bool(self.auto_clear_cards_var.get())
            self.active_hyperparameters["auto_clear_monitor"] = is_enabled
            if is_enabled:
                self._clear_finished_cards()
        except Exception:
            pass

    def _sync_gui_hyperparameters(self):
        self.active_hyperparameters["model_type"] = self.model_type_selector.get()
        self.active_hyperparameters["system_mode"] = self.system_mode_selector.get()
        for param_key, input_widget in self.hyperparameter_inputs.items():
            if param_key in ("model_type", "system_mode"):
                continue
            raw_value = input_widget.get().strip()
            if param_key in ("mutation_type", "selection_type"):
                self.active_hyperparameters[param_key] = raw_value
            elif param_key == "mlp_hidden_nodes":
                cleaned = raw_value.strip("[]() ")
                nodes = [int(x.strip()) for x in cleaned.split(",") if x.strip()]
                if not nodes or any(n <= 0 for n in nodes):
                    raise ValueError("Hidden neurons must be positive integers.")
                self.active_hyperparameters[param_key] = nodes
            elif raw_value.startswith("[") and raw_value.endswith("]"):
                self.active_hyperparameters[param_key] = json.loads(raw_value)
            else:
                try:
                    if "." not in raw_value and "e" not in raw_value.lower():
                        self.active_hyperparameters[param_key] = int(raw_value)
                    else:
                        self.active_hyperparameters[param_key] = float(raw_value)
                except ValueError:
                    self.active_hyperparameters[param_key] = float(raw_value)
        disk_cfg = dict(self.active_hyperparameters)
        # If "ALL" was chosen in GUI, write "ARMAX" to config.json so CLI runs don't hit illegal ':' path errors
        if "ALL" in str(disk_cfg.get("model_type", "")).upper():
            disk_cfg["model_type"] = "ARMAX"
        tmp_cfg = "config.json.tmp"
        with open(tmp_cfg, "w", encoding="utf-8") as config_file:
            json.dump(disk_cfg, config_file, indent=4)
        atomic_replace_retry(tmp_cfg, "config.json")

    def _prompt_multiple_model_files(self, action_label="Select Model(s)"):
        os.makedirs("results", exist_ok=True)
        dialog = ModelSelectionDialog(self, action_label=action_label)
        self.wait_window(dialog)
        return dialog.result or []

    def _dispatch_benchmark_suite(self):
        try:
            self._sync_gui_hyperparameters()
        except ValueError:
            self._set_status("Error: Parameters must be numeric.", "red")
            return

        timestamp_sig = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = os.path.join("results", f"results_benchmarks_{timestamp_sig}")

        self._set_status("Starting benchmarks...", "yellow")
        self._set_buttons_state("disabled")

        run_stop_event = threading.Event()
        self.stop_event = run_stop_event
        with self.queue_lock:
            self.active_stop_events.append(run_stop_event)

        threading.Thread(target=self._execute_benchmark_suite, args=(output_directory, run_stop_event)).start()

    def _execute_benchmark_suite(self, output_dir, run_stop_event=None):
        try:
            run_full_benchmark_suite(
                hyperparameters=self.active_hyperparameters, output_dir=output_dir,
                progress_callback=self._update_status, stop_event=run_stop_event
            )
            if run_stop_event and run_stop_event.is_set():
                self._set_status("Benchmarks stopped by user.", "orange")
            else:
                self._set_status(f"Benchmarks complete. Results in {output_dir}", "cyan")
        except Exception as error:
            self._set_status(f"Error: {str(error)}", "red")
        finally:
            with self.queue_lock:
                if run_stop_event and run_stop_event in self.active_stop_events:
                    self.active_stop_events.remove(run_stop_event)
            self._set_buttons_state("normal")

    def _dispatch_model_evaluation(self):
        selected_files = self._prompt_multiple_model_files(action_label="Evaluate / Compare Model(s)")
        if not selected_files:
            return

        dropdown_dataset = self.dataset_selector.get()
        timestamp_sig = datetime.now().strftime("%Y%m%d_%H%M%S")

        source_datasets = set()
        for fp in selected_files:
            try:
                payload = _parse_serialized_model(fp)
                source_datasets.add(payload.get("dataset", dropdown_dataset))
            except Exception:
                pass

        if source_datasets == {dropdown_dataset}:
            use_source_data = True
        elif source_datasets:
            src_list_str = ", ".join(sorted(source_datasets))
            use_source_data = messagebox.askyesno(
                "Select Evaluation Mode (Source Check vs. Cross-Dataset Inference)",
                f"Selected model(s) were trained on: [{src_list_str}]\n"
                f"Currently selected dropdown dataset: [{dropdown_dataset}]\n\n"
                f"• Click YES to evaluate each model on its original training dataset ({src_list_str}) [Source Check].\n"
                f"• Click NO to run Cross-Dataset Inference on the currently selected dropdown dataset ({dropdown_dataset})."
            )
        else:
            use_source_data = False

        self._set_buttons_state("disabled")

        if use_source_data:
            self._set_status(f"Checking {len(selected_files)} model(s) on their source dataset(s)...", "yellow")
            threading.Thread(
                target=self._execute_source_dataset_check,
                args=(selected_files, timestamp_sig)
            ).start()
        elif len(selected_files) == 1:
            target_filepath = selected_files[0]
            u_signal, y_signal = self.loaded_datasets[dropdown_dataset]
            model_stem = os.path.splitext(os.path.basename(target_filepath))[0]
            output_directory = os.path.join("results", f"Inference_{model_stem}_on_{dropdown_dataset}_{timestamp_sig}")
            self._set_status(f"Running cross-dataset inference with {model_stem} on {dropdown_dataset}...", "yellow")
            threading.Thread(
                target=self._execute_inference_pipeline,
                args=(target_filepath, u_signal, y_signal, dropdown_dataset, output_directory)
            ).start()
        else:
            u_signal, y_signal = self.loaded_datasets[dropdown_dataset]
            output_directory = os.path.join("results", f"Comparison_{len(selected_files)}Models_{dropdown_dataset}_{timestamp_sig}")
            self._set_status(f"Comparing {len(selected_files)} models on {dropdown_dataset}...", "yellow")
            threading.Thread(
                target=self._execute_comparison_pipeline,
                args=(selected_files, u_signal, y_signal, dropdown_dataset, output_directory)
            ).start()

    def _execute_inference_pipeline(self, artifact_filepath, u_signal, y_signal, dataset_name, output_dir):
        try:
            validation_scores, model_payload = execute_inference_pipeline(
                artifact_filepath, u_signal, y_signal, dataset_name,
                output_dir=output_dir, progress_callback=self._update_status
            )
            architecture = model_payload.get("structure", {})
            loaded_model_type = architecture.get("model_type")
            loaded_sys_mode = architecture.get("system_mode")

            def _update_ui():
                if loaded_model_type in ("ARMAX", "MLP", "SVR"):
                    self.model_type_selector.set(loaded_model_type)
                    self._on_model_type_changed(loaded_model_type)

                if loaded_sys_mode in ("SISO", "MISO", "SIMO", "MIMO"):
                    self.system_mode_selector.set(loaded_sys_mode)

                for param_key in ("input_col_idx", "output_col_idx", "na", "nb", "nc", "nk", "mlp_hidden_nodes"):
                    if param_key in architecture and param_key in self.hyperparameter_inputs:
                        self.hyperparameter_inputs[param_key].delete(0, "end")
                        self.hyperparameter_inputs[param_key].insert(0, str(architecture[param_key]))
            self._safe_after(_update_ui)

            summary_msg = (
                f"Inference complete! MSE: {validation_scores['MSE']:.4f} | "
                f"RMSE: {validation_scores['RMSE']:.4f} | R²: {validation_scores['R-Squared']:.4f} "
                f"-> {output_dir}"
            )
            self._set_status(summary_msg, "cyan")
        except Exception as error:
            self._set_status(f"Inference Error: {str(error)}", "red")
        finally:
            self._set_buttons_state("normal")

    def _execute_comparison_pipeline(self, model_filepaths, u_signal, y_signal, dataset_name, output_dir):
        try:
            ranked_models = execute_model_comparison_pipeline(
                model_filepaths, u_signal, y_signal, dataset_name,
                output_dir=output_dir, progress_callback=self._update_status
            )
            winner = ranked_models[0]
            self._set_status(
                f"Comparison complete ({len(ranked_models)} models)! Winner: {winner['label']} (MSE: {winner['scores']['MSE']:.4f}) -> {output_dir}",
                "cyan"
            )
        except Exception as error:
            self._set_status(f"Comparison Error: {str(error)}", "red")
        finally:
            self._set_buttons_state("normal")

    def _execute_source_dataset_check(self, model_filepaths, timestamp_sig):
        try:
            summary_msg = execute_source_dataset_check(
                model_filepaths, self.loaded_datasets,
                base_results_dir="results", timestamp_sig=timestamp_sig,
                progress_callback=self._update_status
            )
            self._set_status(f"Source Check Complete! {summary_msg}", "cyan")
        except Exception as error:
            self._set_status(f"Source Check Error: {str(error)}", "red")
        finally:
            self._set_buttons_state("normal")

    def _dispatch_summary_export(self):
        self._set_buttons_state("disabled")
        self._set_status("Exporting summary sheet...", "yellow")

        def _do_export():
            target_dir = "results"
            try:
                exported_path = generate_results_summary_sheet(
                    base_results_dir=target_dir, loaded_datasets=self.loaded_datasets
                )
                self._set_status(f"Summary sheet exported to: {exported_path}", "cyan")
            except RuntimeError:
                def _ask_parent():
                    chosen_dir = filedialog.askdirectory(
                        title="Select Parent Folder Containing Your Experiment Folders"
                    )
                    if not chosen_dir:
                        self._set_buttons_state("normal")
                        return
                    try:
                        exported_path = generate_results_summary_sheet(
                            base_results_dir=chosen_dir, loaded_datasets=self.loaded_datasets
                        )
                        self._set_status(f"Summary sheet exported to: {exported_path}", "cyan")
                    except Exception as err:
                        self._set_status(f"Export Error: {str(err)}", "red")
                    finally:
                        self._set_buttons_state("normal")
                self._safe_after(_ask_parent)
                return
            except Exception as error:
                self._set_status(f"Export Error: {str(error)}", "red")
            finally:
                self._set_buttons_state("normal")

        threading.Thread(target=_do_export).start()

    def _on_queue_mode_toggled(self):
        try:
            self.is_queue_training_enabled = bool(self.queue_training_var.get())
        except Exception:
            pass
        self._update_queue_ui()

    def _update_queue_ui(self):
        def _apply():
            try:
                pending_count = self.training_queue.qsize()
                is_queue_mode = self.is_queue_training_enabled

                if is_queue_mode:
                    if self.queue_worker_busy:
                        running_desc = ""
                        if self.current_queue_job:
                            running_desc = f" ({self.current_queue_job['model_type']} on {self.current_queue_job['selected_dataset']})"
                        self.queue_status_label.configure(
                            text=f"Queue Active: 1 Running{running_desc} | {pending_count} Pending",
                            text_color="#38BDF8"
                        )
                    else:
                        self.queue_status_label.configure(
                            text="Queue Mode: Idle (0 Pending)",
                            text_color="#34D399"
                        )
                else:
                    if pending_count > 0 or self.queue_worker_busy:
                        self.queue_status_label.configure(
                            text=f"Queue Draining: {pending_count} Pending Remaining",
                            text_color="#FBBF24"
                        )
                    else:
                        self.queue_status_label.configure(
                            text="Mode: Parallel Training (Concurrent)",
                            text_color="#94A3B8"
                        )

                if pending_count > 0:
                    self.clear_queue_btn.pack(side="left", padx=8)
                else:
                    self.clear_queue_btn.pack_forget()
            except Exception:
                pass
        self._safe_after(_apply)

    def _clear_pending_queue(self):
        cleared = 0
        while not self.training_queue.empty():
            try:
                self.training_queue.get_nowait()
                self.training_queue.task_done()
                cleared += 1
            except queue.Empty:
                break
        self._update_queue_ui()

        def _remove_queued_cards():
            for key, card in list(self.active_model_cards.items()):
                if card.get("state") == "queued":
                    try:
                        card["frame"].destroy()
                    except Exception:
                        pass
                    del self.active_model_cards[key]
            self._update_monitor_placeholder()

        self._safe_after(_remove_queued_cards)

        if cleared > 0:
            self._set_status(f"Cleared {cleared} pending job(s) from the training queue.", "orange")

    def _ensure_queue_worker(self):
        with self.queue_lock:
            if self.queue_worker_thread is None or not self.queue_worker_thread.is_alive():
                self.queue_worker_thread = threading.Thread(
                    target=self._run_training_queue_worker,
                    daemon=True
                )
                self.queue_worker_thread.start()

    def _run_training_queue_worker(self):
        while True:
            try:
                job_item = self.training_queue.get(timeout=1.0)
            except queue.Empty:
                with self.queue_lock:
                    self.queue_worker_busy = False
                    self.current_queue_job = None
                    self._update_queue_ui()
                    return

            # If parallel runs are currently active, wait for them to finish before starting queued job
            while True:
                with self.queue_lock:
                    if len(self.active_stop_events) == 0:
                        break
                if self.stop_event.is_set():
                    self.training_queue.task_done()
                    with self.queue_lock:
                        self.queue_worker_busy = False
                        self.current_queue_job = None
                        self._update_queue_ui()
                    return
                time.sleep(0.5)

            run_stop_event = threading.Event()
            self.stop_event = run_stop_event
            with self.queue_lock:
                self.queue_worker_busy = True
                self.current_queue_job = job_item
                self.active_stop_events.append(run_stop_event)
                self.active_runs_count = 1
                self._update_queue_ui()

            # In queue mode, single sequential job gets full worker pool
            job_item["hparams"]["active_jobs"] = 1

            u_signal = job_item["u_signal"]
            y_signal = job_item["y_signal"]
            dataset_name = job_item["selected_dataset"]
            system_mode = job_item["system_mode"]
            model_type = job_item["model_type"]
            output_dir = job_item["output_directory"]
            action_desc = job_item["action_desc"]
            is_batch = job_item["is_batch"]
            hparams = job_item["hparams"]
            batch_dirs = job_item.get("batch_dirs")

            if not is_batch and output_dir is None:
                timestamp_sig = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_id = uuid.uuid4().hex[:6]
                base_dir = "results"
                os.makedirs(base_dir, exist_ok=True)
                folder_name = f"{model_type}_{system_mode}_{dataset_name}_{timestamp_sig}_{run_id}"
                output_dir = os.path.join(base_dir, folder_name)
                os.makedirs(output_dir, exist_ok=True)

            # Update card status in monitor
            if is_batch and batch_dirs:
                first_dir = batch_dirs.get("ARMAX")
                if first_dir:
                    def _mark_first(k=first_dir):
                        c = self.active_model_cards.get(k)
                        if c:
                            c["state"] = "training"
                            c["status_lbl"].configure(text="● Training", text_color="#38BDF8")
                            c["progress_bar"].configure(progress_color=("#2563EB", "#3B82F6"))
                            c["close_btn"].pack_forget()
                    self._safe_after(_mark_first)
            elif output_dir:
                def _mark_running(k=output_dir):
                    c = self.active_model_cards.get(k)
                    if c:
                        c["state"] = "training"
                        c["status_lbl"].configure(text="● Training", text_color="#38BDF8")
                        c["progress_bar"].configure(progress_color=("#2563EB", "#3B82F6"))
                        c["close_btn"].pack_forget()
                self._safe_after(_mark_running)

            remaining = self.training_queue.qsize()

            try:
                if is_batch:
                    self._set_status(
                        f"[Queue Job: Remaining {remaining}] Batch Training: ARMAX + MLP + SVR ({system_mode}) on {dataset_name}...",
                        "yellow"
                    )
                    self._execute_batch_training_pipeline(
                        u_signal, y_signal, dataset_name, system_mode, hparams,
                        run_stop_event=run_stop_event, batch_dirs=batch_dirs
                    )
                else:
                    self._set_status(
                        f"[Queue Job: Remaining {remaining}] {action_desc} {model_type} ({system_mode}) training on {dataset_name}...",
                        "yellow"
                    )
                    self._execute_training_pipeline(
                        u_signal, y_signal, dataset_name, hparams, output_dir, run_stop_event=run_stop_event
                    )
            except Exception as err:
                self._set_status(f"Queue job execution error: {err}", "red")
            finally:
                self.training_queue.task_done()
                with self.queue_lock:
                    self.current_queue_job = None
                    self._update_queue_ui()

            if self.training_queue.empty() and not run_stop_event.is_set():
                if self.is_queue_training_enabled:
                    self._set_status("All queued training jobs completed successfully!", "cyan")

    def _dispatch_training_pipeline(self):
        if self.auto_clear_cards_var.get():
            self._clear_finished_cards()

        try:
            self._sync_gui_hyperparameters()
        except ValueError as err:
            self._set_status(f"Error: {err}", "red")
            return

        selected_dataset = self.dataset_selector.get()
        u_signal, y_signal = self.loaded_datasets[selected_dataset]
        model_type = self.model_type_selector.get()
        system_mode = self.system_mode_selector.get()

        hparams = copy.deepcopy(self.active_hyperparameters)

        resume_target = self.resume_dir
        self.resume_dir = None
        if resume_target:
            action_desc = "Resuming"
            output_directory = resume_target
            # When resuming a specific folder, turn off batch and parse the target model from folder name
            is_batch = False
            if "ALL" in model_type:
                folder_base = os.path.basename(os.path.normpath(resume_target)).upper()
                model_type = "SVR" if "SVR" in folder_base else ("MLP" if "MLP" in folder_base else "ARMAX")
                hparams["model_type"] = model_type
        else:
            action_desc = "Starting"
            output_directory = None
            is_batch = ("ALL" in model_type)

        if self.stop_event.is_set():
            self.stop_event = threading.Event()

        try:
            self.is_queue_training_enabled = bool(self.queue_training_var.get())
        except Exception:
            pass
        is_queue_mode = self.is_queue_training_enabled

        # Precompute output directories and create monitor cards
        timestamp_sig = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dirs = {}
        if is_batch:
            batch_id = uuid.uuid4().hex[:6]
            base_dir = "results"
            os.makedirs(base_dir, exist_ok=True)
            for m_type in ["ARMAX", "MLP", "SVR"]:
                folder_name = f"{m_type}_{system_mode}_{selected_dataset}_{timestamp_sig}_{batch_id}"
                b_dir = os.path.join(base_dir, folder_name)
                os.makedirs(b_dir, exist_ok=True)
                batch_dirs[m_type] = b_dir
                init_stat = "⏱ Queued (Waiting)" if is_queue_mode else ("● Training" if m_type == "ARMAX" else "⏱ Waiting in Batch")
                self._create_model_card(b_dir, m_type, system_mode, selected_dataset, initial_status=init_stat)
        else:
            if output_directory is None:
                run_id = uuid.uuid4().hex[:6]
                base_dir = "results"
                os.makedirs(base_dir, exist_ok=True)
                folder_name = f"{model_type}_{system_mode}_{selected_dataset}_{timestamp_sig}_{run_id}"
                output_directory = os.path.join(base_dir, folder_name)
                os.makedirs(output_directory, exist_ok=True)
            init_stat = "⏱ Queued (Waiting)" if is_queue_mode else "● Training"
            self._create_model_card(output_directory, model_type, system_mode, selected_dataset, initial_status=init_stat)

        if is_queue_mode:
            job_item = {
                "job_id": uuid.uuid4().hex[:6],
                "model_type": model_type,
                "system_mode": system_mode,
                "selected_dataset": selected_dataset,
                "u_signal": u_signal,
                "y_signal": y_signal,
                "hparams": hparams,
                "output_directory": output_directory,
                "batch_dirs": batch_dirs if is_batch else None,
                "action_desc": action_desc,
                "is_batch": is_batch,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }

            self.training_queue.put(job_item)
            pending_count = self.training_queue.qsize()

            with self.queue_lock:
                is_busy = self.queue_worker_busy or len(self.active_stop_events) > 0
            if is_busy:
                self._set_status(
                    f"[Queue: Job Queued] Added {model_type} ({system_mode}) on {selected_dataset} to queue. "
                    f"({pending_count} pending in queue)",
                    "cyan"
                )
            else:
                self._set_status(
                    f"[Queue] Initializing training for {model_type} ({system_mode}) on {selected_dataset}...",
                    "yellow"
                )

            self._update_queue_ui()
            self._ensure_queue_worker()

        else:
            run_stop_event = threading.Event()
            self.stop_event = run_stop_event
            with self.queue_lock:
                self.active_stop_events.append(run_stop_event)
                self.active_runs_count += 1
                cur_runs = self.active_runs_count

            hparams["active_jobs"] = max(1, cur_runs)

            if is_batch:
                self._set_status(
                    f"[Active Jobs: {self.active_runs_count}] Queued Batch Training: ARMAX + MLP + SVR ({system_mode}) on {selected_dataset}...",
                    "yellow"
                )
                threading.Thread(
                    target=self._execute_batch_training_pipeline,
                    args=(u_signal, y_signal, selected_dataset, system_mode, hparams, run_stop_event, batch_dirs),
                    daemon=True
                ).start()
                return

            self._set_status(
                f"[Active Jobs: {self.active_runs_count}] {action_desc} {model_type} ({system_mode}) training on {selected_dataset}...",
                "yellow"
            )
            threading.Thread(
                target=self._execute_training_pipeline,
                args=(u_signal, y_signal, selected_dataset, hparams, output_directory, run_stop_event),
                daemon=True
            ).start()

    def _execute_batch_training_pipeline(self, u_signal, y_signal, dataset_name, system_mode, base_hyperparameters, run_stop_event=None, batch_dirs=None):
        active_stop = run_stop_event or self.stop_event
        models_to_train = ["ARMAX", "MLP", "SVR"]
        completed_model_files = []
        timestamp_sig = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_id = uuid.uuid4().hex[:6]

        if not batch_dirs:
            batch_dirs = {}
            for m in models_to_train:
                f_name = f"{m}_{system_mode}_{dataset_name}_{timestamp_sig}_{batch_id}"
                d_path = os.path.join("results", f_name)
                os.makedirs(d_path, exist_ok=True)
                batch_dirs[m] = d_path

        current_output_dir = None
        try:
            for idx, m_type in enumerate(models_to_train, 1):
                if active_stop and active_stop.is_set():
                    self._set_status("Batch training stopped by user.", "orange")
                    def _cancel_remaining():
                        for rem_m in models_to_train[idx-1:]:
                            rem_dir = batch_dirs.get(rem_m)
                            c = self.active_model_cards.get(rem_dir)
                            if c and c.get("state") not in ("completed", "stopped"):
                                c["state"] = "stopped"
                                c["frame"].configure(border_color="#F59E0B")
                                c["status_lbl"].configure(text="■ Cancelled", text_color="#F59E0B")
                                c["telemetry_lbl"].configure(text="Batch cancelled by user", text_color="#F59E0B")
                                c["close_btn"].pack(side="left")
                        self._update_monitor_placeholder()
                    self._safe_after(_cancel_remaining)
                    return

                output_directory = batch_dirs.get(m_type)
                # If the user clicked 'X' on this model's waiting card to cancel it, skip training it
                if not output_directory or output_directory not in self.active_model_cards:
                    continue
                current_output_dir = output_directory

                self._set_status(
                    f"[Batch {idx}/{len(models_to_train)}] Optimizing {m_type} ({system_mode}) on {dataset_name}...",
                    "yellow"
                )

                def _mark_running(k=output_directory):
                    c = self.active_model_cards.get(k)
                    if c:
                        c["state"] = "training"
                        c["frame"].configure(border_color="#38BDF8")
                        c["status_lbl"].configure(text="● Training", text_color="#38BDF8")
                        c["progress_bar"].configure(progress_color=("#2563EB", "#3B82F6"))
                        c["close_btn"].pack_forget()
                self._safe_after(_mark_running)

                hparams = copy.deepcopy(base_hyperparameters)
                hparams["model_type"] = m_type
                hparams["system_mode"] = system_mode

                execute_kfold_training_pipeline(
                    u_signal, y_signal, dataset_name, hparams, output_directory,
                    progress_callback=self._handle_pipeline_telemetry, stop_event=active_stop
                )

                pkl_files = glob.glob(os.path.join(output_directory, "*_model.pkl"))
                if pkl_files:
                    completed_model_files.append(pkl_files[0])

            if len(completed_model_files) >= 2 and not (active_stop and active_stop.is_set()):
                self._set_status(f"Generating head-to-head comparison for {len(completed_model_files)} models...", "yellow")
                comp_dir = os.path.join("results", f"BatchComparison_{len(completed_model_files)}Models_{dataset_name}_{timestamp_sig}")
                ranked = execute_model_comparison_pipeline(
                    completed_model_files, u_signal, y_signal, dataset_name,
                    output_dir=comp_dir, progress_callback=self._update_status
                )
                winner = ranked[0]
                self._set_status(
                    f"Batch complete! Winner: {winner['label']} (MSE: {winner['scores']['MSE']:.4f}) -> {comp_dir}",
                    "cyan"
                )
            else:
                self._set_status("Batch training complete!", "cyan")
        except Exception as error:
            self._set_status(f"Batch Error: {error}", "red")
            if current_output_dir:
                self._handle_pipeline_telemetry({
                    "type": "job_error",
                    "job_key": current_output_dir,
                    "error_msg": str(error)
                })
            print(f"Batch error: {error}\n{traceback.format_exc()}")
        finally:
            with self.queue_lock:
                if run_stop_event and run_stop_event in self.active_stop_events:
                    self.active_stop_events.remove(run_stop_event)
                self.active_runs_count = max(0, self.active_runs_count - 1)

    def _execute_training_pipeline(self, u_signal, y_signal, dataset_name, hyperparameters, output_dir, run_stop_event=None):
        active_stop = run_stop_event or self.stop_event
        try:
            execute_kfold_training_pipeline(
                u_signal, y_signal, dataset_name, hyperparameters, output_dir,
                progress_callback=self._handle_pipeline_telemetry, stop_event=active_stop
            )
            if not active_stop.is_set():
                self._set_status(f"Training complete! Results in {output_dir}", "cyan")
            else:
                self._set_status("Training stopped. Checkpoints saved.", "orange")
        except Exception as error:
            self._set_status(f"Training error: {error}", "red")
            self._handle_pipeline_telemetry({
                "type": "job_error",
                "job_key": output_dir,
                "error_msg": str(error)
            })
            try:
                with open(os.path.join(output_dir, "crash_log.txt"), "w", encoding="utf-8") as f:
                    f.write(f"CRASH OCCURRED:\n{str(error)}\n\nTRACEBACK:\n{traceback.format_exc()}")
            except Exception:
                pass
            print(f"Thread error: {error}")
        finally:
            with self.queue_lock:
                if run_stop_event and run_stop_event in self.active_stop_events:
                    self.active_stop_events.remove(run_stop_event)
                self.active_runs_count = max(0, self.active_runs_count - 1)

    def _terminate_application(self):
        self.destroy()
        os._exit(0)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = EvolutionarySystemIDApp()
    app.mainloop()