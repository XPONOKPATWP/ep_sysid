"""
system_id_gui.py

GUI for configuring ARMAX parameters, managing datasets,
and running optimization and inference tasks.
"""

import customtkinter as ctk
from tkinter import filedialog
import threading
import os
import glob
import json
from datetime import datetime
import multiprocessing

from optimization_benchmarks import run_full_benchmark_suite
from siso_data_parser import load_cached_uci_benchmarks, fetch_and_parse_uci_dataset, parse_local_csv_file
from training_pipeline import execute_kfold_training_pipeline, execute_inference_pipeline
from config_handler import load_hyperparameters, get_parameter_label

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class EvolutionarySystemIDApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("ARMAX System Identification Tool")
        self.geometry("760x680")

        self.loaded_datasets = load_cached_uci_benchmarks()
        self.dataset_registry_names = list(self.loaded_datasets.keys())
        self.active_hyperparameters = load_hyperparameters("config.json")

        self._construct_user_interface()

    def _construct_user_interface(self):
        # --- Section 1: Data Management Panel ---
        self.data_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.data_panel.pack(pady=(15, 5))

        ctk.CTkLabel(self.data_panel, text="Dataset:", font=("Arial", 14, "bold")).grid(row=0, column=0, padx=10, pady=5)
        self.dataset_selector = ctk.CTkOptionMenu(self.data_panel, values=self.dataset_registry_names, width=200)
        self.dataset_selector.grid(row=0, column=1, columnspan=2, padx=10, pady=5)

        ctk.CTkLabel(self.data_panel, text="Fetch UCI ID:").grid(row=1, column=0, pady=8, padx=10)
        self.uci_input_field = ctk.CTkEntry(self.data_panel, width=90, placeholder_text="e.g. 45")
        self.uci_input_field.grid(row=1, column=1, pady=8, padx=5)
        self.download_uci_btn = ctk.CTkButton(self.data_panel, text="Download", width=100,
                                              command=self._dispatch_uci_download_thread)
        self.download_uci_btn.grid(row=1, column=2, pady=8, padx=5)

        ctk.CTkLabel(self.data_panel, text="Or load custom file:").grid(row=2, column=0, pady=5, padx=10)
        self.load_csv_btn = ctk.CTkButton(self.data_panel, text="Load Local CSV", width=200, fg_color="#3B8ED0",
                                          command=self._prompt_local_csv_load)
        self.load_csv_btn.grid(row=2, column=1, columnspan=2, pady=5, padx=10)

        # --- Section 2: Algorithm Configuration Panel ---
        self.config_panel = ctk.CTkFrame(self)
        self.config_panel.pack(pady=15, padx=20, fill="x")
        self.config_panel.grid_columnconfigure((0, 2), weight=1)

        self.hyperparameter_inputs = {}
        row_idx, col_idx = 0, 0

        # Dynamically generate input fields based on the config schema
        for key, default_val in self.active_hyperparameters.items():
            if key in ("checkpoint_file", "mutation_mu"):
                continue

            param_label = ctk.CTkLabel(self.config_panel, text=f"{get_parameter_label(key)}:")
            param_label.grid(row=row_idx, column=col_idx, padx=10, pady=6, sticky="e")

            input_box = ctk.CTkEntry(self.config_panel, width=80)
            input_box.insert(0, str(default_val))
            input_box.grid(row=row_idx, column=col_idx + 1, padx=10, pady=6, sticky="w")
            self.hyperparameter_inputs[key] = input_box

            col_idx += 2
            if col_idx > 2:
                col_idx = 0
                row_idx += 1

        # --- Section 3: Execution Controls ---
        self.execution_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.execution_panel.pack(pady=10)

        ctk.CTkButton(self.execution_panel, text="Clear Checkpoints", fg_color="orange", hover_color="darkorange",
                      command=self._clear_checkpoints).grid(row=0, column=0, padx=6)

        self.benchmark_btn = ctk.CTkButton(self.execution_panel, text="Run Benchmarks", fg_color="#1f6aa5",
                                           command=self._dispatch_benchmark_suite)
        self.benchmark_btn.grid(row=0, column=1, padx=6)

        self.train_btn = ctk.CTkButton(self.execution_panel, text="Run Optimization", fg_color="green",
                                       hover_color="darkgreen", command=self._dispatch_training_pipeline)
        self.train_btn.grid(row=0, column=2, padx=6)

        self.inference_btn = ctk.CTkButton(self.execution_panel, text="Load Model & Predict", fg_color="#6A4C93",
                                           hover_color="#523A73", command=self._dispatch_inference_pipeline)
        self.inference_btn.grid(row=0, column=3, padx=6)

        ctk.CTkButton(self.execution_panel, text="Exit", fg_color="red", hover_color="darkred",
                      command=self._terminate_application).grid(row=0, column=4, padx=6)

        self.status_display = ctk.CTkLabel(self, text="Ready.", font=("Arial", 12),
                                           wraplength=680)
        self.status_display.pack(pady=10)


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

        # Offload network requests to prevent UI freezing
        threading.Thread(target=self._execute_uci_download, args=(repository_id,)).start()

    def _execute_uci_download(self, repository_id):
        dataset_name, u_signal, y_signal, error_log = fetch_and_parse_uci_dataset(repository_id)
        if dataset_name:
            self._register_parsed_dataset(dataset_name, u_signal, y_signal,
                                          f"Loaded dataset: '{dataset_name}'.")
        else:
            self.status_display.configure(text=f"Download failed: {error_log}",
                                          text_color="red")
        self.download_uci_btn.configure(state="normal")

    def _prompt_local_csv_load(self):
        target_filepath = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not target_filepath:
            return

        dataset_name, u_signal, y_signal, error_log = parse_local_csv_file(target_filepath)
        if dataset_name:
            self._register_parsed_dataset(dataset_name, u_signal, y_signal,
                                          f"Loaded CSV '{dataset_name}' ({len(u_signal)} samples).")
        else:
            self.status_display.configure(text=f"CSV Error: {error_log}", text_color="red")

    def _clear_checkpoints(self):
        binary_checkpoints = glob.glob("checkpoint_*.pkl")
        for file in binary_checkpoints:
            os.remove(file)
        self.status_display.configure(
            text=f"Deleted {len(binary_checkpoints)} checkpoint files.",
            text_color="orange")

    def _update_status(self, message):
        self.status_display.configure(text=message, text_color="yellow")

    def _sync_gui_hyperparameters(self):
        for param_key, input_widget in self.hyperparameter_inputs.items():
            raw_value = input_widget.get().strip()
            self.active_hyperparameters[param_key] = float(raw_value) if '.' in raw_value else int(raw_value)

        with open("config.json", "w") as config_file:
            json.dump(self.active_hyperparameters, config_file, indent=4)

    def _dispatch_benchmark_suite(self):
        try:
            self._sync_gui_hyperparameters()
        except ValueError:
            self.status_display.configure(text="Error: Parameters must be numbers.", text_color="red")
            return

        timestamp_sig = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = f"results_benchmarks_{timestamp_sig}"

        self.status_display.configure(text="Starting benchmarks...", text_color="yellow")
        self.benchmark_btn.configure(state="disabled")
        threading.Thread(target=self._execute_benchmark_suite, args=(output_directory,)).start()

    def _execute_benchmark_suite(self, output_dir):
        try:
            run_full_benchmark_suite(hyperparameters=self.active_hyperparameters, output_dir=output_dir,
                                     progress_callback=self._update_status)
            self.status_display.configure(text=f"Benchmarks complete. Results in {output_dir}",
                                          text_color="cyan")
        except Exception as error:
            self.status_display.configure(text=f"Error: {str(error)}", text_color="red")
        finally:
            self.benchmark_btn.configure(state="normal")

    def _dispatch_inference_pipeline(self):
        target_filepath = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[
                ("Model Files", "*.pkl *.json"),
                ("Binary PKL", "*.pkl"),
                ("JSON", "*.json"),
                ("All Files", "*.*")
            ]
        )
        if not target_filepath:
            return

        selected_dataset = self.dataset_selector.get()
        u_signal, y_signal = self.loaded_datasets[selected_dataset]

        timestamp_sig = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = f"results_inference_{selected_dataset}_{timestamp_sig}"

        self.status_display.configure(text=f"Starting inference for {selected_dataset}...", text_color="yellow")
        self.inference_btn.configure(state="disabled")

        threading.Thread(
            target=self._execute_inference_pipeline,
            args=(target_filepath, u_signal, y_signal, selected_dataset, output_directory)
        ).start()

    def _execute_inference_pipeline(self, artifact_filepath, u_signal, y_signal, dataset_name, output_dir):
        try:
            validation_scores, model_payload = execute_inference_pipeline(
                artifact_filepath, u_signal, y_signal, dataset_name, output_dir=output_dir,
                progress_callback=self._update_status
            )

            # Sync GUI fields to reflect the loaded model's architecture
            architecture = model_payload.get("structure", {})
            for param_key in ("na", "nb", "nc", "nk"):
                if param_key in architecture and param_key in self.hyperparameter_inputs:
                    self.hyperparameter_inputs[param_key].delete(0, "end")
                    self.hyperparameter_inputs[param_key].insert(0, str(architecture[param_key]))

            success_message = (
                f"Inference complete! "
                f"MSE: {validation_scores['MSE']:.4f} | RMSE: {validation_scores['RMSE']:.4f} | R²: {validation_scores['R-Squared']:.4f} "
                f"(Saved to {output_dir})"
            )
            self.status_display.configure(text=success_message, text_color="cyan")
        except Exception as error:
            self.status_display.configure(text=f"Error: {str(error)}", text_color="red")
        finally:
            self.inference_btn.configure(state="normal")

    def _dispatch_training_pipeline(self):
        try:
            self._sync_gui_hyperparameters()
        except ValueError:
            self.status_display.configure(text="Error: Parameters must be numbers.", text_color="red")
            return

        selected_dataset = self.dataset_selector.get()
        u_signal, y_signal = self.loaded_datasets[selected_dataset]

        timestamp_sig = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = f"results_{selected_dataset}_{timestamp_sig}"
        os.makedirs(output_directory, exist_ok=True)

        self.status_display.configure(text=f"Starting training for {selected_dataset}...", text_color="yellow")
        self.train_btn.configure(state="disabled")

        # Evolution loop blocks execution; move to background thread
        threading.Thread(target=self._execute_training_pipeline,
                         args=(u_signal, y_signal, selected_dataset, self.active_hyperparameters,
                               output_directory)).start()

    def _execute_training_pipeline(self, u_signal, y_signal, dataset_name, hyperparameters, output_dir):
        try:
            execute_kfold_training_pipeline(
                u_signal, y_signal, dataset_name, hyperparameters, output_dir, progress_callback=self._update_status
            )
            self.status_display.configure(text=f"Training complete! Results in {output_dir}", text_color="cyan")
        except Exception as error:
            self.status_display.configure(text=f"Training error: {str(error)}", text_color="red")
            print(f"Thread error: {error}")
        finally:
            self.train_btn.configure(state="normal")

    def _terminate_application(self):
        self.destroy()
        os._exit(0)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = EvolutionarySystemIDApp()
    app.mainloop()