import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import webbrowser
import os
import subprocess
import json
import pandas as pd
from datetime import datetime
import tempfile
import shlex
import shutil


def _cmd_quote_for_call(path: str) -> str:
    """Quote a path inside a cmd.exe /c string (embedded \" -> \"\")."""
    return '"' + path.replace('"', '""') + '"'


def _powershell_single_quoted(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def venv_subprocess_kwargs(venv_root: str, script_path: str, model_args_json_file: str):
    """
    Build subprocess.run kwargs that activate the venv then run `python script args`.
    Windows: activate.bat via cmd, else Activate.ps1 via PowerShell.
    Unix: source bin/activate via bash -lc.
    """
    vnorm = os.path.normpath((venv_root or "").strip())
    common = dict(
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    if os.name == "nt":
        bat = os.path.join(vnorm, "Scripts", "activate.bat")
        ps1 = os.path.join(vnorm, "Scripts", "Activate.ps1")
        if os.path.isfile(bat):
            inner = (
                f"call {_cmd_quote_for_call(bat)} && "
                f"python {_cmd_quote_for_call(script_path)} "
                f"{_cmd_quote_for_call(model_args_json_file)}"
            )
            return (
                {
                    **common,
                    "args": inner,
                    "shell": True,
                    "executable": os.environ.get("COMSPEC", "cmd.exe"),
                },
                f"activate.bat ({bat})",
            )
        if os.path.isfile(ps1):
            inner = (
                f". {_powershell_single_quoted(ps1)}; "
                f"python {_powershell_single_quoted(script_path)} "
                f"{_powershell_single_quoted(model_args_json_file)}"
            )
            return (
                {
                    **common,
                    "args": [
                        "powershell.exe",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-NoLogo",
                        "-Command",
                        inner,
                    ],
                },
                f"Activate.ps1 ({ps1})",
            )
        raise FileNotFoundError(
            "Could not find activate.bat or Activate.ps1 under:\n"
            + os.path.join(vnorm, "Scripts")
        )
    activate_sh = os.path.join(vnorm, "bin", "activate")
    if not os.path.isfile(activate_sh):
        raise FileNotFoundError(f"Could not find bin/activate in venv:\n{vnorm}")
    bash = shutil.which("bash")
    if not bash and os.path.isfile("/bin/bash"):
        bash = "/bin/bash"
    if not bash:
        raise FileNotFoundError(
            "bash is required to run `source bin/activate` for this venv, but was not found."
        )
    cmd = (
        f"source {shlex.quote(activate_sh)} && "
        f"python {shlex.quote(script_path)} {shlex.quote(model_args_json_file)}"
    )
    return (
        {**common, "args": [bash, "-lc", cmd]},
        f"source bin/activate ({activate_sh})",
    )


class ModelLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Launcher v0.2")
        self.root.geometry("1000x700")
        self.root_path = str(os.path.dirname(__file__))

        # Initialize variables
        self.wsdir = tk.StringVar()
        self.task = tk.StringVar()
        self.task.trace_add("write", self.task_on_change)
        self.script_dir = tk.StringVar()
        self.script = tk.StringVar()
        self.name = tk.StringVar()
        self.results_folder = tk.StringVar(value="results")
        self.start_year = tk.StringVar()
        self.start_month = tk.StringVar()
        self.end_year = tk.StringVar()
        self.scenario = tk.StringVar()
        self.stochastic = tk.BooleanVar(value=False)
        self.simulations = tk.StringVar()
        self.max_workers = tk.StringVar()
        self.use_venv = tk.BooleanVar(value=False)
        self.venv_dir = tk.StringVar()

        self.available_scripts = []
        
        self.setup_ui()

    def setup_ui(self):
        # Directory
        dir_frame = ttk.LabelFrame(self.root)
        dir_frame.pack(fill='x')
        ttk.Label(dir_frame, text=os.path.basename(self.root_path)).pack(side='left')
        ttk.Label(dir_frame, text=self.root_path, foreground="gray").pack(side='left', padx=5)

        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=5, side='top')
        
        # Home tab
        home_frame = ttk.Frame(notebook)
        notebook.add(home_frame, text="Home")
        self.setup_home_tab(home_frame)
        
        # Input tab
        input_frame = ttk.Frame(notebook)
        notebook.add(input_frame, text="Input Viewer")
        self.setup_input_tab(input_frame)

        # Results tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Results Viewer")
        self.setup_results_tab(results_frame)

    def setup_home_tab(self, parent):
        # Workspace & Task
        ws_task_frame = ttk.LabelFrame(parent, text="Workspace & Task", padding="10")
        ws_task_frame.pack(fill='x', pady=(0, 5))

        # Workspace & Task > Workspace Directory
        ws_frame = ttk.Frame(ws_task_frame)
        ws_frame.pack(fill='x', pady=5)
        ttk.Label(ws_frame, text="Workspace:").pack(side='left')
        ttk.Entry(ws_frame, textvariable=self.wsdir, width=60, state='readonly').pack(
            side='left', fill='x', expand=True, padx=(5, 0))
        ttk.Button(ws_frame, text="Browse", command=self.browse_work_dir).pack(side='right', padx=(5, 0))

        # Workspace & Task > Task Entry
        task_frame = ttk.Frame(ws_task_frame)
        task_frame.pack(fill='x', pady=5)
        ttk.Label(task_frame, text="Task:").pack(side='left')
        self.task_entry = ttk.Entry(task_frame, textvariable=self.task, width=30)
        self.task_entry.pack(side='left', padx=5)

        self.load_button = ttk.Button(task_frame, text="Load Config", command=self.load_task_config, state=tk.DISABLED)
        self.load_button.pack(side='left', padx=5)

        self.save_button = ttk.Button(task_frame, text="Save Config", command=self.save_task_config)
        self.save_button.pack(side='left', padx=5)

        # Model
        model_frame = ttk.LabelFrame(parent, text="Model", padding="10")
        model_frame.pack(fill='x', pady=5)

        # Model > Name
        name_frame = ttk.Frame(model_frame)
        name_frame.pack(fill='x', pady=(0, 0))

        ttk.Label(name_frame, text="Name:").pack(side='left')
        self.name_entry = ttk.Entry(name_frame, textvariable=self.name, width=30)
        self.name_entry.pack(side='left', padx=5)

        # Model > Repository
        repository_frame = ttk.Frame(model_frame)
        repository_frame.pack(fill='x', pady=(0, 5))

        ttk.Label(repository_frame, text="Repository:").pack(side='left')
        ttk.Entry(repository_frame, textvariable=self.script_dir, width=60, state='readonly').pack(
            side='left', fill='x', expand=True, padx=(5, 0))
        ttk.Button(repository_frame, text="Browse", command=self.browse_script_dir).pack(side='right', padx=(5, 0))

        # Model > Select
        select_frame = ttk.Frame(model_frame)
        select_frame.pack(fill='x', pady=(0, 0))

        ttk.Label(select_frame, text="Select:").pack(side='left')
        self.script_menu = ttk.Combobox(select_frame, textvariable=self.script, state='readonly', width=30)
        self.script_menu.pack(side='left', padx=(25, 0))

        venv_frame = ttk.Frame(model_frame)
        venv_frame.pack(fill='x', pady=(8, 0))
        self.venv_check = ttk.Checkbutton(
            venv_frame,
            text="Use virtual environment",
            variable=self.use_venv,
            command=self.on_use_venv_change,
        )
        self.venv_check.pack(side='left')
        ttk.Label(venv_frame, text="Venv root:").pack(side='left', padx=(12, 0))
        self.venv_entry = ttk.Entry(venv_frame, textvariable=self.venv_dir, width=50, state='disabled')
        self.venv_entry.pack(side='left', fill='x', expand=True, padx=(5, 0))
        self.venv_browse_button = ttk.Button(venv_frame, text="Browse", command=self.browse_venv_dir, state='disabled')
        self.venv_browse_button.pack(side='right', padx=(5, 0))
        
        # Run Setting section
        setting_frame = ttk.LabelFrame(parent, text="Run Setting", padding="10")
        setting_frame.pack(fill='x', pady=5)

        setting_grid = ttk.Frame(setting_frame)
        setting_grid.pack(fill='x')

        ttk.Label(setting_grid, text="Scenario:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(setting_grid, textvariable=self.scenario, width=10).grid(row=0, column=1, padx=5)

        ttk.Label(setting_grid, text="Start Year:").grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Entry(setting_grid, textvariable=self.start_year, width=10).grid(row=1, column=1, padx=5)

        ttk.Label(setting_grid, text="Start Month:").grid(row=2, column=0, sticky=tk.W, padx=5)
        ttk.Entry(setting_grid, textvariable=self.start_month, width=10).grid(row=2, column=1, padx=5)

        ttk.Label(setting_grid, text="End Year:").grid(row=3, column=0, sticky=tk.W, padx=5)
        ttk.Entry(setting_grid, textvariable=self.end_year, width=10).grid(row=3, column=1, padx=5)

        self.stochastic_check = ttk.Checkbutton(
            setting_grid,
            text="Stochastic",
            variable=self.stochastic,
            command=self.on_stochastic_change
        )
        self.stochastic_check.grid(row=0, column=2, sticky=tk.W, padx=5, pady=(4, 0))

        ttk.Label(setting_grid, text="Simulations:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=(4, 0))
        self.simulations_entry = ttk.Entry(setting_grid, textvariable=self.simulations, width=10, state=tk.DISABLED)
        self.simulations_entry.grid(row=1, column=3, sticky=tk.W, padx=5, pady=(4, 0))

        ttk.Label(setting_grid, text="max_workers:").grid(row=2, column=2, sticky=tk.W, padx=5, pady=(4, 0))
        self.max_workers_entry = ttk.Entry(setting_grid, textvariable=self.max_workers, width=10, state=tk.DISABLED)
        self.max_workers_entry.grid(row=2, column=3, sticky=tk.W, padx=5, pady=(4, 0))

        ttk.Label(setting_grid, text="Input Folder(s):").grid(row=0, column=4, sticky=tk.W, padx=5, pady=5)
        ttk.Label(setting_grid, text="(separated by ',' or newline)").grid(row=1, column=4, sticky=tk.W, padx=5, pady=5)
        self.input_folder = tk.Text(setting_grid, width=30, height=6)
        self.input_folder.grid(row=0, column=5, rowspan=4, padx=5, pady=5)

        # Run button
        run_frame = ttk.Frame(parent)
        run_frame.pack(fill='x', pady=10)

        self.run_button = ttk.Button(run_frame, text="Run", command=self.run_script)
        self.run_button.pack(side='left', padx=5)

        self.open_results_button = ttk.Button(run_frame, text="Open", command=self.open_results_folder)
        self.open_results_button.pack(side='right', padx=5)

        ttk.Entry(run_frame, textvariable=self.results_folder, width=20, state='readonly').pack(side='right', padx=5)
        ttk.Label(run_frame, text="Results Folder:").pack(side='right', padx=5)

        # Log output
        log_frame = ttk.LabelFrame(parent, text="Execution Log", padding="10")
        log_frame.pack(fill='both', expand=True, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(anchor='nw', fill='both', expand=True)

    def setup_input_tab(self, parent):
        # File actions
        file_actions_frame = ttk.Frame(parent)
        file_actions_frame.pack(fill='x', pady=10)
        ttk.Button(file_actions_frame, text="Refresh", command=self.refresh_input_files).pack(side='left', padx=5)
        ttk.Button(file_actions_frame, text="View Selected File", command=self.view_selected_file).pack(side='left', padx=5)
        self.input_file_dp = tk.StringVar(value="4")
        spinbox = ttk.Spinbox(file_actions_frame, textvariable=self.input_file_dp, from_=0, to=10, width=5)
        spinbox.pack(side='right')
        ttk.Label(file_actions_frame, text="Decimal places to display:").pack(side='right')

        # Data files list
        files_tree_frame = ttk.LabelFrame(parent, text="Data Files", padding="10")
        files_tree_frame.pack(fill='both', expand=True, pady=5)

        # Treeview for file list
        columns = ("Subfolder", "File", "Modified", "Size")
        self.files_tree = ttk.Treeview(files_tree_frame, selectmode="browse", columns=columns, show="headings", height=10)
        
        for col in columns:
            self.files_tree.heading(col, text=col)
            self.files_tree.column(col, width=150)
        
        files_scrollbar = ttk.Scrollbar(files_tree_frame, orient='vertical', command=self.files_tree.yview)
        self.files_tree.configure(yscrollcommand=files_scrollbar.set)
        
        self.files_tree.pack(side='left', fill='both', expand=True)
        files_scrollbar.pack(side='right', fill='y')

    def setup_results_tab(self, parent):
        # Results actions
        results_actions_frame = ttk.Frame(parent)
        results_actions_frame.pack(fill='x', pady=10)
        ttk.Button(results_actions_frame, text="Refresh", command=self.refresh_result_files).pack(side='left', padx=5)
        ttk.Button(results_actions_frame, text="View Selected Result", command=self.view_result_file).pack(side='left', padx=5)
        self.result_file_dp = tk.StringVar(value="4")
        spinbox = ttk.Spinbox(results_actions_frame, textvariable=self.result_file_dp, from_=0, to=10, width=5)
        spinbox.pack(side='right')
        ttk.Label(results_actions_frame, text="Decimal places to display:").pack(side='right')

        # Results tree
        results_tree_frame = ttk.LabelFrame(parent, text="Result Files", padding="10")
        results_tree_frame.pack(fill='both', expand=True, pady=5)

        results_columns = ("Task", "File", "Modified", "Size")
        self.results_tree = ttk.Treeview(results_tree_frame, selectmode="browse", columns=results_columns, show="headings", height=12)

        for col in results_columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=150)
        
        results_scrollbar = ttk.Scrollbar(results_tree_frame, orient='vertical', command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_tree.pack(side='left', fill='both', expand=True)
        results_scrollbar.pack(side='right', fill='y')

    def browse_work_dir(self):
        folder = filedialog.askdirectory(title="Select Workspace Folder")
        if folder:
            self.wsdir.set(folder)

    def browse_script_dir(self):
        folder = filedialog.askdirectory(title="Select Model Repository Folder")
        if folder:
            self.script_dir.set(folder)
            self.refresh_script_menu()

    def browse_venv_dir(self):
        folder = filedialog.askdirectory(title="Select virtual environment folder (e.g. .venv)")
        if folder:
            self.venv_dir.set(folder)

    def refresh_script_menu(self):
        self.available_scripts.clear()

        scr_path = self.script_dir.get()
        scripts = []
        # Retrieve Python scripts
        for file in os.listdir(scr_path):
            if file.endswith(".py") and not file.startswith("_"):
                scr_name = f"{file[:-3]}"
                scripts.append(scr_name)
                self.available_scripts.append((file[:-3], scr_name))

        self.script_menu['values'] = scripts
        self.script.set("")

    def get_input_folders(self):
        raw_text = self.input_folder.get('1.0', 'end')
        raw_text = raw_text.replace('\n',',').replace('\r',',')
        raw_list = raw_text.split(',')
        lst = list(filter(None, [s.strip() for s in raw_list]))
        return lst

    def on_stochastic_change(self):
        state = tk.NORMAL if self.stochastic.get() else tk.DISABLED
        self.simulations_entry.config(state=state)
        self.max_workers_entry.config(state=state)

    def on_use_venv_change(self):
        if self.use_venv.get():
            self.venv_entry.config(state="readonly")
            self.venv_browse_button.config(state=tk.NORMAL)
        else:
            self.venv_entry.config(state="disabled")
            self.venv_browse_button.config(state=tk.DISABLED)

    @property
    def task_dir(self):
        return os.path.join(self.wsdir.get(), '.tasks', self.task.get())

    @property
    def config_json_file(self):
        return f"{self.task_dir}/config.json"

    def task_on_change(self, *args):
        if os.path.exists(self.config_json_file):
            self.load_button.config(state=tk.NORMAL)
        else:
            self.load_button.config(state=tk.DISABLED)
        self.results_folder.set(f"results/{self.task.get()}")

    def load_task_config(self):
        if not os.path.exists(self.config_json_file):
            messagebox.showinfo("Load Config", f"No config file for task '{self.task}'")
            return

        confirmed = messagebox.askyesno(
            "Confirmation", f"You are about to load the config for task {self.task.get()}"
        )
        if not confirmed:
            messagebox.showinfo("Confirmation", "Canceled.")
            return

        with open(self.config_json_file, 'r', encoding='utf-8') as json_file:
            config = json.load(json_file)

        self.script_dir.set(config.get("script_dir", ""))
        self.refresh_script_menu()
        self.script.set(config.get("script", ""))
        self.use_venv.set(bool(config.get("use_venv", False)))
        self.venv_dir.set(config.get("venv_dir", ""))
        self.on_use_venv_change()
        self.stochastic.set(bool(config.get("stochastic", False)))
        self.on_stochastic_change()

        model_args_json_file = config.get("model_args_json_file", None)
        if model_args_json_file:
            with open(model_args_json_file, 'r', encoding='utf-8') as json_file:
                model_args = json.load(json_file)

            self.name.set(model_args.get("model_name", ""))
            self.scenario.set(model_args.get("scenario",""))
            self.start_year.set(model_args.get("start_year",""))
            self.start_month.set(model_args.get("start_month", ""))
            self.end_year.set(model_args.get("end_year", ""))
            self.simulations.set(model_args.get("simulations", ""))
            self.max_workers.set(model_args.get("max_workers", 0))
            self.input_folder.delete('0.0', 'end')
            lst = model_args.get("input_directories", "")
            txt = '\n'.join(lst)
            self.input_folder.insert('0.0', txt)

        messagebox.showinfo("Load Config", "Task config has been loaded.")

    def save_task_config(self, bypass_confirm=False, msg_done=True):
        if not self.task.get():
            return

        if not bypass_confirm:
            confirmed = messagebox.askyesno(
                "Confirmation", f"You are about to save the config for task {self.task.get()}"
            )
            if not confirmed:
                messagebox.showinfo("Confirmation", "Canceled.")
                return

        # make task dir
        os.makedirs(f"{self.task_dir}", exist_ok=True)

        # create the model args dictionary
        start_year, start_month, end_year = 0, 0, 0
        if self.start_year.get(): start_year = int(self.start_year.get())
        if self.start_month.get(): start_month = int(self.start_month.get())
        if self.end_year.get(): end_year = int(self.end_year.get())
        model_args = {
            "model_name": self.name.get(),
            "scenario": self.scenario.get(),
            "start_year": start_year,
            "start_month": start_month,
            "end_year": end_year,
            "simulations": self.simulations.get() if self.stochastic.get() else None,
            "max_workers": int(self.max_workers.get()) if self.stochastic.get() else 0,
            "workspace_directory": self.wsdir.get(),
            "input_directories": self.get_input_folders(),
            "results_directory": self.results_folder.get()
        }
        # write the model args to a JSON file
        ts = str(hex(int(f"{datetime.now().strftime('%y%m%d%H%M%S')}")))[2:]
        model_args_json_file = f"{self.task_dir}/model_args-{ts}.json"
        with open(model_args_json_file, "w", encoding="utf-8") as json_file:
            json.dump(model_args, json_file, ensure_ascii=False, indent=4)

        # create the task config dictionary
        config = {
            "task": self.task.get(),
            "script_dir": self.script_dir.get(),
            "script": self.script.get(),
            "use_venv": self.use_venv.get(),
            "venv_dir": self.venv_dir.get(),
            "stochastic": self.stochastic.get(),
            "model_args_json_file": model_args_json_file,
            "saved_on": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        }
        # write the task config dictionary to a JSON file
        with open(self.config_json_file, "w", encoding="utf-8") as json_file:
            json.dump(config, json_file, ensure_ascii=False, indent=4)
        # make a backup named with timestamp matching model args JSON file
        with open(f"{self.config_json_file[:-5]}-{ts}.json", "w", encoding="utf-8") as json_file:
            json.dump(config, json_file, ensure_ascii=False, indent=4)

        if msg_done:
            messagebox.showinfo("Save Config", "Task config has been saved.")

    def run_script(self):
        try:
            confirmed = messagebox.askyesno(
                "Confirmation", f"You are about to run task {self.task.get()}"
            )
            if not confirmed:
                messagebox.showinfo("Confirmation", "Canceled.")
                return None
            self.save_task_config(bypass_confirm=True, msg_done=False)

            script_path = os.path.join(self.script_dir.get(), f"{self.script.get()}.py")
            with open(self.config_json_file, 'r', encoding='utf-8') as json_file:
                config = json.load(json_file)
            model_args_json_file = config['model_args_json_file']

            if self.use_venv.get():
                vroot = (self.venv_dir.get() or "").strip()
                if not vroot:
                    messagebox.showwarning(
                        "Virtual environment",
                        "Please choose the virtual environment folder (e.g. .venv).",
                    )
                    return None
                try:
                    venv_run_kw, venv_activation_desc = venv_subprocess_kwargs(
                        vroot, script_path, model_args_json_file
                    )
                except FileNotFoundError as e:
                    messagebox.showerror("Virtual environment", str(e))
                    return None
            else:
                venv_run_kw = None
                venv_activation_desc = None

            # Reset the error log file
            errlog_file = f"{self.task_dir}/err_log.txt"
            if os.path.exists(errlog_file):
                os.remove(errlog_file)

            self.write_run_log(f"{'=' * 80}")
            self.write_run_log(f"Starting run: {self.task.get()}")
            self.write_run_log(f"Starting run:")
            self.write_run_log(f"- script: {script_path}")
            self.write_run_log(f"- args: {model_args_json_file}")
            if self.use_venv.get():
                self.write_run_log(f"- venv activation: {venv_activation_desc}")
            self.write_run_log(f"Running...")

            try:
                if venv_run_kw is not None:
                    subprocess.run(**venv_run_kw)
                else:
                    subprocess.run(
                        ["python", script_path, model_args_json_file],
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        check=True,
                    )
                self.write_run_log(f"Script {self.script.get()}.py executed successfully.")
                messagebox.showinfo("Done", "Run completed!")
            except subprocess.CalledProcessError as e:
                self.write_run_log(f"Script {self.script.get()}.py failed with return code: {e.returncode}")
                self.write_run_log(f"Error output:\n{e.stderr}.")
                with open(errlog_file, "w", encoding="utf-8") as f:
                    f.write(e.stderr)
                self.write_run_log(f"Error information has been saved to: {errlog_file}.")
                messagebox.showinfo("Done", f"Run failed!\n\n"
                                    f"Error information has been saved to:\n{errlog_file}.")

            self.write_run_log(f"{'=' * 80}")

        except Exception as e:
            print("Error during run:", e)
            return False
        return True

    def write_run_log(self, log_text: str):
        self.log_text.insert(tk.END, f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ")
        self.log_text.insert(tk.END, f"{log_text}\n")

    def open_results_folder(self):
        results_path = os.path.join(self.wsdir.get(), self.results_folder.get())
        if not os.path.exists(results_path):
            messagebox.showinfo("Message", f"Path doesn't exist:\n{results_path}")
            return
        os.startfile(results_path)

    def refresh_input_files(self):
        # Clear existing items
        for item in self.files_tree.get_children():
            self.files_tree.delete(item)

        input_path = os.path.join(self.wsdir.get(), "input")
        if not os.path.exists(input_path):
            return

        for sub_folder in os.listdir(input_path):
            sub_path = os.path.join(input_path, sub_folder)
            if os.path.isdir(sub_path):
                # Look for input files
                for root, dirs, files in os.walk(sub_path):
                    for file in files:
                        if file.endswith('.csv'):
                            file_path = os.path.join(root, file)
                            stat = os.stat(file_path)
                            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                            size = f"{stat.st_size / 1e3:.0f} KB"
                            self.files_tree.insert("", 'end', values=(sub_folder, file, modified, size))

    def view_selected_file(self):
        selection = self.files_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a file to view.")
            return

        item = self.files_tree.item(selection[0])
        sub_folder, filename, _, _ = item['values']

        input_folder = os.path.join(self.wsdir.get(), "input")
        file_path = os.path.join(input_folder, sub_folder, filename)

        self.open_csv_in_browser(file_path, filename, int(self.input_file_dp.get()))

    def refresh_result_files(self):
        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        results_path = os.path.join(self.wsdir.get(), "results")

        if not os.path.exists(results_path):
            return
        
        for sub_folder in os.listdir(results_path):
            sub_path = os.path.join(results_path, sub_folder)
            if os.path.isdir(sub_path):
                # Look for result files
                for root, dirs, files in os.walk(sub_path):
                    for file in files:
                        if file.endswith('.csv'):
                            file_path = os.path.join(root, file)
                            stat = os.stat(file_path)
                            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                            size = f"{stat.st_size/1e3:.0f} KB"
                            
                            self.results_tree.insert("", 'end', values=(sub_folder, file, modified, size))
    
    def view_result_file(self):
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a result file to view.")
            return
        
        item = self.results_tree.item(selection[0])
        sub_folder, filename, _, _ = item['values']

        results_folder = os.path.join(self.wsdir.get(), "results")
        file_path = os.path.join(results_folder, sub_folder, filename)

        self.open_csv_in_browser(file_path, filename, int(self.result_file_dp.get()))

    def open_csv_in_browser(self, file_path, filename, dp):
        """Open CSV in browser with beautiful HTML formatting"""
        try:
            df = pd.read_csv(file_path, na_filter=False)
            html_content = self.generate_html_from_df(df, file_path, filename, dp)

            # Create temporary HTML file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                f.write(html_content)
                temp_path = f.name

            # Open in browser
            webbrowser.open('file://' + os.path.abspath(temp_path))

        except Exception as e:
            messagebox.showerror("Error", f"Could not open file in browser: {str(e)}")

    @staticmethod
    def generate_html_from_df(df, file_path, filename, dp):
        """Generate a beautiful HTML representation of the DataFrame"""

        # Custom CSS for modern styling with sticky header and enhanced scrolling
        css = """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 20px;
                min-height: 100vh;
            }

            .container {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                max-width: 1200px;
                margin: 0 auto;
            }

            h2 {
                color: #2c3e50;
                margin-bottom: 20px;
                font-weight: 300;
                font-size: 1.5em;
            }

			p {
                margin: 0 0 15px 0;
                color: #6c757d;
                font-size: 0.9em;
            }

            .table-wrapper {
                position: relative;
                max-height: 70vh;
                overflow: auto;
                border-radius: 15px;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
                background: white;
            }

            table {
                width: 100%;
                border-collapse: collapse;
                background: white;
            }

            th {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 12px;
                text-align: left;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                font-size: 0.9em;
                position: sticky;
                top: 0;
                z-index: 100;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                cursor: pointer;
                transition: background 0.3s ease;
                user-select: none;
            }

            th:hover {
                background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
            }

            th.locked {
                background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
                position: sticky;
                left: var(--left-offset, 0px);
                z-index: 101;
            }

            th.locked::after {
                content: '🔒';
                margin-left: 8px;
                font-size: 0.8em;
            }

            td.locked {
                position: sticky;
                left: var(--left-offset, 0px);
                z-index: 99;
                background: #fff8f8;
                border-right: 2px solid #e53e3e;
            }

            tbody tr:nth-child(even) td.locked {
                background: #fef5f5;
            }

            tbody tr:hover td.locked {
                background: #fed7d7;
            }

            td {
                padding: 12px;
                border-bottom: 1px solid #e9ecef;
                transition: background-color 0.3s ease;
            }

            tbody tr:hover {
                background-color: #f8f9fa;
            }

            tbody tr:nth-child(even) {
                background-color: #fdfdfd;
            }

            .numeric {
                text-align: right;
                font-weight: 500;
                color: #495057;
            }

            /* Scroll indicators */
            .scroll-indicator {
                position: fixed;
                right: 20px;
                top: 50%;
                transform: translateY(-50%);
                background: rgba(102, 126, 234, 0.9);
                color: white;
                padding: 10px;
                border-radius: 20px;
                font-size: 0.8em;
                z-index: 1000;
                opacity: 0;
                transition: opacity 0.3s ease;
                pointer-events: none;
            }

            .scroll-indicator.show {
                opacity: 1;
            }

            /* Column lock controls */
            .column-controls {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                border: 1px solid #dee2e6;
            }

            .column-controls h3 {
                margin: 0 0 10px 0;
                color: #495057;
                font-size: 1.1em;
            }

            .column-controls p {
                margin: 0 0 15px 0;
                color: #6c757d;
                font-size: 0.9em;
            }

            .column-selector {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                align-items: center;
            }

            .column-btn {
                background: white;
                border: 2px solid #dee2e6;
                padding: 8px 12px;
                border-radius: 20px;
                cursor: pointer;
                font-size: 0.85em;
                transition: all 0.3s ease;
                user-select: none;
            }

            .column-btn:hover {
                border-color: #667eea;
                background: #f0f4ff;
            }

            .column-btn.locked {
                background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
                color: white;
                border-color: #e53e3e;
            }

            .column-btn.locked::after {
                content: ' 🔒';
            }

            .clear-locks-btn {
                background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 20px;
                cursor: pointer;
                font-size: 0.85em;
                transition: transform 0.2s ease;
            }

            .clear-locks-btn:hover {
                transform: translateY(-1px);
            }
            .nav-buttons {
                position: fixed;
                right: 20px;
                bottom: 20px;
                display: flex;
                flex-direction: column;
                gap: 10px;
                z-index: 1000;
            }

            .nav-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                cursor: pointer;
                font-size: 1.2em;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }

            .nav-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
            }

            /* Enhanced scrollbar styling */
            .table-wrapper::-webkit-scrollbar {
                width: 12px;
                height: 12px;
            }

            .table-wrapper::-webkit-scrollbar-track {
                background: #f1f1f1;
                border-radius: 10px;
            }

            .table-wrapper::-webkit-scrollbar-thumb {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 10px;
            }

            .table-wrapper::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
            }

            /* Progress bar for scroll position */
            .scroll-progress {
                position: fixed;
                top: 0;
                left: 0;
                width: 0%;
                height: 4px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                z-index: 1001;
                transition: width 0.1s ease;
            }

            @media (max-width: 768px) {
                .container {
                    padding: 15px;
                    margin: 10px;
                }

                .stats {
                    flex-direction: column;
                    gap: 15px;
                }

                .nav-buttons {
                    right: 10px;
                    bottom: 10px;
                }

                .nav-btn {
                    width: 40px;
                    height: 40px;
                    font-size: 1em;
                }
            }
        </style>
        """

        # Start HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>CSV Data Viewer</title>
            {css}
        </head>
        <body>
            <div class="container">
                <h2>📊 {filename}</h2>
                <p>{file_path}</p>                

                <!-- Column Lock Controls -->
                <div class="column-controls">
                    <h3>📌 Column Lock Controls</h3>
                    <p>Click on column names below to lock/unlock them. Locked columns will stay visible when scrolling horizontally.</p>
                    <div class="column-selector" id="column-selector">
                        <!-- Column buttons will be added here by JavaScript -->
                    </div>
                </div>

                <div class="scroll-progress"></div>
                <div class="scroll-indicator">Scroll position: <span id="scroll-percent">0%</span></div>

                <div class="table-wrapper" id="table-wrapper">
                    <table>
                        <thead>
                            <tr>
        """

        # Add headers
        for i, col in enumerate(df.columns):
            if i < 10:
                html_content += f'<th data-column-index="{i}" onclick="toggleColumnLock({i})">{col}</th>'
            else:
                html_content += f'<th>{col}</th>'

        html_content += """
                            </tr>
                        </thead>
                        <tbody>
        """

        # Add data rows (limit to first 1000 rows for performance)
        display_df = df.head(1000) if len(df) > 1000 else df

        for _, row in display_df.iterrows():
            html_content += "<tr>"
            for i, col in enumerate(df.columns):
                value = row[col]
                # Check if numeric for styling
                # css_class = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else ""
                if pd.api.types.is_numeric_dtype(df[col]):
                    css_class = "numeric"
                    value = round(value, dp)
                else:
                    css_class = ""
                html_content += f'<td class="{css_class}" data-column-index="{i}">{value}</td>'
            html_content += "</tr>"

        html_content += """
                        </tbody>
                    </table>
                </div>

                <!-- Navigation buttons -->
                <div class="nav-buttons">
                    <button class="nav-btn" onclick="scrollToTop()" title="Go to top">↑</button>
                    <button class="nav-btn" onclick="scrollToBottom()" title="Go to bottom">↓</button>
                    <button class="nav-btn" onclick="scrollToMiddle()" title="Go to middle">⊞</button>
                </div>
        """

        if len(df) > 1000:
            html_content += f"<p style='text-align: center; margin-top: 20px; color: #6c757d;'><em>Showing first 1000 rows out of {len(df)} total rows</em></p>"

        html_content += """
            </div>

            <script>
            // Enhanced scrolling functionality with column locking
            const tableWrapper = document.getElementById('table-wrapper');
            const scrollProgress = document.querySelector('.scroll-progress');
            const scrollIndicator = document.querySelector('.scroll-indicator');
            const scrollPercent = document.getElementById('scroll-percent');
            const columnSelector = document.getElementById('column-selector');

            // Column locking state
            let lockedColumns = [];
            let columnWidths = [];

            // Initialize column selector buttons
            function initializeColumnSelector() {
                const headers = document.querySelectorAll('th');
                columnSelector.innerHTML = '';

                headers.forEach((header, index) => {
                    if (header.textContent && index < 10) {
                        const btn = document.createElement('button');
                        btn.className = 'column-btn';
                        btn.textContent = header.textContent;
                        btn.dataset.columnIndex = index;
                        btn.onclick = () => toggleColumnLock(index);
                        columnSelector.appendChild(btn);
                    }
                });

                // Add clear all button
                const clearBtn = document.createElement('button');
                clearBtn.className = 'clear-locks-btn';
                clearBtn.textContent = 'Clear All Locks';
                clearBtn.onclick = clearAllLocks;
                columnSelector.appendChild(clearBtn);

                // Calculate initial column widths
                calculateColumnWidths();
            }

            // Calculate column widths for proper positioning
            function calculateColumnWidths() {
                const headers = document.querySelectorAll('th');
                columnWidths = Array.from(headers).map(header => header.offsetWidth);
            }

            // Toggle column lock
            function toggleColumnLock(columnIndex) {
                const header = document.querySelector(`th[data-column-index="${columnIndex}"]`);
                const cells = document.querySelectorAll(`td[data-column-index="${columnIndex}"]`);
                const btn = document.querySelector(`button[data-column-index="${columnIndex}"]`);

                if (lockedColumns.includes(columnIndex)) {
                    // Unlock column
                    lockedColumns = lockedColumns.filter(col => col !== columnIndex);
                    header.classList.remove('locked');
                    cells.forEach(cell => cell.classList.remove('locked'));
                    btn.classList.remove('locked');
                } else {
                    // Lock column
                    lockedColumns.push(columnIndex);
                    lockedColumns.sort((a, b) => a - b); // Keep sorted for proper positioning
                    header.classList.add('locked');
                    cells.forEach(cell => cell.classList.add('locked'));
                    btn.classList.add('locked');
                }

                updateLockedColumnsPosition();
            }

            // Update positions of locked columns
            function updateLockedColumnsPosition() {
                let leftOffset = 0;

                lockedColumns.forEach(columnIndex => {
                    const header = document.querySelector(`th[data-column-index="${columnIndex}"]`);
                    const cells = document.querySelectorAll(`td[data-column-index="${columnIndex}"]`);

                    header.style.setProperty('--left-offset', `${leftOffset}px`);
                    cells.forEach(cell => {
                        cell.style.setProperty('--left-offset', `${leftOffset}px`);
                    });

                    leftOffset += columnWidths[columnIndex] || 120; // Default width fallback
                });
            }

            // Clear all locked columns
            function clearAllLocks() {
                lockedColumns.forEach(columnIndex => {
                    const header = document.querySelector(`th[data-column-index="${columnIndex}"]`);
                    const cells = document.querySelectorAll(`td[data-column-index="${columnIndex}"]`);
                    const btn = document.querySelector(`button[data-column-index="${columnIndex}"]`);

                    header.classList.remove('locked');
                    cells.forEach(cell => cell.classList.remove('locked'));
                    btn.classList.remove('locked');
                });

                lockedColumns = [];
            }

            // Auto-lock first column by default (common use case)
            function autoLockFirstColumn() {
                if (document.querySelectorAll('th').length > 0) {
                    toggleColumnLock(0);
                }
            }

            // Update scroll progress and indicator
            function updateScrollProgress() {
                const scrollTop = tableWrapper.scrollTop;
                const scrollHeight = tableWrapper.scrollHeight - tableWrapper.clientHeight;
                const scrollPercentage = scrollHeight > 0 ? (scrollTop / scrollHeight) * 100 : 0;

                scrollProgress.style.width = scrollPercentage + '%';
                scrollPercent.textContent = Math.round(scrollPercentage) + '%';

                // Show indicator when scrolling
                scrollIndicator.classList.add('show');
                clearTimeout(scrollIndicator.hideTimer);
                scrollIndicator.hideTimer = setTimeout(() => {
                    scrollIndicator.classList.remove('show');
                }, 2000);
            }

            // Scroll navigation functions
            function scrollToTop() {
                tableWrapper.scrollTo({ top: 0, behavior: 'smooth' });
            }

            function scrollToBottom() {
                tableWrapper.scrollTo({ top: tableWrapper.scrollHeight, behavior: 'smooth' });
            }

            function scrollToMiddle() {
                const middle = (tableWrapper.scrollHeight - tableWrapper.clientHeight) / 2;
                tableWrapper.scrollTo({ top: middle, behavior: 'smooth' });
            }

            // Keyboard navigation
            document.addEventListener('keydown', function(e) {
                if (e.target.tagName.toLowerCase() !== 'input') {
                    switch(e.key) {
                        case 'Home':
                            e.preventDefault();
                            scrollToTop();
                            break;
                        case 'End':
                            e.preventDefault();
                            scrollToBottom();
                            break;
                        case 'PageUp':
                            e.preventDefault();
                            tableWrapper.scrollBy({ top: -tableWrapper.clientHeight * 0.8, behavior: 'smooth' });
                            break;
                        case 'PageDown':
                            e.preventDefault();
                            tableWrapper.scrollBy({ top: tableWrapper.clientHeight * 0.8, behavior: 'smooth' });
                            break;
                        case 'ArrowUp':
                            if (e.ctrlKey) {
                                e.preventDefault();
                                tableWrapper.scrollBy({ top: -50, behavior: 'smooth' });
                            }
                            break;
                        case 'ArrowDown':
                            if (e.ctrlKey) {
                                e.preventDefault();
                                tableWrapper.scrollBy({ top: 50, behavior: 'smooth' });
                            }
                            break;
                    }
                }
            });

            // Mouse wheel enhancement for smoother scrolling
            tableWrapper.addEventListener('wheel', function(e) {
                e.preventDefault();
                const delta = e.deltaY;
                const scrollAmount = delta * 1.5; // Adjust scroll sensitivity
                tableWrapper.scrollBy({ top: scrollAmount, behavior: 'auto' });
            });

            // Add scroll event listener
            tableWrapper.addEventListener('scroll', updateScrollProgress);

            // Initialize
            updateScrollProgress();

            // Add row highlighting on hover for better readability
            const rows = document.querySelectorAll('tbody tr');
            rows.forEach(row => {
                row.addEventListener('mouseenter', function() {
                    this.style.backgroundColor = '#e3f2fd';
                    this.style.transform = 'scale(1.001)';
                    this.style.transition = 'all 0.2s ease';
                });

                row.addEventListener('mouseleave', function() {
                    this.style.backgroundColor = '';
                    this.style.transform = '';
                });
            });

            // Auto-hide navigation buttons when not needed
            function toggleNavButtons() {
                const navButtons = document.querySelector('.nav-buttons');
                const isScrollable = tableWrapper.scrollHeight > tableWrapper.clientHeight;
                navButtons.style.display = isScrollable ? 'flex' : 'none';
            }

            // Initialize everything
            window.addEventListener('load', () => {
                initializeColumnSelector();
                toggleNavButtons();
                updateScrollProgress();

                // Auto-lock first column for better UX
                setTimeout(autoLockFirstColumn, 100);
            });

            // Recalculate on resize
            window.addEventListener('resize', () => {
                calculateColumnWidths();
                updateLockedColumnsPosition();
                toggleNavButtons();
            });
            </script>
        </body>
        </html>
        """

        return html_content


# ----------------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------------
def main():
    root = tk.Tk()
    app = ModelLauncher(root)
    root.mainloop()

if __name__ == "__main__":
    main()