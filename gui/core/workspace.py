import json
import subprocess
import sys
from pathlib import Path


REQUIRED_DIRS = []  # ["models", "inputs"]
AUTO_CREATE_DIRS = []  # ["runs", "results"]
MANIFEST_FILENAME = "models.json"
VENV_DIR = ".venv"


class Workspace:
    def __init__(self, root: Path):
        self.root = root
        self.manifest: dict = {}
        self.models: list[dict] = []

    @property
    def venv_python(self) -> Path:
        if sys.platform == "win32":
            return self.root / VENV_DIR / "Scripts" / "python.exe"
        return self.root / VENV_DIR / "bin" / "python"

    @property
    def venv_exists(self) -> bool:
        return self.venv_python.exists()

    def validate(self) -> list[str]:
        errors = []
        if not self.root.is_dir():
            errors.append(f"Directory does not exist: {self.root}")
            return errors
        for d in REQUIRED_DIRS:
            if not (self.root / d).is_dir():
                errors.append(f"Missing directory: {d}/")
        for d in AUTO_CREATE_DIRS:
            dir_path = self.root / d
            if not dir_path.is_dir():
                dir_path.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root / MANIFEST_FILENAME
        if not manifest_path.is_file():
            errors.append(f"Missing file: {MANIFEST_FILENAME}")
        else:
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "models" not in data:
                    errors.append(f"{MANIFEST_FILENAME} must have a 'models' key")
                elif not isinstance(data["models"], list):
                    errors.append(f"'models' must be a list")
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON in {MANIFEST_FILENAME}: {e}")
        return errors

    def load_manifest(self) -> None:
        manifest_path = self.root / MANIFEST_FILENAME
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        self.models = self.manifest.get("models", [])

    def resolve_script_path(self, script_path: str) -> Path:
        p = Path(script_path)
        if p.is_absolute():
            return p
        return (self.root / p).resolve()

    def list_input_dirs(self) -> list[str]:
        inputs_dir = self.root / "inputs"
        if not inputs_dir.is_dir():
            return []
        return sorted([
            d.name for d in inputs_dir.iterdir() if d.is_dir()
        ])

    def create_venv(self, on_output=None) -> bool:
        """Create workspace venv. on_output(str) callback for progress messages."""
        def emit(msg):
            if on_output:
                on_output(msg)

        emit("Checking Python installation...")
        try:
            result = subprocess.run(
                [sys.executable, "--version"],
                capture_output=True, text=True, timeout=10
            )
            emit(result.stdout.strip())
        except Exception as e:
            emit(f"Error: Python not found: {e}")
            return False

        venv_path = self.root / VENV_DIR
        if venv_path.exists():
            emit(f"Virtual environment '{VENV_DIR}' already exists.")
            return True

        emit(f"Creating virtual environment in '{VENV_DIR}'...")
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", str(venv_path)],
                check=True, timeout=120
            )
        except Exception as e:
            emit(f"Error creating venv: {e}")
            return False

        emit("Upgrading pip...")
        try:
            subprocess.run(
                [str(self.venv_python), "-m", "pip", "install", "--upgrade", "pip"],
                check=True, timeout=120
            )
        except Exception as e:
            emit(f"Warning: pip upgrade failed: {e}")

        requirements = self.root / "requirements.txt"
        if requirements.exists():
            emit("Installing dependencies from requirements.txt (may take a few minutes)...")
            try:
                subprocess.run(
                    [str(self.venv_python), "-m", "pip", "install", "-r", str(requirements)],
                    check=True, timeout=600
                )
                emit("Dependencies installed successfully.")
            except Exception as e:
                emit(f"Error installing dependencies: {e}")
                return False
        else:
            emit("No requirements.txt found. Skipping dependency installation.")

        emit("Virtual environment ready.")
        return True
