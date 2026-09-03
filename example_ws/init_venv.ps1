$VENV_NAME = ".venv"

# 1. Check Python installation
Write-Host "[STEP 1/5] Checking for Python installation..." -ForegroundColor Cyan
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not added to your system PATH."
    Exit
}

# 2. Create the virtual environment if it doesn't exist
if (-not (Test-Path -Path $VENV_NAME)) {
    Write-Host "[STEP 2/5] Creating virtual environment in '$VENV_NAME'..." -ForegroundColor Green
    python -m venv $VENV_NAME
} else {
    Write-Host "[STEP 2/5] Virtual environment '$VENV_NAME' already exists." -ForegroundColor Yellow
}

# 3. Upgrade pip using the venv's explicit Python path
Write-Host "[STEP 3/5] Upgrading pip..." -ForegroundColor Green
& ".\$VENV_NAME\Scripts\python.exe" -m pip install --upgrade pip

# 4. Install dependencies if requirements.txt is present
if (Test-Path -Path "requirements.txt") {
    Write-Host "[STEP 4/5] Found requirements.txt. Installing dependencies (expected 2-5 minutes, depending on network conditions)..." -ForegroundColor Green
    & ".\$VENV_NAME\Scripts\pip.exe" install -r requirements.txt
} else {
    Write-Host "[STEP 4/5] No requirements.txt found. Skipping installation." -ForegroundColor Yellow
}

# 5. Activate the environment for the current session
Write-Host "[STEP 5/5] Activating virtual environment..." -ForegroundColor Cyan
& ".\$VENV_NAME\Scripts\Activate.ps1"