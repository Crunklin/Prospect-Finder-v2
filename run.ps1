# Navigate to the script directory (project root)
Set-Location -Path $PSScriptRoot

# Ensure Python is available
$python = "python"
try {
  & $python --version | Out-Null
} catch {
  Write-Host "Python not found on PATH. Please install Python 3.10+ and try again." -ForegroundColor Red
  exit 1
}

# Create virtual environment if missing
$venvPath = Join-Path $PSScriptRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts/python.exe"
if (-Not (Test-Path $venvPython)) {
  Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Cyan
  & $python -m venv .venv
}

# Upgrade pip and install requirements if present
Write-Host "Upgrading pip and installing dependencies..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip
if (Test-Path (Join-Path $PSScriptRoot "requirements.txt")) {
  & $venvPython -m pip install -r requirements.txt
}

# Run the FastAPI app with Uvicorn (auto-reload)
$port = 8003
Write-Host "Starting server at http://127.0.0.1:$port/ (Ctrl+C to stop)" -ForegroundColor Green
& $venvPython -m uvicorn app.main:app --reload --port $port --host 127.0.0.1
