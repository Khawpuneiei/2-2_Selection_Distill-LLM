$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    py -3.10 -m venv (Join-Path $projectRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Could not create the Python 3.10 virtual environment." }
}

& $venvPython -m pip install --disable-pip-version-check --progress-bar off --quiet --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not upgrade pip in the project environment." }
# The local RTX 4060 laptop and its installed driver support the CUDA 12.1 wheel.
& $venvPython -m pip install --disable-pip-version-check --progress-bar off --quiet "torch==2.5.1" --index-url "https://download.pytorch.org/whl/cu121"
if ($LASTEXITCODE -ne 0) { throw "Could not install the CUDA 12.1 PyTorch wheel." }
& $venvPython -m pip install --disable-pip-version-check --progress-bar off --quiet -r (Join-Path $projectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Could not install the pinned project requirements." }
& $venvPython -m scripts.check_environment --require-cuda
if ($LASTEXITCODE -ne 0) { throw "The project environment check failed." }
Write-Host "Environment ready. Activate with .\.venv\Scripts\Activate.ps1"
