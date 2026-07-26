param(
    [string]$Name = "PressureMatrixCollector"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    $PythonExe = Join-Path $ProjectRoot "venv\Scripts\python.exe"
}

if (-not (Test-Path $PythonExe)) {
    throw "No virtual environment Python found. Create .venv or venv in the project root first."
}

& $PythonExe -m PyInstaller --version | Out-Null
if ($LASTEXITCODE -ne 0) {
    & $PythonExe -m pip install pyinstaller
}

& $PythonExe -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name $Name `
    --distpath (Join-Path $ProjectRoot "dist") `
    --workpath (Join-Path $ProjectRoot "build") `
    --specpath (Join-Path $ProjectRoot "build") `
    (Join-Path $ProjectRoot "desktop_app.py")

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
}

Write-Host "Built dist\\$Name.exe"