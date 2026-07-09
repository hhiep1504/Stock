param(
    [switch]$Quick,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$UnixVenvPython = Join-Path $ProjectRoot ".venv\bin\python"

if (Test-Path $VenvPython) {
    $PythonBin = $VenvPython
} elseif (Test-Path $UnixVenvPython) {
    $PythonBin = $UnixVenvPython
} else {
    $PythonBin = "python"
}

$ExtraArgs = @()
if ($Quick) {
    $ExtraArgs = @(
        "--epochs", "5",
        "--runs", "1",
        "--max-candidates", "8",
        "--top-candidates", "8",
        "--max-greedy-features", "3"
    )
}

Write-Host "[feature-window-sweep] PROJECT_ROOT=$ProjectRoot"
Write-Host "[feature-window-sweep] PYTHON_BIN=$PythonBin"

& $PythonBin (Join-Path $ProjectRoot "scripts\sweep_feature_windows_weekly.py") `
    --device auto `
    @ExtraArgs `
    @RemainingArgs
