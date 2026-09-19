$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
if (-not $env:MUJOCO_GL) { $env:MUJOCO_GL = "wgl" }
$env:MPLBACKEND = "Agg"
$extra = @(
    (Join-Path $Root "src"),
    (Join-Path $Root "cloud\libero\transcoder_runtime")
)
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = ($extra + $env:PYTHONPATH) -join ";"
} else {
    $env:PYTHONPATH = $extra -join ";"
}
Set-Location $Root
$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
& $python -u (Join-Path $Root "run\pi0.5\infer.py") @args
exit $LASTEXITCODE
