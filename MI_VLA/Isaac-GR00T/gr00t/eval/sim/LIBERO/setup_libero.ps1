# Windows port of setup_libero.sh — creates a dedicated sim venv for LIBERO client.
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LiberoRepo = Resolve-Path (Join-Path $ScriptDir "..\..\..\..\external_dependencies\LIBERO")
$ProjectRepo = Resolve-Path (Join-Path $ScriptDir "..\..\..\..")
$LiberoUvEnv = Join-Path $ScriptDir "libero_uv"
$VenvPython = Join-Path $LiberoUvEnv ".venv\Scripts\python.exe"
$VenvPip = Join-Path $LiberoUvEnv ".venv\Scripts\pip.exe"

Write-Host "LIBERO repo: $LiberoRepo"
Write-Host "Project repo: $ProjectRepo"
Write-Host "Sim env: $LiberoUvEnv"

if (-not (Test-Path (Join-Path $LiberoRepo "requirements.txt"))) {
    throw "LIBERO checkout missing files under $LiberoRepo"
}

if (Test-Path $LiberoUvEnv) {
    Remove-Item -Recurse -Force $LiberoUvEnv
}
New-Item -ItemType Directory -Path $LiberoUvEnv | Out-Null

# Must be Python 3.12 so numpy/mujoco wheels exist (base Anaconda is often 3.13).
function Find-GrootPython {
    if ($env:CONDA_DEFAULT_ENV -eq "groot" -and $env:CONDA_PREFIX) {
        $p = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path $p) { return $p }
    }
    foreach ($root in @(
        "$env:USERPROFILE\anaconda3",
        "$env:USERPROFILE\miniconda3",
        "$env:LOCALAPPDATA\anaconda3",
        "$env:LOCALAPPDATA\miniconda3",
        "C:\ProgramData\anaconda3",
        "C:\ProgramData\miniconda3"
    )) {
        $p = Join-Path $root "envs\groot\python.exe"
        if (Test-Path $p) { return $p }
    }
    return $null
}
$Py = Find-GrootPython
if (-not $Py) {
    throw "Need conda env groot (Python 3.12). Create it first, or activate it and re-run."
}

Write-Host "Creating venv with: $Py"
& $Py -m venv (Join-Path $LiberoUvEnv ".venv")

$PatchedRequirements = Join-Path $LiberoUvEnv "requirements-py312.txt"
$Replacements = @{
    "hydra-core"     = "hydra-core==1.3.2"
    "numpy"          = "numpy==1.26.4"
    "transformers"   = "transformers==4.57.3"
    "opencv-python"  = "opencv-python==4.10.0.84"
    "matplotlib"     = "matplotlib==3.9.4"
    "wandb"          = "wandb==0.18.7"
}

$lines = Get-Content (Join-Path $LiberoRepo "requirements.txt")
$out = foreach ($raw in $lines) {
    $stripped = $raw.Trim()
    if (-not $stripped -or $stripped.StartsWith("#")) {
        $raw
        continue
    }
    $name = ($stripped -split "==", 2)[0].Trim().ToLower()
    if ($Replacements.ContainsKey($name)) { $Replacements[$name] } else { $raw }
}
$out | Set-Content -Encoding utf8 $PatchedRequirements

Write-Host "Installing LIBERO requirements (skip Linux-only egl_probe)..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install --only-binary=:all: numpy==1.26.4
# Core LIBERO deps without robomimic's egl_probe (Linux-only native build).
& $VenvPython -m pip install `
    hydra-core==1.3.2 easydict==1.9 transformers==4.57.3 `
    opencv-python==4.10.0.84 robosuite==1.4.0 bddl==1.0.1 `
    future==0.18.2 matplotlib==3.9.4 cloudpickle==2.1.0 gym==0.25.2 `
    einops==0.8.1 thop==0.1.1.post2209072238 wandb==0.18.7 `
    termcolor imageio imageio-ffmpeg h5py
& $VenvPython -m pip install robomimic==0.2.0 --no-deps
& $VenvPython -m pip install -e $LiberoRepo --config-settings editable_mode=compat

Write-Host "Installing GR00T eval client deps + MuJoCo (Windows uses WGL)..."
& $VenvPython -m pip install `
    torch==2.9.0 torchvision==0.24.0 `
    pydantic av tianshou==0.5.1 `
    numba==0.65.1 llvmlite==0.47.0 `
    tyro pandas dm_tree einops==0.8.1 `
    albumentations==1.4.18 pyzmq `
    transformers==4.57.3 msgpack==1.1.0 msgpack-numpy==0.4.8 `
    gymnasium==0.29.1 numpy==1.26.4 mujoco==3.3.1 glfw

# Windows robosuite quirks: copy mujoco.dll and force WGL instead of EGL.
$site = & $VenvPython -c "import site; print(site.getsitepackages()[-1])"
$mujocoDll = Get-ChildItem -Path (Join-Path $site "mujoco") -Filter mujoco.dll -Recurse | Select-Object -First 1
if ($null -eq $mujocoDll) { throw "mujoco.dll not found after mujoco install" }
Copy-Item $mujocoDll.FullName (Join-Path $site "robosuite\utils\mujoco.dll") -Force
$binding = Join-Path $site "robosuite\utils\binding_utils.py"
$bindingTxt = Get-Content $binding -Raw
$bindingPatched = $bindingTxt -replace 'os\.environ\["MUJOCO_GL"\] = "egl"', 'os.environ["MUJOCO_GL"] = "wgl"'
Set-Content -Path $binding -Value $bindingPatched -NoNewline
New-Item -ItemType Directory -Force C:\tmp | Out-Null

# Expose gr00t from the repo root via a .pth file (no re-resolving groot deps).
& $VenvPython -c @"
import sysconfig, pathlib
purelib = pathlib.Path(sysconfig.get_path('purelib'))
purelib.joinpath('gr00t.pth').write_text(r'$ProjectRepo' + chr(10), encoding='utf-8')
print('Wrote', purelib / 'gr00t.pth')
"@

$env:MUJOCO_GL = "wgl"

Write-Host "Registering / smoke-testing LIBERO env..."
cmd /c "echo n| `"$VenvPython`" -c `"import os; os.environ['MUJOCO_GL']='wgl'; from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs; register_libero_envs(); import gymnasium as gym; env=gym.make('libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate'); env.reset(); env.close(); print('Env OK', type(env))`""

Write-Host ""
Write-Host "Setup done."
Write-Host "Sim python: $VenvPython"
Write-Host "Next: start policy server in conda groot, then run rollout with this python."
