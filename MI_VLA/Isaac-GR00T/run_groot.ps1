# Activate conda env + ensure FFmpeg 7 shared DLLs are first on PATH.
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Command
)

$FfmpegBin = "C:\Users\lijun\tools\ffmpeg\ffmpeg-n7.1-latest-win64-gpl-shared-7.1\bin"
if (Test-Path $FfmpegBin) {
    $env:PATH = "$FfmpegBin;" + $env:PATH
}

& conda activate groot
if ($Command.Count -eq 0) {
    Write-Host "groot env ready. Example:"
    Write-Host '  python scripts/deployment/standalone_inference_script.py --model-path nvidia/GR00T-N1.7-3B --dataset-path demo_data/droid_sample --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT --traj-ids 1 2 --inference-mode pytorch --execution-horizon 8'
} else {
    & $Command[0] @($Command[1..($Command.Count - 1)])
}
