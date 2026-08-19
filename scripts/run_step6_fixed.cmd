@echo off
setlocal
cd /d "%~dp0.."

where conda >nul 2>nul
if errorlevel 1 (
  echo ERROR: conda is not available. Run this script from Anaconda Prompt.
  exit /b 1
)

call conda activate semreg-gs-v1
if errorlevel 1 exit /b 1

where blender >nul 2>nul
if errorlevel 1 (
  echo ERROR: blender is not available on PATH.
  exit /b 1
)

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print(torch.__version__, torch.cuda.get_device_name(0))"
if errorlevel 1 exit /b 1

blender -b --python scripts\render_hssd_multiview.py -- --input data\raw\hssd\107734119_175999932\scene.glb --semantic-config data\raw\hssd\107734119_175999932\semantic_config.json --output outputs\smoke\107734119_175999932\multiview --width 512 --height 512
if errorlevel 1 exit /b 1

python scripts\train_neutral_gaussians.py --gaussians data\processed\semantic_gaussians\107734119_175999932\gaussians.npz --views outputs\smoke\107734119_175999932\multiview\target_views --output outputs\smoke\107734119_175999932\neutral_gaussians_fixed --steps 300 --learning-rate 0.05 --visibility-tolerance 0.03 --minimum-observed-fraction 0.50 --seed 42 --device cuda
if errorlevel 1 exit /b 1

python scripts\render_semantic_gaussians.py --gaussians data\processed\semantic_gaussians\107734119_175999932\gaussians.npz --appearance outputs\smoke\107734119_175999932\neutral_gaussians_fixed\appearance.npz --views outputs\smoke\107734119_175999932\multiview\target_views --semantic-mapping data\processed\targets\107734119_175999932\semantic_mapping.json --output outputs\smoke\107734119_175999932\neutral_gaussians_fixed\renders --mode both --point-radius 1 --device cuda
if errorlevel 1 exit /b 1

echo Step 6 fixed pipeline completed successfully.
echo Training report: outputs\smoke\107734119_175999932\neutral_gaussians_fixed\training_report.json
echo Render report: outputs\smoke\107734119_175999932\neutral_gaussians_fixed\renders\render_report.json
