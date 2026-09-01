@echo off
setlocal
cd /d "%~dp0.."

where conda >nul 2>nul
if errorlevel 1 (
  echo ERROR: conda is not available.
  exit /b 1
)

set "CONDA_ENV=semreg-gs"
if not "%~1"=="" set "CONDA_ENV=%~1"

set "SOURCE_ID=107734119_175999932"
set "TARGET_ID=103997424_171030444"
set "PAIR_ID=%SOURCE_ID%__to__%TARGET_ID%"
set "PROTOCOL=data\processed\pairs\%PAIR_ID%\protocol_consistency_v2.json"
set "GAUSSIANS=data\processed\semantic_gaussians\%TARGET_ID%\gaussians.npz"
set "DONORS=outputs\smoke\%SOURCE_ID%\multiview\donor_reference"
set "TARGET_VIEWS=outputs\pairs\%PAIR_ID%\target_multiview\target_views"
set "CONSISTENCY_VIEWS=outputs\pairs\%PAIR_ID%\consistency_multiview\target_views"
set "MAPPING=data\processed\targets\%TARGET_ID%\semantic_mapping.json"
set "OUTPUT=outputs\pairs\%PAIR_ID%\cross_geometry_dino"
set "DINO_REPO=%USERPROFILE%\.cache\torch\hub\facebookresearch_dinov2_main"
set "DINO_WEIGHTS=%USERPROFILE%\.cache\torch\hub\checkpoints\dinov2_vits14_pretrain.pth"

call conda run -n "%CONDA_ENV%" python scripts\build_source_only_dino.py --gaussians "%GAUSSIANS%" --donor-views "%DONORS%" --semantic-mapping "%MAPPING%" --protocol-manifest "%PROTOCOL%" --dino-repo "%DINO_REPO%" --dino-weights "%DINO_WEIGHTS%" --output "%OUTPUT%\appearance" --device cuda
if errorlevel 1 exit /b 1

for %%M in (global_dino semantic_dino) do (
  call conda run -n "%CONDA_ENV%" python scripts\render_semantic_gaussians.py --gaussians "%GAUSSIANS%" --appearance "%OUTPUT%\appearance\%%M\appearance.npz" --views "%TARGET_VIEWS%" --semantic-mapping "%MAPPING%" --protocol-manifest "%PROTOCOL%" --protocol-split target_heldout --output "%OUTPUT%\renders\%%M" --mode both --point-radius 1 --device cuda
  if errorlevel 1 exit /b 1
  call conda run -n "%CONDA_ENV%" python scripts\render_semantic_gaussians.py --gaussians "%GAUSSIANS%" --appearance "%OUTPUT%\appearance\%%M\appearance.npz" --views "%CONSISTENCY_VIEWS%" --semantic-mapping "%MAPPING%" --protocol-manifest "%PROTOCOL%" --protocol-split target_consistency --output "%OUTPUT%\consistency_renders\%%M" --mode both --point-radius 1 --device cuda
  if errorlevel 1 exit /b 1
)

call conda run -n "%CONDA_ENV%" python scripts\evaluate_cross_geometry_2d.py --protocol-manifest "%PROTOCOL%" --target-views "%TARGET_VIEWS%" --renders-root "%OUTPUT%\renders" --appearance-root "%OUTPUT%\appearance" --semantic-mapping "%MAPPING%" --gaussian-validation "data\processed\semantic_gaussians\%TARGET_ID%\validation.json" --methods global_dino semantic_dino --output "%OUTPUT%\metrics.json"
if errorlevel 1 exit /b 1

call conda run -n "%CONDA_ENV%" python scripts\evaluate_multiview_warp.py --protocol-manifest "%PROTOCOL%" --protocol-split target_consistency --target-views "%CONSISTENCY_VIEWS%" --renders-root "%OUTPUT%\consistency_renders" --depth-cache "outputs\pairs\%PAIR_ID%\cross_geometry_2d\consistency_depth.npz" --methods global_dino semantic_dino --output "%OUTPUT%\consistency_warp_metrics.json" --occlusion-tolerance 0.05 --minimum-correspondences 100
if errorlevel 1 exit /b 1

call conda run -n "%CONDA_ENV%" python scripts\merge_cross_geometry_pilot.py --protocol-manifest "%PROTOCOL%" --metrics-2d "outputs\pairs\%PAIR_ID%\cross_geometry_2d\metrics.json" --warp-2d "outputs\pairs\%PAIR_ID%\cross_geometry_2d\consistency_warp_metrics.json" --metrics-dino "%OUTPUT%\metrics.json" --warp-dino "%OUTPUT%\consistency_warp_metrics.json" --dino-appearance-root "%OUTPUT%\appearance" --output "outputs\pairs\%PAIR_ID%\pilot_four_method_metrics.json"
if errorlevel 1 exit /b 1

echo Source-only DINO cross-geometry evaluation completed.
echo Output: %OUTPUT%
echo Combined pilot: outputs\pairs\%PAIR_ID%\pilot_four_method_metrics.json
