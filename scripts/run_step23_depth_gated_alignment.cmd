@echo off
setlocal
cd /d "%~dp0.."

where conda >nul 2>nul
if errorlevel 1 exit /b 1
where blender >nul 2>nul
if errorlevel 1 (
  echo ERROR: blender is not available.
  exit /b 1
)

set "CONDA_ENV=semreg-gs"
if not "%~1"=="" set "CONDA_ENV=%~1"
set "SOURCE_ID=107734119_175999932"
set "TARGET_ID=103997424_171030444"
set "PAIR_ID=%SOURCE_ID%__to__%TARGET_ID%"
set "PAIR_OUTPUT=outputs\pairs\%PAIR_ID%"
set "TARGET_VIEWS=%PAIR_OUTPUT%\target_multiview\target_views"
set "OUTPUT=%PAIR_OUTPUT%\depth_gated_alignment"
set "DEPTH=%OUTPUT%\heldout_mesh_depth.npz"
set "PROTOCOL=data\processed\pairs\%PAIR_ID%\protocol_consistency_v2.json"
set "MAPPING=data\processed\targets\%TARGET_ID%\semantic_mapping.json"
set "GAUSSIANS=data\processed\semantic_gaussians\%TARGET_ID%\gaussians.npz"

blender -b --python-exit-code 1 --python scripts\export_depth_exr_blender.py -- --views "%TARGET_VIEWS%" --view-names view_00 view_07 --output "%DEPTH%"
if errorlevel 1 exit /b 1

call conda run -n "%CONDA_ENV%" python scripts\render_semantic_gaussians.py --gaussians "%GAUSSIANS%" --appearance "%PAIR_OUTPUT%\cross_geometry_2d\appearance\semantic_2d\appearance.npz" --views "%TARGET_VIEWS%" --semantic-mapping "%MAPPING%" --protocol-manifest "%PROTOCOL%" --protocol-split target_heldout --output "%OUTPUT%\renders\semantic_2d" --mode both --point-radius 1 --depth-cache "%DEPTH%" --occlusion-tolerance 0.05 --device cuda
if errorlevel 1 exit /b 1

call conda run -n "%CONDA_ENV%" python scripts\audit_rendered_semantic_alignment.py --protocol-manifest "%PROTOCOL%" --target-views "%TARGET_VIEWS%" --renders "%OUTPUT%\renders\semantic_2d" --semantic-mapping "%MAPPING%" --output "%OUTPUT%\alignment_metrics.json" --visualizations "%OUTPUT%\visualizations"
if errorlevel 1 exit /b 1

echo Depth-gated semantic alignment completed.
echo Metrics: %OUTPUT%\alignment_metrics.json
