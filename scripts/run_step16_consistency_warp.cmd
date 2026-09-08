@echo off
setlocal
cd /d "%~dp0.."

where conda >nul 2>nul
if errorlevel 1 exit /b 1

set "BLENDER=blender"
where blender >nul 2>nul
if errorlevel 1 set "BLENDER=%USERPROFILE%\scoop\shims\blender.exe"

set "SOURCE_ID=107734119_175999932"
set "TARGET_ID=103997424_171030444"
set "PAIR_ID=%SOURCE_ID%__to__%TARGET_ID%"
set "PROTOCOL=data\processed\pairs\%PAIR_ID%\protocol_consistency_v2.json"
set "VIEWS=outputs\pairs\%PAIR_ID%\consistency_multiview\target_views"
set "BASE=outputs\pairs\%PAIR_ID%\cross_geometry_2d"
set "GAUSSIANS=data\processed\semantic_gaussians\%TARGET_ID%\gaussians.npz"
set "MAPPING=data\processed\targets\%TARGET_ID%\semantic_mapping.json"

for %%M in (global semantic_2d) do (
  call conda run -n semreg-gs python scripts\render_semantic_gaussians.py --gaussians "%GAUSSIANS%" --appearance "%BASE%\appearance\%%M\appearance.npz" --views "%VIEWS%" --semantic-mapping "%MAPPING%" --protocol-manifest "%PROTOCOL%" --protocol-split target_consistency --output "%BASE%\consistency_renders\%%M" --mode both --point-radius 1 --device cuda
  if errorlevel 1 exit /b 1
)

"%BLENDER%" -b --python-exit-code 1 --python scripts\export_depth_exr_blender.py -- --views "%VIEWS%" --view-names consistency_00 consistency_01 --output "%BASE%\consistency_depth.npz"
if errorlevel 1 exit /b 1

call conda run -n semreg-gs python scripts\evaluate_multiview_warp.py --protocol-manifest "%PROTOCOL%" --protocol-split target_consistency --target-views "%VIEWS%" --renders-root "%BASE%\consistency_renders" --depth-cache "%BASE%\consistency_depth.npz" --output "%BASE%\consistency_warp_metrics.json" --occlusion-tolerance 0.05 --minimum-correspondences 100
if errorlevel 1 exit /b 1

echo Consistency warp evaluation completed.
echo Metrics: %BASE%\consistency_warp_metrics.json
