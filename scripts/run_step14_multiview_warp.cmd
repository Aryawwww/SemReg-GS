@echo off
setlocal
cd /d "%~dp0.."

set "BLENDER=blender"
where blender >nul 2>nul
if errorlevel 1 (
  if exist "%USERPROFILE%\scoop\shims\blender.exe" (
    set "BLENDER=%USERPROFILE%\scoop\shims\blender.exe"
  ) else (
    echo ERROR: blender is not available on PATH or in the Scoop shim directory.
    exit /b 1
  )
)

where conda >nul 2>nul
if errorlevel 1 (
  echo ERROR: conda is not available. Run this script from Anaconda Prompt.
  exit /b 1
)

set "SOURCE_ID=107734119_175999932"
set "TARGET_ID=103997424_171030444"
set "PAIR_ID=%SOURCE_ID%__to__%TARGET_ID%"
set "PROTOCOL=data\processed\pairs\%PAIR_ID%\protocol_manifest.json"
set "TARGET_VIEWS=outputs\pairs\%PAIR_ID%\target_multiview\target_views"
set "RESULTS=outputs\pairs\%PAIR_ID%\cross_geometry_2d"

"%BLENDER%" -b --python-exit-code 1 --python scripts\export_depth_exr_blender.py -- --views "%TARGET_VIEWS%" --view-names view_00 view_07 --output "%RESULTS%\heldout_depth.npz"
if errorlevel 1 exit /b 1

call conda run -n semreg-gs-v1 python scripts\evaluate_multiview_warp.py --protocol-manifest "%PROTOCOL%" --target-views "%TARGET_VIEWS%" --renders-root "%RESULTS%\renders" --depth-cache "%RESULTS%\heldout_depth.npz" --output "%RESULTS%\warp_metrics.json" --occlusion-tolerance 0.05 --minimum-correspondences 100
if errorlevel 1 exit /b 1

echo Multi-view warp evaluation completed.
echo Metrics: %RESULTS%\warp_metrics.json
