@echo off
setlocal
cd /d "%~dp0.."

where conda >nul 2>nul
if errorlevel 1 exit /b 1

call conda run -n semreg-gs-v1 python scripts\freeze_coverage_split.py --audit outputs\smoke\107734119_175999932\coverage_audit.json --multiview outputs\smoke\107734119_175999932\multiview --window-manifest outputs\smoke\107734119_175999932\multiview\window_view_manifest.json --scene-id 107734119_175999932 --output data\splits\107734119_175999932_split.json
if errorlevel 1 exit /b 1

call conda run -n semreg-gs-v1 python scripts\prepare_cross_geometry_pair.py --source-scene-id 107734119_175999932 --source-split data\splits\107734119_175999932_split.json --selection-seed 42
if errorlevel 1 exit /b 1

echo Frozen split and downloaded a second HSSD scene.
echo Read data\processed\pairs\^<source^>__to__^<target^>\pair_manifest.json for the selected target scene ID.
