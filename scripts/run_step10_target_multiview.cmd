@echo off
setlocal
cd /d "%~dp0.."

where blender >nul 2>nul
if errorlevel 1 (
  echo ERROR: blender is not available on PATH.
  exit /b 1
)

set "TARGET_ID=103997424_171030444"
set "RAW=data\raw\hssd\%TARGET_ID%"
set "PLAN=configs\camera_plans\%TARGET_ID%.json"
set "OUTPUT=outputs\pairs\107734119_175999932__to__%TARGET_ID%\target_multiview"

blender -b --python-exit-code 1 --python scripts\render_hssd_multiview.py -- --input "%RAW%\scene.glb" --semantic-config "%RAW%\semantic_config.json" --camera-plan "%PLAN%" --output "%OUTPUT%" --width 512 --height 512
if errorlevel 1 exit /b 1

echo Target multiview render completed successfully.
echo Render manifest: %OUTPUT%\render_manifest.json
