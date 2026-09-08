@echo off
setlocal
cd /d "%~dp0.."

where conda >nul 2>nul
if errorlevel 1 (
  echo ERROR: conda is not available. Run this script from Anaconda Prompt.
  exit /b 1
)

set "BLENDER=blender"
where blender >nul 2>nul
if errorlevel 1 (
  if exist "%USERPROFILE%\scoop\shims\blender.exe" (
    set "BLENDER=%USERPROFILE%\scoop\shims\blender.exe"
  ) else (
    echo ERROR: blender is not available.
    exit /b 1
  )
)

set "SOURCE_ID=107734119_175999932"
set "TARGET_ID=103997424_171030444"
set "PAIR_ID=%SOURCE_ID%__to__%TARGET_ID%"
set "RAW=data\raw\hssd\%TARGET_ID%"
set "PLAN=configs\camera_plans\%TARGET_ID%_consistency.json"
set "OUTPUT=outputs\pairs\%PAIR_ID%\consistency_multiview"
set "PAIR_DATA=data\processed\pairs\%PAIR_ID%"

"%BLENDER%" -b --python-exit-code 1 --python scripts\render_hssd_multiview.py -- --input "%RAW%\scene.glb" --semantic-config "%RAW%\semantic_config.json" --camera-plan "%PLAN%" --output "%OUTPUT%" --width 512 --height 512
if errorlevel 1 exit /b 1

call conda run -n semreg-gs python scripts\freeze_consistency_protocol.py --base-protocol "%PAIR_DATA%\protocol_manifest.json" --consistency-multiview "%OUTPUT%" --render-manifest "%OUTPUT%\render_manifest.json" --output "%PAIR_DATA%\protocol_consistency_v2.json"
if errorlevel 1 exit /b 1

echo Consistency views rendered and protocol v2 frozen.
echo Protocol: %PAIR_DATA%\protocol_consistency_v2.json
