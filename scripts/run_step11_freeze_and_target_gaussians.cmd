@echo off
setlocal
cd /d "%~dp0.."

where conda >nul 2>nul
if errorlevel 1 (
  echo ERROR: conda is not available. Run this script from Anaconda Prompt.
  exit /b 1
)

where blender >nul 2>nul
if errorlevel 1 (
  echo ERROR: blender is not available on PATH.
  exit /b 1
)

set "SOURCE_ID=107734119_175999932"
set "TARGET_ID=103997424_171030444"
set "PAIR_ID=%SOURCE_ID%__to__%TARGET_ID%"
set "PAIR_DATA=data\processed\pairs\%PAIR_ID%"
set "PAIR_OUTPUT=outputs\pairs\%PAIR_ID%"
set "TARGET_RAW=data\raw\hssd\%TARGET_ID%"
set "TARGET_PROCESSED=data\processed\targets\%TARGET_ID%"
set "GAUSSIANS=data\processed\semantic_gaussians\%TARGET_ID%"

call conda run -n semreg-gs-v1 python scripts\freeze_cross_geometry_protocol.py --pair-manifest "%PAIR_DATA%\pair_manifest.json" --source-split "data\splits\%SOURCE_ID%_split.json" --coverage-audit "%PAIR_OUTPUT%\target_multiview\coverage_audit.json" --source-multiview "outputs\smoke\%SOURCE_ID%\multiview" --target-multiview "%PAIR_OUTPUT%\target_multiview" --output "%PAIR_DATA%\protocol_manifest.json"
if errorlevel 1 exit /b 1

blender -b --python-exit-code 1 --python scripts\mesh_to_semantic_gaussians.py -- --mesh "%TARGET_RAW%\scene.glb" --semantic-mapping "%TARGET_PROCESSED%\semantic_mapping.json" --output "%GAUSSIANS%" --max-gaussians 100000 --seed 42 --freeze-position --freeze-rotation --freeze-scale
if errorlevel 1 exit /b 1

call conda run -n semreg-gs-v1 python scripts\validate_semantic_gaussians.py --input "%GAUSSIANS%" --source-mesh "%TARGET_RAW%\scene.glb" --report "%GAUSSIANS%\validation.json" --max-distance 0.00001
if errorlevel 1 exit /b 1

echo Cross-geometry protocol and target Gaussians completed successfully.
echo Protocol: %PAIR_DATA%\protocol_manifest.json
echo Validation: %GAUSSIANS%\validation.json
