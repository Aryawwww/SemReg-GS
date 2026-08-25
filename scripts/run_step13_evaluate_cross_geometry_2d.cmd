@echo off
setlocal
cd /d "%~dp0.."

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

call conda run -n semreg-gs-v1 python scripts\evaluate_cross_geometry_2d.py --protocol-manifest "%PROTOCOL%" --target-views "%TARGET_VIEWS%" --renders-root "%RESULTS%\renders" --appearance-root "%RESULTS%\appearance" --semantic-mapping "data\processed\targets\%TARGET_ID%\semantic_mapping.json" --gaussian-validation "data\processed\semantic_gaussians\%TARGET_ID%\validation.json" --output "%RESULTS%\metrics.json"
if errorlevel 1 exit /b 1

echo Cross-geometry held-out evaluation completed successfully.
echo Metrics: %RESULTS%\metrics.json
