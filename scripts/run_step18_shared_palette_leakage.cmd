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
set "PAIR_DATA=data\processed\pairs\%PAIR_ID%"
set "PAIR_OUTPUT=outputs\pairs\%PAIR_ID%"
set "PROTOCOL=%PAIR_DATA%\protocol_consistency_v2.json"
set "DONORS=outputs\smoke\%SOURCE_ID%\multiview\donor_reference"
set "TARGET_VIEWS=%PAIR_OUTPUT%\target_multiview\target_views"
set "MAPPING=data\processed\targets\%TARGET_ID%\semantic_mapping.json"
set "PALETTE=%PAIR_DATA%\frozen_reference_palette.json"

call conda run -n "%CONDA_ENV%" python scripts\build_frozen_reference_palette.py --protocol-manifest "%PROTOCOL%" --donor-views "%DONORS%" --semantic-mapping "%MAPPING%" --output "%PALETTE%"
if errorlevel 1 exit /b 1

call conda run -n "%CONDA_ENV%" python scripts\evaluate_shared_palette_leakage.py --protocol-manifest "%PROTOCOL%" --target-views "%TARGET_VIEWS%" --semantic-mapping "%MAPPING%" --reference-palette "%PALETTE%" --renders-2d "%PAIR_OUTPUT%\cross_geometry_2d\renders" --renders-dino "%PAIR_OUTPUT%\cross_geometry_dino\renders" --output "%PAIR_OUTPUT%\shared_palette_leakage.json"
if errorlevel 1 exit /b 1

echo Shared-palette leakage evaluation completed.
echo Palette: %PALETTE%
echo Metrics: %PAIR_OUTPUT%\shared_palette_leakage.json
