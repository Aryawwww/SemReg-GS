@echo off
setlocal
cd /d "%~dp0.."

where conda >nul 2>nul
if errorlevel 1 exit /b 1

set "CONDA_ENV=semreg-gs"
if not "%~1"=="" set "CONDA_ENV=%~1"
set "SOURCE_ID=107734119_175999932"
set "TARGET_ID=103997424_171030444"
set "PAIR_ID=%SOURCE_ID%__to__%TARGET_ID%"
set "PAIR_OUTPUT=outputs\pairs\%PAIR_ID%"

call conda run -n "%CONDA_ENV%" python scripts\decompose_intervention_spill.py --protocol-manifest "data\processed\pairs\%PAIR_ID%\protocol_consistency_v2.json" --target-views "%PAIR_OUTPUT%\target_multiview\target_views" --semantic-mapping "data\processed\targets\%TARGET_ID%\semantic_mapping.json" --base-renders-2d "%PAIR_OUTPUT%\cross_geometry_2d\renders" --base-renders-dino "%PAIR_OUTPUT%\cross_geometry_dino\renders" --edited-renders "%PAIR_OUTPUT%\semantic_interventions\renders" --output "%PAIR_OUTPUT%\semantic_interventions\spill_decomposition.json"
if errorlevel 1 exit /b 1

echo Spill decomposition completed.
echo Output: %PAIR_OUTPUT%\semantic_interventions\spill_decomposition.json
