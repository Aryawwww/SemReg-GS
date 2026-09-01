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
set "PROTOCOL=data\processed\pairs\%PAIR_ID%\protocol_consistency_v2.json"
set "GAUSSIANS=data\processed\semantic_gaussians\%TARGET_ID%\gaussians.npz"
set "MAPPING=data\processed\targets\%TARGET_ID%\semantic_mapping.json"
set "PAIR_OUTPUT=outputs\pairs\%PAIR_ID%"
set "TARGET_VIEWS=%PAIR_OUTPUT%\target_multiview\target_views"
set "OUTPUT=%PAIR_OUTPUT%\semantic_interventions"

call conda run -n "%CONDA_ENV%" python scripts\build_semantic_interventions.py --protocol-manifest "%PROTOCOL%" --gaussians "%GAUSSIANS%" --semantic-mapping "%MAPPING%" --appearance-2d "%PAIR_OUTPUT%\cross_geometry_2d\appearance" --appearance-dino "%PAIR_OUTPUT%\cross_geometry_dino\appearance" --output "%OUTPUT%\appearance" --delta 0.20 -0.15 0.10
if errorlevel 1 exit /b 1

for %%M in (global semantic_2d global_dino semantic_dino) do (
  for %%C in (wall floor ceiling door window other) do (
    call conda run -n "%CONDA_ENV%" python scripts\render_semantic_gaussians.py --gaussians "%GAUSSIANS%" --appearance "%OUTPUT%\appearance\%%M\%%C\appearance.npz" --views "%TARGET_VIEWS%" --semantic-mapping "%MAPPING%" --protocol-manifest "%PROTOCOL%" --protocol-split target_heldout --output "%OUTPUT%\renders\%%M\%%C" --mode appearance --point-radius 1 --device cuda
    if errorlevel 1 exit /b 1
  )
)

call conda run -n "%CONDA_ENV%" python scripts\evaluate_semantic_interventions.py --protocol-manifest "%PROTOCOL%" --target-views "%TARGET_VIEWS%" --semantic-mapping "%MAPPING%" --base-renders-2d "%PAIR_OUTPUT%\cross_geometry_2d\renders" --base-renders-dino "%PAIR_OUTPUT%\cross_geometry_dino\renders" --edited-renders "%OUTPUT%\renders" --intervention-manifest "%OUTPUT%\appearance\intervention_manifest.json" --output "%OUTPUT%\metrics.json"
if errorlevel 1 exit /b 1

echo Semantic intervention evaluation completed.
echo Metrics: %OUTPUT%\metrics.json
