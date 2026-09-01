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
set "OUTPUT=%PAIR_OUTPUT%\semantic_alignment_audit"

call conda run -n "%CONDA_ENV%" python scripts\audit_rendered_semantic_alignment.py --protocol-manifest "data\processed\pairs\%PAIR_ID%\protocol_consistency_v2.json" --target-views "%PAIR_OUTPUT%\target_multiview\target_views" --renders "%PAIR_OUTPUT%\cross_geometry_2d\renders\semantic_2d" --semantic-mapping "data\processed\targets\%TARGET_ID%\semantic_mapping.json" --output "%OUTPUT%\metrics.json" --visualizations "%OUTPUT%\visualizations"
if errorlevel 1 exit /b 1

echo Semantic alignment audit completed.
echo Metrics: %OUTPUT%\metrics.json
echo Visualizations: %OUTPUT%\visualizations
