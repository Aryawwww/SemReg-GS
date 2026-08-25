@echo off
setlocal
cd /d "%~dp0.."

where conda >nul 2>nul
if errorlevel 1 exit /b 1

conda run -n semreg-gs-v1 python scripts\build_step7_baselines.py --gaussians data\processed\semantic_gaussians\107734119_175999932\gaussians.npz --donor-views outputs\smoke\107734119_175999932\multiview\donor_reference --semantic-mapping data\processed\targets\107734119_175999932\semantic_mapping.json --output outputs\smoke\107734119_175999932\step7_baselines
if errorlevel 1 exit /b 1

conda run -n semreg-gs-v1 python scripts\render_semantic_gaussians.py --gaussians data\processed\semantic_gaussians\107734119_175999932\gaussians.npz --appearance outputs\smoke\107734119_175999932\step7_baselines\global\appearance.npz --views outputs\smoke\107734119_175999932\multiview\target_views --semantic-mapping data\processed\targets\107734119_175999932\semantic_mapping.json --output outputs\smoke\107734119_175999932\step7_baselines\global\renders --mode both --point-radius 1 --device cuda
if errorlevel 1 exit /b 1

conda run -n semreg-gs-v1 python scripts\render_semantic_gaussians.py --gaussians data\processed\semantic_gaussians\107734119_175999932\gaussians.npz --appearance outputs\smoke\107734119_175999932\step7_baselines\semantic_2d\appearance.npz --views outputs\smoke\107734119_175999932\multiview\target_views --semantic-mapping data\processed\targets\107734119_175999932\semantic_mapping.json --output outputs\smoke\107734119_175999932\step7_baselines\semantic_2d\renders --mode both --point-radius 1 --device cuda
if errorlevel 1 exit /b 1

conda run -n semreg-gs-v1 python scripts\evaluate_step7_baselines.py --methods-root outputs\smoke\107734119_175999932\step7_baselines --target-views outputs\smoke\107734119_175999932\multiview\target_views --semantic-mapping data\processed\targets\107734119_175999932\semantic_mapping.json --output outputs\smoke\107734119_175999932\step7_baselines\metrics.json
if errorlevel 1 exit /b 1

echo Step 7 Global and B_sem-2D baselines completed successfully.
echo Metrics: outputs\smoke\107734119_175999932\step7_baselines\metrics.json
