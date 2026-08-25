@echo off
setlocal
cd /d "%~dp0.."

where conda >nul 2>nul
if errorlevel 1 exit /b 1

conda run -n semreg-gs-v1 python scripts\train_dino_gaussian_decoder.py --method global_dino --gaussians data\processed\semantic_gaussians\107734119_175999932\gaussians.npz --donor-views outputs\smoke\107734119_175999932\multiview\donor_reference --target-views outputs\smoke\107734119_175999932\multiview\target_views --semantic-mapping data\processed\targets\107734119_175999932\semantic_mapping.json --output outputs\smoke\107734119_175999932\step7_baselines\global_dino --dino-model dinov2_vits14 --train-view-count 6 --steps 1000 --batch-size 8192 --hidden-dim 128 --learning-rate 0.001 --device cuda
if errorlevel 1 exit /b 1

conda run -n semreg-gs-v1 python scripts\train_dino_gaussian_decoder.py --method semantic_dino --gaussians data\processed\semantic_gaussians\107734119_175999932\gaussians.npz --donor-views outputs\smoke\107734119_175999932\multiview\donor_reference --target-views outputs\smoke\107734119_175999932\multiview\target_views --semantic-mapping data\processed\targets\107734119_175999932\semantic_mapping.json --output outputs\smoke\107734119_175999932\step7_baselines\semantic_dino --dino-model dinov2_vits14 --train-view-count 6 --steps 1000 --batch-size 8192 --hidden-dim 128 --learning-rate 0.001 --device cuda
if errorlevel 1 exit /b 1

conda run -n semreg-gs-v1 python scripts\render_semantic_gaussians.py --gaussians data\processed\semantic_gaussians\107734119_175999932\gaussians.npz --appearance outputs\smoke\107734119_175999932\step7_baselines\global_dino\appearance.npz --views outputs\smoke\107734119_175999932\multiview\target_views --semantic-mapping data\processed\targets\107734119_175999932\semantic_mapping.json --output outputs\smoke\107734119_175999932\step7_baselines\global_dino\renders --mode both --point-radius 1 --device cuda
if errorlevel 1 exit /b 1

conda run -n semreg-gs-v1 python scripts\render_semantic_gaussians.py --gaussians data\processed\semantic_gaussians\107734119_175999932\gaussians.npz --appearance outputs\smoke\107734119_175999932\step7_baselines\semantic_dino\appearance.npz --views outputs\smoke\107734119_175999932\multiview\target_views --semantic-mapping data\processed\targets\107734119_175999932\semantic_mapping.json --output outputs\smoke\107734119_175999932\step7_baselines\semantic_dino\renders --mode both --point-radius 1 --device cuda
if errorlevel 1 exit /b 1

conda run -n semreg-gs-v1 python scripts\evaluate_step7_baselines.py --methods-root outputs\smoke\107734119_175999932\step7_baselines --target-views outputs\smoke\107734119_175999932\multiview\target_views --semantic-mapping data\processed\targets\107734119_175999932\semantic_mapping.json --view-names view_06 view_07 --output outputs\smoke\107734119_175999932\step7_baselines\metrics_heldout.json
if errorlevel 1 exit /b 1

echo Step 7 DINO comparison completed successfully.
echo Held-out metrics: outputs\smoke\107734119_175999932\step7_baselines\metrics_heldout.json
