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
set "GAUSSIANS=data\processed\semantic_gaussians\%TARGET_ID%\gaussians.npz"
set "DONORS=outputs\smoke\%SOURCE_ID%\multiview\donor_reference"
set "TARGET_VIEWS=outputs\pairs\%PAIR_ID%\target_multiview\target_views"
set "MAPPING=data\processed\targets\%TARGET_ID%\semantic_mapping.json"
set "OUTPUT=outputs\pairs\%PAIR_ID%\cross_geometry_2d"

call conda run -n semreg-gs python scripts\build_step7_baselines.py --gaussians "%GAUSSIANS%" --donor-views "%DONORS%" --semantic-mapping "%MAPPING%" --protocol-manifest "%PROTOCOL%" --output "%OUTPUT%\appearance"
if errorlevel 1 exit /b 1

for %%M in (global semantic_2d) do (
  call conda run -n semreg-gs python scripts\render_semantic_gaussians.py --gaussians "%GAUSSIANS%" --appearance "%OUTPUT%\appearance\%%M\appearance.npz" --views "%TARGET_VIEWS%" --semantic-mapping "%MAPPING%" --protocol-manifest "%PROTOCOL%" --protocol-split target_heldout --output "%OUTPUT%\renders\%%M" --mode both --point-radius 1 --device cuda
  if errorlevel 1 exit /b 1
)

echo Cross-geometry Global and B_sem-2D held-out renders completed successfully.
echo Output: %OUTPUT%
