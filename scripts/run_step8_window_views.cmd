@echo off
setlocal
cd /d "%~dp0.."

where blender >nul 2>nul
if errorlevel 1 (
  echo ERROR: blender is not available on PATH.
  exit /b 1
)

where conda >nul 2>nul
if errorlevel 1 exit /b 1

blender -b --python-exit-code 1 --python scripts\generate_window_targeted_views.py -- --input data\raw\hssd\107734119_175999932\scene.glb --semantic-config data\raw\hssd\107734119_175999932\semantic_config.json --output outputs\smoke\107734119_175999932\multiview --width 512 --height 512 --camera-distance 1.25 --lateral-offset 0.30 --window-candidates 3 --maximum-window-area 1.0 --maximum-proxy-distance 0.50
if errorlevel 1 exit /b 1

conda run -n semreg-gs python scripts\audit_view_semantic_coverage.py --donor-candidates outputs\smoke\107734119_175999932\multiview\donor_reference --target-candidates outputs\smoke\107734119_175999932\multiview\target_views --semantic-mapping data\processed\targets\107734119_175999932\semantic_mapping.json --output outputs\smoke\107734119_175999932\coverage_audit.json --donor-count 3 --heldout-count 2 --minimum-pixels 1000
if errorlevel 1 (
  echo Window views were rendered, but semantic coverage still failed.
  echo Read outputs\smoke\107734119_175999932\coverage_audit.json and outputs\smoke\107734119_175999932\multiview\window_view_manifest.json.
  exit /b 1
)

echo Window-targeted rendering and coverage audit passed.
