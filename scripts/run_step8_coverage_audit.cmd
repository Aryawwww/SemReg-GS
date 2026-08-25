@echo off
setlocal
cd /d "%~dp0.."

where conda >nul 2>nul
if errorlevel 1 exit /b 1

conda run -n semreg-gs-v1 python scripts\audit_view_semantic_coverage.py --donor-candidates outputs\smoke\107734119_175999932\multiview\donor_reference --target-candidates outputs\smoke\107734119_175999932\multiview\target_views --semantic-mapping data\processed\targets\107734119_175999932\semantic_mapping.json --output outputs\smoke\107734119_175999932\coverage_audit.json --donor-count 3 --heldout-count 2 --minimum-pixels 1000
if errorlevel 1 (
  echo Coverage audit failed. Read outputs\smoke\107734119_175999932\coverage_audit.json and add cameras for missing classes.
  exit /b 1
)

echo Coverage audit passed.
echo Split manifest: outputs\smoke\107734119_175999932\coverage_audit.json
