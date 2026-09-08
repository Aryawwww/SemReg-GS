@echo off
setlocal
cd /d "%~dp0.."

where conda >nul 2>nul
if errorlevel 1 (
  echo ERROR: conda is not available. Run this script from Anaconda Prompt.
  exit /b 1
)

where blender >nul 2>nul
if errorlevel 1 (
  echo ERROR: blender is not available on PATH.
  exit /b 1
)

set "TARGET_ID=103997424_171030444"
set "RAW=data\raw\hssd\%TARGET_ID%"
set "PROCESSED=data\processed\targets\%TARGET_ID%"
set "PAIR=data\processed\pairs\107734119_175999932__to__%TARGET_ID%\pair_manifest.json"

blender -b --python scripts\audit_hssd_blender.py -- --input "%RAW%\scene.glb" --output "%PROCESSED%\asset_audit.json" --preview "%PROCESSED%\asset_preview.png"
if errorlevel 1 exit /b 1

call conda run -n semreg-gs python scripts\audit_hssd_semantics.py --scene "%RAW%\scene_instance.json" --metadata "%RAW%\hssd_obj_semantics_condensed.csv" --output "%PROCESSED%\semantic_audit.json"
if errorlevel 1 exit /b 1

blender -b --python scripts\create_semantic_mapping_blender.py -- --input "%RAW%\scene.glb" --output "%PROCESSED%\semantic_mapping.json" --preview "%PROCESSED%\semantic_preview.png"
if errorlevel 1 exit /b 1

call conda run -n semreg-gs python scripts\finalize_target_audit.py --pair-manifest "%PAIR%" --asset-audit "%PROCESSED%\asset_audit.json" --semantic-audit "%PROCESSED%\semantic_audit.json" --semantic-mapping "%PROCESSED%\semantic_mapping.json" --asset-preview "%PROCESSED%\asset_preview.png" --semantic-preview "%PROCESSED%\semantic_preview.png"
if errorlevel 1 exit /b 1

echo Step 9 target asset and semantic audit completed successfully.
echo Pair manifest: %PAIR%
