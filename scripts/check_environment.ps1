$ErrorActionPreference = "SilentlyContinue"

Write-Output "=== SemReg-GS environment audit ==="
Write-Output "Date: $(Get-Date -Format s)"

Write-Output "`n=== GPU ==="
nvidia-smi

Write-Output "`n=== GPU summary ==="
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader

Write-Output "`n=== Disk ==="
Get-PSDrive -PSProvider FileSystem |
  Select-Object Name,
    @{Name="UsedGB";Expression={[math]::Round($_.Used / 1GB, 1)}},
    @{Name="FreeGB";Expression={[math]::Round($_.Free / 1GB, 1)}}

Write-Output "`n=== Tools ==="
foreach ($tool in @("conda", "git", "python", "blender", "cmake", "nvcc")) {
  $command = Get-Command $tool -ErrorAction SilentlyContinue
  if ($command) {
    Write-Output "$tool`t$($command.Source)"
  } else {
    Write-Output "$tool`tMISSING"
  }
}

Write-Output "`n=== Conda environments ==="
conda info --envs

Write-Output "`n=== Recommendation ==="
Write-Output "8 GB VRAM: data preparation + low-resolution smoke only."
Write-Output "24 GB VRAM: pilot/main recommendation."
Write-Output "48 GB VRAM: StyleGaussian/TRELLIS.2 and safer high-resolution runs."
