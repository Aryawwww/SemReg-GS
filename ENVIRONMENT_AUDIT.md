# 本机环境审计（2026-08-18 更新）

## 结论

本机**适合**：

- 3D-FRONT 数据整理和 donor-target 配对；
- Blender 离线渲染（安装 Blender 后）；
- 单房间、低分辨率、低 Gaussian 数量的 Smoke test；
- 语义标签投影、指标计算和结果可视化；
- 小型 MLP 的 Global vs Semantic Pilot（需要控制分辨率和 Gaussian 数量）。

本机**不适合直接承担**：

- README 中建议的 24 GB VRAM 完整训练；
- 512–800 px、多视角、大量 Gaussians 的 StyleGaussian 训练；
- MaterialMVP/TRELLIS.2 的完整高分辨率运行；
- 多个主实验并行训练。

## 检测结果

| 项目 | 本机配置 | 实验最低建议 | 判断 |
|---|---:|---:|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop | NVIDIA CUDA GPU | 可用 |
| 显存 | 8 GB | Smoke 8 GB；Main 24 GB | Smoke 可用，Main 不足 |
| CPU | AMD Ryzen 9 7945HX | 8+ cores | 足够 |
| RAM | 约 32 GB | 32–64 GB | 达到最低线 |
| E 盘可用空间 | 约 346 GB | 80 GB+，完整数据更多 | 足够做数据子集 |
| NVIDIA Driver | 566.26 | 支持所选 PyTorch CUDA | 可用 |
| Driver CUDA API | 12.7 | CUDA 11.8/12.x | 可用 |
| Conda | 25.5.1 | 需要 | 已安装 |
| Git | 2.51.0 | 需要 | 已安装 |
| PyTorch CUDA | `semreg-gs-v1`: torch 2.13.0+cu126，CUDA 12.6 可用 | 需要 | 已验证 |
| Blender | 5.2.0 LTS（Scoop） | 数据渲染需要 | 已安装并验证 GLB 导入/渲染 |
| CMake | 未找到 | CUDA 扩展编译需要 | 待安装 |
| CUDA Toolkit / nvcc | 未找到 | 本地编译 rasterizer 需要 | 待安装 |

## 本地 Smoke 限制

```yaml
image_width: 512
image_height: 512
reference_views: 3
target_views: 8
max_gaussians: 100000
batch_size: 1
mixed_precision: fp16
freeze_geometry: true
```

如果仍然 OOM，按以下顺序降低：

1. `max_gaussians: 50000`；
2. 分辨率降到 `384x384`；
3. reference encoder 预先离线提取 feature；
4. 使用 gradient accumulation；
5. 将 StyleGaussian 训练移到云端，不再压缩核心实验。

## 推荐云环境

### 首选：RunPod RTX 4090 24 GB

适合 Pilot 和大部分主实验。选择：

- Ubuntu 22.04；
- RTX 4090 24 GB；
- PyTorch 2.x + CUDA 12.x 模板；
- 100–150 GB persistent volume；
- 使用普通 Pod，而不是 Serverless endpoint。

官方页面显示 RTX 4090 为 24 GB；价格随 Community/Secure Cloud 和实时可用性变化，创建实例前以控制台为准：

- https://www.runpod.io/gpu-models/rtx-4090
- https://www.runpod.io/pricing

### 显存更稳妥：Lambda A6000 48 GB

适合 StyleGaussian、MaterialMVP/TRELLIS.2 或高分辨率实验。Lambda 官方按需实例页面列出 A6000 48 GB，当前页面标价约 USD 1.09/GPU-hour，创建前再次确认：

- https://lambda.ai/instances

### 选择原则

- Global vs Semantic Pilot：RTX 4090 24 GB；
- StyleGaussian semantic decoder：优先 RTX 4090，OOM 再切 A6000 48 GB；
- TRELLIS.2 / 高分辨率 PBR baseline：A6000 48 GB 或 A100 40/80 GB；
- 不建议为第一阶段使用 H100，成本与本实验规模不匹配。

## 下一步安装顺序

1. 安装 Blender 4.x；
2. 安装 Visual Studio 2022 Build Tools（Desktop development with C++）；
3. 安装 CMake；
4. 仅当决定在 Windows 本地编译 3DGS 时安装 CUDA Toolkit；
5. 使用已创建并验证的独立 `semreg-gs-v1` Conda 环境，不污染现有 `hiv` 环境；
6. 云端使用 Linux 编译官方 CUDA rasterizer，作为主要训练环境。

Windows 上编译 CUDA 扩展通常比 Ubuntu 容易出现编译器与 CUDA 版本不匹配，因此本地先完成数据闭环，主训练放到 Linux 云端。

## 已完成的 Stage 0 项目

- [x] 审计 GPU、显存、CPU、RAM、磁盘和驱动；
- [x] 创建并验证 `semreg-gs-v1`（Python 3.10.20）；
- [x] 下载官方 `gaussian-splatting` 代码及 CUDA 子模块；
- [x] 下载官方 `StyleGaussian` 代码；
- [x] 建立 `configs/smoke.yaml`；
- [x] 建立数据、输出和脚本目录；
- [x] 安装 PyTorch 到 `semreg-gs-v1`；
- [ ] 安装 CMake、Visual Studio Build Tools 和 CUDA Toolkit；Blender 5.2.0 LTS 已安装；
- [ ] 申请并接受 3D-FRONT/3D-FUTURE 数据条款；
- [ ] 申请 ScanNet++ 访问。

激活命令：

```powershell
conda activate semreg-gs-v1
```

## Step 1 验证记录（2026-08-18）

通过 `conda run -n semreg-gs-v1` 完成独立环境验证：

```text
Python:          3.10.20
PyTorch:         2.13.0+cu126
PyTorch CUDA:    12.6
CUDA available:  True
GPU:             NVIDIA GeForce RTX 4060 Laptop GPU
VRAM:            8.00 GB
GPU matmul:      passed
pip check:       No broken requirements found
```

README 的 Step 1（激活独立环境并确认 CUDA/PyTorch 能识别 GPU）已完成。Blender、CMake、Visual Studio Build Tools 和 CUDA Toolkit 属于后续资产审计与本地扩展编译准备，不阻塞进入 Step 2 获取单个 HSSD 场景。

## Step 2 数据记录（2026-08-18）

- 已接受 HSSD 的 CC BY-NC 4.0 条款并通过 Hugging Face 官方设备授权登录；
- 只下载场景 `107734119_175999932`，未下载完整数据集；
- 完整 GLB 为 78,510,308 bytes，格式为 glTF 2.0；
- 对应场景配置包含 61 个对象实例；
- 来源、文件大小和 SHA-256 保存在 `data/raw/hssd/107734119_175999932/manifest.json`。

## Step 3 资产审计记录（2026-08-18）

- Blender 5.2.0 LTS 可成功导入并渲染完整 GLB；
- 场景尺寸约为 `8.71 × 11.13 × 2.81 m`；
- 486 个 mesh，426,047 个顶点，530,720 个多边形；
- 181 个材质、115 张图像，包含 64 个图像纹理材质和 20 个 Normal Map；
- 61 个对象实例中 58 个可通过官方 metadata 解析类别；未解析模板 `224-132` 暂归为 `other`；
- 审计输出保存在 `outputs/smoke/107734119_175999932/`。

## Step 4 语义映射记录（2026-08-18）

- 使用 HSSD 官方 semantic config、对象 metadata 和 GLB 节点/材质命名；
- 统一 ID 为 `wall=0 / floor=1 / ceiling=2 / door=3 / window=4 / other=5`；
- 官方命名缺失的结构面仅在 `geometry_*` mesh 上使用高度与法线规则；
- 分类面数：wall 1,375、floor 40、ceiling 22、door 4,044、window 258、other 524,981；
- `FP_GLASS` 当前映射为 window，玻璃门是进入 Pilot 前需要人工复核的已知边界；
- 映射保存在 `data/processed/targets/107734119_175999932/semantic_mapping.json`，预览保存在对应 smoke 输出目录。

## Step 5 多视角渲染记录（2026-08-18）

- 单场景 smoke test 输出 3 个 donor reference 和 8 个 target views；
- 分辨率为 512×512，每个视角包含 `rgb.png / semantic.png / depth.exr / normal.exr / camera.json`；
- Depth 为 32-bit float，Normal 为 32-bit vector，使用 Blender 5.2 Multilayer EXR 输出节点；
- 相机 JSON 保存内参、camera-to-world 和 world-to-camera 4×4 矩阵；
- 11/11 个视角文件完整，RGB 和语义图已视觉检查；
- 当前使用同一场景验证渲染闭环，正式 cross-geometry 实验仍需下载第二个不同场景建立 donor–target pair；
- 输出位于 `outputs/smoke/107734119_175999932/multiview/`。
