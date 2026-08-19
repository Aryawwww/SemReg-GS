# SemReg-GS 实验设计

> **目标**：研究在 source building 与 target CAD 几何不同、没有逐点几何对应的情况下，能否以建筑语义作为 correspondence，将现有建筑照片的区域级外观迁移到 CAD-anchored 3D Gaussians，并保持几何准确、区域可控和跨视角一致。

## 0. 从这里开始：数据与实验执行路线

### 当前数据决策

**不要等待 3D-FRONT 审批后才开始实验。** 当前采用以下四层数据组合：

| 层级 | 数据集 | 在本项目中的作用 | 当前行动 |
|---|---|---|---|
| 即时主数据 | **HSSD** | 可编辑室内场景、语义对象和较完整材质；用于 smoke test 与 pilot | 立即下载 1 个场景，检查 GLB、材质和语义 |
| 最终标准数据 | **3D-FRONT/3D-FUTURE** | 与 RoomPainter 等工作可比；用于扩大主实验 | 保留 Hugging Face 申请，同时尝试阿里天池官方入口 |
| 受控辅助数据 | **OpenRooms** | albedo、roughness、lighting、semantic GT；验证阴影/高光是否被错误当作材质 | 主流程跑通后下载少量场景和 GT |
| 真实域验证 | **ScanNet++** | 真实照片、相机和语义；验证 synthetic-to-real | 最后只下载 5–10 个场景子集 |

HSSD 是当前最实际的 3D-FRONT 替代，但不是永久删除 3D-FRONT：

```text
HSSD：立即完成可编辑场景闭环和主方法 pilot
  ↓
3D-FRONT：获批后扩大规模并与房间级强 baseline 对齐
  ↓
OpenRooms：独立验证材质与光照解耦
  ↓
ScanNet++：验证真实建筑照片泛化
```

### 新成员只需按此顺序执行

- [x] **Step 1 — 环境检查**：`semreg-gs-v1` 已安装 PyTorch 2.13.0+cu126，CUDA 可识别 RTX 4060 Laptop GPU，并通过 GPU 矩阵运算测试。
- [x] **Step 2 — 获取一个 HSSD 场景**：已下载场景 `107734119_175999932` 到 `data/raw/hssd/`，并保存来源、许可和 SHA-256 清单。
- [x] **Step 3 — 资产审计**：Blender 5.2 已成功导入场景；mesh、米制尺度、纹理/PBR 节点和官方对象类别映射均可读取，未映射模板暂归为 `other`。
- [x] **Step 4 — 统一五类语义**：已映射为 `wall / floor / ceiling / door / window / other`，并保存可复现的逐面规则、`semantic_mapping.json` 和语义剖视预览。
- [x] **Step 5 — 多视角渲染**：已修复 Blender compositor 污染 RGB 的问题；128×128 真实 smoke test 已输出 3 张 donor reference 和 8 张 target views，并通过 RGB、semantic、32-bit EXR depth/normal 和 camera JSON 验证。正式 512×512 数据需要用修复脚本重新生成。
- [ ] **Step 6 — CAD-anchored Gaussians**：100,000 个 semantic Gaussians 的初始化、冻结几何验证和语义渲染已通过；修复后的中性灰初始化训练已在 128×128 有效 RGB 上通过快速复验，正式 512×512 一键训练尚待执行。
- [ ] **Step 7 — 最小三方法比较**：在相同场景、相机和训练预算下运行 `Global`、`B_sem-2D`、`Semantic-DINO`。
- [ ] **Step 8 — 完成 10 个 HSSD pair**：报告 region LPIPS/DINO、Semantic Leakage、multi-view error 和 CAD surface distance；不满足 Go/No-Go 判据时先修数据。
- [ ] **Step 9 — 加入强 baseline**：依次运行 `StyleGaussian-global → MaterialMVP whole-room → MaterialMVP semantic-submesh → TRELLIS.2`。
- [ ] **Step 10 — 扩大与验证**：3D-FRONT 获批后做 100–300 pairs；OpenRooms 做材质/光照消融；ScanNet++ 做真实域测试。

### 当前 Smoke Test：实验设计与实际结果（更新于 2026-08-19）

这一节记录已经实际运行的实验，不是未来计划。当前已完成 **Step 1–5 和 Step 6 的几何/语义部分**；修复后的 RGB 管线与中性外观训练已通过 128×128 快速复验，但正式 512×512 重渲染与复训尚未执行。尚未比较 Global、B_sem-2D 与 Semantic-DINO，因此当前结果只能证明数据、相机、几何、语义和低分辨率训练管线可用，不能证明论文假设成立。

#### 本轮实验目的

在开始 3DGS 训练前，先回答以下工程问题：

1. HSSD 的完整室内 GLB 能否在本机 Blender 中正确加载；
2. mesh、尺度、材质、纹理和对象语义是否可读取；
3. 能否将原始类别统一成 `wall / floor / ceiling / door / window / other`；
4. RGB、semantic mask、depth、normal 和相机参数能否在同一相机下同步输出；
5. 当前 8 GB GPU 机器是否足以承担后续低分辨率、低 Gaussian 数量的 smoke test。

#### 实验输入与控制变量

| 项目 | 本轮设置 |
|---|---|
| 数据集 | HSSD 官方 Hugging Face 仓库，CC BY-NC 4.0 |
| Scene ID | `107734119_175999932` |
| 原始场景 | 78,510,308 bytes 的 glTF 2.0 GLB |
| 场景角色 | 同一场景同时用于 donor-view 与 target-view 渲染管线测试 |
| 渲染器 | Blender 5.2.0 LTS / Eevee + Workbench semantic pass |
| 图像分辨率 | 512 × 512 |
| Donor views | 3：living room、bedroom、kitchen |
| Target views | 8：living room ×2、bedroom、kitchen、hallway ×2、bathroom、utility room |
| 相机高度 | 1.50 m |
| 相机焦距 | 28 mm，36 mm sensor；`fx = fy = 398.22 px` |
| 几何控制 | 所有模态共享同一 GLB、同一相机内外参，不修改几何 |
| 随机种子 | `42`，供后续 Gaussian 采样与训练沿用 |

当前使用同一场景并不是 cross-geometry 实验，而是为了隔离并验证渲染与标签管线。正式实验必须使用不同 scene ID 的 donor 和 target。

#### 数据获取与许可结果

- Hugging Face 设备授权登录成功；
- 只下载一个完整 HSSD 场景，没有下载完整 38.5 GB 数据集；
- 同时保存 `scene_instance.json`、官方 semantic config、semantic lexicon 和 condensed object semantics；
- `manifest.json` 记录来源仓库、远程路径、文件大小、CC BY-NC 4.0 和 SHA-256；
- 原始数据保持在 `data/raw/hssd/107734119_175999932/`，处理结果写入 `data/processed/`，没有修改原始文件。

#### 资产审计结果

| 指标 | 实际结果 | 判断 |
|---|---:|---|
| 场景尺寸 | 8.71 × 11.13 × 2.81 m | 符合室内米制尺度 |
| Mesh 数量 | 486 | 可读取 |
| 顶点数 | 426,047 | 可读取 |
| 多边形数 | 530,720 | 可读取 |
| 材质数 | 181 | 可读取 |
| 图像数 | 115 | 可读取 |
| Image Texture 材质 | 64 | 可读取 |
| Normal Map 节点 | 20 | 可读取 |
| Principled BSDF | 181 | 可读取 |
| Object instances | 61 | 可读取 |
| 成功解析类别的实例 | 58 / 61 | 3 个未解析实例暂归 `other` |

未解析的 3 个实例共享模板 `224-132`。这一问题不会阻塞六类建筑语义 smoke test，但进入 Pilot 前必须检查它是否属于 wall、floor、ceiling、door 或 window；否则继续保留为 `other`。

#### 六类语义映射设计

统一标签固定为：

| ID | 类别 | 预览颜色 | 主要映射依据 |
|---:|---|---|---|
| 0 | wall | red | `wall` GLB 名称；其余 `geometry_*` 垂直结构面 |
| 1 | floor | green | `geometry_*` 中接近 z=0 m 的水平结构面 |
| 2 | ceiling | blue | `ceiling` 名称；接近 z=2.8 m 的水平结构面 |
| 3 | door | orange | `FP_DOOR / DOORFRAME / DOORHANDLE` |
| 4 | window | cyan | `FP_GLASS / GLASSBORDER / window` |
| 5 | other | gray | 家具、设备及无法可靠映射的对象 |

映射严格采用“官方 metadata 和 GLB 命名优先，结构几何规则补充，其他全部回退为 `other`”。法线/高度规则只作用于 `geometry_*`，避免把桌面、床面等水平家具误标成 floor。

逐面映射结果：

| 类别 | Polygon 数量 |
|---|---:|
| wall | 1,375 |
| floor | 40 |
| ceiling | 22 |
| door | 4,044 |
| window | 258 |
| other | 524,981 |
| **合计** | **530,720** |

`other` 占比高是预期现象，因为完整房间中的家具模型包含大量高细节三角形，而建筑壳体由较少的大三角面组成。后续采样 Gaussians 时应按表面积而不是 polygon 数量分配，避免家具细分程度影响类别采样比例。

当前将 `FP_GLASS` 映射为 window。该规则可能把玻璃门误标为 window，是进入 Pilot 前需要人工复核的已知边界。

#### 多视角多模态渲染设计

每个视角保存：

```text
<view_id>/
├── rgb.png          # 512×512 PBR appearance render
├── semantic.png     # 六类语义着色图
├── depth.exr        # 32-bit float Depth layer
├── normal.exr       # 32-bit vector Normal layer
└── camera.json      # intrinsics + camera-to-world + world-to-camera
```

Blender 5.2 修改了 compositor API。旧实现把 RGB 与 depth/normal compositor pass 交错执行，生成的 11 张 RGB 均退化为只有 0/1 的近黑图。修复后使用新版 `CompositorNodeTree`、`directory/file_name` 和显式 `FLOAT Depth / VECTOR Normal` sockets 输出 EXR；RGB、geometry 和 semantic 三类 pass 完全分离，RGB 从原始 `Render Result` 显式保存。脚本还会检查动态范围、8-bit 色阶数和跨视角 SHA-256，发现 0/1 图或全部视角相同时立即失败。

#### 多视角渲染结果

| 检查项 | 实际结果 |
|---|---:|
| Donor reference views | 3 / 3 完整 |
| Target views | 8 / 8 完整 |
| 每视角文件数 | 5 |
| 修复验证分辨率 | 128 × 128；正式 512 × 512 待重渲染 |
| 相机矩阵 | 4 × 4 camera-to-world 与 world-to-camera |
| EXR 文件头 | 有效 OpenEXR magic `762f3101` |
| 完整视角数 | 11 / 11 |
| RGB 自动验证 | `passed`；11 个视角具有 11 个不同 SHA-256 |
| RGB 动态范围 | 10 个丰富视角为 165–230 个 8-bit 色阶；`view_04` 正对低纹理表面，为 5 个色阶但不是 0/1 图 |
| RGB 非黑帧 | 修复后的 128×128 代表性视角已视觉检查通过 |
| RGB–semantic 对齐 | 修复后的代表性视角已视觉检查通过 |

语义 PNG 中约 76.8% 像素恰好等于六种类别 RGB，其余主要是抗锯齿边缘和黑色背景。训练前应执行 nearest-palette 解码，或额外输出关闭抗锯齿的单通道 class-ID mask；不能直接把所有非精确 RGB 当作新类别。

#### 当前可复现产物

```text
data/raw/hssd/107734119_175999932/
├── scene.glb
├── scene_instance.json
├── semantic_config.json
├── semantic_lexicon.json
├── hssd_obj_semantics_condensed.csv
└── manifest.json

data/processed/targets/107734119_175999932/
└── semantic_mapping.json

outputs/smoke/107734119_175999932/
├── asset_audit.json
├── semantic_audit.json
├── asset_preview.png
├── semantic_preview.png
└── multiview/
    ├── render_manifest.json
    ├── donor_reference/reference_00..02/
    └── target_views/view_00..07/

outputs/smoke/107734119_175999932/multiview_fix_test/
├── rgb_validation.json
├── render_manifest.json
├── donor_reference/reference_00..02/
└── target_views/view_00..07/

outputs/smoke/107734119_175999932/neutral_gaussians_fix_test/
├── appearance.npz
├── checkpoint.pt
├── training_report.json
└── renders/view_00..07/
```

关键脚本：

```text
scripts/download_hssd_smoke.py
scripts/audit_hssd_blender.py
scripts/audit_hssd_semantics.py
scripts/create_semantic_mapping_blender.py
scripts/render_hssd_multiview.py
scripts/train_neutral_gaussians.py
scripts/render_semantic_gaussians.py
scripts/run_step6_fixed.cmd
```

#### 当前结论

**已支持的结论：**

1. 本机环境可以完成 HSSD 单场景资产处理；修复后的多模态渲染已在 128×128 验证通过，512×512 正式重渲染待执行；
2. HSSD 场景具备可读取的几何、PBR 资源、对象实例和可统一的建筑语义；
3. 六类映射与 RGB/semantic/depth/normal/camera 同步输出闭环已跑通，RGB 退化问题现在可被自动检测；
4. CAD-anchored Gaussian 初始化与几何保持验证已通过；修复后的训练器能够在有效 RGB 上从中性灰收敛，并保持几何参数冻结。

**尚不支持的结论：**

1. 尚未证明 semantic conditioning 优于 global conditioning；
2. 尚未测量 Semantic Leakage、region LPIPS/DINO 或 multi-view warp error；
3. 尚未在正式 512×512 有效 RGB 上完成中性外观训练与指标复验；
4. 尚未完成不同几何的 donor–target pair；
5. 尚未与 StyleGaussian、MaterialMVP 或 TRELLIS.2 比较。

因此当前里程碑是 **data/camera/geometry/semantic pipeline passed; fixed-RGB low-resolution training passed**。正式 512×512 外观复训仍待完成，更不是 **SemReg-GS method validated**。

#### Step 6 实际结果

- 从 target mesh 按三角面面积采样并导出了 100,000 个 Gaussian；六类计数为 wall 40,102、floor 9,505、ceiling 7,651、door 6,224、window 268、other 36,250。
- `semantic_id` 与 `source_face_id` 验证通过；Gaussian center 到来源三角面的最大重建误差为 `1.134145e-6 m`，低于 `1e-5 m` 阈值。
- 旧中性外观 smoke 训练使用 8 个 target views，仅优化 RGB/SH-DC，冻结 xyz、normal、rotation、scale 和 semantic ID；程序覆盖 83,113 / 100,000 个 Gaussian（83.11%），但输入 RGB 只有 0/1，因此 `8.788e-5 → 5.689e-5` 的 L1 下降无外观质量意义，不能作为有效实验结果。
- 外观与语义双模式渲染成功输出 8 个视角；固定半径点投影的像素覆盖率为 12.52%–50.60%，语义空间结构与相机方向一致。
- 当前实现是便携式 point-zbuffer smoke baseline，不是各向异性 CUDA Gaussian rasterizer。RGB 导出代码已经修复：128×128 测试的 11 个视角具有 11 个不同哈希，丰富视角包含 165–230 个 8-bit 色阶，RGB、depth、normal 和 semantic 均成功输出。
- 训练器已修复另一个实验有效性问题：旧实现直接用目标 RGB 初始化参数，导致初始 loss 虚低；新实现固定从 `RGB=0.5` 中性灰开始，只优化 RGB/SH-DC，并在训练前检查 0/1 图、重复哈希、相机尺寸和最低 Gaussian 覆盖率。
- 修复后的 128×128 快速复训使用 8 个 target views 和 CUDA，观察到 53,623 / 100,000 个 Gaussian（53.62%）。100 步训练将 L1 从 `0.268841` 降至 `0.001578`，最终 MSE 为 `1.95285e-5`、PSNR 为 `47.09 dB`；随后 8 个外观/语义视角均成功渲染。该结果验证训练代码正确性，但不是正式 512×512 结果，也不是独立测试集泛化指标。
- `scripts/run_step6_fixed.cmd` 已封装 Anaconda 环境激活、512×512 多模态重渲染、RGB 验证、300 步中性外观训练和双模式渲染；任何阶段失败都会停止，防止无效数据继续进入训练。

#### 下一实验

运行 `scripts/run_step6_fixed.cmd`，重新生成 512×512 `multiview/`，确认 `rgb_validation.json` 为 `passed`，并将正式中性外观输出到 `neutral_gaussians_fixed/`。检查正式训练的覆盖率、初始/最终 L1、MSE、PSNR 和代表性渲染；全部通过后将 Step 6 标记完成，再进入 Step 7 的 `Global`、`B_sem-2D` 与 `Semantic-DINO` 三方法比较。

### Step 1–8 的最小完成产物

```text
outputs/pilot/<pair_id>/
├── input/
│   ├── donor_reference_rgb/
│   ├── donor_reference_masks/
│   ├── target_mesh.glb
│   └── semantic_mapping.json
├── global/
├── semantic_2d/
├── semantic_dino/
├── ground_truth/
├── comparison_grid.png
├── metrics.json
└── camera_path.mp4
```

完成标志不是“生成了一张好看的图”，而是同一个 donor–target pair 能复现三种方法，并得到区域准确性、泄漏、跨视角一致性和几何保持的可比较结果。

## 1. 最小论文命题

### 最小 paper claim

> **We investigate reference-conditioned appearance transfer between geometrically different indoor scenes by using architectural semantics as the correspondence between real images and CAD-anchored 3D Gaussians.**

中文：

> **我们研究如何利用建筑语义作为不同几何场景之间的对应关系，将真实建筑照片中的区域级外观迁移到没有目标照片的 CAD-anchored 3D Gaussian 场景。**

这里的核心不是简单的 `semantic-aware StyleGaussian`，而是：

```text
Geometry(source building) != Geometry(target CAD)

source wall  <---- architectural semantics ----> target wall
source floor <---- architectural semantics ----> target floor
source door  <---- architectural semantics ----> target door
```

### 研究问题

给定：

- 一个没有目标照片的室内 CAD mesh；
- CAD 表面语义：`wall / floor / ceiling / door / window`；
- 另一个建筑的 1–5 张参考照片；

能否将参考照片中同类建筑部件的外观迁移到 CAD-anchored Gaussians，使结果比全局风格迁移更：

1. **语义正确**：wall 的外观只影响 wall，而不是门窗或地面；
2. **跨视角稳定**：移动相机时不闪烁、不漂移、不出现明显接缝；
3. **几何忠实**：Gaussian 中心仍受 CAD 表面约束；
4. **接近参考建筑**：各语义区域的颜色、纹理和整体视觉特征接近参考图。

### 核心假设

> 与向整个场景广播一个全局风格向量相比，按 `wall / floor / ceiling / door / window` 聚合参考特征，并只条件化对应语义的 Gaussians，会显著提高区域对应准确率、参考相似度和跨视角稳定性。

统计假设写为：

```text
H1: Leakage_semantic < Leakage_global
H2: RegionSimilarity_semantic > RegionSimilarity_global
H3: MultiViewError_semantic <= MultiViewError_global
```

不预先写死“必须提升 20%”。主实验使用配对场景差值、95% confidence interval、effect size 和配对统计检验判断效果是否稳定。

### 三项预期贡献

1. **Architectural semantic correspondence**：将参考照片分解为 region-specific appearance representations，而不是一个全局 style code。
2. **Cross-geometry transfer**：source 与 target 不需要形状或逐点对应，只依赖 wall-to-wall、floor-to-floor 等建筑语义对应。
3. **CAD-constrained Gaussian generation**：Gaussian `xyz` 固定或投影约束在 CAD 表面，normal 受 CAD 约束，实现 appearance generation without geometry drift。

### 建议论文名称

**SemReg-GS: Semantic Region-Guided Appearance Transfer to CAD-Anchored 3D Gaussians**

本实验第一版只预测 Gaussian 外观参数（RGB/SH 或外观 feature）；不同时解决完整 PBR、建筑融合和 CAD 生成。PBR 分解可以作为第二阶段扩展。

必须正面区分 MaterialMVP：MaterialMVP 已经完成 `mesh + reference image -> illumination-invariant multi-view PBR textures`。因此本文不能把“reference-conditioned appearance transfer”本身当成贡献；贡献必须落在**不同建筑几何之间的语义 correspondence、区域可控性和 CAD-constrained Gaussian representation**上。

---

## 2. 数据库选择与替代策略

### 2.1 即时主数据：HSSD

3D-FRONT 授权等待期间，HSSD 用于第一轮 smoke test 和 pilot：

- 211 个高质量合成室内场景和 18,656 个物体模型；
- 提供可加载的 3D 场景资产，适合检查 mesh、材质和语义；
- 数据规模虽小，但足以验证 cross-geometry semantic transfer；
- 采用 CC BY-NC 4.0，使用和发布结果时必须遵守非商业条款。

第一轮只选 10–20 个结构清楚的场景。原始类别必须通过 `semantic_mapping.json` 显式映射为 `wall / floor / ceiling / door / window / other`；无法可靠映射的对象标为 `other`。

来源：

- HSSD 官方主页：https://3dlg-hcvc.github.io/hssd/
- HSSD 数据入口：https://huggingface.co/datasets/hssd/hssd-models

### 2.2 最终标准数据：3D-FRONT + 3D-FUTURE

这是第一版最合适的数据，而不是 Objaverse。

理由：

- **与 RoomPainter 最接近**：RoomPainter 在 3D-FRONT 室内场景上进行实验，是当前最接近“整个房间一致纹理生成”的强工作；
- 数据包含室内场景 mesh、房间布局、结构/对象语义和高质量纹理；
- 可以将原始纹理去除，把 mesh 当作“未来 CAD”；
- 可以从另一个房间渲染“现有建筑照片”；
- 可以人工构造迁移后的目标材质，因此拥有像素级和区域级 ground truth；
- 比直接使用真实照片更容易判断方法究竟是否有效，而不是只看结果是否好看。

相关来源：

- 3D-FRONT 论文：https://arxiv.org/abs/2011.09127
- 3D-FRONT 数据镜像（需要接受数据条件）：https://huggingface.co/datasets/Gen3DF/3DFront
- RoomPainter：https://arxiv.org/abs/2412.16778
- RoomPainter CVPR 页面：https://openaccess.thecvf.com/content/CVPR2025/html/Huang_RoomPainter_View-Integrated_Diffusion_for_Consistent_Indoor_Scene_Texturing_CVPR_2025_paper.html

当前 Hugging Face 请求正在等待维护者审批。等待期间使用 HSSD，不把审批作为 Stage 1 的阻塞条件。同时可以检查阿里天池官方入口；不得使用绕过授权的非官方网盘镜像。

- 3D-FRONT 阿里天池入口：https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset

### 2.3 受控辅助数据：OpenRooms

OpenRooms 不替代完整可编辑场景，而用于判断模型迁移的是材质，还是照片中的固定阴影和高光。它提供同一 CAD 的不同材质/光照版本，以及 albedo、roughness、depth、normal、lighting 和 45 类 semantic labels。

```text
same material + different lighting -> 材质表示应保持稳定
same lighting + different material -> 迁移结果应随材质明显变化
```

完整重新渲染所需的部分 SVBRDF 来自 Adobe Stock。第一阶段只使用官方发布的 renderings 和 ground truth，不重新分发受限材质。

- OpenRooms 官方仓库：https://github.com/ViLab-UCSD/OpenRooms

### 2.4 真实域验证：ScanNet++ v2

ScanNet++ 不作为第一阶段训练集，而用于验证合成数据训练的方法能否处理真实建筑照片。

它提供：

- 1000+ 个真实室内场景；
- 已配准的高分辨率 DSLR 图像；
- 激光扫描 mesh；
- wall、floor、ceiling 等语义标签；
- 相机内外参；
- 官方 3DGS 示例和 2D–3D 语义投影工具。

为了控制下载量，第一轮只申请并下载 **5–10 个训练/验证场景的低分辨率 DSLR、mesh 和 semantics**，不要下载完整 1.5 TB 数据。

来源：

- 官方主页：https://scannetpp.mlsg.cit.tum.de/scannetpp/
- 数据结构：https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation
- 官方工具：https://github.com/scannetpp/scannetpp

### 2.5 其他备选数据的边界

| 数据集 | 可以帮助什么 | 为什么不作为当前首选 |
|---|---|---|
| Infinigen Indoors | 程序化产生 Blender 场景、语义和 GT | 搭建与渲染成本较高，作为 HSSD 不可用时的开放式 fallback |
| Structured3D | 房间结构、布局和语义监督 | 不够直接支持完整可编辑 PBR/CAD-to-GS 主流程 |
| InteriorVerse | PBR、几何和空间变化光照研究 | 需要签署协议并邮件申请，不能解决当前等待问题 |
| Hypersim | 强图像级材质、光照和语义 GT | 数据量大，完整源场景资产存在商业资产限制 |

不能因为数据含 RGB 和 semantic mask，就默认它能替代 3D-FRONT。主数据至少需要 target geometry、相机、多视角渲染能力和可映射的建筑语义。

### 2.6 Objaverse 的用途

MaterialMVP 公开 paper checkpoint 使用 Objaverse 数据，TRELLIS.2 训练配置使用 Objaverse-XL。它们适合单体 3D 资产的 PBR/纹理预训练，但不适合直接作为本实验的房间级主数据库。

因此 Objaverse 只用于：

- 运行 MaterialMVP/TRELLIS.2 零样本 baseline；
- 如果后续需要，预训练“参考图 → 材质/外观 feature”模块；
- 不用于主要结论。

来源：

- MaterialMVP 官方代码：https://github.com/ZebinHe/MaterialMVP
- MaterialMVP 论文：https://arxiv.org/abs/2503.10289
- TRELLIS.2 官方代码：https://github.com/microsoft/TRELLIS.2

### 数据选择结论

| 用途 | 数据库 | 原因 |
|---|---|---|
| Smoke test 与 pilot | HSSD | 可立即启动；场景质量高、资产可编辑、语义可映射 |
| 扩大训练和最终定量评估 | 3D-FRONT/3D-FUTURE | 与 RoomPainter 可比；可构造迁移真值 |
| 材质/光照解耦 | OpenRooms | 同 CAD 不同材质/光照及 albedo、roughness、lighting GT |
| 真实参考图泛化 | ScanNet++ v2 | 真实 DSLR、语义 mesh、相机和官方 3DGS 工具齐全 |
| 资产级强 baseline | Objaverse / Objaverse-XL | MaterialMVP 与 TRELLIS.2 的公开训练数据来源 |

---

## 3. 受控数据集如何构造

直接随机选两个房间并比较结果没有可靠 ground truth。必须构造一个**已知正确答案的 cross-geometry transfer benchmark**。

### 3.1 基本单元

每个样本由四部分组成：

```text
donor room D（现有建筑）
    └── 1–5 张 reference RGB + semantic masks

target room T（未来 CAD，几何不同）
    └── 无纹理 mesh + 每个面/每个 Gaussian 的 semantic label

oracle transferred target T*（只用于监督和评估）
    └── 把 D 的 wall/floor/... 材质按语义施加到 T 后的渲染

held-out target cameras
    └── 方法训练时不可看到的 T* 视角
```

### 3.2 配对规则

1. Pilot 从 HSSD 选择 bedroom、living room 或 corridor；3D-FRONT 获批后沿用相同规则扩展；第一版只选一种房型。
2. 过滤损坏 mesh、缺失纹理和语义不完整的场景。
3. target 与 donor 必须是不同 scene ID，且布局/几何不能相同。
4. 两者至少同时包含 `wall / floor / ceiling`；door/window 可以先作为可选类别。
5. 第一版优先选择材质相对单一的房间，避免一面墙内同时出现多种材质。

### 3.3 构造 oracle ground truth

对 donor 的每个语义类别提取一组材质：

```text
M_D = {
  wall: donor_wall_material,
  floor: donor_floor_material,
  ceiling: donor_ceiling_material,
  door: donor_door_material,
  window: donor_window_material
}
```

然后通过 triplanar projection 或统一的 world-space UV scale，把这些材质施加到 target 的对应类别，得到 `T*`。方法输入只能看到 donor reference images 和未着色 target；不能看到 `T*` 的纹理或渲染。

这样可以准确测量：模型是否从参考图恢复了 donor 材质，并迁移到了不同几何的 target。

但 oracle 不能永远是“一类一个均匀材质”，否则任务会退化为识别语义后复制纹理。因此数据难度分三级推进。

### 3.4 数据难度三级设计

#### Level 1：Uniform material

```text
wall -> one material
floor -> one material
ceiling -> one material
```

只用于 Smoke test，验证语义、UV、Gaussian 和渲染闭环，不用于主要创新结论。

#### Level 2：Intra-class variation

同一个语义类别内部存在外观变化，例如：

- wall：painted plaster / wood panel / concrete；
- floor：light oak / dark oak / stone；
- 同一 wall 上出现 panel、边框或局部 variation；
- 多张 reference 对同一类别提供不同可见区域。

模型需要学习 region appearance distribution，而不是给整个类别复制一个均值。

#### Level 3：Geometry-dependent appearance

加入：

- 大尺度图案及 texture scale；
- 墙角、门框、窗边等 boundary interaction；
- 视角相关效果；
- 窗口附近光照变化；
- 缺失语义区域和遮挡；
- real reference -> unseen CAD。

论文证据链按 `controlled -> complex -> real` 展开。Level 1 只证明系统可运行；Level 2 是主要定量结论；Level 3 证明现实价值。

### 3.5 第一轮数据规模

先做小规模可行性验证，不要立即训练大模型。

| 阶段 | 场景对数量 | reference views | target GT views | 用途 |
|---|---:|---:|---:|---|
| Smoke test | 1 | 3 | 8 | 验证数据、相机、语义和渲染流程 |
| Pilot | 10 | 3 | 12 | 比较 global 与 semantic conditioning |
| Main | 100–300 | 1/3/5 | 24 | 训练、验证和消融 |

推荐按 **donor scene ID 和 target scene ID 同时隔离**划分 `70% train / 10% val / 20% test`，防止同一材质或同一房间泄漏到测试集。

---

## 4. 数据目录约定

```text
Experient/
├── README.md
├── configs/
│   ├── smoke.yaml
│   ├── pilot.yaml
│   └── main.yaml
├── data/
│   ├── raw/
│   │   ├── hssd/
│   │   ├── 3d_front/
│   │   ├── 3d_future/
│   │   ├── openrooms/
│   │   └── scannetpp/
│   ├── processed/
│   │   ├── donors/
│   │   ├── targets/
│   │   ├── oracle_targets/
│   │   └── semantic_gaussians/
│   └── splits/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── external/
│   ├── gaussian-splatting/
│   ├── StyleGaussian/
│   ├── MaterialMVP/
│   └── TRELLIS.2/
├── scripts/
│   ├── prepare_3dfront.py
│   ├── prepare_hssd.py
│   ├── prepare_openrooms.py
│   ├── build_pairs.py
│   ├── render_references.py
│   ├── create_oracle_transfer.py
│   ├── mesh_to_semantic_gaussians.py
│   ├── project_semantics.py
│   └── evaluate.py
├── src/
│   ├── reference_encoder.py
│   ├── semantic_pooling.py
│   ├── gaussian_decoder.py
│   ├── renderer.py
│   └── losses.py
└── outputs/
    ├── smoke/
    ├── pilot/
    └── main/
```

所有下载的数据必须保持在 `data/raw/`，处理脚本只写入 `data/processed/`，不要修改原数据。

---

## 5. 可实现的模型设计

### 5.1 CAD → semantic Gaussians

1. 从 target mesh 的三角面按面积采样点。
2. Gaussian center 固定在采样点；初始 rotation 与面法线对齐。
3. scale 由相邻采样距离和局部三角形尺寸初始化。
4. opacity 使用统一初值。
5. 每个 Gaussian 继承所在三角面的 semantic ID。
6. 第一版训练期间**冻结 center、rotation 和 scale**，只学习外观；这样 CAD geometry deviation 理论上为零。

每个 Gaussian 的输入：

```text
g_i = [position_i, normal_i, semantic_embedding_i, reference_token_sem(i)]
```

### 5.2 参考照片 → 语义区域特征

受控 3D-FRONT 实验直接使用渲染出的真值 semantic mask，先排除分割误差。

真实 ScanNet++ 实验再使用预训练语义分割模型，并把类别映射到五类结构标签。

处理步骤：

特征实验严格按以下顺序进行：

```text
RGB / texture statistics baseline
        ↓
DINOv2 dense visual features（主方法）
        ↓
CLIP（仅作为可选附加实验）
```

DINOv2 是第一选择，因为任务关注局部纹理、颜色和 dense visual similarity；CLIP 更偏全局语义概念，第一版不同时引入两个大型 feature encoder。

具体步骤：

1. 使用 DINOv2 提取 dense feature map；
2. 将 semantic mask 下采样到 feature map 分辨率；
3. 对同一语义的所有像素和多个 reference views 做 masked attention pooling；
4. 得到 `z_wall, z_floor, z_ceiling, z_door, z_window`；
5. 缺失类别使用可学习的 `unknown` token，并在 loss 中屏蔽其 reference matching 项。
6. 分别测试 1 / 3 / 5 张 reference，报告性能随参考图数量变化的曲线。

### 5.3 语义条件化 Gaussian 解码器

第一版采用小型 MLP，避免一开始改动大型 diffusion：

```text
[position encoding, normal, semantic embedding, z_sem]
                         ↓
                    4-layer MLP
                         ↓
           RGB / SH coefficients + opacity residual
```

如果 pilot 证明 semantic conditioning 有效，再借鉴 StyleGaussian：

- 保留每个 Gaussian 的 feature embedding；
- 以 StyleGaussian decoder 为初始化；
- 将单一 global style code 改成 semantic-specific reference codes；
- 渲染时，每个 Gaussian 只读取其类别对应的 code；
- decoder 在多个 target camera 上联合训练。

这一步是论文方法与 StyleGaussian 的关键区别。

---

## 6. 训练目标

对 oracle target 的多视角渲染计算：

```text
L_total =
    1.0 * L_rgb
  + 0.2 * L_ssim
  + 0.1 * L_lpips
  + 0.2 * L_semantic_region
  + 0.1 * L_reference_feature
  + 0.05 * L_view_regularization
```

### 损失定义

- `L_rgb`：预测渲染与 oracle GT 的 L1；
- `L_ssim`：保持结构和低频外观；
- `L_lpips`：感知相似度；
- `L_semantic_region`：分别在 wall/floor/... mask 内计算误差，再对类别平均，防止大面积 wall 淹没 door/window；
- `L_reference_feature`：预测区域渲染与 donor 对应区域的 DINO/CLIP feature 距离；
- `L_view_regularization`：同一表面点在不同相机下解码后的非视角相关颜色保持稳定。

第一版不要加入复杂对抗损失、SDS 或完整 PBR loss。

---

## 7. Baseline 与消融

### 7.1 必须实现的 baseline

| ID | 方法 | 作用 |
|---|---|---|
| B0 | Neutral CAD-GS | 只验证几何和相机流程，不做迁移 |
| B1 | Global reference code | 所有 Gaussian 使用同一个参考图全局 feature；这是最重要对照 |
| B_sem-2D | Simple semantic statistics | semantic mask + mean RGB / colour histogram / shallow texture statistics -> 对应 Gaussian；回答是否真的需要学习模型 |
| B2 | StyleGaussian-global | 在 CAD-anchored GS 上使用 StyleGaussian 风格嵌入/解码，不加入语义区域 |
| B3 | MaterialMVP zero-shot | **最重要、最危险 baseline**；输入 target mesh + donor reference，比较 PBR 质量、跨视角一致性与区域可控性 |
| B4 | MaterialMVP semantic-submesh | 对 wall/floor/... 子 mesh 分别运行 MaterialMVP 后合并；避免只比较其不擅长的整房间设置 |
| B5 | TRELLIS.2 texturing zero-shot | 测试强基础模型在房间 mesh 上的零样本能力 |
| Ours | Semantic reference codes | 每个 Gaussian 使用同类区域的 reference token |

MaterialMVP 的比较必须回答：即使它在 PBR realism 和 relighting 上更强，本文是否能在 `cross-geometry architectural correspondence`、局部编辑控制和 Semantic Leakage 上显著更好。

RoomPainter 是重要的场景级相关工作，但它是 text-conditioned mesh texturing，并非 image-conditioned Gaussian transfer。若官方实现可稳定复现，将 reference image 自动 caption 后作为额外定性 baseline；不要因复现困难阻塞主实验。

### 7.2 必须做的消融

| 消融 | 改动 | 回答的问题 |
|---|---|---|
| A1 No semantics | 五个 token 合并成一个 | 语义区域是否真正有用？ |
| A2 Predicted masks | GT mask 换成预训练分割结果 | 对分割误差是否稳健？ |
| A3 One reference | 只输入 1 张图 | 少视图条件下是否仍有效？ |
| A4 No view regularization | 去掉跨视角正则 | 多视角稳定性来自哪里？ |
| A5 Unfreeze geometry | 允许 center/scale 更新 | 外观优化是否损害 CAD 几何？ |
| A6 No semantic embedding | 仅输入 reference token | 模型是否真正利用类别身份？ |

---

## 8. 评价指标

### 8.1 图像质量

- PSNR、SSIM、LPIPS：在 held-out target views 对 oracle GT 计算；
- DINO cosine / CLIP-I：按语义区域分别计算参考相似度；
- 所有区域指标先按类别平均，再按场景平均。

### 8.2 语义泄漏

定义 `Semantic Leakage Score`：改变 donor wall reference、保持其他类别不变，测量非 wall 区域的输出变化。

```text
Leakage(wall) =
  mean(|I_before - I_after| outside wall mask)
  ------------------------------------------------
  mean(|I_before - I_after| inside wall mask) + eps
```

越低越好。对五类分别计算。这是最能支持“受控区域级迁移”贡献的指标。

报告形式：

```text
Global leakage: mean ± std
Ours leakage:   mean ± std
Paired delta:   mean difference + 95% CI
Effect size:    paired Cohen's d 或 rank-biserial correlation
Test:           paired t-test；非正态时用 Wilcoxon signed-rank
```

不人为设定 20% 为 meaningful threshold。

### 8.3 多视角一致性

- 用 target depth 和 camera pose 将相邻视图 warp 到同一视图；
- 只在可见且非遮挡区域计算 warped LPIPS/L1；
- 若资源允许，再加入 MEt3R 作为学习型多视角一致性指标。

### 8.4 CAD 几何保持

- Gaussian center 到原 CAD surface 的 point-to-mesh distance；
- 冻结几何时该值应接近数值误差；
- 报告 95th percentile 和 maximum distance。

### 8.5 人工评价

真实 ScanNet++ reference 没有迁移 ground truth，因此进行盲测：

- 20–30 名参与者；
- 每对展示 reference、global baseline 和 ours 的同一路径视频；
- 询问：区域匹配、参考相似、跨视角稳定、总体真实感；
- 随机左右位置并报告 preference rate 与置信区间。

---

## 9. 实验执行顺序

### Stage 0：环境与数据许可

- [ ] 接受 HSSD 数据许可并只下载 1 个场景；
- [ ] 保存 HSSD、OpenRooms 和后续数据的原始 LICENSE/TERMS 记录；
- [ ] 申请 3D-FRONT/3D-FUTURE 数据访问；
- [ ] 申请 ScanNet++ 访问，仅下载少量场景；
- [ ] 克隆官方 3DGS 和 StyleGaussian；
- [ ] 记录 CUDA、PyTorch、GPU 型号和依赖版本；
- [ ] 不要在许可未确认前重新分发原始数据。

**完成标准**：一个 HSSD 房间可在 Blender 中正确加载，几何、语义和纹理没有错位；3D-FRONT 是否已获批不影响进入 Stage 1。

### Stage 1：单样本数据闭环

- [ ] 选择 1 个 donor 和 1 个不同几何的 target；
- [ ] 输出五类 semantic mesh masks；
- [ ] 渲染 3 张 donor reference；
- [ ] 构造 oracle transferred target；
- [ ] 渲染 8 个 target GT views；
- [ ] 将 target mesh 转为 semantic Gaussians；
- [ ] 验证每个 Gaussian 的 label 与 mesh face 一致。

**完成标准**：可以可视化 wall/floor/... 五种颜色的 Gaussian scene，并能从任意相机渲染。

### Stage 2：最简单、当前最高优先级的关键实验

先比较三个方法：

1. `Global`: 一个全局 reference code；
2. `B_sem-2D`: semantic mask + 简单 RGB/texture statistics；
3. `Semantic`: 五个 DINO semantic reference codes。

其余网络、训练步数、相机和损失完全相同。

输出每个 pair 的：

```text
Global render | B_sem-2D render | Semantic render | GT render
Region LPIPS  | Region DINO      | Leakage         | Multi-view error
```

**完成标准**：在配对 pilot scenes 上，Semantic 相对 Global 的 leakage 差值置信区间支持降低趋势，并提高 region similarity；同时必须说明它相对简单 B_sem-2D 的增益。如果 B_sem-2D 已达到相近结果，优先增强 Level 2 数据难度，而不是立刻增加网络规模。

如果此阶段没有优势，应先检查：

- semantic mask 是否准确；
- donor material 是否在 reference 中可见；
- Gaussian 标签是否正确；
- oracle transfer 的 UV scale 是否一致；
- global baseline 是否已经足够表达简单材质。

不要直接增加大模型来掩盖数据问题。

### Stage 3：StyleGaussian 改造

- [ ] 复现 StyleGaussian 官方例子；
- [ ] 在一个 CAD-anchored Gaussian target 上跑通 global style；
- [ ] 将 global style embedding 改为 semantic-specific embeddings；
- [ ] 使用 Gaussian semantic ID 选择对应 embedding；
- [ ] 多视角联合解码；
- [ ] 与 Stage 2 的小型 MLP 比较。

**完成标准**：语义版本在保持跨视角稳定的同时，降低非目标区域变化。

### Stage 4：强 baseline

- [ ] MaterialMVP paper checkpoint：整个 room shell 一次推理；
- [ ] MaterialMVP：按 semantic submesh 分开推理再合并；
- [ ] TRELLIS.2 texturing：整个 target mesh；
- [ ] StyleGaussian global；
- [ ] 若可复现，RoomPainter + reference caption。

所有方法使用相同 target geometry、reference images 和 held-out cameras。

### Stage 5：真实照片泛化

- [ ] 从 ScanNet++ 选 5–10 个结构清楚的房间；
- [ ] 使用官方工具将 mesh semantics 投影到 DSLR 图像；
- [ ] 从真实照片提取 semantic reference features；
- [ ] 迁移到 3D-FRONT target CAD 或自建简单 CAD room；
- [ ] 输出固定相机路径视频；
- [ ] 完成人工偏好测试。

---

## 10. 资源与时间估计

### 最小硬件

- 1 张 24 GB GPU：足够做 512–800 px 的 pilot 和小型 decoder；
- 32–64 GB RAM；
- 3D-FRONT/3D-FUTURE 预留约 40–80 GB；
- ScanNet++ 只下载子集，完整数据超过 TB 级，不适合第一轮。

### 建议进度

| 周 | 目标 |
|---|---|
| 1 | 数据许可、一个房间加载、相机与语义验证 |
| 2 | donor-target 配对、oracle transfer、semantic Gaussian 转换 |
| 3 | Global 与 Semantic 小型模型 |
| 4 | pilot 评估、修正数据和指标 |
| 5–6 | StyleGaussian semantic conditioning |
| 7 | MaterialMVP/TRELLIS.2 baseline |
| 8 | ScanNet++ 真实图验证、视频和用户研究准备 |

---

## 11. Go / No-Go 判据

进入完整论文实验前，pilot 必须满足：

1. Semantic 方法的 region DINO similarity 在配对场景上稳定优于 Global；
2. `Leakage_semantic < Leakage_global`，并报告 paired delta、95% CI、effect size 和统计检验；不使用任意百分比阈值；
3. Semantic 相对 B_sem-2D 有可解释增益，或明确显示简单方法在哪种 Level 2/3 情况下失效；
4. held-out view LPIPS / warped consistency 不劣于 Global；
5. CAD point-to-surface distance 接近零；
6. 多个 donor-target pair 上趋势一致，而不是只成功一个案例；
7. 1 / 3 / 5 references 形成可解释的性能曲线。

如果只提高“好看程度”，但没有降低语义泄漏或提高区域对应，则还不足以支持论文贡献。

---

## 12. 最终论文中需要回答的问题

1. 为什么不能直接使用 StyleGaussian 的全局风格迁移？
2. 语义区域条件到底改善了什么，可否通过 leakage 指标定量证明？
3. CAD anchoring 是否确实保证了几何不被外观优化破坏？
4. 当参考图缺少 ceiling/window 时如何处理？
5. 合成 3D-FRONT 训练能否泛化到 ScanNet++ 真实照片？
6. MaterialMVP 在整个房间和分语义子 mesh 上分别表现如何；本文为何不是重复 image-conditioned PBR generation？
7. 结果是否在未见过的相机视角保持稳定？

8. 简单的 semantic RGB/texture statistics 是否已经足够，学习式 DINO feature transfer 的必要性在哪里？
9. Level 1、Level 2、Level 3 难度提升时，各方法如何退化？

这些问题都能得到清楚答案时，实验才形成一条完整的论文证据链。

---

## 13. 版本与来源记录

本文档于 **2026-08-18** 根据以下官方论文、项目和数据文档更新：

- 3D Gaussian Splatting：https://arxiv.org/abs/2308.04079
- StyleGaussian：https://github.com/Kunhao-Liu/StyleGaussian
- MaterialMVP：https://github.com/ZebinHe/MaterialMVP
- RoomPainter：https://openaccess.thecvf.com/content/CVPR2025/html/Huang_RoomPainter_View-Integrated_Diffusion_for_Consistent_Indoor_Scene_Texturing_CVPR_2025_paper.html
- TRELLIS.2：https://github.com/microsoft/TRELLIS.2
- 3D-FRONT：https://arxiv.org/abs/2011.09127
- HSSD：https://3dlg-hcvc.github.io/hssd/
- OpenRooms：https://github.com/ViLab-UCSD/OpenRooms
- ScanNet++：https://scannetpp.mlsg.cit.tum.de/scannetpp/

开始实现前应再次检查各仓库最新 commit、模型权重可用性、数据下载条款和学术许可。
