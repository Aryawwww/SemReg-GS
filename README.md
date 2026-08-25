# SemReg-GS 实验记录

## 目标

研究在 source building 与 target CAD 几何不同、没有逐点对应时，能否用建筑语义
`wall / floor / ceiling / door / window / other` 作为 correspondence，将参考照片的
区域外观迁移到 CAD-anchored 3D Gaussians，同时保持几何、区域可控性和跨视角一致性。

当前实现是便携式 point-zbuffer smoke baseline，不是正式各向异性 CUDA Gaussian
rasterizer。单个 source→target pair 的 cross-geometry pilot 已跑通 Global 与 B_sem-2D，
但尚不足以支持跨场景总体方法结论。

## 当前状态

| 阶段 | 状态 | 已得到的结果 |
|---|---|---|
| 环境与 HSSD 下载 | 部分失效 | HSSD、Blender 和 DINOv2 缓存仍在；原 `semreg-gs-v1` Conda 环境当前不存在，需恢复 PyTorch/CUDA 环境 |
| Source 资产与语义审计 | 完成 | 486 meshes、530,720 polygons；六类逐面映射可复现 |
| 原始 3 donor / 8 target 多模态渲染 | 完成 | 512×512 RGB/semantic/depth/normal/camera；11/11 RGB 哈希不同 |
| CAD-anchored Gaussian 初始化 | 完成 | 100,000 Gaussians；最大几何重建误差 `1.134145e-6 m` |
| Step 6 中性外观训练 | 完成 | 几何冻结；300 steps；83.11% Gaussians 可见；训练拟合 PSNR `58.76 dB` |
| Step 7 单场景 baseline | 完成 | Global、B_sem-2D、Global-DINO、Semantic-DINO 已做原始 3/8 视角比较 |
| Window 语义覆盖修复 | 完成 | coverage gate passed；donor/train/held-out 六类均超过 1,000 pixels |
| Source split 冻结 | 完成 | `data/splits/107734119_175999932_split.json` |
| 第二场景选择与审计 | 完成 | target `103997424_171030444`；1,027 meshes、2,617,276 polygons，六类均存在 |
| Target 多视角与协议 | 完成 | 8/8 RGB 哈希不同；六类 coverage passed；3 source donor、6 target train、2 target held-out 已冻结 |
| Target CAD Gaussians | 完成 | 100,000 Gaussians，六类齐全；最大锚定误差 `2.049921e-6 m`，几何字段冻结 |
| Cross-geometry 2D transfer | 完成（单 pair pilot） | Global 与 B_sem-2D 已在 `view_00/view_07` 评估；B_sem-2D leakage 降低 17.21 个百分点，但 PSNR 下降 0.87 dB |
| Multi-view consistency | 完成（单 pair pilot） | 独立 living-room 重叠视角已冻结；5,828 个双向对应点；B_sem-2D warp L1 `0.02858` |
| Source-only DINO | 代码完成、未运行 | donor-only ridge decoder 与评估入口已实现；因原 PyTorch Conda 环境缺失尚无正式结果 |
| 强 baseline 和多 pair | 未完成 | StyleGaussian、MaterialMVP、TRELLIS.2、10 pairs 均未运行 |

## 已完成实验及结果

### 1. Source 场景与六类语义

- Source scene：`107734119_175999932`，尺寸约 `8.71 × 11.13 × 2.81 m`。
- 资产：486 meshes、426,047 vertices、530,720 polygons、181 materials、61 instances。
- 58/61 instances 成功解析；模板 `224-132` 的 3 个未解析实例暂归 `other`。
- Polygon 标签：wall 1,375、floor 40、ceiling 22、door 4,044、window 258、other 524,981。
- `FP_GLASS → window` 可能包含玻璃门，是进入正式数据集前需人工复核的边界。

### 2. 多模态渲染与 Gaussian 初始化

- 原始正式集合为 3 donor + 8 target，分辨率 512×512。
- 每个视角包含 `rgb.png / semantic.png / depth.exr / normal.exr / camera.json`。
- RGB 验证 passed：11 个视角有 11 个不同 SHA-256；旧 compositor 产生的 0/1
  近黑 RGB 已确认无效，不应用于任何结论。
- 按三角面面积采样 100,000 个 Gaussians：wall 40,102、floor 9,505、ceiling 7,651、
  door 6,224、window 268、other 36,250。
- `semantic_id`、`source_face_id` 和 CAD anchoring 验证通过；训练只优化 RGB/SH-DC，
  xyz、normal、rotation、scale 和 semantic ID 冻结。

### 3. Step 7 原始单场景结果

下列结果来自原始 3 donor / 8 target 设置；DINO held-out 为 `view_06/view_07`。

| 方法 | 评估 | L1 ↓ | PSNR ↑ | 代理 leakage ↓ |
|---|---|---:|---:|---:|
| Global | 8 targets | 0.29539 | 9.96 dB | 100.00% |
| B_sem-2D | 8 targets | 0.30148 | 9.10 dB | 54.99% |
| Global | held-out 2 views | 0.30719 | 9.94 dB | 100.00% |
| B_sem-2D | held-out 2 views | 0.34810 | 8.40 dB | 60.72% |
| Global-DINO | held-out 2 views | **0.14428** | **14.54 dB** | 95.55% |
| Semantic-DINO | held-out 2 views | 0.14493 | 14.22 dB | **94.33%** |

结论：B_sem-2D 明显降低颜色原型 leakage，但重建误差变差；Semantic-DINO 相对
Global-DINO 只降低 `1.22` 个百分点 leakage，同时 L1 增加约 `0.00065`、PSNR
下降 `0.32 dB`。因此尚不能声称 semantic conditioning 全面优于 global conditioning。

该实验仍是同一场景的 6-view supervision，可能记忆 target 外观；原 held-out 中
floor、ceiling、window 为 0 pixels，不能支持六类总体结论。

### 4. Window coverage 与冻结 split

为补足 window，生成了 3 个 window donor 和 6 个 window target 候选，并使用带
`0.50 m` 距离 gate 的 ray-cast proxy 标签传播。该 window 标签是可审计 proxy，
不是原始逐面 GT。

最终冻结选择：

- donor：`reference_00 / reference_01 / reference_window_02`
- held-out：`view_05 / view_window_heldout_02`
- 其余候选作为 train

最低覆盖类别为 held-out door，共 16,672 pixels，高于 1,000-pixel gate。

### 5. Target 场景、多视角与冻结协议

- Target scene：`103997424_171030444`，包含 `living room / kitchen / bathroom /
  bedroom / office / hallway / dining room`，没有 `utilityroom`。
- scene-aware 相机配置已替代固定房间计划；缺失、额外或重复房间会直接报错，不会
  静默替换。
- Target 生成 8 个 512×512 多模态视角，每个视角具有
  `rgb.png / semantic.png / depth.exr / normal.exr / camera.json`。
- RGB audit passed：8 个视角有 8 个不同 SHA-256，动态范围与 8-bit 色值数量正常。
- Target coverage audit passed；正式 target split 为：
  - train：`view_01 / view_02 / view_03 / view_04 / view_05 / view_06`
  - held-out：`view_00 / view_07`
- Held-out 六类均超过 1,000 pixels；最低类别为 door，共 17,096 pixels。
- `protocol_manifest.json` 固定了 3 个 source donor、6 个 target train、2 个 target
  held-out 及其 55 个模态文件哈希。协议规定 appearance 只能来自 source donor，
  target held-out 仅用于评估。

### 6. Target CAD-anchored Gaussians

- 从 target CAD 按三角面面积采样 100,000 个 Gaussians。
- 六类计数：wall 36,845、floor 10,465、ceiling 10,333、door 5,309、window 2,941、
  other 34,107。
- `semantic_id / source_face_id / source_triangle / barycentric` 等字段验证通过。
- center-to-source-triangle 最大误差为 `2.049921e-6 m`，低于 `1e-5 m` gate。
- position、rotation、scale 全部冻结；source mesh SHA-256 与初始化 metadata 一致。

### 7. Cross-geometry Global 与 B_sem-2D pilot

两个方法只读取冻结的 `reference_00 / reference_01 / reference_window_02`，逐文件校验
哈希后，将 donor 外观赋给 target CAD Gaussians。评估只使用冻结的 target held-out
`view_00 / view_07`，指标范围是 point-zbuffer 实际覆盖的 52,126 pixels。

| 方法 | L1 ↓ | PSNR ↑ | 代理 leakage ↓ |
|---|---:|---:|---:|
| Global | **0.25652** | **10.80 dB** | 47.40% |
| B_sem-2D | 0.25953 | 9.93 dB | **30.19%** |

B_sem-2D 相对 Global 将 leakage 降低 `17.21` 个百分点，但 L1 增加 `0.00300`，
PSNR 下降 `0.87 dB`。这支持“语义原型提高区域可控性”的有限结论，但不支持其
重建质量全面优于 Global。当前 point renderer 的 held-out 平均覆盖率约 10.22%，
因此数值不能与完整各向异性 Gaussian rasterizer 结果等价比较。

### 8. 独立 multi-view consistency pilot

原 held-out `view_00` 与 `view_07` 位于不同房间，没有共同视野。为避免修改主
train/held-out split，另建并冻结了两个只用于 multi-view evaluation 的 living-room
视角 `consistency_00 / consistency_01`；二者禁止参与外观训练。

双向 depth/camera 重投影得到 5,828 个通过 0.05 m occlusion gate 的对应点：

| 方法 | Warp L1 ↓ | Warp RMSE ↓ | 对应点 |
|---|---:|---:|---:|
| Global | 0.00000 | 0.00000 | 5,828 |
| B_sem-2D | 0.02858 | 0.09609 | 5,828 |

Global 是全场常量颜色，warp error 天然为 0；该结果只表示跨视角颜色恒定，不能解释为
视觉质量最好。B_sem-2D 的非零误差主要来自语义边界、离散 point coverage 和最近邻
重投影，需在正式 Gaussian rasterizer 中复核。

### 9. Source-only DINO 当前状态

旧 `train_dino_gaussian_decoder.py` 使用 target train RGB 监督 decoder，不符合当前
冻结协议，因此旧 Global-DINO/Semantic-DINO 不能作为正式 cross-geometry 结果。

新的 `build_source_only_dino.py` 已实现：仅使用冻结 source donor 的 DINO patch token
与 donor RGB 拟合 ridge RGB decoder，再以 global/class DINO code 为 target Gaussians
赋色；代码显式记录 `target_rgb_accessed: false`，并接入 held-out 与 consistency 评估。
本机已有 `facebookresearch_dinov2_main` 和 `dinov2_vits14_pretrain.pth` 缓存，但原
`semreg-gs-v1` Conda 环境已不存在，现存 `semreg-gs` 又没有 PyTorch，所以该步骤尚未运行。

## 已运行但未形成完整实验的内容

### 7 donor / 16 target 临时重跑

2026-08-24 在包含全部候选的目录上又运行了一次：

- 中性训练完成：16 targets、300 steps、83,169/100,000 Gaussians 可见（83.17%）。
- 中性 appearance/semantic 渲染完成：16 views。
- baseline appearance 重建完成：读取了 7 donor，六类都有观测。
- Global 渲染完成：16 views。
- **Semantic-2D 的 render report 仍是旧的 8-view 文件。**
- **`step7_baselines/metrics.json` 仍是旧的 3/8 实验结果。**

这次 7/16 运行没有按冻结 split 分离 donor/train/held-out，会把候选集合混在一起；
即使补完 Semantic-2D 和 metrics，也只能作为目录完整性测试，不能作为正式实验结果。
后续不要把旧 metrics 与新 appearance/render 混合引用。

### 其他无效或仅工程验证的运行

- 旧 0/1 RGB 的中性训练 loss 下降无意义，已废弃。
- 128×128、100-step 中性训练只验证修复后的代码能收敛，不是正式结果。
- 16-view 中性训练是训练拟合检查，不是 held-out 泛化实验。

## 尚未完成

1. 恢复包含 PyTorch/CUDA 的 `semreg-gs-v1` 或等价环境，运行 source-only
   Global-DINO 与 Semantic-DINO；不能使用旧 target-supervised DINO 结果替代。
2. 加入 region LPIPS 与 evaluation-only region DINO 指标，并冻结模型版本、权重哈希、
   resize/crop 和 mask aggregation 规则。
3. 将 point-zbuffer baseline 替换或补充为正式各向异性 CUDA Gaussian rasterizer，
   重新报告 coverage、held-out 指标与 warp error。
4. 扩展到至少 10 个 HSSD pairs，报告 paired delta、95% CI、effect size 和统计检验。
5. 加入 StyleGaussian、MaterialMVP whole-room、MaterialMVP semantic-submesh、TRELLIS.2。
6. 之后再扩展 3D-FRONT，并用 OpenRooms 检查材质/光照解耦、ScanNet++ 检查真实域。

## 下一步执行顺序

### Step A–C — Target 数据与 Gaussians（已完成）

scene-aware 相机、多模态渲染、coverage audit、协议冻结和 target CAD Gaussians 均已完成。
主 held-out 和额外 consistency views 使用不同用途字段，不允许混入训练。

### Step D — 完成 source-only 四方法比较

Global 与 B_sem-2D 已完成。下一项是恢复 PyTorch 环境后运行：

```bat
scripts\run_step17_source_only_dino.cmd
```

该入口不得读取 target RGB；完成后将 DINO 的 held-out L1/PSNR/leakage 与 consistency
warp 指标和 2D 两方法合并成同一 pilot 表。

### Step E — 感知指标与正式 rasterizer

冻结 LPIPS/DINO evaluation protocol，并迁移到正式各向异性 Gaussian rasterizer。
Global 的常量颜色 warp=0 必须和 held-out 视觉误差共同解释，不能单独作为方法优势。

### Step F — 扩展与统计

pilot 通过后扩展到 10 pairs，再加入强 baseline。主要判断依据是跨 pair 的 region
DINO/LPIPS、leakage、multi-view consistency 和 geometry preservation，不能依据单张
最好看的图。

## 关键入口与产物

```text
scripts/run_step6_fixed.cmd           原始 3/8 source smoke pipeline
scripts/run_step7_baselines.cmd       Global 与 B_sem-2D（默认扫描整个目录，正式实验慎用）
scripts/run_step7_dino.cmd            原始单场景 DINO smoke comparison
scripts/run_step8_window_views.cmd    window-targeted cameras 与 coverage audit
scripts/run_step8_prepare_pair.cmd    冻结 source split并选择第二场景
scripts/run_step9_target_audit.cmd    target 资产与语义审计
scripts/run_step10_target_multiview.cmd       target scene-aware 多模态渲染
scripts/run_step11_freeze_and_target_gaussians.cmd  冻结 pair protocol并初始化 target Gaussians
scripts/run_step12_cross_geometry_2d.cmd      source-only Global/B_sem-2D 迁移与 held-out 渲染
scripts/run_step13_evaluate_cross_geometry_2d.cmd  held-out L1/PSNR/leakage/geometry 评估
scripts/run_step15_consistency_views.cmd      独立重叠 consistency views 与 protocol v2
scripts/run_step16_consistency_warp.cmd       双向 depth/camera warp consistency
scripts/run_step17_source_only_dino.cmd       donor-only DINO 两方法（待恢复 PyTorch 环境）

outputs/smoke/107734119_175999932/    source smoke 与历史结果
data/splits/107734119_175999932_split.json
data/processed/pairs/107734119_175999932__to__103997424_171030444/pair_manifest.json
data/processed/pairs/107734119_175999932__to__103997424_171030444/protocol_manifest.json
data/processed/pairs/107734119_175999932__to__103997424_171030444/protocol_consistency_v2.json
data/processed/semantic_gaussians/103997424_171030444/validation.json
outputs/pairs/107734119_175999932__to__103997424_171030444/cross_geometry_2d/metrics.json
outputs/pairs/107734119_175999932__to__103997424_171030444/cross_geometry_2d/consistency_warp_metrics.json
```

`pair_manifest.json` 的旧状态字段仍为 `audited_pending_multiview_render`，但后续不可变
`protocol_manifest.json`、`protocol_consistency_v2.json` 和产物哈希已经记录真实进度；
后续应以这些 protocol 为实验依据，不应回退引用旧状态字段或单场景 7/16 metrics。

## 数据与许可

- HSSD：当前 smoke/pilot 主数据，CC BY-NC 4.0。
- 3D-FRONT/3D-FUTURE：获批后用于扩大规模和对齐房间级 baseline。
- OpenRooms：用于材质与光照解耦消融。
- ScanNet++：用于 synthetic-to-real 验证。

正式使用任何外部模型或数据前，应再次核对仓库版本、权重许可和数据条款。
