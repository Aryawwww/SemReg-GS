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
| Source-only DINO | 完成（单 pair pilot） | donor-only ridge decoder 已运行；3 donor、4,078 patches、六类均有样本，`target_rgb_accessed=false`；四方法汇总已生成 |
| Shared-palette leakage | 完成（单 pair pilot） | 方法无关 donor palette 已冻结；Global/Global-DINO 均为 47.40%，B_sem-2D/Semantic-DINO 均为 30.19%，修复了旧指标跨方法不可比问题 |
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

### 9. Source-only DINO 四方法 pilot

旧 target-supervised DINO 训练代码与单场景 Step 7 入口已删除；它们不符合当前冻结协议，
不能作为正式 cross-geometry 结果。

新的 `build_source_only_dino.py` 已运行：仅使用冻结 source donor 的 DINO patch token
与 donor RGB 拟合 ridge RGB decoder，再以 global/class DINO code 为 target Gaussians
赋色。运行使用 `dinov2_vits14`、384 维特征、ridge `0.01`，共 4,078 个有效 patch；
wall/floor/ceiling/door/window/other 分别为 1,051/133/624/574/807/889，无缺失类回退。
DINO 权重 SHA-256 为
`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`，报告明确记录
`target_rgb_accessed: false`。

协议锁定的四方法结果如下；held-out 指标均基于 point renderer 覆盖的 52,126 pixels，
warp 均基于 5,828 个双向对应点。

| 方法 | L1 ↓ | PSNR ↑ | 当前代理 leakage* ↓ | Warp L1 ↓ |
|---|---:|---:|---:|---:|
| Global | 0.25652 | **10.80 dB** | 47.40% | 0.00000 |
| B_sem-2D | 0.25953 | 9.93 dB | 30.19% | 0.02858 |
| Global-DINO | 0.25652 | **10.80 dB** | 72.18% | 0.00000 |
| Semantic-DINO | **0.25615** | 10.09 dB | 30.19% | **0.02496** |

Semantic-DINO 相对 Global-DINO 的 L1 仅改善 `0.00038`，PSNR 反而下降 `0.71 dB`；
相对 B_sem-2D，L1 改善 `0.00338`、PSNR 提高 `0.16 dB`、warp L1 降低 `0.00362`。
这些差异来自单 pair、低覆盖 point-zbuffer pilot，尚不足以证明 learned DINO feature
优于简单 semantic prototype。

`*` 上表中的旧 method-specific leakage 不能跨方法解释：它将每个方法的预测 RGB 与该方法自己的 donor class
prototype 比较。Global 与 Global-DINO 的 held-out L1/PSNR 完全相同，却分别得到 47.40%
与 72.18% leakage，说明该数值受到 prototype 定义影响；Semantic-2D 与 Semantic-DINO
恰好同为 30.19% 也不代表两者 controllability 相同。在冻结方法无关的 reference palette
或实施单区域干预测试前，不再用该 leakage 排名方法或判断 Gate B。

### 10. Shared-palette leakage 复核

Step 18 从协议冻结的 3 个 source donor 生成唯一 reference palette，四方法共享该 palette。
协议与 donor 哈希校验通过，且 `target_rgb_accessed=false`。

| 方法 | Micro leakage ↓ | Macro leakage ↓ |
|---|---:|---:|
| Global | 47.40% | 83.33% |
| Global-DINO | 47.40% | 83.33% |
| B_sem-2D | **30.19%** | **15.86%** |
| Semantic-DINO | **30.19%** | **15.86%** |

修正后 Global 与 Global-DINO 完全一致，证明旧 72.18% Global-DINO leakage 是
method-specific prototype 造成的评估伪差异。两个 semantic 方法的 confusion matrix
逐元素完全相同；其主要错误来自 `other`（51.93%）、door（22.08%）和 window（11.00%）。
原因不是 DINO 没有提取到特征，而是当前 Semantic-DINO 最终只解码每类一个 RGB，和
B_sem-2D 一样是 class-constant 表示；DINO 的类内信息在求均值时被丢弃。因此当前实现
只能证明 semantic class selector 的区域隔离价值，不能证明 learned feature 的必要性。

下一步先运行 Step 19 的单类别干预，直接测 target response、non-target spill 与
selectivity。若 semantic 方法相对 global 方法没有明显 spill 优势，应先修复 renderer
边界/标签；若有优势，再设计保留类内变化的 multi-prototype 或 spatial decoder，避免
继续比较两个本质上都是 class-constant 的方法。

### 11. Semantic intervention 结果

Step 19 对六类分别施加固定 RGB edit。Global 方法没有 class selector，因此请求任一类
编辑都会改变全部 Gaussians；semantic 方法只改变对应 `semantic_id`。

| 方法 | Target response L1 ↑ | Non-target spill L1 ↓ | Selectivity ↑ |
|---|---:|---:|---:|
| Global | 0.14902 | 0.14902 | 0.5000 |
| Global-DINO | 0.15033 | 0.15033 | 0.5000 |
| B_sem-2D | 0.12569 | **0.00979** | 0.9422 |
| Semantic-DINO | 0.12602 | 0.00979 | **0.9422** |

semantic selector 将平均非目标串扰降低约 93.5%，支持区域可控性的机制性结论；两个
semantic 方法仍无实质差异。wall 是明显异常类：spill `0.04487`、selectivity `0.7641`，
而 ceiling/window 的 selectivity 均超过 `0.995`。下一步用 Step 20 在 0/2/4/8/16 px
边界腐蚀下复算指标；若 wall spill 随 margin 快速下降，主要原因是 point splat、proxy
semantic 与可见表面边界错位，而不是区域内部 appearance 污染。

### 12. Boundary sensitivity 结果

Step 20 已完成。wall spill 从 margin 0 的 `0.04487` 到 margin 16 的 `0.04411`，仅下降
`1.7%`，selectivity 仍为 `0.7716`；因此 wall 问题不是普通的几像素边界污染。
相对地，ceiling/window/other 的 spill 在 16 px 时分别下降 `93.3% / 84.8% / 100%`，
符合边界误差特征。floor spill 从 `0.00765` 增至 `0.01296`，说明错误集中在腐蚀后仍
保留的内部区域。两个 semantic 方法的完整曲线继续一致。

该结果将原因收窄到 target CAD Gaussian `semantic_id` 与 held-out ray-cast proxy semantic
之间的区域级不一致，尤其可能是结构 wall 与 `other` 家具/遮挡面、floor 可见面之间的
标签定义或可见性差异。下一步 Step 21 直接比较 rendered semantic ID 与 held-out proxy，
输出 target→rendered confusion、逐类 precision/recall/IoU、逐视角结果和 mismatch heatmap；
在定位具体混淆方向前不应修改 appearance decoder。

### 13. Semantic alignment audit 结果

Step 21 在 52,126 个 point-renderer covered pixels 上得到总体 agreement `69.81%`；
`view_00` 为 `78.82%`，`view_07` 仅 `40.24%`。最大混淆是 target proxy `other`
被 rendered Gaussian 标为 `wall`（10,810 pixels）与 `floor`（2,609 pixels）。其中
`view_07` 单独贡献 6,261 个 `other→wall` pixels。

wall recall 为 `97.55%`，但 precision 仅 `55.53%`；floor recall 为 `92.59%`，precision
仅 `24.02%`。这说明 wall/floor 并非没有覆盖自身 target region，而是 CAD semantic
surface 大量落在 proxy 定义的 `other` 区域。ceiling/window IoU 分别为 `95.25% / 85.56%`，
说明其标签定义相对一致。mismatch heatmap 也显示 `view_07` 是大面积内部不一致，不是边缘带。

因此 proxy-based intervention spill 同时混合了两件事：表示本身是否跨 semantic_id 修改，
以及 rendered CAD semantic 是否同意 held-out proxy label。Step 22 将两者分解：以 rendered
semantic ID 计算 intrinsic spill，同时统计 proxy apparent spill 中有多少变化恰好位于
`rendered=edited class, proxy!=edited class`。只有 intrinsic spill 才直接衡量表示的区域隔离；
proxy disagreement 应作为数据/标签对齐指标单独报告。

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

1. 修正并冻结方法无关的 leakage/controllability protocol：统一 reference prototype，
   并加入“只改变一个 semantic donor，测量目标类响应与非目标类串扰”的 intervention 指标。
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

### Step D — Source-only 四方法比较（已完成）

Global、B_sem-2D、Global-DINO 与 Semantic-DINO 已完成并汇总到：

```text
outputs/pairs/107734119_175999932__to__103997424_171030444/pilot_four_method_metrics.json
```

该结果通过 source-donor-only 与 `target_rgb_accessed=false` 检查，但当前 leakage 定义
未通过跨方法可比性审计，因此四方法比较只可引用 L1/PSNR/warp，并注明 point renderer
覆盖限制。

### Step E — 先修正 controllability 指标，再加入感知指标

先冻结方法无关的 leakage 与 semantic intervention protocol；随后冻结 LPIPS/DINO
evaluation protocol，并迁移到正式各向异性 Gaussian rasterizer。
Global 的常量颜色 warp=0 必须和 held-out 视觉误差共同解释，不能单独作为方法优势。

### Step F — 扩展与统计

pilot 通过后扩展到 10 pairs，再加入强 baseline。主要判断依据是跨 pair 的 region
DINO/LPIPS、leakage、multi-view consistency 和 geometry preservation，不能依据单张
最好看的图。

## 关键入口与产物

```text
scripts/run_step6_fixed.cmd           原始 3/8 source smoke pipeline
scripts/run_step8_window_views.cmd    window-targeted cameras 与 coverage audit
scripts/run_step8_prepare_pair.cmd    冻结 source split并选择第二场景
scripts/run_step9_target_audit.cmd    target 资产与语义审计
scripts/run_step10_target_multiview.cmd       target scene-aware 多模态渲染
scripts/run_step11_freeze_and_target_gaussians.cmd  冻结 pair protocol并初始化 target Gaussians
scripts/run_step12_cross_geometry_2d.cmd      source-only Global/B_sem-2D 迁移与 held-out 渲染
scripts/run_step13_evaluate_cross_geometry_2d.cmd  held-out L1/PSNR/leakage/geometry 评估
scripts/run_step15_consistency_views.cmd      独立重叠 consistency views 与 protocol v2
scripts/run_step16_consistency_warp.cmd       双向 depth/camera warp consistency
scripts/run_step17_source_only_dino.cmd       donor-only DINO 两方法、评估及四方法汇总（已运行）
scripts/merge_cross_geometry_pilot.py         四方法协议校验与 held-out/warp 汇总
scripts/run_step18_shared_palette_leakage.cmd 方法共享的冻结 donor palette 与 leakage 重评估
scripts/run_step19_semantic_interventions.cmd 单类别编辑的 target response/non-target spill 评估
scripts/run_step20_boundary_sensitivity.cmd   semantic intervention 的边界腐蚀敏感性评估
scripts/run_step21_semantic_alignment_audit.cmd rendered Gaussian semantic 与 held-out proxy 对齐审计
scripts/run_step22_spill_decomposition.cmd     intrinsic spill 与 proxy-label apparent spill 分解

outputs/smoke/107734119_175999932/    source smoke 与历史结果
data/splits/107734119_175999932_split.json
data/processed/pairs/107734119_175999932__to__103997424_171030444/pair_manifest.json
data/processed/pairs/107734119_175999932__to__103997424_171030444/protocol_manifest.json
data/processed/pairs/107734119_175999932__to__103997424_171030444/protocol_consistency_v2.json
data/processed/semantic_gaussians/103997424_171030444/validation.json
outputs/pairs/107734119_175999932__to__103997424_171030444/cross_geometry_2d/metrics.json
outputs/pairs/107734119_175999932__to__103997424_171030444/cross_geometry_2d/consistency_warp_metrics.json
outputs/pairs/107734119_175999932__to__103997424_171030444/cross_geometry_dino/metrics.json
outputs/pairs/107734119_175999932__to__103997424_171030444/cross_geometry_dino/consistency_warp_metrics.json
outputs/pairs/107734119_175999932__to__103997424_171030444/pilot_four_method_metrics.json
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
