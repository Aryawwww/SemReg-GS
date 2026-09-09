# SemReg-GS

## 项目目标

研究 source building 与 target CAD 几何不同、没有逐点对应时，能否用建筑语义
`wall / floor / ceiling / door / window / other` 建立外观对应，将少量 source reference 的
区域外观迁移到 CAD-anchored 3D Gaussians，同时保持 CAD 几何、区域可控性和跨视角一致性。

当前要证明的是“语义类别选择器是否带来跨几何区域控制”，不是 photorealism，也不是
learned feature 一定优于简单颜色原型。总体设计见 `PROJECT_PLAN.md`，当前进度与下一步见
`MILESTONES.md`；冻结的 protocol manifest 和产物哈希是实验事实的最终依据。

## 当前结论

已完成一个冻结 HSSD pilot pair：

```text
107734119_175999932 → 103997424_171030444
```

该 pair 使用 3 个 source donor、6 个 target train、2 个 target held-out，以及独立的
consistency views。target 使用 100,000 个 CAD-anchored Gaussians；几何与 semantic ID
冻结，最大 center-to-surface 误差为 `2.049921e-6 m`。

在 point-zbuffer 覆盖的 52,126 个 held-out pixels 上：

| 方法 | L1 ↓ | PSNR ↑ | Shared-palette macro leakage ↓ | Warp L1 ↓ |
|---|---:|---:|---:|---:|
| Global | 0.25652 | **10.80 dB** | 83.33% | 0.00000 |
| B_sem-2D | 0.25953 | 9.93 dB | **15.86%** | 0.02858 |
| Global-DINO | 0.25652 | **10.80 dB** | 83.33% | 0.00000 |
| Semantic-DINO | **0.25615** | 10.09 dB | **15.86%** | **0.02496** |

单类别 intervention 的宏平均结果：

| 方法 | Target response L1 ↑ | Non-target spill L1 ↓ | Selectivity ↑ |
|---|---:|---:|---:|
| Global | 0.14902 | 0.14902 | 0.5000 |
| Global-DINO | 0.15033 | 0.15033 | 0.5000 |
| B_sem-2D | 0.12569 | **0.00979** | 0.9422 |
| Semantic-DINO | 0.12602 | 0.00979 | **0.9422** |

可支持的结论：

- semantic selector 在该 pilot 中显著提高区域隔离和编辑选择性；
- 它没有提高像素重建质量，Global 的 held-out PSNR 更高；
- B_sem-2D 与 Semantic-DINO 几乎相同，因为当前 DINO 最终仍将每类压缩成一个 RGB，
  不能证明 learned feature transfer 的必要性；
- Global 的 warp error 为 0 来自全场常量颜色，不能解释为视觉质量更好；
- point renderer 的 held-out 平均覆盖率约 10.22%，结果不能替代正式各向异性 rasterizer
  或多 pair 统计。

## 已定位的评估问题

rendered Gaussian semantic 与 held-out ray-cast proxy semantic 的总体 agreement 为
`69.81%`，`view_07` 仅 `40.24%`；主要混淆为 proxy `other` 被标为 `wall/floor`。
spill decomposition 显示两个 semantic 方法均为 `intrinsic_spill_l1=0`、
`intrinsic_selectivity=1.0`。此前的 apparent spill 几乎全来自标签/可见性不一致，而非
表示内部跨类污染。已实现 mesh-depth occlusion gate 入口，但尚未运行。

正式实验必须分开报告 intrinsic spill、proxy-label apparent spill 和 semantic alignment，
不能使用旧的 method-specific prototype leakage 排名方法。

## 当前工作

当前唯一执行重点是审计四个已下载的 HSSD GLB：
`102344094 / 102344328 / 102344439 / 102815835`。

每个场景需完成 Blender 加载、六类逐面映射、semantic preview、polygon/area 统计和可用
房间检查，通过后才能冻结 10-pair cohort。同时补齐 Blender headless 与正式 3DGS CUDA
extension smoke test。

暂不开展：100–300 pairs、StyleGaussian 改造、TRELLIS.2、photorealism 优化，或继续比较
当前 class-constant DINO。

## 关键入口与产物

```text
scripts/run_step9_target_audit.cmd             HSSD 资产与语义审计
scripts/run_step10_target_multiview.cmd        scene-aware 多模态渲染
scripts/run_step11_freeze_and_target_gaussians.cmd  冻结协议并初始化 CAD Gaussians
scripts/run_step17_source_only_dino.cmd        source-only DINO pilot
scripts/run_step18_shared_palette_leakage.cmd  方法无关 leakage
scripts/run_step19_semantic_interventions.cmd  单类别 intervention
scripts/run_step21_semantic_alignment_audit.cmd semantic alignment
scripts/run_step22_spill_decomposition.cmd     intrinsic/proxy spill 分解
scripts/run_step23_depth_gated_alignment.cmd   mesh-depth gate 复核
scripts/download_hssd_smoke.py                 HSSD 单场景/批量下载

configs/hssd_expansion.json
configs/hssd_candidates_batch01.json
data/processed/pairs/107734119_175999932__to__103997424_171030444/protocol_manifest.json
data/processed/pairs/107734119_175999932__to__103997424_171030444/protocol_consistency_v2.json
data/processed/semantic_gaussians/103997424_171030444/validation.json
outputs/pairs/107734119_175999932__to__103997424_171030444/pilot_four_method_metrics.json
```

## 数据范围

- HSSD（CC BY-NC 4.0）：当前主实验数据；
- 3D-FRONT/3D-FUTURE：获批后仅作跨数据集扩展，不阻塞主线；
- ScanNet++：后期真实域验证；
- OpenRooms：后期材质/光照解耦消融。
