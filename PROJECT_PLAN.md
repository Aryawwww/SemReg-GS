# SemReg-GS 总体实验设计

## 研究问题

在 source 与 target CAD 几何不同且没有逐点对应时，建筑语义能否作为 appearance
correspondence，使参考外观迁移到 CAD-anchored 3D Gaussians，并实现可量化的区域控制而
不造成几何漂移？

研究贡献需要覆盖：cross-geometry semantic correspondence、region-specific
controllability、CAD-constrained geometry、few-reference practicality，以及从 uniform
material 到类内变化和真实照片的难度递进。仅做 mesh texturing、global style、附加
semantic label 或展示好看的渲染，不足以回答该问题。

## 实验单位与冻结协议

一个实验单位是冻结的 `source_scene_id → target_scene_id` pair：

- source 与 target 为不同 scene ID；正式新 pair 默认不复用场景；
- source 只提供 3 个 donor views；target 使用 6 train、2 held-out views；
- consistency views 独立冻结，不参与外观构建；
- 六类固定为 `wall / floor / ceiling / door / window / other`；
- 每个 split 每类至少 1,000 pixels；target held-out RGB 不得参与 appearance 构建；
- target 使用 100,000 个 CAD-anchored Gaussians；position、rotation、scale、semantic ID
  冻结，最大 center-to-surface 误差不超过 `1e-5 m`；
- 所有方法共享相机、mask、renderer、edit magnitude 和评估协议。

每个 pair 保存 pair/protocol manifest、文件哈希、多模态视角、coverage audit、Gaussian
validation 和失败原因。未通过 gate 的场景不得静默剔除。

## 分阶段实验

### Level 1：语义选择器是否有效

确认性主比较为 `B_sem-2D − Global-RGB`。逐类改变 donor appearance，比较 target-region
response 与 non-target spill。

主要终点：macro non-target spill L1、macro selectivity，以及分开报告的 intrinsic spill
与 proxy-label apparent spill。

次要终点：held-out L1/PSNR、shared-palette leakage、per-class response、semantic
alignment、coverage、warp error 和 geometry error。Region LPIPS/DINO 只有在模型、权重
哈希、预处理和 mask aggregation 冻结后才进入确认性统计。

### Renderer 复核

先在冻结 pilot 上用正式各向异性 CUDA Gaussian rasterizer 做等价性检查，再运行 10-pair
cohort。不得更改 split 来改善结果。若主要方向反转，优先诊断 coverage、visibility 和
semantic alignment，不进入大型模型开发。

### Level 2：learned feature 是否必要

仅在 Level 1 跨 pair 成立后启动。加入类内多材质、空间变化、纹理尺度与 1/3/5
references，并将 class-constant DINO 改为 multi-prototype 或 spatial decoder。
主比较为 `Semantic-DINO − B_sem-2D`；只有 learned 方法在冻结指标上稳定增益，才支持
learned feature transfer 的必要性。

### 强 baseline 与外部有效性

顺序为 MaterialMVP whole-room、MaterialMVP semantic-submesh、StyleGaussian，再考虑
TRELLIS.2。之后用 ScanNet++ 验证真实域；3D-FRONT 若获批，仅作跨数据集扩展。

## 统计与报告

10-pair 实验按 pair 和 edited semantic class 配对，报告 mean/median/std、逐 pair/逐类
结果、paired delta、bootstrap 95% CI、effect size、统计检验、geometry/coverage/alignment
gate、像素质量代价和失败案例。

paired Cohen's d 用于近似正态差值；非正态时报告 rank-biserial correlation，并以
Wilcoxon 替代 paired t-test。不预设任意“20% improvement”阈值。主要终点需方向稳定、
95% CI 不跨 0，且不能只由一个场景或类别驱动。

## 决策门

- **Gate A — 数据闭环**：不同几何 pair 具有正确 donor、mask、多模态 target、冻结 split
  和 CAD-anchored semantic Gaussians。
- **Gate B — 语义假设**：B_sem-2D 相对 Global 在多 pair 主要可控性终点上稳定改善，
  geometry gate 全部通过。
- **Gate C — 学习模型必要性**：Semantic-DINO 在 Level 2 稳定优于 B_sem-2D；否则转向
  benchmark/metric 或重新设计模型。
- **Gate D — 强 baseline**：相对 MaterialMVP 等方法仍在跨几何对应、局部控制和 CAD-GS
  表示上有明确优势。
- **Gate E — 真实价值**：真实 reference 到 unseen CAD 仍保持区域控制和跨视角稳定性。

具体状态、依赖和当前行动仅在 `MILESTONES.md` 维护。
