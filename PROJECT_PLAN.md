# SemReg-GS 项目计划

## 当前唯一目标

在修改 StyleGaussian、运行 TRELLIS.2 或追求真实建筑 photorealism 之前，先完成最小证据闭环：

```text
1 donor + 1 different-geometry target
                    ↓
Level-1 oracle target + semantic Gaussians
                    ↓
Global vs B_sem-2D vs Semantic
                    ↓
renders + Region LPIPS + Region DINO + Leakage
```

如果 Semantic 在这个闭环中没有表现出区域可控性优势，项目应立刻检查假设和数据，而不是投入两个月改大型网络。

## 研究定位

### 不足以成为贡献

- mesh + reference image -> appearance/PBR；MaterialMVP 已经非常强；
- 给 StyleGaussian 增加 semantic label；这会被认为 incremental；
- 对整个 CAD 场景套一个 global style；
- 只展示好看的渲染，不测 controllability。

### 需要证明的贡献

1. **Cross-Geometry Semantic Appearance Correspondence**：source 与 target 几何不同，只通过建筑语义建立对应；
2. **Region-specific controllability**：只改变 wall reference 时，floor/ceiling/door 不应明显变化；
3. **CAD-constrained Gaussians**：appearance optimization 不造成 geometry drift；
4. **Few-reference practicality**：量化 1 / 3 / 5 张 reference 的效果；
5. **Complexity progression**：从 uniform material 到 intra-class variation，再到真实照片。

## Work Packages

### WP0：环境与数据访问

状态：进行中。

- [x] 本机硬件审计；
- [x] 创建 `semreg-gs` Python 3.10 环境；
- [x] 下载 3DGS 与 StyleGaussian 官方代码；
- [x] 接受 HSSD CC BY-NC 4.0 条款并下载两个 pilot 场景；
- [x] 提交 3D-FRONT Hugging Face 访问申请；
- [ ] 3D-FRONT 申请获批（可选跨数据集扩展，不阻塞主线）；
- [ ] 安装 PyTorch、Blender、CMake；
- [ ] 按 `configs/hssd_expansion.json` 下载并审计 HSSD 候选场景；
- [ ] 若 3D-FRONT 获批，只下载并校验扩展实验所需文件；
- [ ] 记录数据许可和具体版本。

完成条件：至少十个不同几何 HSSD pair 通过资产、语义、相机覆盖和许可审计；
3D-FRONT 是否获批不作为完成条件。

### WP1：Level-1 单样本数据闭环

预计：3–5 个工作日。

- [ ] 选择一个 donor bedroom；
- [ ] 选择一个不同 scene ID、不同布局的 target bedroom；
- [ ] 统一五类标签；
- [ ] 生成 donor 的 3 张 reference RGB 和 semantic mask；
- [ ] 用 world-space/triplanar projection 构造 target oracle；
- [ ] 渲染 target 的 8 张 held-out GT；
- [ ] 将 target CAD mesh 转成 semantic Gaussians；
- [ ] 以类别颜色渲染 Gaussians，人工检查 label；
- [ ] 验证 Gaussian center 到 CAD surface 的距离。

交付物：

```text
outputs/stage1/<pair_id>/
├── donor_reference/
├── donor_masks/
├── target_cad/
├── target_oracle/
├── semantic_gaussians/
└── validation_report.json
```

### WP2：关键假设实验

状态：单 pair 四方法 point-zbuffer pilot 已完成；当前 leakage 定义未通过跨方法可比性审计，Gate B 尚未通过。

预计：5–7 个工作日。

实现顺序：

1. [x] `Global-RGB`：全局 mean/std 或 histogram；
2. [x] `B_sem-2D`：每类 mean/std、histogram、简单 texture descriptor；
3. [x] `Global-DINO`：一个全局 DINO token；
4. [x] `Semantic-DINO`：每类一个 DINO token；
5. 保持 renderer、decoder、训练视角、步数和 loss 一致。

进入多 pair 前必须完成：

- [ ] 用所有方法共享的冻结 reference palette 重算 leakage；
- [x] 完成 semantic intervention：semantic macro selectivity 约 0.942，global 约 0.500；
- [x] 完成边界腐蚀敏感性评估：wall spill 在 16 px 仅下降 1.7%，属于区域内部语义不一致；
- [x] 实现 rendered Gaussian semantic 与 held-out proxy semantic 的 confusion/IoU 审计（待运行）；
- [x] 完成 semantic alignment audit：overall agreement 69.81%，view_07 仅 40.24%，主要为 other→wall/floor；
- [x] 实现 intrinsic spill 与 proxy-label apparent spill 分解（待运行）；
- [x] 完成 spill 分解：intrinsic spill=0，proxy apparent spill 约 100% 由标签/可见性不一致解释；
- [x] 为 point renderer 实现可选 mesh-depth occlusion gate，并增加 gated alignment 复核入口（待运行）；
- [ ] 冻结 evaluation-only Region DINO/LPIPS 的模型、权重哈希、预处理和 mask aggregation；
- [ ] 用正式 Gaussian rasterizer 复核低覆盖 point-zbuffer 结果。

先运行 1 pair，之后扩展到 10 pairs。每个方法必须保存固定相机路径和相同评估 masks。

主要比较：

```text
Semantic-DINO vs Global-DINO    -> semantic conditioning 的价值
Semantic-DINO vs B_sem-2D       -> learned feature transfer 的必要性
Global-DINO vs StyleGaussian    -> baseline 实现合理性
```

完成条件：输出配对场景的 Region LPIPS、Region DINO、Leakage、multi-view error，以及 paired difference 图。

### WP3：Level-2 数据难度

预计：1–2 周。

- [ ] 同一 semantic class 中加入多材质与空间 variation；
- [ ] 加入大尺度 texture pattern 和不同 scale；
- [ ] 增加 1 / 3 / 5 reference 设置；
- [ ] donor/target scene ID 严格隔离；
- [ ] 扩展到 100–300 pairs；
- [ ] 对 Global、B_sem-2D、Semantic 做完整配对统计。

只有当 B_sem-2D 在 Level 1 已经接近 Semantic 时，Level 2 才是下一步；不要用更大网络代替数据难度。

### WP4：StyleGaussian 语义改造

预计：1–2 周；必须在 WP2 支持核心假设后启动。

- [ ] 复现官方 StyleGaussian global baseline；
- [ ] CAD-anchored Gaussians 代替照片重建 Gaussians；
- [ ] global style embedding -> semantic-specific embeddings；
- [ ] Gaussian semantic ID 选择对应 reference code；
- [ ] 联合多视角 decoder；
- [ ] 比较 MLP 与 StyleGaussian decoder。

### WP5：最强 baseline

MaterialMVP 优先级高于 TRELLIS.2。

- [ ] MaterialMVP whole-room zero-shot；
- [ ] MaterialMVP semantic-submesh + merge；
- [ ] 比较 albedo/MR realism、multi-view consistency、reference similarity；
- [ ] 重点比较 Semantic Leakage 和局部编辑 controllability；
- [ ] TRELLIS.2 whole-room/per-region texturing；
- [ ] RoomPainter reference-caption 定性结果（不阻塞主实验）。

如果 Ours 只在“好看”上与 MaterialMVP 接近，但不能在 cross-geometry controllability 上建立清晰优势，则论文定位仍然不足。

### WP6：Level-3 与真实域

- [ ] ScanNet++ 真实照片；
- [ ] predicted semantic masks；
- [ ] missing semantic regions；
- [ ] windows/lighting/boundary interaction；
- [ ] real image -> unseen CAD；
- [ ] 用户研究和固定相机视频。

## 统计计划

### Primary endpoint

`Semantic Leakage Score`，按 donor-target pair 和 edited semantic class 配对。

### Secondary endpoints

- Region DINO similarity；
- Region LPIPS；
- warped multi-view consistency；
- CAD point-to-surface distance；
- 1 / 3 / 5 reference efficiency curve。

### 报告要求

- mean、median、std；
- paired delta；
- bootstrap 95% CI；
- paired Cohen's d；非正态时 rank-biserial correlation；
- paired t-test；明显非正态时 Wilcoxon；
- 不预先设定任意“20% improvement”阈值；
- 同时报告失败场景，不能只展示最佳案例。

## 决策门

### Gate A：数据闭环

通过条件：不同几何 donor-target pair 有正确 reference、semantic mask、oracle GT 和 semantic Gaussians。

### Gate B：语义假设

通过条件：在方法无关的冻结指标上，Semantic 相对 Global 降低 intervention spill、提高
target-region response 与 region similarity，且跨配对场景趋势稳定。当前 method-specific
nearest-prototype leakage 不用于 Gate B 判定。

### Gate C：学习模型必要性

通过条件：Semantic-DINO 在 Level 2 复杂外观上优于 B_sem-2D；否则贡献应转向 benchmark/metric 或重新设计模型。

### Gate D：强 baseline

通过条件：MaterialMVP 即使 PBR/relighting 较强，Ours 仍在建筑语义 correspondence、局部控制和 CAD-GS 表示上建立明确优势。

### Gate E：真实价值

通过条件：ScanNet++ reference 在 unseen CAD 上保留语义区域控制，并具有可接受的跨视角稳定性。

## 近期行动顺序

1. 在已恢复的 `semreg-gs` 中将 CPU-only PyTorch 换为 CUDA build、补装
   `huggingface_hub`，并复核 Blender 本地渲染；
2. 从 HSSD 候选池逐批下载场景，先做资产与六类语义审计，不先批量渲染；
3. 将通过审计的场景写入 `configs/hssd_expansion.json`，冻结 pair 后再生成多视角数据；
4. 将当前单 pair 协议扩展到至少 10 个不同几何 HSSD pairs；
5. 对每个 pair 运行相同的四方法、held-out、intervention、alignment 与 geometry gates；
6. 汇总 paired delta、95% CI、effect size，并报告失败场景而非静默剔除；
7. 3D-FRONT 审批继续并行等待；若获批，仅作为跨数据集扩展；
8. HSSD 多 pair 证据闭环完成后，再投入 StyleGaussian、MaterialMVP 或 TRELLIS.2。
