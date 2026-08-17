# SemReg-GS 实验设计

> **目标**：验证“语义区域对应”是否能把现有建筑照片的外观，受控地迁移到几何不同、由 CAD 定义的未来室内空间，并在 CAD-anchored 3D Gaussians 中保持几何准确和跨视角一致。

## 1. 最小论文命题

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

### 建议论文名称

**SemReg-GS: Semantic Region-Guided Appearance Transfer to CAD-Anchored 3D Gaussians**

本实验第一版只预测 Gaussian 外观参数（RGB/SH 或外观 feature）；不同时解决完整 PBR、建筑融合和 CAD 生成。PBR 分解可以作为第二阶段扩展。

---

## 2. 数据库选择

### 2.1 主数据库：3D-FRONT + 3D-FUTURE

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

### 2.2 真实域验证：ScanNet++ v2

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

### 2.3 Objaverse 的用途

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
| 主要训练和定量评估 | 3D-FRONT/3D-FUTURE | 房间 mesh、语义、纹理齐全；与 RoomPainter 可比；可构造迁移真值 |
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

1. 从 3D-FRONT 选择 bedroom、living room 或 corridor；第一版只选一种房型。
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

### 3.4 第一轮数据规模

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
│   │   ├── 3d_front/
│   │   ├── 3d_future/
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

1. 使用 DINOv2/CLIP 图像编码器提取 dense feature map；
2. 将 semantic mask 下采样到 feature map 分辨率；
3. 对同一语义的所有像素和多个 reference views 做 masked attention pooling；
4. 得到 `z_wall, z_floor, z_ceiling, z_door, z_window`；
5. 缺失类别使用可学习的 `unknown` token，并在 loss 中屏蔽其 reference matching 项。

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
| B2 | StyleGaussian-global | 在 CAD-anchored GS 上使用 StyleGaussian 风格嵌入/解码，不加入语义区域 |
| B3 | MaterialMVP zero-shot | 输入 target mesh + donor reference；测试最接近的 image-to-PBR 方法 |
| B4 | TRELLIS.2 texturing zero-shot | 测试强基础模型在房间 mesh 上的零样本能力 |
| Ours | Semantic reference codes | 每个 Gaussian 使用同类区域的 reference token |

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

- [ ] 申请 3D-FRONT/3D-FUTURE 数据访问；
- [ ] 申请 ScanNet++ 访问，仅下载少量场景；
- [ ] 克隆官方 3DGS 和 StyleGaussian；
- [ ] 记录 CUDA、PyTorch、GPU 型号和依赖版本；
- [ ] 不要在许可未确认前重新分发原始数据。

**完成标准**：一个 3D-FRONT 房间可在 Blender 中正确加载，语义和纹理没有错位。

### Stage 1：单样本数据闭环

- [ ] 选择 1 个 donor 和 1 个不同几何的 target；
- [ ] 输出五类 semantic mesh masks；
- [ ] 渲染 3 张 donor reference；
- [ ] 构造 oracle transferred target；
- [ ] 渲染 8 个 target GT views；
- [ ] 将 target mesh 转为 semantic Gaussians；
- [ ] 验证每个 Gaussian 的 label 与 mesh face 一致。

**完成标准**：可以可视化 wall/floor/... 五种颜色的 Gaussian scene，并能从任意相机渲染。

### Stage 2：最简单的关键实验

只训练两个模型：

1. `Global`: 一个全局 reference code；
2. `Semantic`: 五个 semantic reference codes。

其余网络、训练步数、相机和损失完全相同。

**完成标准**：在 10 个 pilot pairs 上，Semantic 在至少两个核心指标上稳定优于 Global，尤其是 semantic leakage 和 region DINO similarity。

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

1. Semantic 方法的 region DINO similarity 优于 Global；
2. Semantic Leakage Score 至少相对降低 20%；
3. held-out view LPIPS 不劣于 Global；
4. CAD point-to-surface distance 接近零；
5. 三种不同 donor-target pair 上都能看到相同趋势，而不是只成功一个案例。

如果只提高“好看程度”，但没有降低语义泄漏或提高区域对应，则还不足以支持论文贡献。

---

## 12. 最终论文中需要回答的问题

1. 为什么不能直接使用 StyleGaussian 的全局风格迁移？
2. 语义区域条件到底改善了什么，可否通过 leakage 指标定量证明？
3. CAD anchoring 是否确实保证了几何不被外观优化破坏？
4. 当参考图缺少 ceiling/window 时如何处理？
5. 合成 3D-FRONT 训练能否泛化到 ScanNet++ 真实照片？
6. MaterialMVP/TRELLIS.2 在整个房间和分语义子 mesh 上分别表现如何？
7. 结果是否在未见过的相机视角保持稳定？

这七个问题都能得到清楚答案时，实验才形成一条完整的论文证据链。

---

## 13. 版本与来源记录

本文档于 **2026-08-17** 根据以下官方论文、项目和数据文档制定：

- 3D Gaussian Splatting：https://arxiv.org/abs/2308.04079
- StyleGaussian：https://github.com/Kunhao-Liu/StyleGaussian
- MaterialMVP：https://github.com/ZebinHe/MaterialMVP
- RoomPainter：https://openaccess.thecvf.com/content/CVPR2025/html/Huang_RoomPainter_View-Integrated_Diffusion_for_Consistent_Indoor_Scene_Texturing_CVPR_2025_paper.html
- TRELLIS.2：https://github.com/microsoft/TRELLIS.2
- 3D-FRONT：https://arxiv.org/abs/2011.09127
- ScanNet++：https://scannetpp.mlsg.cit.tum.de/scannetpp/

开始实现前应再次检查各仓库最新 commit、模型权重可用性、数据下载条款和学术许可。
