# SemReg-GS 里程碑

更新日期：2026-09-09

总体实验设计见 `PROJECT_PLAN.md`。本文件只维护状态、验收条件和执行顺序。

## M0 — 可复现环境

状态：**部分完成**。

已完成 Python 3.10.20、PyTorch 2.13.0+cu126、CUDA Toolkit 12.6、MSVC 14.44 和 HSSD
访问验证。

剩余：验证 Blender headless；编译 3DGS CUDA extension 并完成单视角 render smoke；冻结
Python、Torch、CUDA、MSVC、Blender 和关键包版本。

通过条件：新终端按文档激活后，环境检查、Blender headless、extension import 和单视角
渲染全部成功。

## M1 — 单 pair 最小证据闭环

状态：**已完成，仅为 point-zbuffer pilot**。

冻结 pair：`107734119_175999932 → 103997424_171030444`。

已完成资产与语义审计、多模态视角、100,000 CAD Gaussians、冻结协议、四方法、held-out、
shared-palette leakage、multi-view warp、intervention、边界敏感性、semantic alignment 和
spill decomposition。

结论边界：semantic selector 的区域隔离得到机制性支持，但像素质量未优于 Global；当前
class-constant DINO 不能证明 learned feature 必要性；point renderer 结果不能等同正式
3DGS。M1 冻结为实现与协议参考。

可选复核：运行已实现的 mesh-depth occlusion gate，检查 `other→wall/floor` mismatch。

## M2 — HSSD 扩展第一波资产审计

状态：**进行中，当前唯一执行重点**。

20 个候选已完成 metadata-only 预审计；四个完整 GLB 已下载：

| Scene ID | 房间数 | 物体数 | metadata 解析率 |
|---|---:|---:|---:|
| `102344094` | 11 | 130 | 97.7% |
| `102344328` | 13 | 286 | 95.8% |
| `102344439` | 19 | 430 | 94.7% |
| `102815835` | 15 | 328 | 94.5% |

每个场景运行 Blender asset audit、六类逐面映射、semantic preview、类别 polygon/area 统计
和可用房间检查。不得仅凭 metadata 解析率判定通过。

通过条件：至少 3/4 场景稳定加载、六类均存在、预览无明显整片误标，并生成完整 audit
report。失败场景保留报告并从后续候选替换。

## M3 — 冻结 10-pair HSSD cohort

状态：**未开始，依赖 M2**。

目标为 10 个不同几何 pair（包含 M1 pilot），优先不复用场景。若场景不足，必须在结果
产生前声明复用策略。

通过条件：10/10 pair 通过文件哈希、split 隔离、六类 coverage 和 geometry preservation
gate；失败与替换过程写入 cohort flow。

## M4 — 多 pair 确认性实验

状态：**未开始，依赖 M3**。

运行冻结的 `B_sem-2D − Global-RGB`，输出逐 pair/逐类结果、paired delta、bootstrap 95%
CI、effect size、像素质量代价、各 gate 与失败案例。

通过条件：主要可控性终点稳定改善、95% CI 不跨 0、结果不由单一场景或类别驱动，且
geometry gate 全部通过。

## M5 — 正式 Gaussian rasterizer 复核

状态：**未开始；M0 完成后可并行准备**。

先复核 pilot，再运行 10-pair cohort。通过条件是正式 renderer 的 coverage、视觉完整性和
指标可报告，且 M4 主要方向保持；若反转，返回 renderer/visibility/alignment 诊断。

## M6 — Level 2 与 learned feature

状态：**仅在 Gate B 通过后启动**。

加入类内变化、纹理尺度和 1/3/5 references，使用 multi-prototype 或 spatial decoder。
只有 `Semantic-DINO − B_sem-2D` 在冻结指标上稳定增益，才通过 Gate C。

## M7 — 强 baseline 与外部有效性

状态：**后置**。

依次运行 MaterialMVP whole-room、MaterialMVP semantic-submesh、StyleGaussian，再考虑
TRELLIS.2；最后使用 ScanNet++ 检查真实域。3D-FRONT 不阻塞 HSSD 主线。

## 当前执行顺序

```text
M0 Blender/CUDA smoke ───────────────┐
                                     │
M2 四场景 GLB 审计 → M3 10 pairs → M4 Global vs B_sem-2D
                                     │
                                     └→ M5 正式 renderer 复核
                                                  ↓ Gate B
                                      M6 learned feature
                                                  ↓
                                      M7 强 baseline / 真实域
```

当前最小可执行工作：完成 M2 四场景 GLB 审计。当前不做 100–300 pairs、StyleGaussian
改造、TRELLIS.2、photorealism 优化或 class-constant DINO 的重复比较。
