# Yang et al. (AsiaCCS 2021) 复现与改进报告

> **复现论文**: Yang et al., AsiaCCS 2021 — *Robust Roadside Physical Adversarial Attack Against Deep Learning in LiDAR Perception Modules*
>
> **攻击类型**: Appearing Attack（在路边放置对抗物体，使检测器误检出幻影车辆）
>
> **目标模型**: PointRCNN（白盒）、PointPillar / PV-RCNN（原论文黑盒目标）

---

## 1. 原论文方法

根据原论文及综述文献 (Guesmi & Shafique, 2024) 描述，Yang et al. 的方法如下：

- **白盒攻击**：假设可访问目标模型 PointRCNN 内部，通过操作 3D 网格 (mesh) 顶点生成对抗点云。使用可微分 LiDAR 渲染器（Möller-Trumbore 光线-三角形求交）将 mesh 渲染为点云，通过梯度下降优化 mesh 形状。
- **黑盒攻击**：不访问模型内部，使用遗传进化算法 (genetic algorithm) 搜索 mesh 参数，目标模型为 PointPillar 和 PV-RCNN。
- **物理验证**：将对抗物体 3D 打印后放置在路边，在 LGSVL 仿真器 + Baidu Apollo 和实际道路测试中验证。

原论文中 mesh 参数化的具体技术细节（是否使用 sigmoid 重参数化等）未公开开源代码，但其引用了 Tu et al. (CVPR 2020) 的 mesh 对抗攻击框架。

---

## 2. 复现实现

### 2.1 复现方案

由于原论文未开源代码，我们参照 Tu et al. (CVPR 2020) 的公开方法实现了 mesh 参数化攻击，具体如下：

- **Mesh 表示**：icosphere 模板网格（162 顶点、320 面）
- **sigmoid 重参数化**（来自 Tu et al.）：`vᵢ = R(b ⊙ sign(v̄₀ᵢ) ⊙ σ(|v̄₀ᵢ| + Δv̄ᵢ)) + c ⊙ tanh(t̄)`，其中 σ 为 sigmoid 函数，b 控制形状边界，c 控制平移边界，sign 将顶点锁定在初始象限防止自交
- **可微 LiDAR 渲染器**：Möller-Trumbore 光线求交，模拟 Velodyne HDL-64E 激光扫描
- **Loss 函数**：L_cls (CW-margin 前景分类) + L_loc (框中心对齐) + L_size (框尺寸匹配) + L_feat (特征匹配) + L_lap (Laplacian mesh smoothing)
- **白盒优化器**：Adam 梯度下降
- **黑盒搜索**：CMA-ES 进化策略（原论文用遗传算法，我们选用 CMA-ES 作为替代）

### 2.2 复现结果：ASR 极低

| 方案 | 优化算法 | ASR |
|------|----------|-----|
| Mesh 白盒 | Adam | **0.1%** (4/3720) |
| Mesh 黑盒 | CMA-ES | **0%** |
| Mesh 两阶段白盒 | Adam（分阶段） | **≈0%** |

### 2.3 失败原因分析

梯度从 RPN loss 反传到优化变量 δv 需穿过两层间接映射：

1. **可微渲染器**：光线-三角形求交的梯度通过有限差分近似获得，噪声大、方向不稳定
2. **sigmoid 重参数化**：sigmoid 对大值区域梯度饱和，tanh 类似；sign 函数在零点不可微——整个重参数化层压缩和扭曲了搜索空间

此外还有一个可能被忽视的因素：**原论文做的是 appearing attack，但 sigmoid 重参数化来自 Tu et al. 的 hiding attack**。Hiding attack 只需让已有检测的置信度降到阈值以下，优化目标相对容易；appearing attack 需要从零生成高置信度检测，对优化信号的质量要求高得多。Tu et al. 的方法在 hiding 场景下有效，但搬到 appearing 场景后，渲染器 + 重参数化的梯度噪声就不足以支撑更困难的优化目标了。

> Tu et al. 自己也提到：*"In this black box setting, we find re-parameterization unnecessary for gradient-free optimization."* 即 sigmoid 重参数化对黑盒并无帮助。

---

## 3. 改进方案：直接点优化（PointOpt）

### 3.1 核心思路

去掉 mesh 和渲染器，直接以 N 个点的 (x, y, z) 坐标作为优化变量。梯度从 RPN loss 经 PointNet2 骨干网络直达输入点坐标，无中间层噪声。

```
原方案: δv, t → sigmoid 重参数化 → mesh 顶点 → 可微渲染器 → 点云 → 检测器 → Loss
改进后: 点坐标 (N, 3) ──────────────────────────→ 注入场景 → 检测器 → Loss
```

### 3.2 白盒 PointOpt

| 项目 | 设计 |
|------|------|
| **优化变量** | N=400 个点的 (x, y, z)，共 1200 维 |
| **优化器** | Adam, lr=0.01, CosineAnnealing 衰减 |
| **Loss** | L_cls + L_loc + L_size + L_feat + L_uni (点均匀性正则，替代 Laplacian) |
| **梯度来源** | RPN 逐点输出（因 ROI Pooling 不可微，不用 RCNN 级别梯度） |
| **初始化** | GT 车辆 LiDAR 扫描提取 → 去中心化（起点即类车形状） |
| **约束** | bbox clamp, 半径 [1.95, 0.8, 0.78]m |
| **训练帧** | KITTI val 中注入距离 5-20m 的帧，每步随机采样 8 帧 |

### 3.3 黑盒 PointOpt

与白盒相同的参数化，优化算法替换为 CMA-ES，不需要模型梯度。

| 项目 | 设计 |
|------|------|
| **优化变量** | N=200 个点的 (x, y, z)，共 600 维 |
| **搜索算法** | CMA-ES (CMA_diagonal 模式) |
| **适应度** | 注入 → PointRCNN 推理 → 注入位置附近检测置信度 + L2 正则 |
| **sigma0** | 0.15 |
| **popsize** | 48，多 GPU 并行 + per-candidate early termination |
| **每代评估** | 8 帧/候选 |

### 3.4 对比

| | 复现方案 (Mesh) | 改进方案 (PointOpt) |
|--|-----------------|---------------------|
| **搜索空间** | 间接（δv → sigmoid → 渲染 → 点云） | 直接（点坐标即搜索变量） |
| **梯度链路** | 穿过渲染器 + sigmoid，噪声大 | PointNet2 → 输入点，干净 |
| **空间一致性** | 不一致（δv 变化 ≠ 点云几何变化） | 完全一致（搜索方向 = 点移动方向） |
| **初始化** | 零向量（标准球体） | GT 车辆扫描（类车形状） |
| **约束** | sigmoid 隐式（扭曲空间） | bbox clamp（线性，不扭曲） |
| **正则化** | Laplacian mesh smoothing | 点均匀性 |

---

## 4. 速度优化（黑盒）

黑盒首版在 8 GPU 上每代约 85s，300 代需约 7 小时。优化后：

| 措施 | 效果 |
|------|------|
| popsize 128 → 48 | 推理量减少 62.5% |
| n_eval_samples 20 → 8 | 再减少 60% |
| per-candidate early termination | 劣质候选 2-3 帧即跳过，省 50-70% |
| 单帧 detect() 替代 detect_batch() | 消除 FPS 点云填充开销 |
| **总计** | 每代从 ~85s 降至 ~5s，总时间降至约 25-50 分钟 |

---

## 5. 实验结果

| 攻击方案 | 优化算法 | 参数化 | 需要模型梯度 | ASR |
|----------|----------|--------|-------------|-----|
| Mesh 白盒（复现） | Adam | mesh → render → 点云 | 是 | 0.1% (4/3720) |
| Mesh 黑盒（复现） | CMA-ES | mesh → render → 点云 | 否 | 0% |
| Mesh 两阶段（复现） | Adam 分段 | mesh → render → 点云 | 是 | ≈0% |
| **PointOpt 白盒（改进）** | Adam | 直接点坐标 | 是 | **39.5%** (1471/3720) |
| **PointOpt 黑盒（改进）** | CMA-ES | 直接点坐标 | 否 | 运行中 |

---

## 6. 结论

1. 按照原论文 + Tu et al. 的 mesh 参数化路径复现 appearing attack，ASR 仅 0.1%。核心原因是可微渲染器梯度噪声 + sigmoid 搜索空间扭曲，且该技术原本设计用于更简单的 hiding attack 场景
2. 改进方案 PointOpt 去掉 mesh 和渲染器，直接优化点坐标，白盒 ASR 从 0.1% 提升至 **39.5%**
3. 黑盒同样采用 PointOpt 参数化配合 CMA-ES，搜索空间与目标空间一致，有望取得有效攻击成功率
4. 黑盒不依赖模型梯度，只需检测输出，更贴近实际黑盒威胁模型
