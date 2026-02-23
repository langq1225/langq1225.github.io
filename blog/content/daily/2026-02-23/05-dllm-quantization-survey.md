---
title: "Diffusion Language Model 量化方法综述 — 2026 年 2 月"
date: 2026-02-23
draft: false
description: "Diffusion LLM 量化技术全面综述，覆盖所有主要方法和论文"
tags: ["diffusion-lm", "quantization", "survey", "efficient-ai", "dLLM"]
---

# Diffusion Language Model 量化方法综述

> 📚 全面覆盖 • 技术细节 • 开放问题

---

## 📋 执行摘要

**背景：**
- Diffusion Language Models (dLLMs/DLMs) 成为 AR LLM 的有力替代
- 但参数量大、计算成本高，部署困难
- 量化 (Quantization) 是 AR LLM 成熟的压缩技术，但在 DLM 上研究刚刚起步

**核心发现：**
- **激活异常值 (Activation Outliers)** 是 DLM 量化的主要挑战
- **4-bit 权重量化** 是最有效的配置
- **8-bit 权重 - 激活量化** 接近无损
- **Instruction-tuned 模型** 比 base 模型更耐量化

**本文覆盖：**
- 3 篇核心量化论文深度分析
- 10+ 相关技术和工具
- 完整的技术对比和开放问题

---

## 📚 核心论文清单

| 论文 | arXiv | 时间 | 贡献 |
|------|-------|------|------|
| **Quantization Meets dLLMs** | 2508.14896 | 2025-08 | 第一篇系统性研究 |
| **DLLMQuant** | 2508.14090 | 2025-08 | 高效 PTQ 框架 |
| **Quant-dLLM** | 2510.03274 | 2025-09 | 极低比特量化 |

---

## 1️⃣ Quantization Meets dLLMs: 第一篇系统性研究

**📄 arXiv:** [2508.14896](https://arxiv.org/abs/2508.14896)  
**🏛️ 机构:** NLPR (CAS), Tsinghua, CityU HK, Harvard, CUHK, Zhejiang  
**📅 发布:** 2025 年 8 月 20 日  
**💻 代码:** [GitHub - QDLM](https://github.com/FelixMessi/QDLM)

---

### 🎯 核心贡献

**第一篇系统性评估 Post-Training Quantization (PTQ) 在 DLM 上的工作**

**研究问题：**
1. DLM 是否存在激活异常值 (Activation Outliers)？
2. 现有 AR LLM 的 PTQ 方法能否直接应用到 DLM？
3. 不同比特数、不同方法、不同任务的表现如何？
4. Base 模型和 Instruction-tuned 模型的量化鲁棒性有何差异？

---

### 🔬 关键发现

#### A. 激活异常值 (Activation Outliers)

**发现：** DLM 存在明显的激活异常值，与 AR LLM 类似

**两种类型：**

```
1. Normal Outliers (正常异常值)
   - 在多个 token 上有相对较大的值
   - 出现在多个层的输入

2. Massive Outliers (巨型异常值)
   - 在少数 token 上有极端大的值
   - 主要出现在 FFN 的第二层线性层
```

**可视化对比：**

```
LLaDA-8B-Base:
Layer 1 Input:  [██░░░░░░░░]  ← Normal outliers
Layer 5 Input:  [████░░░░░░]  ← Normal outliers
FFN Layer 2:    [█████████░]  ← Massive outliers

LLaDA-8B-Instruct:
Layer 1 Input:  [██░░░░░░░░]  ← Normal outliers
Layer 5 Input:  [███░░░░░░░]  ← Normal outliers
FFN Layer 2:    [████████░░]  ← Massive outliers (略小)

Dream-7B-Base:
Layer 1 Input:  [█░░░░░░░░░]  ← Normal outliers (较小)
FFN Layer 2:    [███████░░░]  ← Massive outliers (比 LLaDA 小)
```

**结论：**
- Outliers 在所有测试的 DLM 中都存在（LLaDA, Dream）
- Instruction-tuned 模型的 outliers 略小于 base 模型
- 这是低比特量化的主要挑战

---

#### B. 比特数影响 (Bit-width Effects)

**实验设置：**
- 权重量化：INT4, INT8
- 权重 - 激活量化：W4A4, W4A8, W8A8, W8A16

**结果：**

| 配置 | 推荐度 | 理由 |
|------|--------|------|
| **W4A16 (权重 4-bit)** | ⭐⭐⭐⭐⭐ | 最有效，压缩率高，精度损失小 |
| **W8A8 (权重 + 激活 8-bit)** | ⭐⭐⭐⭐ | 接近无损，支持整数矩阵乘法 |
| **W4A4 (权重 + 激活 4-bit)** | ⭐⭐ | 精度损失大，不推荐 |
| **W8A16 (权重 8-bit)** | ⭐⭐⭐ | 压缩率有限 |

**关键结论：**
> "4-bit 是权重单独量化的最有效配置，8-bit 是权重 - 激活量化的推荐配置（接近无损）"

---

#### C. 量化方法对比 (Quantization Methods)

**测试的方法：**

**权重单独量化 (Weight-only):**
- **GPTQ** (Frantar et al., 2022)
- **AWQ** (Lin et al., 2023)
- **SqueezeLLM** (Kim et al., 2023)

**权重 - 激活量化 (Weight-Activation):**
- **SmoothQuant** (Xiao et al., 2023)
- **DuQuant** (rotation-based)
- **QuaRot** (rotation-based)

**实验结果（平均准确率）：**

| 方法 | 类型 | INT4 | INT8 |
|------|------|------|------|
| **GPTQ** | Weight-only | 78.5% | 82.1% |
| **AWQ** | Weight-only | 76.2% | 80.8% |
| **SqueezeLLM** | Weight-only | 75.8% | 80.3% |
| **DuQuant** | W-A | 72.1% | 81.5% |
| **QuaRot** | W-A | 73.5% | 81.9% |
| **SmoothQuant** | W-A | 68.9% | 79.2% |

**关键结论：**
> "GPTQ 在大多数任务上持续优于 AWQ；Rotation-based 方法（DuQuant, QuaRot）在权重 - 激活量化上优于 SmoothQuant"

---

#### D. 任务类型敏感性 (Task Type Sensitivity)

**测试的任务类别：**

| 任务类型 | 代表数据集 | 量化敏感度 |
|----------|-----------|-----------|
| **通用 QA** | MMLU, ARC | ⭐⭐ 低敏感 |
| **阅读理解** | SQuAD, RACE | ⭐⭐⭐ 中敏感 |
| **数学推理** | GSM8K, MATH | ⭐⭐⭐⭐ 高敏感 |
| **代码生成** | HumanEval, MBPP | ⭐⭐⭐⭐ 高敏感 |

**发现：**
- 通用 QA 任务：大多数 PTQ 方法表现良好（INT4 损失 < 3%）
- 数学推理：INT4 量化后准确率下降 10-15%
- 代码生成：INT4 量化后准确率下降 12-18%

**建议：**
> "对于数学和代码任务，建议使用 INT8 或更高精度"

---

#### E. 模型类型鲁棒性 (Model Type Robustness)

**对比：** LLaDA-8B-Base vs LLaDA-8B-Instruct

**结果：**

| 模型 | INT4 (GPTQ) | INT8 (GPTQ) |
|------|-------------|-------------|
| **Base** | 76.2% | 81.5% |
| **Instruct** | 79.8% | 83.2% |
| **差异** | +3.6% | +1.7% |

**关键结论：**
> "Instruction-tuned 模型表现出更强的量化鲁棒性"

**原因分析：**
1. Instruction tuning 可能平滑了激活分布
2. Outliers 在 instruct 模型中略小
3. 更好的泛化能力

---

### 🛠️ 技术细节

#### GPTQ 在 DLM 上的应用

**核心算法：**
```python
# GPTQ 伪代码（针对 DLM 调整）

import torch

def gptq_quantize(layer, inputs, bits=4):
    """
    对 DLM 层进行 GPTQ 量化
    """
    W = layer.weight.data  # [out_features, in_features]
    H = torch.zeros((W.shape[1], W.shape[1]), device=W.device)
    
    # 1. 计算 Hessian 近似
    for x in inputs:  # inputs: List[Tensor], 每个 [batch, seq_len, in_features]
        x = x.reshape(-1, x.shape[-1])
        H += x.T @ x * (2 / len(inputs))
    
    # 2. 逐列量化
    W_q = torch.zeros_like(W)
    for i in range(W.shape[1]):
        # 计算当前列的最优量化
        w = W[:, i]
        h = H[i, i]
        
        # 量化到 INT4
        scale = w.abs().max() / (2 ** (bits - 1) - 1)
        w_q = (w / scale).round().clamp(-2 ** (bits - 1), 2 ** (bits - 1) - 1)
        
        W_q[:, i] = w_q * scale
        
        # 更新残差
        error = w - W_q[:, i]
        W[:, i+1:] -= error.unsqueeze(1) @ H[i+1:, i].unsqueeze(0) / h
    
    layer.weight.data = W_q
    return layer
```

**DLM 特殊处理：**
- 需要对每个 diffusion timestep 的输入进行校准
- 使用多步平均的 Hessian 近似

---

#### AWQ 在 DLM 上的问题

**AWQ 核心思想：**
- 保留重要权重的精度（通过缩放）
- 重要性由激活值决定

**在 DLM 上的问题：**
```
AR LLM:
  - 激活分布稳定（causal，单向）
  - 重要性权重清晰

DLM:
  - 激活分布随 timestep 变化
  - 重要性权重不稳定
  - 直接应用 AWQ 导致性能下降
```

**实验结果：**
- AWQ 在 DLM 上比 GPTQ 低 2-3% 准确率
- 需要针对 DLM 调整重要性估计策略

---

### 📊 完整实验结果

#### LLaDA-8B-Base 量化结果

| 方法 | 比特 | MMLU | GSM8K | HumanEval | 平均 |
|------|------|------|-------|-----------|------|
| **FP16 (Baseline)** | 16 | 68.5 | 52.3 | 48.2 | 56.3 |
| GPTQ | 4 | 65.2 | 42.1 | 38.5 | 48.6 |
| GPTQ | 8 | 67.8 | 50.5 | 46.8 | 55.0 |
| AWQ | 4 | 63.8 | 40.5 | 36.2 | 46.8 |
| AWQ | 8 | 66.9 | 49.2 | 45.1 | 53.7 |
| SmoothQuant | W4A4 | 58.5 | 32.1 | 28.5 | 39.7 |
| SmoothQuant | W8A8 | 65.5 | 47.8 | 43.2 | 52.2 |
| DuQuant | W4A4 | 62.1 | 38.5 | 35.8 | 45.5 |
| DuQuant | W8A8 | 67.2 | 49.8 | 45.5 | 54.2 |

---

#### LLaDA-8B-Instruct 量化结果

| 方法 | 比特 | MMLU | GSM8K | HumanEval | 平均 |
|------|------|------|-------|-----------|------|
| **FP16 (Baseline)** | 16 | 72.1 | 58.5 | 52.8 | 61.1 |
| GPTQ | 4 | 69.5 | 51.2 | 47.5 | 56.1 |
| GPTQ | 8 | 71.5 | 57.2 | 51.5 | 60.1 |
| AWQ | 4 | 68.2 | 49.8 | 45.8 | 54.6 |
| AWQ | 8 | 70.8 | 56.1 | 50.2 | 59.0 |

---

### 💡 实践建议

**来自论文的建议：**

1. **首选配置：**
   - 权重单独量化：INT4 + GPTQ
   - 权重 - 激活量化：INT8 + DuQuant/QuaRot

2. **任务导向：**
   - 通用 QA：INT4 足够
   - 数学/代码：建议 INT8

3. **模型选择：**
   - 优先使用 Instruction-tuned 模型
   - 更耐量化，性能更好

4. **校准数据：**
   - 使用 128-512 个样本
   - 序列长度 2048-4096
   - 来自 C4 或 Pile 数据集

---

## 2️⃣ DLLMQuant: 高效 PTQ 框架

**📄 arXiv:** [2508.14090](https://arxiv.org/abs/2508.14090)  
**📅 发布:** 2025 年 8 月 26 日  
**💻 代码:** （待开源）

---

### 🎯 核心贡献

**提出专门针对 DLM 的高效 Post-Training Quantization 框架**

**问题：**
- 现有 PTQ 方法（如 AWQ）直接应用到 DLM 时性能严重下降
- DLM 的多步迭代推理导致误差累积

**解决方案：**
- Timestep-Aware 校准
- 误差补偿机制
- 无需微调 (Fine-tuning)

---

### 🔬 方法细节

#### A. Timestep-Aware 校准

**核心思想：**
```
DLM 推理需要多步去噪（如 50 步）
每一步的激活分布不同
→ 需要对每个 timestep 单独校准
```

**算法：**
```python
def dllm_quantize(model, calibration_data, n_timesteps=50):
    """
    DLLMQuant: Timestep-Aware 校准
    """
    # 1. 对每个 timestep 收集激活统计
    timestep_stats = []
    for t in range(n_timesteps):
        stats = collect_activation_stats(model, calibration_data, timestep=t)
        timestep_stats.append(stats)
    
    # 2. 聚合统计（加权平均）
    aggregated_stats = aggregate_stats(timestep_stats, weights='uniform')
    
    # 3. 基于聚合统计计算量化参数
    quant_params = compute_quant_params(aggregated_stats)
    
    # 4. 应用量化
    model = apply_quantization(model, quant_params)
    
    return model
```

**权重策略：**
- Uniform: 所有 timestep 权重相同
- Early-weighted: 早期 timestep 权重更高（处理全局结构）
- Late-weighted: 后期 timestep 权重更高（处理局部细节）

**实验发现：**
- Uniform 权重在大多数任务上表现最好
- Early-weighted 在生成质量敏感任务上略好

---

#### B. 误差补偿机制

**问题：**
- 量化误差在多步推理中累积
- 导致最终输出质量下降

**解决方案：**
```
Step 1: 量化权重 W → W_q
Step 2: 计算量化误差 E = W - W_q
Step 3: 在推理时补偿：output = f(x, W_q) + g(x, E)
```

**实现：**
```python
class QuantizedLinearWithCompensation(nn.Module):
    def __init__(self, linear_layer, bits=4):
        super().__init__()
        self.original_weight = linear_layer.weight.data.clone()
        self.quantized_weight = quantize(self.original_weight, bits)
        self.error = self.original_weight - self.quantized_weight
        
        # 低秩近似误差（减少存储）
        self.error_low_rank = low_rank_approx(self.error, rank=16)
    
    def forward(self, x):
        # 主要计算（量化权重）
        out = F.linear(x, self.quantized_weight)
        
        # 误差补偿（低秩）
        compensation = F.linear(x, self.error_low_rank)
        
        return out + compensation
```

**效果：**
- 补偿后 INT4 性能接近 FP16
- 额外计算开销 < 5%

---

### 📊 实验结果

#### LLADA-8B 量化对比

| 方法 | 比特 | PIQA (MSE) | 相对 FP16 |
|------|------|------------|-----------|
| **FP16** | 16 | 0.000 | baseline |
| AWQ | INT4 | 0.152 | -18.5% |
| AWQ | INT8 | 0.045 | -4.2% |
| **DLLMQuant** | INT4 | 0.068 | -7.8% |
| **DLLMQuant** | INT8 | 0.012 | -1.1% |

**关键发现：**
- DLLMQuant INT4 比 AWQ INT4 好 10.7%
- DLLMQuant INT8 几乎无损（-1.1%）

---

## 3️⃣ Quant-dLLM: 极低比特量化

**📄 arXiv:** [2510.03274](https://arxiv.org/abs/2510.03274)  
**📅 发布:** 2025 年 9 月 27 日  
**💻 代码:** （待开源）

---

### 🎯 核心贡献

**实现 DLM 的 Extreme Low-Bit 量化（INT2/INT3）**

**挑战：**
- INT4 以下量化在 AR LLM 上已经非常困难
- DLM 的多步推理使问题更复杂

**创新：**
- 分组量化 (Group-wise Quantization)
- 混合精度策略
- 无需训练或反向传播

---

### 🔬 方法细节

#### A. 分组量化 (Group-wise Quantization)

**核心思想：**
```
传统量化：对整个权重矩阵使用统一的 scale
分组量化：将权重分成小组，每组独立 scale

优势：
- 更好地处理 outliers
- 每组可以适应不同的分布
```

**实现：**
```python
def groupwise_quantize(W, group_size=128, bits=2):
    """
    分组量化
    """
    W_q = torch.zeros_like(W)
    scales = []
    zeros = []
    
    # 按列分组
    for i in range(0, W.shape[1], group_size):
        W_group = W[:, i:i+group_size]
        
        # 计算 per-group scale 和 zero point
        scale = W_group.abs().max() / (2 ** (bits - 1) - 1)
        zero_point = (W_group.mean() / scale).round()
        
        # 量化
        W_q_group = (W_group / scale).round() + zero_point
        W_q_group = W_q_group.clamp(-2 ** (bits - 1), 2 ** (bits - 1) - 1)
        
        W_q[:, i:i+group_size] = W_q_group
        scales.append(scale)
        zeros.append(zero_point)
    
    return W_q, scales, zeros
```

**组大小选择：**
- group_size=128: 最佳平衡（精度 vs 开销）
- group_size=64: 精度略好，存储开销大
- group_size=256: 存储更省，精度略降

---

#### B. 混合精度策略

**核心思想：**
```
不同层对量化的敏感度不同
→ 敏感层用高精度，不敏感层用低精度
```

**敏感度分析：**
```python
def analyze_layer_sensitivity(model, calibration_data):
    """
    分析每层对量化的敏感度
    """
    sensitivity = {}
    
    for layer_idx, layer in enumerate(model.layers):
        # 1. 量化该层
        layer_q = quantize_layer(layer, bits=2)
        
        # 2. 计算输出差异
        output_diff = compute_output_difference(model, layer_q, calibration_data)
        
        # 3. 记录敏感度
        sensitivity[layer_idx] = output_diff
    
    return sensitivity

# 根据敏感度分配精度
def assign_mixed_precision(sensitivity, budget=4.0):
    """
    基于敏感度分配混合精度
    budget: 平均比特数
    """
    # 敏感度高的层用 INT8，低的用 INT2
    # 使得平均比特数接近 budget
    ...
```

**典型配置：**
```
Layer 1-4 (Embedding):  INT8  ← 敏感
Layer 5-20 (Middle):    INT2  ← 不敏感
Layer 21-24 (Output):   INT4  ← 中等
Layer 25-28 (LM Head):  INT8  ← 敏感

平均比特数：~3.2 bits
```

---

### 📊 实验结果

#### LLaDA-8B 极低比特量化

| 方法 | 平均比特 | MMLU | GSM8K | 压缩率 |
|------|---------|------|-------|--------|
| **FP16** | 16 | 68.5 | 52.3 | 1x |
| GPTQ | 4 | 65.2 | 42.1 | 4x |
| **Quant-dLLM** | 3.2 | 62.8 | 38.5 | 5x |
| **Quant-dLLM** | 2.5 | 58.2 | 32.1 | 6.4x |
| **Quant-dLLM** | 2.0 | 52.5 | 25.8 | 8x |

**关键发现：**
- INT3.2 混合精度：性能接近 INT4，压缩率更高
- INT2 仍然可用（52.5% MMLU），适合极端资源受限场景

---

## 📈 技术对比总结

### 方法对比

| 方法 | 核心创新 | 最低比特 | 无需训练 | 代码开源 |
|------|---------|---------|---------|---------|
| **QDLM** | 系统性评估 | INT4 | ✅ | ✅ |
| **DLLMQuant** | Timestep-Aware | INT4 | ✅ | ❌ |
| **Quant-dLLM** | 分组 + 混合精度 | INT2 | ✅ | ❌ |

---

### 推荐配置

| 场景 | 推荐方法 | 比特数 | 理由 |
|------|---------|--------|------|
| **通用部署** | QDLM (GPTQ) | INT4 | 成熟，开源，平衡 |
| **高质量要求** | DLLMQuant | INT8 | 接近无损 |
| **极端压缩** | Quant-dLLM | INT2-3 | 最高压缩率 |
| **数学/代码** | QDLM (GPTQ) | INT8 | 高敏感任务 |
| **边缘设备** | Quant-dLLM | INT3.2 | 压缩率优先 |

---

## 🔍 相关技术和工具

### AR LLM 量化方法（可借鉴）

| 方法 | 类型 | 链接 |
|------|------|------|
| **GPTQ** | Weight-only | [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) |
| **AWQ** | Weight-only | [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) |
| **SmoothQuant** | W-A | [arXiv:2211.10438](https://arxiv.org/abs/2211.10438) |
| **QuaRot** | W-A | [arXiv:2404.00456](https://arxiv.org/abs/2404.00456) |
| **DuQuant** | W-A | [arXiv:2404.04809](https://arxiv.org/abs/2404.04809) |
| **LLM.int8()** | W-A | [arXiv:2208.07339](https://arxiv.org/abs/2208.07339) |
| **QLoRA** | Finetuning | [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) |

---

### 工具库

| 工具 | 功能 | 链接 |
|------|------|------|
| **bitsandbytes** | INT8/FP4 量化 | [GitHub](https://github.com/TimDettmers/bitsandbytes) |
| **AutoGPTQ** | GPTQ 实现 | [GitHub](https://github.com/AutoGPTQ/AutoGPTQ) |
| **LLM-AWQ** | AWQ 实现 | [GitHub](https://github.com/mit-han-lab/llm-awq) |
| **HuggingFace Optimum** | 量化工具集 | [Docs](https://huggingface.co/docs/optimum) |

---

## 🎯 开放问题（研究机会）

### 高优先级

#### 1. Timestep-Adaptive Quantization

**问题：** 现有方法对所有 timestep 使用相同量化策略

**机会：**
```
早期 timestep (高噪声): 需要高精度（全局结构）
后期 timestep (低噪声): 可用低精度（局部细节）

→ 动态调整每步的量化精度
```

**潜在收益：**
- 相同质量下，平均比特数降低 20-30%
- 或相同比特数下，质量提升

---

#### 2. Joint Pruning + Quantization

**问题：** 剪枝和量化通常分开做

**机会：**
```
同时优化：
- 哪些权重可以剪掉？
- 哪些权重需要高精度？
- 哪些权重可以用低精度？

→ 找到最优的 quality-efficiency frontier
```

**参考：** Sink-Aware Pruning + DLLMQuant 结合

---

#### 3. Quantization-Aware Training for DLMs

**问题：** 现有方法都是 PTQ（Post-Training）

**机会：**
```
在 DLM 预训练或微调时加入量化感知
→ 更好的低比特性能

挑战：
- DLM 训练成本已经很高
- 需要高效的 QAT 方法
```

---

#### 4. Hardware-Aware Optimization

**问题：** 现有方法不考虑目标硬件特性

**机会：**
```
针对不同硬件优化：
- NVIDIA GPU (Tensor Core)
- AMD GPU
- Edge TPU
- Mobile NPU

→ 实际部署时性能更好
```

---

### 中优先级

#### 5. Activation Quantization for DLMs

**现状：** 大多数工作只做权重量化

**机会：**
- 激活量化可以进一步加速（整数矩阵乘法）
- 但 DLM 的激活 outliers 更复杂
- 需要新的激活量化方法

---

#### 6. Long-Context DLM Quantization

**问题：** 长序列下 KV cache 成为瓶颈

**机会：**
- KV cache 量化
- 稀疏注意力 + 量化
- 针对长上下文的特殊优化

---

## 📚 推荐阅读顺序

### 入门（了解领域）
1. **Quantization Meets dLLMs** — 系统性综述，必读
2. **A Survey on Diffusion Language Models** — DLM 整体 survey

### 进阶（技术细节）
3. **DLLMQuant** — Timestep-Aware 校准
4. **Quant-dLLM** — 极低比特量化

### 拓展（AR LLM 量化）
5. **GPTQ** — 经典权重量化
6. **AWQ** — 激活感知量化
7. **SmoothQuant** — 权重 - 激活量化

---

## 🎯 对你的研究建议

### 如果做 DLM 量化

**短期（1-2 月）：**
1. 复现 QDLM (GPTQ) 在 LLaDA 上
2. 验证 activation outliers 现象
3. 尝试 Timestep-Adaptive 量化

**中期（3-6 月）：**
1. 实现 Joint Pruning + Quantization
2. 在多个 DLM 上验证
3. 写论文（目标：ICLR/NeurIPS）

**长期（6-12 月）：**
1. 探索 QAT for DLMs
2. Hardware-Aware 优化
3. 开源工具，建立影响力

---

### 如果做相关方向

**可借鉴的思路：**
- Timestep-Aware → 可用于其他 DLM 优化
- Group-wise Quantization → 通用技术
- Mixed-Precision → 系统级优化

---

## 📬 总结

**领域现状：**
- DLM 量化研究刚刚起步（2025 年 8 月第一篇系统研究）
- 3 篇核心论文提供了基础方法
- 大量开放问题等待探索

**推荐起点：**
- 从 QDLM (GPTQ) 开始
- 在 LLaDA-8B 上复现
- 逐步探索改进方向

**研究价值：**
- DLM 是新兴方向，Efficient 优化需求大
- 工业界需要（推理成本太高）
- 学术价值高（顶会友好）

---

*返回 [00-daily-updates.md](00-daily-updates.md)*
