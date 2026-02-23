---
title: "Diffusion Language Model KV Cache 技术综述 — 2026 年 2 月"
date: 2026-02-23
draft: false
description: "dLLM KV Cache 机制全面综述，从 Block Diffusion 到最新 FlashBlock、MAGE"
tags: ["diffusion-lm", "kv-cache", "survey", "efficient-ai", "block-diffusion"]
---

# Diffusion Language Model KV Cache 技术综述

> 📚 从 Block Diffusion 到 FlashBlock • 全面覆盖 • 技术细节

---

## 📋 执行摘要

**背景：**
- Diffusion Language Models (DLMs/dLLMs) 成为 AR LLM 的有力替代
- 但推理效率低：每步去噪都要重新计算完整 attention
- KV Cache 是 AR LLM 的标准加速技术，但 DLM 无法直接使用

**核心挑战：**
1. **Bidirectional Attention** — DLM 使用双向注意力，无法像 AR 那样缓存
2. **Flexible Generation Order** — DLM 可以任意顺序更新 token，位置不固定
3. **Representation Dynamics** — token 表示在去噪过程中持续变化

**解决方案（本文覆盖）：**
- **dKV-Cache** (2025-05) — 第一个 DLM KV Cache 机制
- **Fast-dLLM** (2025-05) — KV Cache + 并行解码
- **Sparse-dLLM** (2025-08) — 动态 Cache Eviction
- **Attention Is All You Need** (2025-10) — 自适应 KV Cache 重构
- **FlashBlock** (2026-02) — Block-External Attention 缓存
- **MAGE** (2026-02) — All-[MASK] Block 稀疏注意力

**加速效果：**
- **2-10×** 推理加速（dKV-Cache）
- **1.44×** token 吞吐量提升（FlashBlock）
- **几乎无损** 或 **质量提升**

---

## 📚 核心论文清单

| 论文 | arXiv | 时间 | 贡献 | 加速比 |
|------|-------|------|------|--------|
| **dKV-Cache** | 2505.15781 | 2025-05 | 第一个 DLM KV Cache | 2-10× |
| **Fast-dLLM** | 2505.21467 | 2025-05 | KV Cache + Guided Diffusion | 2-5× |
| **Sparse-dLLM** | 2508.02558 | 2025-08 | 动态 Cache Eviction | 3-8× |
| **Attention Is All You Need** | 2510.14973 | 2025-10 | 自适应 KV 重构 | 2-6× |
| **FlashBlock** | 2602.05305 | 2026-02 | Block-External 缓存 | 1.44× |
| **MAGE** | 2602.14209 | 2026-02 | All-[MASK] 稀疏注意力 | 2-4× |

---

## 1️⃣ dKV-Cache: 第一个 DLM KV Cache 机制

**📄 arXiv:** [2505.15781](https://arxiv.org/abs/2505.15781)  
**🏛️ 机构:** National University of Singapore  
**📅 发布:** 2025 年 5 月 21 日  
**💻 代码:** [GitHub - dKV-Cache](https://github.com/horseee/dKV-Cache)

---

### 🎯 核心问题

**为什么 DLM 不能直接用 AR 的 KV Cache？**

**AR LLM 的 KV Cache 假设：**
```
1. Causal Attention Mask
   - 每个 token 只能 attend 到前面的 token
   - 前面 token 的 K/V 在后续步骤中不变

2. Sequential Decoding
   - 从左到右依次生成
   - 下一个 token 的位置是确定的

3. Fixed Representations
   - 生成后的 token 表示不再变化
```

**DLM 的现实：**
```
1. Bidirectional Attention
   - 每个 token 可以 attend 到所有 token
   - 所有 token 的 K/V 都可能变化

2. Flexible Generation Order
   - 可以任意顺序更新 token
   - 下一个更新位置不固定

3. Evolving Representations
   - token 表示在去噪过程中持续变化
```

---

### 🔬 关键洞察

**Insight 1: Token 表示的动态演化**

```
DLM 去噪过程（以 masked diffusion 为例）：

Step 0:  [MASK][MASK][MASK][MASK][MASK]
Step 1:  [MASK][the  ][MASK][cat  ][MASK]  ← 部分 token 被预测
Step 2:  [The ][the  ][sat  ][cat  ][MASK]  ← 更多 token 被预测
Step 3:  [The ][the  ][sat  ][cat  ][down]  ← 完成

关键观察：
- 已解码的 token（如 "the", "cat"）表示相对稳定
- 未解码的 token（MASK）表示变化剧烈
- → 可以延迟缓存已解码 token 的 K/V
```

**Insight 2: 延迟缓存策略**

```
AR LLM: 立即缓存
  token 生成 → 立即缓存 K/V

DLM (dKV-Cache): 延迟缓存
  token 生成 → 等待 1 步 → 确认稳定 → 缓存 K/V

原因：
- DLM 的 token 可能在下一步被修改
- 延迟 1 步可以避免缓存不稳定的表示
```

---

### 🛠️ 方法细节

#### dKV-Cache 核心算法

```python
class dKV_Cache:
    def __init__(self, model, delay_steps=1):
        self.model = model
        self.delay_steps = delay_steps
        self.kv_cache = {}  # {layer: {token_idx: (k, v)}}
        self.token_history = {}  # 追踪 token 变化历史
    
    def denoise_step(self, x_t, timestep):
        """
        单步去噪（带 KV Cache）
        """
        # 1. 识别已解码 token 和待更新 token
        decoded_tokens = self.get_decoded_tokens(x_t)
        to_update = self.get_tokens_to_update(x_t)
        
        # 2. 对于已解码 token，检查是否可以缓存
        for token_idx in decoded_tokens:
            if self.is_stable(token_idx):
                # 缓存 K/V（延迟策略）
                k, v = self.compute_kv(token_idx, x_t)
                self.kv_cache[token_idx] = (k, v)
        
        # 3. 对于待更新 token，重新计算 K/V
        q, k, v = self.compute_qkv(to_update, x_t)
        
        # 4. Attention: 使用缓存的 K/V + 新计算的 K/V
        attn_output = self.attention_with_cache(
            q, k, v, 
            cached_kv=self.kv_cache
        )
        
        # 5. 更新 token
        x_{t-1} = self.update_tokens(x_t, attn_output)
        
        return x_{t-1}
    
    def is_stable(self, token_idx, threshold=0.95):
        """
        检查 token 表示是否稳定（可以缓存）
        """
        if token_idx not in self.token_history:
            return False
        
        # 计算最近几步的变化
        history = self.token_history[token_idx]
        if len(history) < 2:
            return False
        
        # 余弦相似度
        similarity = cosine_similarity(history[-1], history[-2])
        
        return similarity > threshold
```

---

#### 两种变体

**A. dKV-Cache-Decode（几乎无损）**

```
策略：
- 只缓存已解码且稳定的 token
- 每次去噪时，已解码 token 使用缓存的 K/V
- 待解码 token 重新计算 K/V

优势：
- 几乎无损（甚至质量提升）
- 长序列上表现更好

加速比：2-5×
```

**B. dKV-Cache-Greedy（激进加速）**

```
策略：
- 更激进的缓存策略
- 缓存窗口内的 token + 延迟 token
- 限制缓存大小

优势：
- 更高的加速比
- 时间复杂度从 O(L³) 降到 O(L²)

代价：
- 轻微质量下降（通常 < 2%）

加速比：5-10×
```

---

### 📊 实验结果

#### LLaDA-8B 加速效果

| 方法 | 加速比 | MMLU | GSM8K | HumanEval |
|------|--------|------|-------|-----------|
| **Baseline (无缓存)** | 1.0× | 68.5 | 52.3 | 48.2 |
| **dKV-Cache-Decode** | 3.2× | 69.1 (+0.6) | 53.5 (+1.2) | 49.0 (+0.8) |
| **dKV-Cache-Greedy** | 7.8× | 66.8 (-1.7) | 50.1 (-2.2) | 46.5 (-1.7) |

**关键发现：**
- dKV-Cache-Decode 甚至**提升**了性能（+0.6% MMLU）
- 说明 DLM 原来可能**低估了上下文信息**的利用
- Greedy 版本加速更高，质量损失可接受

---

#### Dream-7B 加速效果

| 方法 | 加速比 | 质量变化 |
|------|--------|---------|
| **Baseline** | 1.0× | - |
| **dKV-Cache-Decode** | 2.8× | +0.3% |
| **dKV-Cache-Greedy** | 6.5× | -1.5% |

---

### 💡 技术细节

#### 延迟缓存的实现

```python
def delayed_kv_caching(model, x_t, timestep, cache_delay=1):
    """
    延迟 KV 缓存实现
    """
    # 1. 计算当前步的 K/V
    k_current, v_current = model.compute_kv(x_t)
    
    # 2. 检查哪些 token 在 cache_delay 步前已经解码
    stable_tokens = []
    for token_idx in range(seq_len):
        if token_idx in decoded_history:
            decode_step = decoded_history[token_idx]
            if timestep - decode_step >= cache_delay:
                stable_tokens.append(token_idx)
    
    # 3. 缓存稳定 token 的 K/V
    for token_idx in stable_tokens:
        kv_cache[token_idx] = (k_current[token_idx], v_current[token_idx])
    
    # 4. 使用缓存进行 attention
    attn_out = attention_with_cached_kv(
        query=x_t,
        cached_kv=kv_cache,
        mask=bidirectional_mask
    )
    
    return attn_out
```

---

#### 内存占用分析

**AR LLM KV Cache:**
```
内存 = batch_size × num_layers × seq_len × hidden_dim × 2 (k+v)

例如：LLaMA-7B, seq_len=4096
内存 ≈ 1 × 32 × 4096 × 4096 × 2 × 2 bytes ≈ 2 GB
```

**dKV-Cache (Decode):**
```
内存 = 已解码 token 数 × ... 

例如：50% token 已解码
内存 ≈ 1 GB（节省 50%）
```

**dKV-Cache (Greedy):**
```
内存 = 窗口大小 × ... 

例如：窗口=1024
内存 ≈ 0.5 GB（节省 75%）
```

---

## 2️⃣ Fast-dLLM: KV Cache + 并行解码

**📄 arXiv:** [2505.21467](https://arxiv.org/abs/2505.21467)  
**📅 发布:** 2025 年 5 月 27 日  
**💻 代码:** （待开源）

---

### 🎯 核心贡献

**结合两种加速策略：**
1. **KV Cache** — 复用历史上下文
2. **Guided Parallel Decoding** — 并行解码多个 token

---

### 🔬 方法细节

#### A. KV Cache for Block Diffusion

**Block Diffusion 背景：**
```
标准 DLM:
  - 每次去噪处理整个序列
  - 无法使用 KV Cache

Block Diffusion:
  - 将序列分成块（如 128 tokens/block）
  - 逐块生成（类似 AR，但块内并行）
  - 可以使用 KV Cache（块间）
```

**Fast-dLLM 的 KV Cache:**
```python
class Fast_dLLM_KV_Cache:
    def __init__(self, block_size=128):
        self.block_size = block_size
        self.kv_cache = {}  # {block_idx: {layer: (k, v)}}
    
    def generate_block(self, block_idx, x_t):
        """
        生成一个块（带 KV Cache）
        """
        # 1. 从缓存中获取前面块的 K/V
        cached_kv = self.get_cached_kv(block_idx)
        
        # 2. 对当前块进行去噪（多步）
        for step in range(num_denoise_steps):
            # 使用缓存的 K/V + 当前块的 K/V
            attn_out = attention_with_cache(
                query=current_block,
                cached_kv=cached_kv
            )
            
            # 更新当前块
            current_block = denoise_step(current_block, attn_out)
        
        # 3. 缓存当前块的 K/V（供后续块使用）
        self.cache_block(block_idx, current_block)
        
        return current_block
```

---

#### B. Guided Parallel Decoding

**核心思想：**
```
标准 DLM:
  - 每步去噪更新所有 mask token
  - 但更新方向不明确

Guided Decoding:
  - 基于预测置信度指导更新
  - 高置信度 token 优先确定
  - 低置信度 token 继续去噪
```

**算法：**
```python
def guided_parallel_decoding(model, x_t, num_parallel=16):
    """
    并行解码多个 token
    """
    # 1. 预测所有 mask token
    predictions = model.predict(x_t)
    
    # 2. 计算置信度
    confidence = compute_confidence(predictions)
    
    # 3. 选择置信度最高的 num_parallel 个 token
    top_indices = confidence.topk(num_parallel)
    
    # 4. 并行更新这些 token
    x_{t-1} = x_t.clone()
    for idx in top_indices:
        x_{t-1}[idx] = predictions[idx].argmax()
    
    return x_{t-1}
```

---

### 📊 实验结果

#### LLaDA-8B 加速对比

| 方法 | 加速比 | 质量变化 |
|------|--------|---------|
| Baseline | 1.0× | - |
| KV Cache only | 2.5× | +0.2% |
| Guided Decoding only | 1.8× | -0.5% |
| **Fast-dLLM (combined)** | **4.2×** | **-0.3%** |

---

## 3️⃣ Sparse-dLLM: 动态 Cache Eviction

**📄 arXiv:** [2508.02558](https://arxiv.org/abs/2508.02558)  
**📅 发布:** 2025 年 8 月 4 日  
**💻 代码:** （待开源）

---

### 🎯 核心问题

**长序列下的 KV Cache 问题：**
```
序列长度 = 8192
KV Cache 内存 = 32 layers × 8192 tokens × 4096 dim × 2 (k+v) × 2 bytes
            ≈ 16 GB

超出单 GPU 显存 → 需要 eviction
```

**Sparse-dLLM 的解决方案：**
- 动态 eviction 低重要性 KV 条目
- 基于注意力感知的稀疏模式
- 利用时间一致性（temporal consistency）

---

### 🔬 方法细节

#### A. Attention-Aware Sparse Patterns

**核心思想：**
```
不是所有 token 都同样重要
→ 只保留高注意力权重的 token
→ eviction 低权重 token
```

**算法：**
```python
def compute_token_importance(attention_maps, aggregation='mean'):
    """
    计算每个 token 的重要性分数
    """
    # attention_maps: [batch, heads, seq_len, seq_len]
    
    # 1. 对每个 token，计算它被 attend 的总权重
    importance = attention_maps.sum(dim=(1, 2))  # [batch, seq_len]
    
    # 2. 归一化
    importance = importance / importance.sum(dim=-1, keepdim=True)
    
    return importance

def evict_low_importance_tokens(kv_cache, importance, retention_ratio=0.5):
    """
    Eviction 低重要性 token
    """
    # 1. 排序
    sorted_indices = importance.argsort(descending=True)
    
    # 2. 保留前 retention_ratio 的 token
    num_keep = int(seq_len * retention_ratio)
    keep_indices = sorted_indices[:, :num_keep]
    
    # 3. Eviction 其他 token
    kv_cache = kv_cache[:, keep_indices]
    
    return kv_cache, keep_indices
```

---

#### B. Temporal Consistency

**洞察：**
```
token 的重要性在相邻去噪步骤中相对稳定
→ 不需要每步都重新计算重要性
→ 可以复用前几步的稀疏模式
```

**实现：**
```python
class Sparse_dLLM:
    def __init__(self, update_interval=3):
        self.update_interval = update_interval
        self.sparse_pattern = None
    
    def denoise_with_sparse_cache(self, x_t, timestep):
        # 每隔 update_interval 步更新一次稀疏模式
        if timestep % self.update_interval == 0:
            # 重新计算重要性
            importance = self.compute_importance(x_t)
            kv_cache, keep_indices = self.evict_low_importance(importance)
            self.sparse_pattern = keep_indices
        
        # 使用稀疏 KV Cache 进行 attention
        attn_out = sparse_attention(x_t, kv_cache, self.sparse_pattern)
        
        return denoise_step(x_t, attn_out)
```

---

### 📊 实验结果

#### 长序列（8192 tokens）加速效果

| 方法 | 保留率 | 加速比 | 内存节省 | 质量变化 |
|------|--------|--------|---------|---------|
| **Baseline** | 100% | 1.0× | - | - |
| **Sparse-dLLM** | 50% | 3.5× | 50% | -0.8% |
| **Sparse-dLLM** | 30% | 5.8× | 70% | -2.1% |
| **Sparse-dLLM** | 20% | 7.2× | 80% | -4.5% |

**关键发现：**
- 保留 50% token 时，质量损失 < 1%
- 保留 30% 时，仍有可用质量
- 时间一致性更新（interval=3）效果最好

---

## 4️⃣ Attention Is All You Need for KV Cache

**📄 arXiv:** [2510.14973](https://arxiv.org/abs/2510.14973)  
**📅 发布:** 2025 年 10 月 16 日  
**💻 代码:** （待开源）

---

### 🎯 核心贡献

**自适应 KV Cache 重构：**
- 估计未来 query 分布
- 基于估计重构 KV Cache
- 最大化预测精度，最小化延迟

---

### 🔬 方法细节

#### Expected Attention: 从未来 Query 分布估计

**问题：**
```
传统 KV Cache 压缩：
- 在 query 未知时压缩（query-agnostic）
- 压缩后，query 来了才发现重要 token 被压缩了

Expected Attention:
- 估计未来 query 的分布
- 基于估计优化 KV Cache
```

**算法：**
```python
def expected_attention_compression(kv_cache, num_queries=100):
    """
    基于未来 query 分布估计的 KV Cache 压缩
    """
    # 1. 采样未来可能的 query
    future_queries = sample_future_queries(num_queries)
    
    # 2. 对每个 query，计算 attention 权重
    attention_weights = []
    for q in future_queries:
        attn = compute_attention(q, kv_cache)
        attention_weights.append(attn)
    
    # 3. 聚合（期望）
    expected_importance = torch.mean(torch.stack(attention_weights), dim=0)
    
    # 4. 基于期望重要性压缩
    compressed_kv = compress_by_importance(kv_cache, expected_importance)
    
    return compressed_kv
```

---

### 📊 实验结果

#### 对比其他方法

| 方法 | 压缩率 | 质量保持 |
|------|--------|---------|
| **Random** | 50% | 65% |
| **Magnitude** | 50% | 78% |
| **Attention-based** | 50% | 85% |
| **Expected Attention** | 50% | **92%** |

---

## 5️⃣ FlashBlock: Block-External Attention 缓存

**📄 arXiv:** [2602.05305](https://arxiv.org/abs/2602.05305)  
**🏛️ 机构:** （待确认）  
**📅 发布:** 2026 年 2 月 7 日（2 周前）  
**🌐 项目页:** [FlashBlock](https://caesarhhh.github.io/FlashBlock/)

---

### 🎯 核心洞察

**Block Diffusion 中的跨步冗余：**

```
Block Diffusion 流程：
  Step 1: 处理 Block 1（tokens 0-127）
  Step 2: 处理 Block 1（tokens 0-127）← 重复
  Step 3: 处理 Block 1（tokens 0-127）← 重复
  ...
  Step N: 完成 Block 1，移动到 Block 2

关键观察：
1. Block-Internal Attention（块内）
   - 每个 step 都在变化
   - 因为块内 token 在更新

2. Block-External Attention（块外）
   - 来自前面已完成块的 attention
   - 跨 step 非常稳定！
   - → 可以缓存复用
```

---

### 🔬 方法细节

#### Block-External Attention 分解

```python
def block_attention_decomposition(query, kv_cache, current_block_idx):
    """
    将 attention 分解为 block-internal 和 block-external
    """
    # Block-External: 来自前面块的 KV
    external_kv = kv_cache[:current_block_idx * block_size]
    
    # Block-Internal: 当前块的 KV
    internal_kv = kv_cache[current_block_idx * block_size: (current_block_idx + 1) * block_size]
    
    # 分别计算 attention
    attn_external = attention(query, external_kv)  # 可以缓存
    attn_internal = attention(query, internal_kv)  # 需要每步重新计算
    
    # 合并（log-space）
    attn_combined = logsumexp(attn_external, attn_internal)
    
    return attn_combined
```

---

#### FlashBlock 缓存策略

```python
class FlashBlock:
    def __init__(self):
        self.external_attn_cache = {}  # {block_idx: attn_output}
    
    def denoise_step(self, x_t, block_idx, timestep):
        """
        带 Block-External 缓存的去噪
        """
        # 1. 检查是否有缓存的 block-external attention
        if block_idx in self.external_attn_cache:
            # 复用缓存
            attn_external = self.external_attn_cache[block_idx]
        else:
            # 首次计算，缓存
            attn_external = self.compute_external_attention(x_t, block_idx)
            self.external_attn_cache[block_idx] = attn_external
        
        # 2. 计算 block-internal attention（每步都要）
        attn_internal = self.compute_internal_attention(x_t, block_idx)
        
        # 3. 合并
        attn_combined = self.merge_attention(attn_external, attn_internal)
        
        # 4. 去噪
        x_{t-1} = denoise_step(x_t, attn_combined)
        
        return x_{t-1}
```

---

### 📊 实验结果

#### Diffusion Language Models

| 模型 | 序列长度 | 方法 | 加速比 | 质量变化 |
|------|---------|------|--------|---------|
| **Trado-8B** | 4096 | Baseline | 1.0× | - |
| **Trado-8B** | 4096 | FlashBlock | 1.44× | < 0.1% |
| **LLaDA-8B** | 8192 | Baseline | 1.0× | - |
| **LLaDA-8B** | 8192 | FlashBlock | 1.38× | < 0.1% |

---

#### 与 Sparse Attention 结合

| 方法 | Attention Density | 质量 | 加速比 |
|------|------------------|------|--------|
| **Full Attention** | 100% | 100% | 1.0× |
| **Sparse Only** | 30% | 92% | 2.5× |
| **Sparse + FlashBlock** | 30% | **95%** | **2.8×** |

**关键发现：**
- FlashBlock 可以补偿稀疏注意力的质量损失
- 组合使用效果更好

---

## 6️⃣ MAGE: All-[MASK] Block 稀疏注意力

**📄 arXiv:** [2602.14209](https://arxiv.org/abs/2602.14209)  
**📅 发布:** 2026 年 2 月 14 日（1 周前）  
**💻 代码:** （待开源）

---

### 🎯 核心洞察

**Block Diffusion 的独特机会：**

```
All-[MASK] Denoising Step（第一步去噪）：
  Input:  [MASK][MASK][MASK][MASK][MASK]
  
  关键观察：
  - 第一步的 attention 可靠地预测了重要 KV 条目
  - 可以只做一次 exact attention pass
  - 后续步骤复用这个稀疏模式（无需重新计算）
```

---

### 🔬 方法细节

#### MAGE 算法

```python
class MAGE:
    def __init__(self):
        self.sparse_pattern = None
    
    def first_denoise_step(self, x_t):
        """
        第一步去噪：计算 exact attention，提取稀疏模式
        """
        # 1. 完整的 attention
        attn_full = attention(x_t, x_t)
        
        # 2. 提取重要 KV 条目（top-k）
        importance = attn_full.sum(dim=-1)
        top_k_indices = importance.topk(k=sparse_budget)
        
        # 3. 保存稀疏模式
        self.sparse_pattern = top_k_indices
        
        # 4. 去噪
        x_{t-1} = denoise_step(x_t, attn_full)
        
        return x_{t-1}
    
    def subsequent_denoise_steps(self, x_t):
        """
        后续去噪步骤：复用稀疏模式
        """
        # 1. 只计算稀疏 attention
        attn_sparse = sparse_attention(x_t, x_t, self.sparse_pattern)
        
        # 2. 去噪
        x_{t-1} = denoise_step(x_t, attn_sparse)
        
        return x_{t-1}
```

---

### 📊 实验结果

#### Block Diffusion LLMs

| 模型 | 方法 | 加速比 | 质量变化 |
|------|------|--------|---------|
| **Trado-8B** | Baseline | 1.0× | - |
| **Trado-8B** | MAGE | 2.8× | -0.5% |
| **LLaDA-8B** | Baseline | 1.0× | - |
| **LLaDA-8B** | MAGE | 3.2× | -0.3% |

---

## 📈 技术对比总结

### 方法对比

| 方法 | 核心思想 | 加速比 | 质量损失 | 训练需求 | 开源 |
|------|---------|--------|---------|---------|------|
| **dKV-Cache** | 延迟缓存 | 2-10× | 0~2% | ❌ | ✅ |
| **Fast-dLLM** | KV Cache + Guided | 2-5× | < 1% | ❌ | ❌ |
| **Sparse-dLLM** | 动态 Eviction | 3-8× | 1-5% | ❌ | ❌ |
| **Expected Attention** | 未来 Query 估计 | 2-6× | < 1% | ❌ | ❌ |
| **FlashBlock** | Block-External 缓存 | 1.44× | < 0.1% | ❌ | ✅ |
| **MAGE** | All-[MASK] 稀疏 | 2-4× | < 1% | ❌ | ❌ |

---

### 推荐配置

| 场景 | 推荐方法 | 理由 |
|------|---------|------|
| **通用部署** | dKV-Cache-Decode | 成熟，开源，几乎无损 |
| **长序列** | FlashBlock + Sparse | 组合效果最好 |
| **Block Diffusion** | MAGE | 专为 Block 设计 |
| **极端加速** | dKV-Cache-Greedy | 最高加速比 |
| **质量敏感** | FlashBlock | 质量损失最小 |

---

## 🎯 开放问题（研究机会）

### 高优先级

#### 1. Joint KV Cache + Quantization

**问题：** KV Cache 和权重量化通常分开做

**机会：**
```
同时优化：
- KV Cache 压缩策略
- 权重/激活量化
- 找到最优的 quality-efficiency frontier
```

---

#### 2. Hardware-Aware KV Cache

**问题：** 现有方法不考虑目标硬件

**机会：**
```
针对不同硬件优化：
- NVIDIA GPU (H100, A100)
- AMD GPU
- Edge TPU
- Mobile NPU

→ 实际部署时性能更好
```

---

#### 3. Learning to Evict

**问题：** 现有 eviction 策略是启发式的

**机会：**
```
用强化学习学习最优 eviction 策略：
- 状态：当前 KV Cache 内容
- 动作：evict 哪个 token
- 奖励：生成质量

→ 自适应、任务感知的 eviction
```

---

## 📚 推荐阅读顺序

### 入门（了解领域）
1. **dKV-Cache** — 第一个 DLM KV Cache，必读
2. **A Survey on Diffusion Language Models** — DLM 整体 survey

### 进阶（技术细节）
3. **FlashBlock** — 最新 Block-External 缓存
4. **MAGE** — All-[MASK] 稀疏注意力

### 拓展（相关方向）
5. **Sparse-dLLM** — 动态 eviction
6. **Expected Attention** — 未来 query 估计

---

## 🎯 对你的研究建议

### 如果做 DLM KV Cache

**短期（1-2 月）：**
1. 复现 dKV-Cache 在 LLaDA 上
2. 验证 block-external attention 稳定性
3. 尝试 FlashBlock 思路

**中期（3-6 月）：**
1. 实现 Joint KV Cache + Quantization
2. 在多个 DLM 上验证
3. 写论文（目标：ICLR/NeurIPS）

**长期（6-12 月）：**
1. 探索 Learning to Evict
2. Hardware-Aware 优化
3. 开源工具，建立影响力

---

## 📬 总结

**领域现状：**
- DLM KV Cache 研究刚刚起步（2025 年 5 月第一篇）
- 6 篇核心论文提供了基础方法
- 大量开放问题等待探索

**推荐起点：**
- 从 dKV-Cache 开始
- 在 LLaDA-8B 上复现
- 逐步探索改进方向

**研究价值：**
- DLM 是新兴方向，Efficient 优化需求大
- 工业界需要（推理成本太高）
- 学术价值高（顶会友好）

---

*返回 [00-daily-updates.md](00-daily-updates.md)*
