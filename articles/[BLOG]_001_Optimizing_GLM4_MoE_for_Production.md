# Optimizing GLM4-MoE for Production: 65% Faster TTFT with SGLang

## 📌 元数据

- **来源**: https://lmsys.org/blog/2026-01-21-novita-glm4/
- **作者**: Novita AI & LMSYS Org
- **日期**: 2026-01-21
- **阅读日期**: 2025-02-01
- **分类**: `APP`
- **标签**: #MoE #GLM4 #SGLang #Optimization #Inference #Production #Performance

---

## 📖 文章概述

Novita AI 基于 SGLang 开发了一套端到端的 GLM4-MoE 生产优化方案，通过 Shared Experts Fusion、Qknorm Fusion、Async Transfer 和 Suffix Decoding 等技术，在 H200 集群上实现了 **TTFT 降低 65%**、**TPOT 提升 22%** 的显著性能提升。

---

## 🎯 核心内容

### 主要观点

1. **端到端优化策略**
   - 不仅仅优化单个组件，而是从整个推理 pipeline 入手
   - 解决从 kernel 执行效率到跨节点数据传输调度的全链路瓶颈
   - 所有优化均在实际生产环境中验证（H200 集群，TP8 + FP8 配置）

2. **四大核心优化**
   - **Shared Experts Fusion**: 合并共享专家到路由 MoE 结构
   - **Qknorm Fusion**: 融合 QK 归一化和 RoPE 操作
   - **Async Transfer**: 异步数据传输优化
   - **Suffix Decoding**: 针对代理编码场景的模式复用加速

3. **性能提升**
   - TTFT (Time to First Token): **最多降低 65%**
   - TPOT (Time Per Output Token): **提升 22%**
   - 在代理编码工作负载下表现尤为突出

### 技术要点

#### 1. Shared Experts Fusion（共享专家融合）

**原理**:
- GLM4.7 有 160 个路由专家 + 1 个共享专家
- 每个 token 选择 top-8 路由专家
- 原先分别处理共享专家和路由专家
- 优化后将共享专家合并到路由 MoE 结构中，从 161 个专家中选择 top-9

**效果**:
- TTFT 提升 **23.7%**
- ITL (Inter-Token Latency) 提升 **20.8%**
- 在 TP8 + FP8 配置下（intermediate size = 192），显著提升 SM 利用率
- 减少内存 I/O 开销

**适用场景**:
- 中小 intermediate size 的 MoE 模型
- GPU 算力较强但内存带宽受限的场景

#### 2. Qknorm Fusion（QK 归一化融合）

**原理**:
- QK 归一化和 RoPE 都是按头（head-wise）计算
- 将两个操作融合到单个 kernel
- 适配 GLM4-MoE 的特殊情况：只有半个维度旋转

**来源**:
- 基于 Qwen-MoE 的优化思路
- PR: SGLang #15141, #15305

**效果**:
- 减少 kernel 启动开销
- 提升计算效率

#### 3. Async Transfer（异步传输）

**问题背景**:
- 在 PD (Prefill-Decode) disaggregation + overlapping schedule 场景下
- 虽然吞吐量提升 10%，但 TTFT 显著下降
- 原因：数据传输延迟到下一批 kernel 启动后才开始
- 对于 92 层的 GLM4.7，kernel 启动耗时数百毫秒甚至超过 1 秒

**优化方案**:
- 提前传输：在对应 GPU 操作完成后立即调度传输
- 独立线程：将传输放在单独线程中，不阻塞主线程
- 精心处理数据竞争

**效果**:
- 在重负载下，TTFT 最多可节省 **1 秒**
- 特别适合层数多、kernel 启动频繁的模型

#### 4. Suffix Decoding（后缀解码）

**背景**:
- 代理编码场景（如 Cursor、Claude Code）存在大量可复用的代码模式
- 传统 Speculative Decoding 需要训练额外的 draft 模型，工程复杂度高

**创新点**:
- **完全无模型（Model-free）**: 不依赖额外的模型权重
- **模式复用**: 利用历史输出序列的模式预测即将到来的 token
- **智能匹配**: 当当前请求的后缀与历史模式匹配时，沿历史序列进行推测

**数据验证**:
分析了 22 个 Claude Code 会话（17,487 对话轮次）：
- **39.3% 的输出存在模式重复**
- 高度结构化的代理行为
- 固定短语频繁出现："Let me...", "Now let me..." 等
- 数据集已开源：Agentic Code Dataset on Hugging Face

**效果**:
- TPOT 从 25.13ms 降至 19.63ms
- **提升 21.90%**（平均值）
- **提升 22.70%**（中位值）

### 重要发现

1. **融合操作的价值**
   - 在现代 GPU 上，减少 kernel 启动次数比提升单个 kernel 效率更重要
   - 内存带宽往往是瓶颈，而非计算能力

2. **异步优化的重要性**
   - 数据传输的延迟对 TTFT 影响巨大
   - 对于多层模型，kernel 启动开销不容忽视

3. **场景特定优化**
   - 代理编码场景的特殊性（模式重复）为优化提供了机会
   - 通用优化不如针对性优化效果好

4. **生产环境 vs 实验环境**
   - 理论优化在实际部署中可能遇到意想不到的问题
   - 必须在真实负载下验证性能提升

---

## 💡 个人思考

### 有启发的点

1. **系统优化思维**
   - 不只关注单个组件，而是全链路优化
   - 从 kernel 到传输调度，每个环节都可能成为瓶颈
   - 这与之前 TATER 论文的思路一致：系统性优化胜过局部优化

2. **工程实践的价值**
   - 论文和实际生产之间存在巨大差距
   - 真实场景的数据（如 39.3% 模式重复）极具价值
   - 开源数据集有助于社区研究

3. **MoE 模型的优化空间**
   - Shared Experts Fusion 证明 MoE 结构还有优化空间
   - 中小模型在合适优化下可以媲美大模型性能
   - 这与第二篇论文（Scaling Lessons）的结论相呼应

4. **场景驱动的优化**
   - Suffix Decoding 是专门针对代理编码场景的优化
   - 特定场景的特性可以被利用来大幅提升性能
   - "通用不如专用"在工程中屡试不爽

### 疑问

1. **通用性**
   - 这些优化对其他 MoE 模型（如 Mixtral、DeepSeek）的效果如何？
   - 非 MoE 模型能否借鉴类似的思路？

2. **成本考虑**
   - Async Transfer 增加了实现复杂度，bug 风险如何控制？
   - Suffix Decoding 需要维护历史缓存，内存开销如何？

3. **适用场景**
   - 非编码场景下，Suffix Decoding 的效果如何？
   - 对话、翻译等场景是否也有类似的模式重复？

4. **技术栈依赖**
   - 这些优化是否依赖 SGLang 的特定实现？
   - vLLM、TensorRT-LLM 等其他框架能否实现类似优化？

### 与其他文章的关联

- **Scaling Lessons (PAPER_002)**:
  - 都关注计算效率和性能优化
  - 小模型优化的思路一致：通过优化而非扩大规模来提升性能

- **TATER (PAPER_001)**:
  - 都强调系统优化的重要性
  - TATER 关注搜索经验回收，本文关注推理 pipeline 优化
  - 共同主题：更聪明地使用计算资源

- **MoE 相关论文**:
  - Shared Experts Fusion 是 MoE 架构优化的实例
  - 与 DeepSeek、Mixtral 等 MoE 模型相关

- **Speculative Decoding 研究**:
  - Suffix Decoding 是一种新的投机解码思路
  - 无需额外模型，降低了工程复杂度

---

## 📎 关键摘录

> "We introduce an end-to-end performance optimization strategy that addresses bottlenecks across the entire inference pipeline — from kernel execution efficiency to cross-node data transfer scheduling."

> "Through the integration of Shared Experts Fusion and Suffix Decoding, we observe substantial gains in key production metrics, including up to 65% reduction in Time-to-First-Token (TTFT) and 22% improvement in Time-Per-Output-Token (TPOT) under agentic coding workloads."

> "Suffix Decoding takes a fundamentally different approach—it is completely model-free: No dependency on additional model weights, Leverages patterns from previously generated output sequences to predict upcoming tokens."

> "By analyzing 22 Claude Code sessions (17,487 conversation turns), we discovered: 39.3% output pattern repetition: High frequency of similar tool calls and response patterns."

---

## 🔗 相关资源

- **原文**: https://lmsys.org/blog/2026-01-21-novita-glm4/
- **SGLang GitHub**: https://github.com/sgl-project/sglang
- **Novita Labs 实现**: novitalabs/sglang (glm_suffix branch)
- **数据集**: Agentic Code Dataset on Hugging Face
- **相关 PR**:
  - SGLang PR #13873: Shared Experts Fusion
  - SGLang PR #15141: Qknorm Fusion
  - SGLang PR #15305: Qknorm Fusion Fix
  - SGLang PR #14782: Async Transfer
- **相关论文**:
  - Snowflake Engineering Blog: SuffixDecoding at Production Scale
  - NeurIPS Paper: SuffixDecoding
- **相关技术**:
  - Speculative Decoding
  - Mixture of Experts (MoE)
  - KV Cache Optimization
  - CUDA Graph

---

## 📊 补充说明

**优化配置总结**:

核心优化标志（SGLang Runtime）:
```bash
--tp-size 8
--kv-cache-dtype fp8_e4m3
--attention-backend fa3
--chunked-prefill-size 16384
--enable-flashinfer-allreduce-fusion
--enable-fused-qk-norm-rope
--enable-shared-experts-fusion
--disaggregation-async-transfer
```

投机解码配置（代理编码工作负载）:
```bash
--speculative-algorithm NEXTN
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

Suffix Decoding 配置（可选）:
```bash
--speculative-algorithm SUFFIX
--speculative-suffix-cache-max-depth 64
--speculative-suffix-max-spec-factor 1.0
--speculative-suffix-min-token-prob 0.1
```

**Benchmark 配置**:
- 输入长度: 4096
- 输出长度: 1000
- 请求率: 14 req/s
- 模型: GLM-4.7 FP8 (TP8)

**关键性能指标**:
- **TTFT (Time to First Token)**: 首个 token 的延迟，用户体验的关键
- **TPOT (Time Per Output Token)**: 每个 token 生成时间，影响整体速度

**实践建议**:
1. 对于 MoE 模型部署，优先考虑 Shared Experts Fusion
2. 对于多层模型，Async Transfer 可以显著降低 TTFT
3. 对于代理编码场景，Suffix Decoding 是无本万利的优化
4. 所有优化都应在生产负载下验证，实验室数据可能误导

**局限性**:
- 优化主要针对 SGLang 框架
- 需要特定硬件支持（H200 + FP8）
- Suffix Decoding 对非编码场景效果未知
- Async Transfer 增加了实现复杂度

**未来方向**:
- 将这些优化应用到其他 MoE 模型
- 探索其他场景的模式重复（如对话、翻译）
- 降低优化的工程复杂度
- 自动化优化配置选择
