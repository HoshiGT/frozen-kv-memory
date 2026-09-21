# 文献调研结果（2026-09-21）

对照 `HANDOFF-literature.md`。排除 `github.com/HoshiGT/frozen-kv-memory`。每条都有 arXiv ID。不确定标「存疑」。

交叉核验稿（deep-research，24/24 claims 留下）在 `LITERATURE-deep-research.md`，多出 CCM / Simple Context Compression / What Kinds of Tokens / JustMem 等近邻。Q4 完整表以本文件为准。

**总判断**：Q1、Q2 的精确实验还没人做过。架构近邻很多，尤其是「压缩远期 + 精确近期」和「冻结 decoder / 只训槽位」。Q3 的指标污染诊断也没人写过。项目还剩的贡献主要在这三块测量，不在「又一个 memory slot 压缩器」。

HANDOFF 里 AutoCompressor 写成 `2308.15022`，实际是 **2305.14788**。

---

## Q1 有没有按 token 类型拆过压缩损失？

**结论：没有找到同等拆解。**

没有人把待预测 token 分成「上下文里出现过 / 可复制」vs「从未出现」，再分别报 recovery / CE。也没有按上下文出现频次分组报恢复率。

### 最近的前鉴（机制相关，不是同一实验）

| 论文 | arXiv | 做了什么 | 差在哪 |
|---|---|---|---|
| Deng et al. *A Silver Bullet or a Compromise for Full Attention?* | [2412.17483](https://arxiv.org/abs/2412.17483) | gist 压缩在 RAG/摘要上近无损，**synthetic recall 崩**。三种失败：lost by the boundary（段首 PPL 高）、**lost if surprise**（跟主题无关的针更容易丢）、**lost along the way**（32 位 UUID 复制中途丢）。gist 重建准确率 CR=4 时 77%，CR=32 时 10%。 | 任务级 + 探针，不是 next-token CE 按「该 token 是否在上下文出现过」分组 |
| Ge et al. ICAE | [2307.06945](https://arxiv.org/abs/2307.06945) | 128 slot 还原 512 token：正常文本 BLEU 99.3 / loss 0.01；**patterned random 3.5 / 1.63**；**完全随机 0.2 / 4.55**。语义可压，无结构数字压不动。 | 还原原文，不是预测下一段；按内容类型不是按 copyability |
| Hsieh et al. RULER | [2404.06654](https://arxiv.org/abs/2404.06654) | 长上下文按 retrieval / multi-hop / aggregation 拆任务 | 任务类型，不是 token 类型 |
| Ge et al. FastGen | [2310.01801](https://arxiv.org/abs/2310.01801) | 按 **head** 分 local / special-token / 全上下文 | head 类型，不是被预测 token 类型 |
| Zhang et al. H2O | [2306.14048](https://arxiv.org/abs/2306.14048) | Heavy hitter 跟 token 共现频率相关 | 解释 eviction，没有按 copyability 报 recovery |
| Li et al. kv-distill | [2503.10337](https://arxiv.org/abs/2503.10337) | extractive vs abstractive 任务对比 | 任务级 |
| Petrov et al. *Getting to the Gist of Gisting* | [2504.08934](https://arxiv.org/abs/2504.08934) | gist 在 1× 压缩时都不能无损拷贝上下文 | 架构失败，不是 token 分组 |

**还剩的贡献**：把「理解可压、一串数字不能」从任务级观察推进到 **token 级 CE/recovery 分层**，并量化「只出现一次的 token 扛走了大部分信息缺口」。Deng / ICAE 已经摸到同一堵墙，没人用我们这套指标把它量出来。

---

## Q2 有没有把压缩 vs 检索的贡献分开量化过？

**结论：没有找到同等消融。**

没有人在固定 512 条检索 KV 的前提下只换背后的压缩状态，并报：无状态 / 训练状态 / 随机状态 / oracle 检索 这四档。

### 架构近邻（机制撞车，测量没撞）

| 论文 | arXiv | 关系 |
|---|---|---|
| Yuan et al. *Strong Drafts Need Compact Memories* | [2608.30252](https://arxiv.org/abs/2608.30252) | **架构最像**：压缩远期 + 精确近期窗口；adaptor 可训、**draft backbone 冻结**。用途是 speculative decoding 的 draft 侧，target 仍用完整 KV。有 sliding window vs MASW vs full-KV，没有「固定检索、换压缩状态」 |
| Jie et al. SpeCache | [2503.16163](https://arxiv.org/abs/2503.16163) | CPU 上完整 KV + VRAM 里低比特副本做重要性估计 + 推测预取。检索侧高度重合，没有学出来的抽象状态 |
| Xiao et al. InfLLM | [2402.04617](https://arxiv.org/abs/2402.04617) | 训练免费：远期 KV 分块、代表 token 检索、拼上 sink + 局部窗口。有检索块数消融，没有压缩状态 vs 检索的加法分解 |
| Kim et al. CCM | [2312.03414](https://arxiv.org/abs/2312.03414) | 在线把累积 KV 压进固定记忆；**主干权重冻结**，只加 conditional LoRA。纯压缩，没有检索臂 |
| He et al. CAMELoT | [2402.13449](https://arxiv.org/abs/2402.13449) | 冻结 LLM + 非参数联想记忆检索。训练免费。PPL 指标。没有压缩槽位 |
| Tang et al. Quest | [2406.10774](https://arxiv.org/abs/2406.10774) | query-aware 页检索。纯检索 |
| Liu et al. RetrievalAttention | [2409.10516](https://arxiv.org/abs/2409.10516) | CPU 上 ANNS 检索 1–3% KV。纯检索 |
| Mohtashami & Jaggi Landmark | [2305.16300](https://arxiv.org/abs/2305.16300) | landmark token 选块再 attend。要微调 LLaMA |
| HybridKV | [2604.05887](https://arxiv.org/abs/2604.05887) | 静态头 pruning + 动态头 chunk retrieval。多模态。组件消融不是压缩 vs 检索 |
| RocketKV | [2502.14051](https://arxiv.org/abs/2502.14051) | SnapKV 粗驱逐 + 动态稀疏；**Exact-TopK oracle** 给稀疏注意力上界。oracle 检索有，压缩状态没有 |
| CompAct | [2407.08223](https://arxiv.org/abs/2407.08223) 存疑需核 PDF | RAG 文档压缩；有 oracle 文档基线。文本压缩不是 KV 状态 |

CompAct 的 arXiv 以摘要检索为准，ID 标存疑。

**还剩的贡献**：加法分解本身（检索 46 点 / 压缩 38 点 / oracle 再 19 点，随机状态 −40%）看起来是新的。架构故事「压缩 + 检索」已经拥挤。

---

## Q3 有没有人指出过「微调 backbone 污染相对压缩指标」？

**结论：没有找到这个混淆的诊断。** 冻结 backbone 作为设计选择很常见；没人说它是相对指标的必要条件。

| 论文 | arXiv | 态度 |
|---|---|---|
| ICAE | 2307.06945 | **decoder = 原 LLM 冻结**；encoder 用 LoRA + memory token embedding。为了槽位兼容，不是为了保护相对指标 |
| CCM | 2312.03414 | 不微调全部权重，只加 conditional LoRA |
| 500xCompressor | 2408.03094 | 压缩 token 给原 LLM 用，不必微调原模型 |
| LCIRC | 2502.06139 | 「不必重训整个模型」 |
| Strong Drafts | 2608.30252 | 只训 adaptor，draft backbone 冻结（省训练成本） |
| CAMELoT | 2402.13449 | 完全训练免费 |
| Deng et al. | 2412.17483 | 用 base 不用 SFT，「避免 SFT 混杂」。最近，但仍是任务选择，不是「lower 基线被抬高、分母缩小」 |
| Petrov et al. Gist of Gisting | 2504.08934 | 有 Full / No-context 基线，但 **gist 训练时微调全部参数**。正是会被污染的设置，作者没诊断 |
| AutoCompressor | 2305.14788 | 微调整个 LM |
| Gist Tokens | 2304.08467 | 跟 instruction tuning 一起训 |
| SP-KV | 2605.14037 | LLM 和 utility predictor **联合训练**。作者认为暴露稀疏性能减少 train–test mismatch，方向相反 |
| Activation Beacon | 2401.03462 | plug-in 模块；主干大体冻结，beacon 可训 |

**还剩的贡献**：把「冻结」从工程便利改写成 **相对 recovery 指标的鉴定条件**，并用 LoRA 把 lower 从 3.951 拉到 3.370、recovery 从 +66% 掉到 +8% 这一组数。限定：只打击相对指标；报绝对任务准确率的工作不受影响。

---

## Q4 2023–2026 主要工作一览

格式：arXiv | 方法 | 压缩比 | 指标 | backbone | 数据

### 必覆盖

| arXiv | 方法 | 压缩比 | 指标 | backbone | 数据 |
|---|---|---|---|---|---|
| 2306.14048 H2O | 累积注意力 heavy hitter + 近期窗口 eviction | ~5×（留 20%） | PPL + 下游 acc + 吞吐 | 冻结 | OpenWebText, XSum 等 |
| 2404.14469 SnapKV | 用 prompt 末观察窗估计重要 KV，压缩 prefill | ~8× 内存 | LongBench, NIAH | 冻结 | LongBench 16 集, 380K NIAH |
| 2406.02069 PyramidKV | 按层金字塔分配 KV 预算 | 层间不等，总预算固定 | LongBench, NIAH | 冻结 | Llama-3 等 |
| 2309.17453 StreamingLLM | attention sink（开头几 token）+ 滑动窗口 | 缓存恒定（如 4+窗口） | PPL（可到 4M token） | 冻结；可加 sink token 预训练 | Llama-2/MPT/Falcon/Pythia |
| 2406.10774 Quest | query-aware 页关键性，只 load Top-K 页 | 页级稀疏，NIAH 可 ~1% | LongBench, passkey, 延迟 | 冻结 | LongChat, Yarn-Llama, LongBench |
| 2402.04617 InfLLM | 远期分块 + 代表 token 检索 + 局部/sink | GPU 侧恒定（局部+k 块） | 长依赖, 1M token | 冻结、训练免费 | 长序列外推 |
| 2402.02750 KIVI | K 按 channel、V 按 token 的 2-bit 量化 | ~8× bit；峰值内存 2.6× | LongBench, NIAH, 吞吐 | 冻结 | Llama/Falcon/Mistral |
| 2304.08467 Gist Tokens | 改 attention mask，把 prompt 压进 gist token | 最高 26× prompt | 指令跟随 win rate | **微调** LM | Alpaca+ 等 |
| 2307.06945 ICAE | LoRA encoder → memory slots；**decoder 冻结** | 4×（512→128） | AE BLEU/EM/CE，PwC GPT-4 评判 | decoder 冻，encoder LoRA | Pile, PwC |
| 2305.14788 AutoCompressor | 段级 summary vector 当 soft prompt，递归 | 每段 50 vector，训到 30k | **PPL** + ICL acc | **微调** OPT/Llama-2 | Pile, RedPajama |
| 1911.05507 Compressive Transformer | Transformer-XL + 学出来的压缩记忆 | 记忆再压一层 | **PPL** 17.1 WT103, 0.97 bpc Enwik8 | 从头训 | WT103, Enwik8 |
| 2207.06881 RMT | 段间 memory token 递归 | 内存 token 数很小 | **PPL** WT103 + 算法任务 | 从头训 / 加在 Tr-XL 上 | WT103 |

### 相邻、值得放进 related work

| arXiv | 方法 | 压缩比 | 指标 | backbone | 数据 |
|---|---|---|---|---|---|
| 2305.17118 Scissorhands | persistence of importance eviction | 最高 5×（+4bit → 20×） | PPL, 下游 | 冻结 | 生成任务 |
| 2310.01801 FastGen | 按 head 画像自适应 eviction | ~35–50% cache | AlpacaEval, GSM8K, HumanEval | 冻结 | 多任务 |
| 2403.05527 GEAR | 量化 + 低秩残差 + 稀疏 outlier | 4-bit 近无损 | 生成质量, 吞吐 2.38× | 冻结 | LLM 生成 |
| 2405.14366 MiniCache | 跨层 KV 合并 | 最高 5.02× | ShareGPT, 多模型 | 冻结 | LLaMA-2/3, Mistral 等 |
| 2410.10819 DuoAttention | retrieval head 全 KV，streaming head 恒定窗口 | MHA 2.55× 内存 | 长上下文, 3.3M on A100 | 冻结（识别头用合成数据） | Llama-3 |
| 2409.10516 RetrievalAttention | CPU ANNS 检索 KV | 只碰 1–3% | 128K on 4090 | 冻结 | 8B, 128K |
| 2401.03462 Activation Beacon | 层内激活压成 beacon | 8× KV，4K→400K | NIAH, 文档理解, few-shot | plug-in，主干大体冻 | 长上下文任务 |
| 2402.13449 CAMELoT | 冻结 LLM + 联想记忆巩固/检索 | 窗口可小到 128 | **PPL**（Arxiv −29.7%） | 冻结、训练免费 | PG-19, Arxiv, WT103 |
| 2403.09636 DMC | 在线决定 append vs merge KV | 4× 保性能，8× 小降 | MMLU, CS-QA, HumanEval, 吞吐 | **继续预训练**（无新参数） | Llama-2 7/13/70B |
| 2305.16300 Landmark | landmark token 选块 | 扩到 32k+ | 长上下文 | **微调** LLaMA-7B | 长上下文 |
| 2404.07143 Infini-attention | 局部注意力 + 压缩线性记忆 | 有界记忆，无限输入 | PPL, 1M passkey, 500K 摘要 | 训 Infini-attn | 1B/8B |
| 2310.05736 LLMLingua | 硬提示 token 删除 | 最高 20× | GSM8K, BBH, ShareGPT | 冻目标 LLM | 上述 |
| 2310.06839 LongLLMLingua | 长提示的 query-aware 硬压缩 | 高倍 | 长 ICL/RAG | 冻目标 | 长提示 |
| 2408.03094 500xCompressor | 文本→极少 special token 的 KV | 6–480× | QA 能力保留 62–73% | 原 LLM 不必微调 | Arxiv / ArxivQA |
| 2407.09252 COCOM | RAG 多文档压成 context embedding | ξ=4/16/128 | RAG QA, 解码加速 5.69× | 训压缩器 | RAG |
| 2405.13792 xRAG | 检索向量当一个 token | 极端（1 token） | RAG | 模态融合训练 | RAG |
| 2203.08913 Memorizing Transformers | kNN 检索过去的 KV | 外存 + 局部 | 长 LM | 训 kNN 记忆 | 长文档 |
| 2312.03414 CCM | 在线压累积 KV；conditional LoRA | 5× 记忆仍达满上下文 | 对话/个性化/多任务 | 主干冻 | 在线交互 |
| 2502.06139 LCIRC | 递归压缩超窗上下文再注入 | 递归到窗内 | 长文 + query-dependent | 不必重训全模型 | NAACL 2025 任务 |
| 2503.16163 SpeCache | CPU 满 KV + 低比特索引 + 推测预取 | VRAM 10× | LongBench, NIAH | 冻结 | Mistral/Llama-3 |
| 2605.14037 SP-KV | 学 utility predictor 决定写不写全局 KV | 动态 3–10× | NLL, RULER, 下游 | **联合微调** | LongPPL, RULER |
| 2607.05399 Benchmarking KV-Cache | KIVI/TurboQuant/SnapKV/CaM 统一评 | 方法相关 | 任务质量 + 吞吐/TTFT | 冻结 | LongBench 子集 |
| 2608.30252 Strong Drafts | draft 侧压缩记忆 + 精确近期 | draft 内存 −70% | SD 加速 2.08×/3.33× | draft 冻，adaptor 训 | 长前缀摘要，32K |
| 2412.17483 Silver Bullet | 统一 gist 架构，失败模式 | 4–32× | PPL + RAG/QA/recall | 继续预训练 base | SlimPajama, RULER, ∞Bench |
| 2504.08934 Gist of Gisting | gist 长上下文失败；GistPool | 可变 ξ | 答案 PPL + Gemini judge | **全参微调** | SQuAD 到 FairytaleQA |
| 2503.10337 kv-distill | 问题无关 KV 蒸馏 + 保留重要 token | 最高 99% 长度 | extractive/QA/摘要 | PEFT adaptor | 长短上下文 |

### 已在 HANDOFF「确认撞车」里的，不再展开

SpeCache 2503.16163、Strong Drafts 2608.30252、SP-KV 2605.14037、Benchmarking 2607.05399、Compressive / RMT / LCIRC 2502.06139。

---

## 和本项目特别像、但贡献切面不同

1. **ICAE（2307.06945）**：memory slots + 冻结 decoder + 只训很少参数。目标是还原/答题，不是冻结主干下的相对 recovery；也没有 copy vs novel 分层。
2. **CCM（2312.03414）**：在线压 KV、主干冻、只加 LoRA。没有检索臂，没有 token 类型拆解。
3. **Strong Drafts（2608.30252）**：压缩远期 + 精确近期 + 冻 backbone。用在 draft 侧加速，target 仍是满 KV；指标是 SD 接受长度不是 nats recovery。
4. **Activation Beacon（2401.03462）**：细粒度 gist/beacon 插在原文里，继续自回归。Deng 2412.17483 把它归进 Fine-KV。
5. **AutoCompressor / RMT / LCIRC**：递归压缩，HANDOFF 结论 5（滚动稀释近期）的背景。

---

## 还剩什么（对照 HANDOFF 第四节）

| 问题 | 文献状态 | 建议 |
|---|---|---|
| Q1 token 类型拆解 | **空**。最近是 Deng 的失败模式和 ICAE 的随机文本还原 | 这是最干净的剩余贡献。写的时候主动 cite 这两篇，说我们把同一现象做成了 next-token recovery 分层 |
| Q2 压缩 vs 检索加法分解 | **空**。架构拥挤 | 贡献在测量不在系统。cite Strong Drafts / InfLLM / SpeCache / RocketKV oracle |
| Q3 冻结保护相对指标 | **空**。冻结是常见设计，诊断不是 | 方法论文段。cite ICAE/CCM（他们冻了但没说为什么必须冻）和 Gist of Gisting（他们微调了 Full/No-context） |
| 系统本身 | 不新 | 不要把「冻结槽位 + offload 检索」当主贡献 |

没找到编造的引用。CompAct 的 arXiv ID 标了存疑。
