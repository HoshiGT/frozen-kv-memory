# 交叉核验文献报告（deep-research）

对照 `HANDOFF-literature.md`。排除 `github.com/HoshiGT/frozen-kv-memory`。

这是后台 deep-research 跑完后的核验稿：4 个研究子代理 + 2 个独立核验，**24/24 条 claims 留下**。Q4 必覆盖清单里有一批只进了 uncertainty 附录、没进正文表格。更全的一览表见 `LITERATURE-survey.md`。

**状态：Partial**（Q1–Q3 的“没人做过”是在已读 HTML 范围内；不是全 arXiv 穷尽证明。）

---

## 总判断

已读的 2023–2026 论文里，没有人把压缩上下文的 next-token recovery / PPL / accuracy 按「目标 token 是否在上下文出现过」或按出现次数分组。混合系统会对比「只读压缩记忆」vs「调回原文」，但没有固定检索命中、只换压缩状态，也没有把压缩 vs 精确远期 KV 检索拆开。冻结目标 LLM 是 plug-and-play / 防 decoder 泄漏的设计选择；没人主张「微调 backbone 会抬高无上下文基线，从而污染相对压缩指标」。

---

## Q1 Copyable vs novel tokens

近邻切的是别的轴，不是 copyability。

**Deng et al. gist 压缩** [2412.17483]：满注意力下段内 token PPL 比较均匀，压缩后段首更高（lost by the boundary）。压缩比 8 时，主题相关的词针 50.7 vs 意外针 35.8（−14.9）；相关数字针 69.2 vs 意外 59.0（−10.2）。意外信息更容易丢。

**What Kinds of Tokens Benefit from Distant Text?** [2406.11238]：按 n-gram 在原文 vs 新增上下文里的出现次数分层 token PPL。新出现比例与 token-PPL 下降相关（Yi 上 Spearman：K=2k 时 0.530，K=32k 时 0.356，p<0.005）。n-gram 在加长上下文里出现越多，越吃得到超长窗口。**KV / prompt 都没压缩**，这是未压缩长上下文 LM 分析。

**The Perplexity Paradox** [2602.15843]：提示压缩的 keep/drop，n=723 token。代码语法 PPL 是实词的 79×，数学数值 0.79×；留下的和删掉的之间有 71,000× PPL 差。测的是压缩机删哪些 prompt token，不是压缩 cache 之后的后缀 PPL。

**Optical Context Compression** [2512.03643]：分别报重建 PPL（还原原文）、续写 PPL（压缩 1000 前缀再写 1000 后缀）、事实回忆 log-likelihood。截断可以续写 PPL 好看但丢掉远处事实。指标停在任务级，没有 per-token copyability。

**ICAE** [2307.06945]：k=128 slot 时，≤300 token 上下文 BLEU/EM 接近 100%；长度 500 时 median BLEU >0.98、median EM ~0.6。续写 PPL：原文 9.01 vs 记忆槽 9.50（Δ +0.49）。预训练是 AE（还原上下文）+ LM（预测后续），不是把续写 token 标成 copyable vs never-appeared。

### 核验时扫过、仍不是 Q1 的

- FastGen [2310.01801]：按 special / punctuation / local / frequent **给缓存 token 分类以便驱逐**，报整体 attention-score recovery（35% 压缩时 >95%），不是生成 token 的分组 PPL。
- KVzip [2505.23416]：用 teacher-forced「重复前文」的注意力给 KV 打分再驱逐。
- 500xCompressor [2408.03094]：还原 Rouge-L/BLEU vs 抽取 QA F1/EM。
- ShotKV / KVFundaBench [2502.01941]：按**任务**拆压缩掉点（算术掉 17.4%–43.3%）。
- CompressKV [2508.02401]：copy-and-paste head vs semantic retrieval head，决定留哪些缓存 token。

---

## Q2 压缩 vs 检索

精确协议（固定检索到的原文 KV，只换压缩状态 / 随机状态 / oracle）没找到。下面是最近的分解。

### JustMem [2609.19877]（agent 记忆，不是 KV）

压缩原子记忆 + 指回原始会话。LongMemEval-S：

| Access | Overall acc | Assistant-output acc |
|---|---|---|
| Compact Reading（只读压缩记忆） | 74.40% | 32.14% |
| Fixed Top-1 Recovery（每问调回最高会话） | 80.60% | 92.86% |
| Adaptive Replay（planner 认为需要保真才调回） | 83.40% | 94.64% |

路由上界（500 题，其它 planner 输出固定，oracle 用题型标签）：

| Access | Acc | Recall@10 |
|---|---|---|
| Fixed Lookup | 71.20% | 91.80% |
| Fixed Compose | 76.60% | 96.60% |
| Fixed Replay | 63.40% | 87.40% |
| Adaptive | 83.40% | 96.00% |
| Taxonomy oracle | 84.20% | 96.20% |

### Infini Memory [2606.10677]

主题文档摘要 + BM25 调回原文分区。维护 vs 检索轮流固定：

| Setting | Accuracy |
|---|---|
| Summary-only retrieval | 41.7% |
| Hybrid Summary+BM25 | 76.0% |
| Agentic | 79.3% |
| Hybrid reader，关掉 split/merge | 69.3% |

摘要缺细粒度事实，分区检索能补上大部分缺口；两边单独都不够。

### SpeCache [2503.16163]

VRAM 低比特 KV 当索引，CPU 上 16-bit 原文 KV 按 top-k 预取。k=0 就是 KIVI。

| Setting | Score |
|---|---|
| 1-bit KIVI (g=64), LongBench avg | 24.7 |
| SpeCache，同样 0.11× GPU KV | 41.9 |
| 1-bit Qasper without SpeCache | 23.9 |
| 1-bit Qasper with SpeCache | 43.2 |

### StreamKV [2511.07278]

语义段编码；压 frame-level KV，summary vector 不压，再按问题调块。有无 summary 不是「固定检索、换压缩状态」。

| Compression | w/o summary | w/ summary |
|---|---|---|
| 0% | 60.52 | 61.20 |
| 90% | 53.85 | 56.72 |

### Strong Drafts / MASW [2608.30252]

压缩远期槽 + **精确近期原文 KV**，远期原文 KV **丢掉不再调回**。target verifier 仍用完整原文 KV。消融是窗口大小 / adaptor / pretrain vs SFT，以及 full-KV SD vs SWA vs MASW。

### 核验时扫过、仍不是 Q2 的

- InfLLM [2402.04617] Table 4：lookup 开/关（R.KV 96.8 vs w/o Lookup 0.4）。代表 token 是索引，生成时 attend 的是调回来的原文 KV 块。
- Quest [2406.10774]：页 min/max 打分 vs query-aware 稀疏的 oracle；全文 KV 仍在，元数据不是压缩生成记忆。
- REFORM [2506.01215]：换递归压缩机（H2O / StreamingLLM / TOVA），检索 embedding 来自压缩 pass，检索集合不固定。
- InfMem [2602.02704]：检索塑造联合压缩；消融是 chunk size / early stop / thinking，不是冻检索换状态。
- HybridKV [2604.05887] Table 4：全静态 pruning vs 全动态 retrieval vs 混合头（WebQA 73.00 / 75.50 / 76.00），是头策略不是两套可加的记忆。

---

## Q3 冻结 backbone 与相对指标

没人写出「微调 backbone 抬高无上下文基线 → 相对压缩分数被污染」。近邻：

**CCM** [2312.03414]（最近）：naive 全模型压缩微调会让 LM 不看记忆也能答当前输入。LLaMA-7B/MetaICL 上 **无上下文训练 loss 从 2.69 降到 1.84**，测试 loss 仍 2.59（overfitting on inputs）。于是冻住 LM，只训 ⟨COMP⟩ token 上的 conditional LoRA。这是**训练动力学**诊断，不是评测公式被污染。

**Simple Context Compression** [2510.20797]：用  
`(compressed − no-context) / (full-context − no-context)`  
避免「不用上下文也能答的题」抬高压缩机分数。teacher 在训练混合上 LoRA 过，encoder 和 decoder 各有 LoRA。公式几乎就是 recovery，但他们微调两端，校正的是**题集属性**，不是「必须冻 backbone」。

**ICAE / ARC-Encoder** [2307.06945] [2510.20535]：冻 decoder 是为了 plug-and-play、保住 5-shot QA；对照 Gisting / AutoCompressor 必须微调推理 LLM。ARC-Encoder 把「不改 decoder」当成评测约束，理由是微调会伤压缩任务之外的能力。

**500xCompressor** [2408.03094]：encoder/decoder 都用原 LLM，decoder 不加参数，「信息没存进 decoder」。他们的 leakage 检查是 Pile 与 Llama 预训练重叠，用 cutoff 之后的 Arxiv 测，不是抬高的 no-context 基线。

**COCOM** [2407.09252]：**反方向**——冻 decoder 会妨碍用压缩上下文，所以微调 decoder，并用微调过的 closed-book LLM 当下界。批评冻 decoder 的压缩机调参不算真 zero-shot，要跟 soft prompt 这类同类调参比，不是相对指标污染。

**Gist of Gisting** [2504.08934]：单独训 No-context 基线当下界，因为有些问题常识就能答。是边界设计，不是诊断。

**ComprExIT** [2602.03784]：未训练 Zero-shot [w/o context] 当信息全丢的下界，压缩跑在冻结 LLM 上，同样没有相对指标混淆论证。

---

## Q4 正文里核过的卡片

| Paper | arXiv | Method | Ratio | Metric | Backbone | Datasets |
|---|---|---|---|---|---|---|
| H2O | 2306.14048 | 近期 + Heavy Hitter eviction | 20% heavy hitters | 吞吐最高 29× vs DeepSpeed/HF；核对准确率 | 冻结 | OPT-6.7B/30B 吞吐；OPT, LLaMA, GPT-NeoX |
| SnapKV | 2404.14469 | prompt 末观察窗选重要 KV | 8.2× 内存，16K 输入 3.6× 生成 | 准确率相当；NIAH 几乎不掉 | 训练免费 | 16 个长序列集；NIAH 到 380K |
| PyramidKV | 2406.02069 | 浅层多预算、深层少预算 | 12% 对齐满 KV；也测 0.7%、128 条 | LongBench 12% 对齐；0.7% 时 TREC +20.5；NIAH 100.0 | — | LongBench, TREC, NIAH, Llama-3-70B |
| ICAE | 2307.06945 | LoRA encoder → memory slots；原 LLM 当 decoder | Llama 上 4× | 重建 BLEU/EM vs 续写 PPL | decoder 冻结 | Pile；PwC |
| LCIRC | 2502.06139 | 递归 Perceiver + query-dependent 压缩 | — | — | 部分层训、部分冻 | — |
| SpeCache | 2503.16163 | 低比特 GPU KV 引导预取 16-bit CPU KV | 0.11× GPU KV | LongBench, Qasper | — | LongBench, Qasper |
| SP-KV | 2605.14037 | 联合 NTP 训 utility predictor；局部窗默认 128 必留，更老的 KV 过阈值才写入 | 通常 3–10×（不固定） | next-token prediction | LLM 和 predictor 联合继续预训练 | — |
| Strong Drafts / MASW | 2608.30252 | sink ∪ 最近 W 条原文 KV ∪ 学出来的远期槽；远期原文丢掉 | — | speculative decoding vs full-KV SD / SWA | adaptor；pretrain/SFT 消融 | — |
| 500xCompressor | 2408.03094 | 提示压进原 LLM 的 encoder/decoder | 6–480× | 能力保留 62–73%；leakage 是预训练重叠 | encoder+decoder 都冻 | cutoff 后 Arxiv / ArxivQA |
| COCOM | 2407.09252 | RAG context embedding | 解码最高 5.69× | 微调 closed-book 当下界 | decoder 微调 | RAG |
| JustMem | 2609.19877 | 压缩原子记忆 + 溯源会话 | — | task acc；Recall@10 | — | LongMemEval-S |
| Infini Memory | 2606.10677 | 主题摘要 + BM25 原文分区 | — | accuracy | 维护 vs 检索轮流固定 | LongMemEval (S*) |
| StreamKV | 2511.07278 | 段 KV 压缩 + 不压的 summary vector；按问题调块 | 0–90% | QA | — | 流式视频 QA |

下面这些在核验里读过摘要/HTML，**没打进上面这张核过的表**，细节以 `LITERATURE-survey.md` 为准：StreamingLLM 2309.17453、Quest 2406.10774、InfLLM 2402.04617、KIVI 2402.02750、Gist 2304.08467、AutoCompressor 2305.14788、Compressive Transformer 1911.05507、RMT 2207.06881、Scissorhands 2305.17118、FastGen 2310.01801、GEAR 2403.05527、MiniCache 2405.14366、DuoAttention 2410.10819、RetrievalAttention 2409.10516、Activation Beacon 2401.03462、CAMELoT 2402.13449、DMC 2403.09636、Landmark 2305.16300、Infini-attention 2404.07143、LLMLingua 2310.05736、LongLLMLingua 2310.06839、xRAG 2405.13792、Memorizing Transformers 2203.08913、LongMem 2306.07174、Benchmarking KV-Cache Optimizations 2607.05399。

---

## 还剩的贡献

**(a) Token 类型拆解天花板仍空。** 最近是位置/针主题、未压缩 n-gram 频率 PPL、prompt keep/drop 分类 PPL、整段还原 vs 整段续写。

**(b) 检索 vs 压缩 vs oracle 原文 KV 的加法分解仍空。** 最近是 JustMem 压缩 vs 调回 vs 题型 oracle、Infini Memory 摘要维护 vs BM25、SpeCache 在低比特副本上加原文 KV、StreamKV 的 summary 开关、MASW 压缩远期槽且丢掉（不调回）远期原文 KV。

**(c) 「微调抬高 no-context 基线 → 相对 recovery 被污染」仍空。** 最近是 CCM 的训练 overfitting（无上下文 train loss 2.69→1.84）、以及 [2510.20797] 几乎同构的 `(comp − noctx) / (full − noctx)`（但他们 LoRA 了 encoder 和 decoder）。

对本项目特别要紧的三篇近邻：**CCM 2312.03414**、**Simple Context Compression 2510.20797**、**What Kinds of Tokens 2406.11238**。写 related work 时这三篇要正面 cite。

---

## Sources

- [S1] 本轮已读的 2023–2026 主文（FastGen 2310.01801, ICAE 2307.06945, gist 2412.17483, KVzip 2505.23416, Optical Context Compression 2512.03643, Perplexity Paradox 2602.15843, What Kinds of Tokens 2406.11238）
- [S2] [2412.17483](https://arxiv.org/abs/2412.17483) A Silver Bullet or a Compromise for Full Attention?
- [S3] [2406.11238](https://arxiv.org/abs/2406.11238) What Kinds of Tokens Benefit from Distant Text?
- [S4] [2602.15843](https://arxiv.org/abs/2602.15843) The Perplexity Paradox
- [S5] [2512.03643](https://arxiv.org/abs/2512.03643) Optical Context Compression Is Just (Bad) Autoencoding
- [S6] [2307.06945](https://arxiv.org/abs/2307.06945) ICAE
- [S7] [S8] [2609.19877](https://arxiv.org/abs/2609.19877) JustMem
- [S9] [2606.10677](https://arxiv.org/abs/2606.10677) Infini Memory
- [S10] [2503.16163](https://arxiv.org/abs/2503.16163) SpeCache
- [S11] [2511.07278](https://arxiv.org/abs/2511.07278) StreamKV
- [S12] [2608.30252](https://arxiv.org/abs/2608.30252) Strong Drafts Need Compact Memories
- [S13] [2312.03414](https://arxiv.org/abs/2312.03414) Compressed Context Memory
- [S14] [2408.03094](https://arxiv.org/abs/2408.03094) 500xCompressor
- [S15] ICAE 2307.06945（与 ARC-Encoder 对照）
- [S16] [2510.20797](https://arxiv.org/abs/2510.20797) Simple Context Compression
- [S17] [2407.09252](https://arxiv.org/abs/2407.09252) COCOM
- [S18] [2510.20535](https://arxiv.org/abs/2510.20535) ARC-Encoder
- [S19] [2306.14048](https://arxiv.org/abs/2306.14048) H2O
- [S20] [2404.14469](https://arxiv.org/abs/2404.14469) SnapKV
- [S21] [2406.02069](https://arxiv.org/abs/2406.02069) PyramidKV
- [S22] ICAE 2307.06945v4
- [S23] [2502.06139](https://arxiv.org/abs/2502.06139) LCIRC
- [S24] [2605.14037](https://arxiv.org/abs/2605.14037) Self-Pruned KV Attention

## 覆盖缺口（核验原文）

- Q1 的「没有」不是穷尽证明；附录-only 或代码-only 的拆解可能漏掉。
- Q2 没找到「固定检索命中、只换压缩状态 / 随机状态」这一精确协议。
- Q3 没找到相对评测公式被污染的原话；CCM 最近，但是训练 overfitting。
- Q4 正文表不完整，必覆盖名单的其余条目在 uncertainty 里核对过 ID，展开表见 `LITERATURE-survey.md`。
- 本仓库 `github.com/HoshiGT/frozen-kv-memory` 按交接要求排除，不当证据。
