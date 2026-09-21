# TODO

昨天那份已全部做完（预算杠杆、天花板成因、混合架构、检索策略）。这是 2026-09-21 下午起的新队列。

## 当前最好的配置

```bash
python hybrid.py --k 64 --anchors 512 --probe 128 --hops 8
# 7:1 压缩，speculative 检索 +86.8%，纯抽象只有 +69.4%
```

架构：64 抽象槽常驻 + 原文 KV offload + 按 speculative query 调入 512 条。

## 已完成（2026-09-21 下午）

1. ~~训练 anchor-aware 的记忆模块~~ → **负面**。hybrid 没涨，solo 掉 33 点。
   分工是自发的，梯度本来就不会往 anchor 已覆盖的地方使劲。原 ckpt 仍是最优。
2. ~~1.7B 跨规模验证~~ → speculative +86.2%（0.6B 是 +86.8%），保持。
3. ~~打包~~ → `memory.py` + `README-usage.md`，自检 7 MB 常驻 / 448 MB offload / +85.2%。
4. ~~更新报告页~~ → v3 已重写，主线改为"压缩 + 检索"。

## 现在在查

**长上下文下选择策略失效**：8192 token 时 speculative 掉到 +72.2%，
而 oracle 仍有 +94.7%——不是装不下，是选不准。

判别法：8192 配 512 anchor 只有 6.25% 覆盖率，4096 配 512 是 12.5%。
按比例给到 1024 条，覆盖率追平：

- **恢复** → 是覆盖率问题，加预算能解决
- **不恢复** → 选择本身在长上下文下退化，需要新方法

## Airi 落地：等 10 月中加内存条

目标模型是 **Qwen3.6-35B-A3B**，推理端用 **FreeToken**（专家在 CPU、激活参数在 GPU）。
现在 15G 内存不够（FreeToken 会 pin 全部专家 19G），等 10 月中加条子。

**关键判断：FreeToken 是 torch 生态的，路线大概率通。**

```
依赖: torch, transformers, triton, safetensors, gguf ...
freetoken/kvcache/: mha_pool.py  dsa_pool.py  hybrid_swa_pool.py  cache_status.py
```

KV 是 torch tensor，而且有独立的 cache pool 抽象——和 llama.cpp 完全不同
（llama.cpp 不暴露 KV 注入，这条路是死的）。

**训练和推理可以分离**：记忆模块只是个 3.67M 的权重文件。

1. 租 96G 卡，PyTorch bf16 训 35B 的记忆模块（~75GB 显存，2–3 小时，
   主要时间在下 70GB 权重；GGUF 训不了）
2. 本地 FreeToken 推理时加载它

**接口调查已完成**（2026-09-21，见 `docs/freetoken-integration.md`）：
`store_kv` / `k_cache` / `v_cache` 都现成，`qwen3_5_moe` 就是目标架构；
唯一缺的是 embedding 入口，而 `gemma4` 的 `mm_embeds` 是现成模板，改十几行。

**加内存后要先确认的**（在花钱之前）：
- `freetoken/kvcache/*_pool.py` 能不能写入外部构造的 KV
- 能不能用 `inputs_embeds` 喂 memory token（compress 需要）
- 能不能拿到某些位置的 KV 切片（anchor 需要）

这三样有一样不行，就得改 FreeToken 或者换推理端。**先验接口，再训模型。**

## 文献工作（论文的前置条件，明天专门做）

2026-09-21 傍晚粗查了三个自以为是贡献的方向，**两个半站不住**：

| 方向 | 真实状况 |
|---|---|
| 任务 gap 决定压缩效果 | **已有系统研究**，有专门 benchmark（arXiv 2607.05399：GovReport 14.2× vs NarrativeQA 几乎压不动） |
| 滚动记忆有效窗口有限 | 现象**已知**（Compressive Transformer / RMT / LCIRC 一脉） |
| 冻结主干才测得准 | 搜不到对应工作——但**搜索第一条是我们自己的仓库**，没有证明力 |

还撞上 Self-Pruned KV（arXiv 2605.14037）：学一个 utility predictor 决定保留哪些 KV，
backbone 联合微调，3–10×。与我们的 anchor 选择方向重合。

**尚未查过、可能才是真贡献的两条**：

- **天花板的成分分解**：按"答案是否可从上下文复制"、按出现频次拆开 recovery
- **贡献分解**：检索 46.1 / 压缩 38.8 / oracle 上界 19.0，用完美选择定天花板

这两条是"把方法拆开看里面装了什么"，不是提新方法，少见得多。

**方法论论点需要限定**：LoRA 污染只对**相对指标**成立
（`recovery = (lower−comp)/(lower−upper)`，微调抬高 lower 缩小分母）。
报绝对指标（任务准确率）的工作不受此影响。不加这个限定，第一个审稿人就会指出来。

**要做的**：把 KV 压缩这几年的主要工作过一遍建表，
而不是搜三下。在写任何论文内容之前做。

## 还没做

- anchor 预算扩展律：512 → 1024 → 2048，什么时候追平 upper
- Matryoshka 嵌套（解决 k=16 角色冲突，优先级低）
- 换领域验证：现在的 ckpt 只在真实对话语料上训过

## 环境备忘

- `source ~/ctfenv/bin/activate`
- 仓库 https://github.com/HoshiGT/frozen-kv-memory **private**，确认后可公开
- `recur.py` 已废弃，位置约定有 bug，用 `train_recur.py`
- ★ 位置要累积（`accum=True`），不要重置——值 4 个点
- ★ 任何只保留部分 KV 的方案都必须带 attention sink，否则 −400%
