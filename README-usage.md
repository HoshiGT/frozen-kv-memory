# 用法

`memory.py` 是实验结论的可用形态。`hybrid.py` 是测量装置，跑八个对照条件用来互相比较；
这里只剩赢的那一个。

## 它做什么

把长上下文拆成两种记忆，各管各擅长的一半：

```
    [ 64 个抽象槽 ]  常驻显存     要点、风格、文章要往哪走
    [ 全部原文 KV ]  offload      每一个具体细节，逐字
    [ 按需调入 512 ]  临时        此刻真正需要的那部分
```

分工不是设计出来的，是测出来的（`diagnose.py`）：

| token 类型 | 占比 | 只给抽象状态 |
|---|---|---|
| 前文没出现过 | 20.4% | **+231%**（比原文 KV 还好） |
| 前文出现过 | 79.6% | +59% |
| 前文只出现过一次 | 9.6% | **+46%** |

抽象状态天生不存细节——理解可以压缩，一串数字不能。所以细节交给原文 KV，
但只在用得上的时候调进来。

## 快速开始

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from memory import HybridMemory

model = AutoModelForCausalLM.from_pretrained("./qwen3-0.6b", dtype=torch.bfloat16).cuda().eval()
tok   = AutoTokenizer.from_pretrained("./qwen3-0.6b")
mem   = HybridMemory.load("ckpt/memtok_k32_deep_frozen.pt", model, tok)

handle = mem.compress(long_ids)             # 压一次，之后反复用
past   = mem.recall(handle, prefix_ids)     # 每次生成前取一次
out    = model(next_ids, past_key_values=cache_from(past, model.config),
               position_ids=...)            # offset 见 mem.offset(handle)
```

自检：

```bash
python memory.py --ctx 4096 --n 8
```

## 显存账

anchor 随上下文线性增长以维持覆盖率，所以显存是 **O(n/8)**，不是 O(1)：

| 上下文 | 全量 KV | 本方案显存 | 压缩比 | 质量 |
|---|---|---|---|---|
| 4096 | 448M | 63M | 7.1:1 | +87.8% |
| 8192 | 896M | 119M | 7.5:1 | +78.4% |
| 131072 | 14336M | 1799M | 8.0:1 | — |

固定 anchor 数则显存恒定，但质量随长度掉（8192 时 +72.2%）。自己选。

真正的收益是 O(n) 的大头从显存挪到了内存/磁盘——稀缺的是显存。

## 数字（Qwen3-0.6B，4096 token 真实对话，7:1）

| 方案 | recovery |
|---|---|
| 同预算全部给抽象槽 | +69.4% |
| 按稀有度选 anchor | +76.3% |
| 按 surprisal 选 anchor | +83.6% |
| **本实现（speculative 选）** | **+86.8%** |
| 完美选择（oracle，不可部署） | +103.2% |

`recovery = (无上下文 − 本方案) / (无上下文 − 全部原文KV)`，单位 nats。
超过 100% 表示比把全部原文 KV 都给它还好。

## 参数

只有两个值得调：

- **`n`**（每次调入多少，默认 512）——最重要的一个。
  128→+74.5%，256→+79.2%，512→+86.8%。显存换质量，线性代价。
- **`k`**（抽象槽数，默认 64）——**不要加大**。
  64→128 只涨 0.6 个点，抽象早就饱和了；同样的预算花在 `n` 上收益大十倍。

`draft`（默认 128）不用动：16/64/128/512 实测分别是 73.8/74.7/74.8/74.6%，
早就饱和，而它直接决定每次 recall 的延迟。

## 三个写死的东西

都不是偏好问题，改错了代价很大：

1. **位置跨段累积**，不重置。重置少 4 个点（我最初就写错了）。
2. **sink 永远锚住**开头 4 个位置。不带 sink 时部分 KV 会崩到 **−400%**。
3. **query 用模型想象的下文**，不用真实前缀。想象的反映走向，实测更好。

## 已知限制

- **上下文越长越吃力**：2048/4096/8192 分别是 +89.5%/+87.8%/+72.2%。
  但 oracle 在 8192 仍有 +94.7%——瓶颈是选不准，不是装不下。
- **多主题上下文更难**：单主题 +88.4%，多主题 +84.4%。
  同样 oracle 两者持平（104.2 vs 103.9），所以还是选择问题。
- **只在 Qwen3-0.6B 上验证过**，且 checkpoint 是在真实对话语料上训的。
  换领域需要重训记忆模块（主干不动，`train_memtok.py`，几十分钟）。
- `recall` 每次要跑一次短生成，有延迟。真要上生产应该缓存检索结果，
  隔若干 token 再刷新（实测每 16 token 刷新 +73.4%，每 128 token +71.8%，
  但那是 128 anchor 的设置，512 下没单独测过）。

## 复现

```bash
python gap.py                                   # 确认信息缺口存在
python train_memtok.py --no-lora --mem-depth --lr 5e-3 --k-random 16,128 --steps 1000
python budget_sweep.sh                          # 预算不是瓶颈
python diagnose.py --k 64 --hops 8              # 天花板由什么构成
python hybrid.py --k 64 --anchors 512 --probe 128 --hops 8   # 全部对照
```

细节和所有被推翻的假设都在 `README.md`。
