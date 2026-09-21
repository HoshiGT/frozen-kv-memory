# TODO

昨天那份已全部做完（预算杠杆、天花板成因、混合架构、检索策略）。这是 2026-09-21 下午起的新队列。

## 当前最好的配置

```bash
python hybrid.py --k 64 --anchors 512 --probe 128 --hops 8
# 7:1 压缩，speculative 检索 +86.8%，纯抽象只有 +69.4%
```

架构：64 抽象槽常驻 + 原文 KV offload + 按 speculative query 调入 512 条。

## 已完成（2026-09-21）

**上午—下午**
1. ~~anchor-aware 训练~~ → 负面，分工是自发的
2. ~~1.7B / 4B 跨规模~~ → 三个规模点全部持平，规模线结案
3. ~~打包~~ → `memory.py` + `README-usage.md`
4. ~~报告页~~ → v4（已修正 64× 的错误说法）

**傍晚—晚上**
5. ~~有效窗口~~ → 只有一段；滚动在稀释，`memory.py` 已改成一次压缩
6. ~~分层 / 主题召回~~ → 都不如不加
7. ~~跨域~~ → 状态域特定（对话 +53.5% / 小说 −54.4%），重训 12 分钟可解
8. ~~三种子方差~~ → k16 极差 51 点，混合架构 1.5–2.3 点；结论已按方差重新分类
9. ~~文献调研~~ → 交给 Grok，`docs/LITERATURE-*.md`，Q1/Q2 仍空、Q3 需降级
10. ~~FreeToken 接口调查~~ → `docs/freetoken-integration.md`，缺一个 embedding 入口

## 下一步

**A. 论文（定位已收敛）**

主贡献是两个诊断，不是新方法：
- **(a)** 按 token 可复制性拆解天花板
- **(b)** 检索 / 压缩 / oracle 的加法分解

(c) 冻结主干的测量论点**要降级并正面引用**
CCM [2312.03414]（观察过同一现象，归因为训练过拟合）和
No Mean Feat [2510.20797]（用了几乎相同的归一化公式）。
用"据我们所知"，不用"首次"。

**不要**写"我们独立想到、实现中才发现前人工作"——时间上我们在后面，
这种声明会被读成辩解，而且诊断类工作本来就不需要主张方法首创权。
那个过程属于 README 和 git 历史，不属于论文。

还缺：LongBench / RULER（选 gap 大的任务）、H2O / SnapKV baseline、8B 规模点。

**B. Airi 落地**：等 10 月中加内存，按 `docs/freetoken-integration.md` 验三个接口，
通过再租 96G 训 35B 记忆模块。**先验接口再训模型。**

**C. 工程余量**：offload KV 的 4bit 量化（896M→224M）、检索结果跨 token 缓存。
质量已基本榨干，剩下的都是延迟和内存。

## 环境备忘

- `source ~/ctfenv/bin/activate`
- 仓库 https://github.com/HoshiGT/frozen-kv-memory **private**，确认后可公开
- `recur.py` 已废弃，位置约定有 bug，用 `train_recur.py`
- ★ 位置要累积（`accum=True`），不要重置——值 4 个点
- ★ 任何只保留部分 KV 的方案都必须带 attention sink，否则 −400%
