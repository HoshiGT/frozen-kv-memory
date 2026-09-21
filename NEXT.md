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
