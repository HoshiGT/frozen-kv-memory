# TODO

昨天那份已全部做完（预算杠杆、天花板成因、混合架构、检索策略）。这是 2026-09-21 下午起的新队列。

## 当前最好的配置

```bash
python hybrid.py --k 64 --anchors 512 --probe 128 --hops 8
# 7:1 压缩，speculative 检索 +86.8%，纯抽象只有 +69.4%
```

架构：64 抽象槽常驻 + 原文 KV offload + 按 speculative query 调入 512 条。

## 1. 训练 anchor-aware 的记忆模块 ★ 主线

现在用的 ckpt 是**纯抽象**训练出来的——它不知道将来身边会有 anchor，
所以它在努力存一些 anchor 本来就能精确提供的东西，这部分是纯浪费。

训练时就把 anchor 放进 past，抽象槽的梯度自然会转向"anchor 拿不到的那部分"：

- `diagnose.py` 已经指出分工线在哪：novel token 归抽象（+231%），
  seen 尤其 seen×1 归 anchor（+46%）
- anchor 用 surprisal 选（encoding-time，训练循环里够快；speculative 太慢）
- anchor 的 KV 是 detach 的原文，梯度只流经抽象槽
- 预期：抽象槽专注 gist 后，同预算下整体还能再涨

风险：可能没有提升，因为 seg_b_loss 的梯度本来就会避开 anchor 已覆盖的部分。
那样的话结论是"分工是自发的，不需要显式训练"，也值得记。

## 2. 1.7B 跨规模验证

混合架构目前只在 0.6B 上测过。需要确认：
- +86.8% 是否随规模保持或提升
- surprisal / speculative 的优劣是否翻转（2048 上下文时 surprisal 已反超过一次）
- 1.7B 在 8GB 卡上要 `--pred-len 192`，否则 tail 的 logits 就 311MB

## 3. 打包成能用的东西

现在全散在实验脚本里。最小可用形态：

```
compress(ctx) -> (abstract_state, offloaded_kv)
retrieve(state, prefix, n) -> kv_to_page_in
```

配 `README-usage.md`。这是 MVP 真正能交付的那一层。

## 4. 更新报告页

https://claude.ai/artifact/WV3RJaDs97ZkdWCUuEvePZ
里面全是 2026-09-20 的数字（128:1 写的 +50.8%），今天之后**整页需要重写**：
主线已经从"压缩"变成"压缩 + 检索"。

## 5. 可选的扩展

- anchor 预算的扩展律（512 → 1024 → 2048，什么时候追平 upper）
- 16 hops 以上：8192 时 speculative 掉到 +72.2%，oracle 仍有 +94.7%，
  说明选择策略在长上下文下失效得比容量快，值得单独查
- Matryoshka 嵌套（昨天记的，解决 k=16 角色冲突，现在优先级低）

## 环境备忘

- `source ~/ctfenv/bin/activate`
- 仓库 https://github.com/HoshiGT/frozen-kv-memory **private**，确认后可公开
- `recur.py` 已废弃，位置约定有 bug，用 `train_recur.py`
- ★ 位置要累积（`accum=True`），不要重置——值 4 个点
- ★ 任何只保留部分 KV 的方案都必须带 attention sink，否则 −400%
