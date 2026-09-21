# 接到 FreeToken 上（调查结果，2026-09-21）

目标：Airi 的主模型是 Qwen3.6-35B-A3B，推理端用 FreeToken（专家在 CPU、激活在 GPU）。
本方案要能在那上面跑。

**结论：路是通的，缺一个 embedding 入口，其余接口都现成。**

本地内存 15G 不够跑 FreeToken（会 pin 全部专家 19G），要等 10 月中加条子才能实测。
以下是当时直接照做的清单。

## 已经有的

`freetoken/kvcache/base.py`，`BaseKVCachePool`：

```python
def store_kv(self, k: torch.Tensor, v: torch.Tensor,
             out_loc: torch.Tensor, layer_id: int) -> None: ...
def k_cache(self, index: int) -> torch.Tensor: ...
def v_cache(self, index: int) -> torch.Tensor: ...
```

- **写入自定义 KV**：`store_kv` 能指定位置（`out_loc`）和层（`layer_id`），
  正好是注入 memory token 的 KV 和 anchor 所需要的
- **读回 KV 切片**：`k_cache` / `v_cache` 返回 torch tensor，anchor 检索直接可用
- 模型支持：`freetoken/models/qwen3_5_moe/` 就是 35B-A3B 那个架构

## 缺的那一个

`freetoken/models/qwen3_5_moe/model.py:82`：

```python
def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    x = self.embed_tokens.forward(input_ids)
```

只收 token id，而 memory token 不对应任何 token id——它是学出来的向量。

**有现成模板**：`freetoken/models/gemma4/model.py:71-83` 的 `mm_embeds`
做的是同一件事——embed 之后把特定位置（图像 token）的向量换成外部传入的。
memory token 就是同一种东西，照抄那个模式即可：

```python
mm_embeds = getattr(batch, "mm_embeds", None)
if mm_embeds is not None and self._image_token_id is not None:
    slots = (input_ids == self._image_token_id)
    assert slots.sum() == mm_embeds.shape[0]
    x[slots] = mm_embeds.to(x.dtype)
```

对我们而言：保留 k 个特殊 token id 当占位符，
把 `mem.emb`（或 deep 版每层的偏置）从 `batch` 上传进来替换。

## 验证顺序（加完内存后，在花钱之前做）

1. FreeToken 能加载 35B-A3B 并正常出字（基线）
2. `store_kv` 写进去的 KV 能被后续 token attend 到
   ——用一段原文的真实 KV 写进去，看输出是否等同于直接喂那段原文
3. embedding 注入改动生效：喂 k 个随机向量，确认前向不崩、KV 形状对
4. 三样都通过，再考虑租 96G 卡训 35B 的记忆模块

**顺序不能反。** 记忆模块训完注不进去，那几十块钱就白花了。

## 训练端另说

训练需要 PyTorch 全量（bf16 约 70GB 权重 + 激活 ≈ 75GB），本地永远跑不了，
96G 卡够。训练和推理用不同的栈没关系——记忆模块只是个 3.67M 的权重文件。

但注意**记忆模块是模型特定的**，而且**域特定的**
（同一个 ckpt：对话 +53.5%，小说 −54.4%）。
所以要用 Airi 自己的对话记录训，不能拿别的凑。

## 与本地改动的关系

本地 FreeToken 已改过内存调度（冷专家换出到 swap），
与这里要动的 embedding 入口在不同位置，不冲突。

（顺带：冷专家换出和我们的 anchor offload 是同一个思路的两个实例——
使用频率长尾的东西，冷的挪到慢存储，用到再调回。）
