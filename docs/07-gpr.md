# 07 · GPR 变体：异构 token、VAFT 与 HEPO

| | |
|---|---|
| **读这篇前** | [04 SFT](./04-sft.md)、[05 RL](./05-rl.md)（要先认识主流程） |
| **读完能回答** | GPR 和主流程差在哪？U/E/I/O token 是什么？哪些设计接上了代码、哪些没有？ |
| **对应源码** | `data/amazon18_data_process_gpr.py`、`convert_dataset_gpr.py`、`sft_gpr.py`、`rl_gpr.py`、`data.py` 的 `SidSFTDataset_GPR` |
| **预计耗时** | 45 分钟（只读差异） |

> 先给结论：**这是一条研究性分支，代码里既有"设计意图"，也有尚未接线的部分。**
> 本篇会把两者分开讲清楚。主流程不依赖这一章。

---

## 1. GPR 想解决什么

主流程把所有信息都压进 SID：

```text
用户历史: <a_12><b_7><c_3>, <a_9><b_201><c_44>
```

但真实推荐里还有三类信息被丢掉了：

| 信息 | 例子 | 主流程 | GPR 变体 |
|---|---|---|---|
| 用户画像 | 高评分用户 / 低评分用户 | 无 | `[USER_HIGH_RATING]` |
| 场景上下文 | 首页 / 搜索 / 浏览 | 无 | `[CTX_HOMEPAGE]` |
| item 类型 | 自然流量 / 广告位 | 无 | `[O_TOKEN]` / `[I_TOKEN]` |
| item 价值 | 价格、评论数 | 无 | `final_value`（用于加权） |

GPR 的做法是：**把这些信息也变成 token，和 SID 拼在同一个序列里**，
让 LLM 在一个统一的"异构 token 空间"里建模。

```text
[USER_HIGH_RATING] [CTX_SEARCH] The user has interacted with items
[O_TOKEN]<a_12><b_7><c_3>, [I_TOKEN]<a_9><b_201><c_44> ...
### Response:
[O_TOKEN]<a_42><b_80><c_160>
```

对应两个训练技巧：

- **VAFT**（Value-Aware Fine-Tuning）：SFT 阶段按 item 价值给样本加权；
- **HEPO**（Hierarchy Enhanced Policy Optimization）：RL 阶段按 SID 层级给部分奖励。

---

## 2. 数据层：模拟出来的异构特征

### 2.1 预处理脚本

`data/amazon18_data_process_gpr.py` 与主流程基本一致，额外做三件事：

#### (a) item_type

`create_item_features_amazon18_style`（`data/amazon18_data_process_gpr.py:390`）：

```python
if price > 50.0 or (brand and brand != 'Unknown'):
    item_type = 'I'     # 广告/品牌 item
else:
    item_type = 'O'     # 自然 item
```

这是一个**启发式规则**，不是真实广告位数据。

#### (b) final_value

```python
final_value = log1p(price) + log1p(num_reviews)
# 再 min-max 归一化到 [0, 1]
```

直觉：价格越高、评论越多，item 的"价值"越高。训练时用它给样本加权。

#### (c) user token

`create_user_features`（`data/amazon18_data_process_gpr.py:476`）：

```python
avg_rating > 4.5  -> [USER_HIGH_RATING]
avg_rating > 3.5  -> [USER_MID_RATING]
else              -> [USER_LOW_RATING]
```

输出 `<dataset>.user.json`。

#### (d) e_token

`generate_interaction_list_json2csv_style`：

```python
e_token = random.choice(['[CTX_BROWSE]', '[CTX_SEARCH]', '[CTX_HOMEPAGE]'])
```

**随机生成的**，不是真实场景日志。这是为了模拟"上下文 token"的建模方式。

### 2.2 输出差异

`.inter` 多一列 `e_token`：

```text
user_id:token	item_id_list:token_seq	item_id:token	e_token:token
0	12 45 78	99	[CTX_SEARCH]
```

`convert_dataset_gpr.py` 在 CSV 里多写两列：

| 列 | 用途 |
|---|---|
| `user_id_original_str` | 查 `.user.json` 得到 U token |
| `e_token` | 直接作为 E token |

其余字段与主流程一致。

---

## 3. SFT 侧：异构 prompt 与 VAFT

### 3.1 特殊 token 怎么加

`sft_gpr.py:191`：

```python
new_special_tokens = [
    '[USER_HIGH_RATING]', '[USER_MID_RATING]', '[USER_LOW_RATING]', '[USER_UNKNOWN]',
    '[CTX_BROWSE]', '[CTX_SEARCH]', '[CTX_HOMEPAGE]',
    '[O_TOKEN]', '[I_TOKEN]',
]
tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})
model.resize_token_embeddings(len(tokenizer))

# 然后再加 SID token
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))
```

顺序很重要：先加特殊 token，再加 SID token；两次 resize 都会改变词表大小。

### 3.2 异构 prompt 的构造

`data.py:469` 的 `SidSFTDataset_GPR.pre`：

```python
user_id = str(row.get('user_id_original_str', ''))
u_token = self.user_features.get(user_id, '[USER_UNKNOWN]')
e_token = row.get('e_token', '[CTX_HOMEPAGE]')

for item_id, sid in zip(history_item_ids, history_sids):
    item_type = self.item_features.get(item_id, {}).get('item_type', 'O')
    prefix = '[O_TOKEN]' if item_type == 'O' else '[I_TOKEN]'
    history_str += f"{prefix}{sid}"

input_text = f"{u_token} {e_token} The user has interacted with items ..."
target = f"{target_prefix}{target_sid}\n"
```

最终 prompt：

```text
### User Input:
[USER_HIGH_RATING] [CTX_SEARCH] The user has interacted with items
[O_TOKEN]<a_12><b_7><c_3>, [I_TOKEN]<a_9><b_201><c_44> in chronological order.
Can you predict the next possible item that the user may expect?

### Response:
[O_TOKEN]<a_42><b_80><c_160>
```

### 3.3 VAFT 损失

`VAFT_Trainer.compute_loss`（`sft_gpr.py:90`）：

```python
loss_per_token = CrossEntropyLoss(reduction='none')(...)
loss_per_seq = loss_per_token.view(B, -1)
seq_loss = (loss_per_seq * mask).sum(1) / (num_valid + 1e-9)

value_weights = torch.log1p(final_values)
weighted_loss = (seq_loss * value_weights).mean()
```

公式：

$$
\mathcal{L}_{\text{VAFT}} =
\frac{1}{B}\sum_{i=1}^{B}
\underbrace{\log(1 + v_i)}_{w_i}
\cdot
\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}
\ell_{i,t}
$$

其中 $v_i \in [0,1]$ 是目标 item 的归一化价值。

设计意图：让高价值 item 的预测误差占更大权重。

> **一个必须知道的细节**：$v_i=0$ 时 $w_i=\log 1=0$，该样本完全不产生梯度。
> 如果数据里有很多 `final_value=0` 的 item，它们会被静默忽略。

### 3.4 但 VAFT 在当前脚本里真的生效吗？

这是读研究代码时最重要的一类问题。答案是：**按当前 `sft_gpr.py` 的接线，没有生效。**

原因链：

1. `sft_gpr.py:247` 使用的是 `SidSFTDataset`，不是 `SidSFTDataset_GPR`；
2. `SidSFTDataset` 的返回值里**没有** `final_value` 字段；
3. `hf_train_dataset` 的列来自 `train_data[0].keys()`（`ConcatDataset` 的第 0 个数据集），
   因此也不会有 `final_value` 列；
4. `VAFT_Trainer.compute_loss` 里 `inputs.pop("final_value", None)` 得到 `None`，
   直接 `return super().compute_loss(...)`。

所以当前脚本实际训练的是"普通 SFT + 额外的对齐任务"，
VAFT 只是一个**预留但未接线**的能力。

要让 VAFT 生效，至少需要：

```python
train_data1 = SidSFTDataset_GPR(..., category=<正确目录名>)
```

并且保证 `data/<category>/<category>.user.json` 与 `.item.json` 路径正确
（`SidSFTDataset_GPR.__init__` 里把 `category` 直接拼进路径，
而 `sft_gpr.py` 传进来的是 `"industrial and scientific items"` 这种长描述，
不是数据集名，路径会对不上）。

### 3.5 `sft_gpr.py` 实际用了哪些数据集

| Dataset | 任务 | 是否有 `final_value` |
|---|---|---|
| `SidSFTDataset` | 历史 SID → 下一个 SID | 否 |
| `SidItemFeatDataset` | SID ↔ 标题 | 否 |
| `FusionSeqRecDataset` | 历史 SID → 标题 | 否 |
| `SFTData` | 历史标题 → 目标标题 | 否 |
| `TitleHistory2SidSFTDataset` | 历史标题 → 目标 SID | 否 |
| `UserPreference2sidSFTDataset` | 用户偏好文本 → SID（可选） | 否 |

可以看到，GPR 的 SFT 变体实际上是通过**更多自然语言任务**（标题历史、偏好文本）
来增强语义 grounding，而不是靠异构 token。

---

## 4. RL 侧：HEPO 分层 reward

### 4.1 配置

`rl_gpr.py` 与 `rl.py` 的差别主要在 reward：

```python
if reward_type == "sasrec":
    reward_fun = [cf_reward, hepo_reward]   # 组合 reward
```

即"协同过滤分数 + 层级匹配分数"相加。

### 4.2 HEPO 的定义

`rl_gpr.py:88`：

```python
def parse_sid(sid):
    return re.findall(r'\[.*?\]', sid)

if comp_parts[0] == target_parts[0]:
    reward = 0.2
    if comp_parts[1] == target_parts[1]:
        reward = 0.5
        if comp_parts[2] == target_parts[2]:
            reward = 1.0
```

| 匹配层级 | reward |
|---|---|
| 第 1 层命中 | 0.2 |
| 前 2 层命中 | 0.5 |
| 前 3 层全中 | 1.0 |
| 第 1 层就不中 | 0.0 |

直觉：SID 的前缀有语义，答对前缀也应该得到部分奖励，
这比"只有完全命中才给 1"的稀疏 reward 更容易学习。

### 4.3 一个实现层面的坑

`parse_sid` 用的是 `\[.*?\]`，匹配的是**方括号** `[a_1]`；
而当前仓库的 SID token 是**尖括号** `<a_1>`。

按现有 `info.txt` 的格式：

```python
parse_sid("<a_42><b_80><c_160>")  # -> []
```

于是 `comp_parts` 为空，HEPO reward 恒为 0。
如果你的 SID 构建脚本输出的是 `[a_1][b_2][c_3]`，这段代码才会按预期工作。

**复现建议**：先打印一条 `item_name` 和 `parse_sid(item_name)`，
确认格式匹配后再启用 HEPO；否则它只是一个 0 reward。

---

## 5. 主流程 vs GPR 变体：逐项对照

| 维度 | 主流程 | GPR 变体 |
|---|---|---|
| 数据列 | 无 user/e token | `user_id_original_str`, `e_token` |
| item 特征 | title/description/brand/categories | 额外 `item_type`, `final_value` |
| 特殊 token | 只有 SID token | SID + U/E/I/O token |
| SFT 数据集 | 3 个 | 5–6 个（含标题历史、偏好） |
| 损失 | 标准 CE | 设计上 VAFT；当前接线仍是标准 CE |
| RL reward | rule / ranking | `cf_reward + hepo_reward` |
| 入口 | `sft.sh` / `rl.sh` | 需手动改脚本指向 `*_gpr.py` |

---

## 6. CPU 上检查 GPR 特征

```bash
# 把 <gpr_output> 换成 amazon18_data_process_gpr.py 的 --output_path
python3 - <<'PY'
import json
p = '<gpr_output>/Industrial_and_Scientific/Industrial_and_Scientific.item.json'
item = json.load(open(p))
vals = [v.get('final_value', None) for v in item.values()]
types = [v.get('item_type') for v in item.values()]
print('items:', len(item))
print('with final_value:', sum(x is not None for x in vals))
print('value=0 count:', sum(x == 0 for x in vals if x is not None))
from collections import Counter
print('item_type:', Counter(types))
PY
```

重点看 `value=0` 的比例——如果很高，VAFT 会丢掉大量样本。

---

## 7. 读 GPR 代码的建议

1. 先把它当成"主流程 + 额外字段"，不要一上来就啃全部；
2. 遇到 `final_value` / `item_type`，先确认数据文件里是否真的有；
3. 遇到 `reward_type == "sasrec"`，先打印 `parse_sid` 的结果；
4. 任何"论文里说有、代码里没接线"的地方，都以**实际调用链**为准；
5. 如果要做实验，先从 `sft_gpr.py` 的 `train_datasets` 列表改起，
   一次只启用一个变体，保证能定位问题。

---

## 8. 下一步

回到主流程的代码地图与练习：
[08-code-map.md](./08-code-map.md)。
