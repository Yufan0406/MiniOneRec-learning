# 08 · 源码地图、阅读顺序与 CPU 实验

| | |
|---|---|
| **读这篇前** | [02 数据管线](./02-data-pipeline.md)（至少知道四类文件） |
| **读完能拿到** | 文件/函数地图、按调用链的阅读顺序、6 个 CPU 练习、FAQ、术语表、公式速查 |
| **对应源码** | 全仓库 |
| **预计耗时** | 先花 15 分钟看 §1–§3，其余当工具页随时查 |

本篇是工具页：把关键函数按调用链串起来，给出不依赖 GPU 的练习，并集中回答常见问题。

---

## 1. 文件地图

### 1.1 主流程

| 文件 | 作用 | 先看哪个函数 |
|---|---|---|
| `sft.sh` | SFT 启动脚本（示例超参） | 全文 |
| `sft.py` | SFT 训练入口 | `train`（:90） |
| `data.py` | 所有 Dataset 与 prompt 模板 | `SidSFTDataset`（:397） |
| `rl.sh` | RL 启动脚本 | 全文 |
| `rl.py` | RL 训练入口、reward 定义 | `train`（:30） |
| `minionerec_trainer.py` | GRPO Trainer（生成、reward、loss） | `ReReTrainer`（:121） |
| `LogitProcessor.py` | 约束解码的 logits mask | `ConstrainedLogitsProcessor`（:24） |
| `evaluate.sh` | 多卡评估编排 | 全文 |
| `evaluate.py` | 单卡生成与结果落盘 | `main`（:38） |
| `split.py` / `merge.py` | 切分/合并评估结果 | 全文 |
| `calc.py` | HR/NDCG/CC | `gao`（:12） |

### 1.2 SID 构建

| 文件 | 作用 |
|---|---|
| `rq/models/vq.py` | 单层向量量化 + 损失 |
| `rq/models/rq.py` | 残差量化（多层） |
| `rq/models/rqvae.py` | Encoder + RQ + Decoder |
| `rq/models/layers.py` | MLP、k-means、Sinkhorn |
| `rq/trainer.py` | RQ-VAE 训练与模型选择 |
| `rq/rqvae.py` | RQ-VAE 入口 |
| `rq/rqkmeans_faiss.py` | FAISS 版 RQ-KMeans |
| `rq/rqkmeans_constrained.py` | 均衡约束 RQ-KMeans |
| `rq/rqkmeans_plus.py` | RQ-KMeans+（残差连接 + 热启动） |
| `rq/generate_indices.py` | RQ-VAE → `.index.json` |
| `rq/generate_indices_plus.py` | RQ-KMeans+ → `.index.json` |
| `rq/generate_indices_flash.py` | FLASH-SimHash 入口（`--format minionerec` 直出 `.index.json`） |
| `rq/flash_sid/simhash.py` | 随机超平面 → 语义码（`SimHashTokenizer.encode`） |
| `rq/flash_sid/export.py` | 冲突统计 + 追加 `<d_rank>` 后导出 index |
| `rq/flash_sid/cli.py` | 命令行、输入校验、manifest |
| `sid_generation/prq_kmeans.py` | PRQ-KMeans（投影残差 + top-k 软分配） |
| `sid_generation/rq_opq.py` | RQ-OPQ（RQ 前缀 + OPQ/PQ 子码） |
| `sid_generation/sid_common.py` | 归一化、投影残差、`build_indexes`、NPZ 读写 |
| `sid_generation/sid_cli.py` | `fit` / `encode` 子命令与参数 |
| `sid_generation/sid_publish.py` | 目录级原子发布与文件清单校验 |

> 新方法的原理、CLI、实测冲突率见 [10 新 SID 构造](./10-sid-generators.md)；
> 上游自带的英文说明是 `rq/FLASH_SID.md` 与 `sid_generation/README.md`。

### 1.3 数据准备

| 文件 | 作用 |
|---|---|
| `data/amazon18_data_process.py` | Amazon18 → `.inter` / `.item.json` |
| `data/amazon23_data_process.py` | Amazon23 版本 |
| `data/amazon18_data_process_gpr.py` | GPR 版本（多 e_token / item_type / final_value） |
| `rq/text2emb/amazon_text2emb.py` | title+description → `.npy` |
| `convert_dataset.py` | `.inter` + `.index.json` → 训练 CSV |
| `convert_dataset_gpr.py` | GPR 版本转换 |

### 1.4 GPR 变体

| 文件 | 作用 |
|---|---|
| `sft_gpr.py` | SFT 变体（VAFT + 更多对齐任务） |
| `rl_gpr.py` | RL 变体（CF + HEPO reward） |
| `data.py` 的 `SidSFTDataset_GPR`（:469） | 异构 token prompt（当前未被 `sft_gpr.py` 使用） |

---

## 2. 按调用链阅读（推荐顺序）

### 2.1 SFT 调用链

```text
sft.sh
└── sft.py::train
    ├── TokenExtender.get_new_tokens
    ├── tokenizer.add_tokens + model.resize_token_embeddings
    ├── SidSFTDataset.__init__
    │   └── pre -> get_history -> encode -> labels
    ├── SidItemFeatDataset.__init__
    ├── FusionSeqRecDataset.__init__
    ├── ConcatDataset
    ├── HFDataset.from_dict
    ├── transformers.Trainer
    │   └── DataCollatorForSeq2Seq
    └── trainer.train / save_model
```

建议断点位置：

1. `sft.py:159`（`tokenizer.add_tokens` 之后）：看词表从多少变成多少；
2. `data.py:457`（`labels` 构造之后）：看 `-100` 的边界；
3. `data.py:463`（返回前）：看 `input_ids` 长度和截断。

### 2.2 RL 调用链

```text
rl.sh
└── rl.py::train
    ├── 构建 SidDataset / RLTitle2SidDataset / RLSeqTitle2SidDataset
    ├── 收集 prompt2history / history2target
    ├── 定义 reward_fun
    ├── GRPOConfig
    └── ReReTrainer
        ├── __init__: 构建 hash trie（info_file）
        ├── _prepare_inputs
        │   ├── generate（ConstrainedLogitsProcessor）
        │   ├── ref_per_token_logps
        │   ├── reward_funcs -> rewards
        │   └── 组内归一化 -> advantages
        └── compute_loss
            ├── per_token_logps
            ├── per_token_kl
            └── policy loss (+ beta * KL)
```

建议断点位置：

1. `minionerec_trainer.py:482`：看 `generation_config` 的 `num_beams`；
2. `minionerec_trainer.py:940`：看一条 completion 和对应 target；
3. `minionerec_trainer.py:972`：看组内 mean/std/advantage；
4. `minionerec_trainer.py:1049`：看 `per_token_kl` 是否非负。

### 2.3 评估调用链

```text
evaluate.sh
├── split.py::split
├── evaluate.py::main
│   ├── 读 info -> prefixID
│   ├── 构建 hash_dict
│   ├── EvalSidDataset
│   └── evaluate -> model.generate(ConstrainedLogitsProcessor)
├── merge.py::merge
└── calc.py::gao
```

---

## 3. 关键张量形状总表

| 变量 | 形状 | 产生位置 |
|---|---|---|
| `input_ids` (SFT) | `(B, T)` | `DataCollatorForSeq2Seq` |
| `labels` (SFT) | `(B, T)` | `SidSFTDataset.pre` |
| `logits` (SFT) | `(B, T, V)` | LLM |
| `prompt_ids` (RL) | `(B, P)` | `_prepare_inputs` |
| `completion_ids` (RL) | `(B*G, C)` | `generate` |
| `per_token_logps` | `(B*G, C)` | `_get_per_token_logps` |
| `rewards_per_func` | `(B*G, num_reward_funcs)` | reward 计算 |
| `rewards` | `(B*G,)` | 加权求和 |
| `advantages` | `(B*G,)` | 组内归一化 |
| `latent` (RQ) | `(B, e_dim)` | encoder |
| `distance` (VQ) | `(B, K)` | 最近邻 |
| `indices` (RQ) | `(B, L)` | 残差量化 |

记号：`B` batch，`G` 每个 prompt 的候选数，`T` 序列长度，`P` prompt 长度，
`C` completion 长度，`V` 词表大小，`K` 码本大小，`L` 层数。

---

## 4. CPU 实验（不需要 GPU / 模型权重）

### 4.1 手工还原一条 SFT 样本

不依赖 `transformers`，用字符级模拟"token"：

```python
import csv, ast

row = next(csv.DictReader(open(
    'data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv')))

history = ast.literal_eval(row['history_item_sid'])
target = row['item_sid']

prompt = (
    "### User Input: \n"
    f"The user has interacted with items {', '.join(history)} in chronological order. "
    "Can you predict the next possible item that the user may expect?\n\n"
    "### Response:\n"
)
print('prompt =', prompt)
print('target =', target)
print('history 层数 =', [len(h.split('><')) for h in history[:3]])
```

对照 `data.py:403-466`，确认：

- `history` 是用 `", "` 连接；
- prompt 末尾有 `### Response:\n`；
- target 后面会再加一个 `\n` 和 EOS。

### 4.2 用字符串模拟约束前缀树

```python
info = [
    "<a_42><b_80><c_160>",
    "<a_42><b_80><c_177>",
    "<a_236><b_231><c_226>",
]

trie = {}
for sid in info:
    parts = sid.replace("><", ">|<").split("|")
    for i in range(len(parts)):
        key = tuple(parts[:i])
        trie.setdefault(key, set()).add(parts[i])

print(trie[()])                    # 第一层允许的 token
print(trie[("<a_42>",)])           # 第二层
print(trie[("<a_42>", "<b_80>")])  # 第三层
```

观察：`<a_42><b_80>` 之后允许 `<c_160>` 和 `<c_177>`，但不可能出现 `<c_999>`。

### 4.3 手算 GRPO advantage

```python
rewards = [0.0, -0.2463, -0.1952, -0.1681]   # 组内 4 个候选，第 1 个命中
mean = sum(rewards) / len(rewards)
var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
std = var ** 0.5
advantages = [(r - mean) / (std + 1e-4) for r in rewards]
print(mean, std, advantages)
```

思考：如果 4 个 reward 全是 0，advantage 是多少？为什么这时没有梯度？

### 4.4 手算 HR/NDCG

```python
target = "<a_42><b_80><c_160>"
preds = ["<a_9><b_1><c_1>", "<a_42><b_80><c_177>", target]
min_id = preds.index(target)          # 2
for k in [1, 3, 5]:
    hr = int(min_id < k)
    ndcg = hr / __import__('math').log2(min_id + 2)
    print(k, hr, round(ndcg, 4))
```

### 4.5 统计 SID 冲突与使用率

见 [03-sid-construction.md](./03-sid-construction.md) 第 7 节。

### 4.6 静态追踪调用关系

```bash
# 谁调用了某个函数
grep -rn "SidSFTDataset(" --include=*.py .

# 哪些文件定义了 Dataset
grep -n "^class .*Dataset" data.py

# 所有 reward 分支
grep -n "reward_type ==" rl.py rl_gpr.py
```

---

## 5. 常见问题（FAQ）

### Q1. 没有 GPU 能学这个项目吗？

能读、能验证数据管线和指标，但**不能真正训练**。
SFT 至少需要一张能放下 base model 的 GPU；RL 还需要额外的参考模型和 16 个 beam 的显存。
CPU 上最值得做的是第 4 节的四个实验。

### Q2. 只有一张 GPU，怎么改 batch？

- `sft.sh`：把 `--nproc_per_node` 改成 1，`micro_batch_size` 降到 1–4，
  `batch_size` 设成 `micro_batch_size × gradient_accumulation_steps`；
- `rl.sh`：把 `num_processes` 改成 1，`train_batch_size` 必须能被 `num_generations` 整除；
- 评估：降低 `--batch_size` 和 `--num_beams`。

### Q3. 为什么复现的 HR 比论文低很多？

按可能性排序：

1. `CC` 不为 0（约束解码失效，常见于 Qwen Instruct + transformers 新版本）；
2. 评估用的 `info.txt` 与模型训练时的 `.index.json` 不是同一次构建；
3. 没有加载 SFT 保存的 `final_checkpoint`（缺少新 token 的词表）；
4. prompt 模板与训练不一致；
5. beam 数、`max_new_tokens`、`length_penalty` 与论文不同。

### Q4. 为什么 `data.py` 里到处是 `eval()`？

CSV 里的 list 被存成了字符串（`"['<a_1>', '<a_2>']"`），`eval` 用来还原。
它只能用于可信数据；生产环境应改用 `ast.literal_eval` 或 JSON。

### Q5. `dedup` 字段有什么用？

它标记"目标 item 是否等于历史最后一个 item"。
Dataset 会返回这个字段，但主流程的 `calc.py` **没有**用它过滤样本；
只有显式传 `dedup=True` 的 Dataset 才会跳过这些样本。

### Q6. SID 层数/码本大小怎么选？

- 容量：$K^L$ 必须显著大于 item 数；3 层 × 256（$1.68\times10^7$）对几万以内 item 足够；
- 冲突率：去重前看 collision rate，目标接近 0；
- 成本：每多一层，SFT 多一个监督 token，RL 多一步解码；
- 语义：层数太多会让后面几层只拟合噪声，重建误差收益递减。
- 方法：四种经典方法的入口见 03 章；FLASH / PRQ-KMeans / RQ-OPQ 见 [10 章](./10-sid-generators.md)。
  想要"零训练、几秒出结果"用 FLASH（但冲突率高），想要"CPU 上质量可用"用 PRQ-KMeans。

### Q7. 新增 item 怎么办？

1. 用同一个文本编码器算 embedding；
2. 用同一套码本量化得到 SID；
3. 更新 `.index.json` 和 `info.txt`；
4. 如果出现了新的 SID token，需要扩展 tokenizer 词表并重训/微调 embedding；
5. **必须重建约束前缀树**，否则模型永远生成不到新 item。
6. 注意**身份 token 的 rank 是全库重算的**：新增 item 可能改变同组其他 item 的
   `<d_rank>`，所以不能"只追加一行 index"，必须整库重新导出
   （见 [10 §6](./10-sid-generators.md)）。

### Q8. 为什么不直接用 LLM 生成 item 标题，再检索？

那是"生成式检索"的另一条路线。MiniOneRec 选择生成 **SID**，因为：

- SID 是闭集，可以用前缀树保证合法性；
- 一个 item 只需 3 个 token，推理成本低；
- 与协同过滤信号解耦，便于替换量化算法。

### Q9. `train_from_scratch=True` 是什么？

`AutoConfig.from_pretrained` + `AutoModelForCausalLM.from_config`，
随机初始化整个模型。只用于消融"预训练知识是否重要"，正常训练不要开。

### Q10. `sync_ref_model` 在做什么？

GRPO 的 KL 项需要一个参考策略 $\pi_{\text{ref}}$。默认它是 SFT checkpoint；
`sync_ref_model=True` 会周期性地把当前策略复制给参考模型，
让 KL 约束跟随策略移动（类似"滚动基线"）。

---

## 6. 术语表

| 术语 | 含义 |
|---|---|
| SID (Semantic ID) | 由 item 语义量化得到的离散 token 序列 |
| RQ | Residual Quantization，残差量化 |
| RQ-VAE | 用 VAE + 残差量化学习码本 |
| Codebook | 每层的 $K$ 个聚类中心/嵌入向量 |
| Collision | 两个 item 得到相同 SID |
| Constrained decoding | 只允许生成合法 SID 前缀的解码 |
| Beam search | 保留 top-B 条候选的序列解码 |
| GRPO | Group Relative Policy Optimization |
| Advantage | 组内归一化后的相对奖励 |
| KL penalty | 约束新策略不要偏离参考策略 |
| VAFT | Value-Aware Fine-Tuning，按 item 价值加权 SFT |
| HEPO | Hierarchy Enhanced Policy Optimization，按 SID 层级给 reward |
| CC | 评估中非法预测的计数 |

---

## 7. 公式速查

**下一个 item 的概率**

$$
p(c_i \mid h) = \prod_{l=1}^{L} p_\theta\left(c_i^{(l)} \mid c_i^{(<l)}, h\right)
$$

**SFT 损失**

$$
\mathcal{L}_{\text{SFT}} = -\sum_{l=1}^{L}\log p_\theta\left(c_i^{(l)} \mid c_i^{(<l)}, h\right)
$$

**VQ 损失**

$$
\mathcal{L}_{\text{VQ}} = \|x - \hat{x}\|^2 + \|x - \mathrm{sg}[\hat{x}]\|^2 + \beta\|\mathrm{sg}[x] - \hat{x}\|^2
$$

**GRPO advantage**

$$
A_i = \frac{r_i - \operatorname{mean}(r_{1:G})}{\operatorname{std}(r_{1:G}) + \epsilon}
$$

**GRPO 损失**

$$
\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B}\frac{1}{|c_i|}\sum_{t=1}^{|c_i|}
\left(A_i - \beta\left[e^{\Delta_{i,t}} - \Delta_{i,t} - 1\right]\right),
\quad \Delta_{i,t} = \log\pi_{\text{ref}} - \log\pi_\theta
$$

**HR@K**

$$
\mathrm{HR@K} = \frac{1}{N}\sum_{u}\mathbb{1}[\text{rank}_u < K]
$$

**NDCG@K（单相关 item）**

$$
\mathrm{NDCG@K} = \frac{1}{N}\sum_{u}\mathbb{1}[\text{rank}_u < K]\cdot\frac{1}{\log_2(\text{rank}_u+2)}
$$

---

## 8. 一页式检查清单

准备跑一次完整流程前，逐项确认：

- [ ] `.npy` 行数 == `.item.json` key 数 == `.index.json` key 数
- [ ] `.index.json` 的 SID 层数与 `--num_emb_list` 一致
- [ ] 冲突率已检查（目标接近 0）
- [ ] `convert_dataset.py` 用的 `data_dir` 与 SID 构建用的是同一份数据
- [ ] `sft.sh` 的 `--sid_index_path` / `--item_meta_path` 指向同一套文件
- [ ] `rl.sh` 的 `--model_path` 是 SFT 的 `final_checkpoint`
- [ ] `eval` 的 `info_file` 与训练时的 `.index.json` 来自同一次构建
- [ ] `CC == 0`
- [ ] `git status` 中没有 `docs/`（本文档不应被提交）

---

## 9. 结语

如果你能独立回答下面三个问题，说明已经掌握了这套代码的主干：

1. 一条 CSV 样本从哪几个文件读取字段，经过哪些函数，最终变成什么形状的张量？
2. 约束解码为什么能保证生成合法 item？它的信息来自哪个文件？
3. GRPO 的 advantage 为什么要在组内归一化？全零 reward 的组会发生什么？

接下来建议做一件小事：**选一个 item，手工追踪它的 SID 从 `.npy` 到 `.index.json`
再到一条训练样本的全过程**，并把每一步的中间值打印出来。
这比再读十遍论文都有效。

---

## 10. 相关章节

| 想做的事 | 去哪 |
|---|---|
| 真的把流水线跑起来（各阶段命令、参数、排错表） | [11 实操手册](./11-hands-on.md) |
| 了解 FLASH / PRQ-KMeans / RQ-OPQ 三种新 SID 构造 | [10 新 SID 构造](./10-sid-generators.md) |
| 不熟悉 SID 的命名与 0/1-based | [03 §4](./03-sid-construction.md) |
