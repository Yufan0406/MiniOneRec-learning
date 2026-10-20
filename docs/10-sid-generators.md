# 10 · 新的 SID 构造方法：FLASH-SimHash、PRQ-KMeans 与 RQ-OPQ

| | |
|---|---|
| **读这篇前** | [03 SID 构建](./03-sid-construction.md)（残差量化、0-based、冲突率） |
| **读完能回答** | 这三种新方法各自"训练"什么？为什么 FLASH 的冲突率能到 60%？第 4 个 token 从哪来？换一套 SID 之后哪些东西必须重做？ |
| **对应源码** | `rq/generate_indices_flash.py`、`rq/flash_sid/`、`sid_generation/`、`tests/test_flash_sid*.py` |
| **上游文档** | `rq/FLASH_SID.md`、`sid_generation/README.md`、`sid_generation/VALIDATION.md` |
| **预计耗时** | 精读 60–90 分钟；只跑命令 15 分钟 |

---

## 1. 这一章补的是什么缺口

03 章讲了四种"经典"做法：RQ-VAE、RQ-KMeans（FAISS）、constrained RQ-KMeans、RQ-KMeans+。
它们的共同点是：**都要拟合某种量化器**（神经网络或 k-means），都假设"3 层 × 256 码本"，
并且都把"冲突处理"当成一个补丁（只给冲突的 item 追加 rank）。

上游新增的三种方法换了三个角度：

| 新增方法 | 换的角度 | 代价 |
|---|---|---|
| FLASH-SimHash | **完全免训练**：随机超平面 + 符号位，纯 NumPy | 冲突率高（实测 63.6%），语义分辨率低 |
| PRQ-KMeans | **换距离与残差定义**：投影残差 + top-k 软分配 | 仍是 k-means，拟合慢于 SimHash |
| RQ-OPQ | **换量化结构**：RQ 前缀 + OPQ 旋转 + 乘积量化 | 需要 FAISS，token 序列变长（5 个） |

一句话定位：**03 章讲"语义量化怎么做对"，这一章讲"不训练量化器时还能怎么做、以及这些方法对下游数据格式提了什么新要求"。**

> 注意：这三个实现都只做 **SID 构造（tokenization）**。
> FLASH 论文里的并行解码器、语义对齐训练没有实现；RQ-OPQ 也只是 OneSearch 的 item-only 适配。
> 所以它们**不能**直接宣称"效果更好"，只能作为 SID 质量的对照实验。

---

## 2. 七种方法总表

| 方法 | 入口 | 关键依赖 | 需要拟合 | 拟合对象 | 默认 token 数 | 冲突处理 |
|---|---|---|---|---|---|---|
| RQ-VAE | `rq/rqvae.sh` → `rq/generate_indices.py` | torch | 是（GPU 友好） | 编码器 + L 层码本 | 3 | 冲突项重新编码，追加 rank |
| RQ-KMeans | `rq/rqkmeans_faiss.py` | faiss | 是 | L 层质心 | 3 | 冲突率偏高，靠下一层/去重 |
| constrained RQ-KMeans | `rq/rqkmeans_constrained.sh` | polars + `k_means_constrained` | 是 | 带容量约束的质心 | 3 | 冲突项加额外层 |
| RQ-KMeans+ | `rq/rqkmeans_plus.sh` | 同上 + torch | 是 | 质心 + 残差连接/热启动 | 3 | 同上 |
| **FLASH-SimHash** | `rq/generate_indices_flash.py` | **只用 numpy** | **否** | 无（固定随机投影） | **3 语义 + 1 身份** | 固定追加 `<d_rank>` |
| **PRQ-KMeans** | `sid_generation/prq_kmeans.py` | numpy | 是（纯 CPU，秒级/分钟级） | 投影残差质心 | 3（+1 身份） | `append` / `error` / `allow` |
| **RQ-OPQ** | `sid_generation/rq_opq.py` | numpy + faiss-cpu | 是 | 3 层 RQ 质心 + OPQ 旋转 + PQ 码本 | 3 + 2（+1 身份） | 同上 |

> 表中的"token 数"是**默认配置**。长度直接决定：beam 搜索的层数、约束 trie 的深度、
> 一条样本的监督 token 数，以及 `data.py` 里 `''.join(sids)` 拼出来的字符串长度。

---

## 3. FLASH-SimHash：零训练的语义哈希

### 3.1 原理：随机超平面的符号位

给每个语义位置 $l$ 抽 $m = \log_2 K$ 个随机高斯超平面 $g_{l,b} \in \mathbb{R}^{d}$，
item 的 embedding $x$ 在该位置上得到

$$
c_l(x) = \sum_{b=0}^{m-1} 2^{b} \cdot \mathbb{1}\!\left[\, g_{l,b}^{\top} x > 0 \,\right],
\qquad b=0 \text{ 是最低位}
$$

直觉：**一个超平面把空间切成两半，$m$ 个超平面切成 $2^m$ 个扇区**。
两个向量夹角越小，落在同一扇区的概率越高——这就是 angle-preserving LSH（SimHash）。
$K=256$ 时 $m=8$，三个位置相当于三张互不相关的 8-bit 哈希表。

三个关键设计决定（读代码时容易误以为是 bug）：

1. **不归一化、不中心化**。输入直接进 float32 矩阵乘，模长会影响符号位。
2. **三个位置互相独立**，没有残差、没有码本，所以第 2 层不会"修正"第 1 层的误差。
3. **码是 0-based**，而论文公式写的是 1-based（03 章 §4.2 踩过同样的坑）。

### 3.2 代码走读：`rq/flash_sid/simhash.py`

| 位置 | 在做什么 | 值得注意的细节 |
|---|---|---|
| `SimHashTokenizer.__init__` | 抽超平面 | `planes` 形状 `(num_codebooks, log2(K), d)`；先用 `default_rng(seed).standard_normal` 在 **float64** 抽样，再 `astype(np.float32)`——直接抽 float32 会得到与参考实现不同的超平面 |
| `_configure` | 参数校验 | `codebook_size` 必须是 2 的幂（2 ~ 2³⁰），因为要按 bit 打包 |
| `encode` | 前向 | 输入按 `batch_size=4096` 分块；`scores = values @ projection_matrix`，reshape 成 `(B, L, m)`；`np.greater(scores, 0) * bit_weights` 求和 |
| `encode` 校验 | 防御 | 拒绝零向量行、非有限值、float32 转换溢出、维度不匹配 |
| `save` / `load` | 落盘 | NPZ 里存 `metadata` + `planes`（只读），`os.replace` 原子替换 |

`projection_matrix = planes.reshape(-1, d).T` 这一步把"每层每 bit 一个超平面"
压成**一次矩阵乘**完成全部 $L \times m$ 个符号判断——这是纯 NumPy 也能 0.1 秒跑完 3,686 条的原因。

### 3.3 实测：把自带数据跑一遍

```bash
python rq/generate_indices_flash.py \
  --embeddings data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy \
  --item-meta data/Amazon/index/Industrial_and_Scientific.item.json \
  --output-dir /tmp/flash_out --dataset Industrial_and_Scientific
```

`Industrial_and_Scientific`（3,686 items × 2,560 维，float16）在这台 Mac 上的结果：

| 指标 | 数值 | 怎么读 |
|---|---|---|
| 编码耗时 | 0.075 s | 纯矩阵乘，CPU 足够 |
| 原始唯一 SID | 1,343 / 3,686 | 只有 36.4% 的 SID 是唯一的 |
| 原始冲突率 | **63.56%** | `(N − 唯一数) / N` |
| 最大冲突组 | 68 个 item 共享一个 SID | 需要 `<d_rank>` 兜底，容量默认 256 |
| 第 1/2/3 层用到的码 | 38 / 54 / 94（共 256） | 利用率仅 15% / 21% / 37% |
| 各层熵 | 3.15 / 3.03 / 4.63 bit | 远低于理论 8 bit，说明哈希远未均分 |
| 导出后 | 3,686 个 4-token SID，唯一率 100% | 唯一性完全由身份 token 提供 |

这组数字值得记住：**FLASH 的"语义"信息量很小，唯一性来自第 4 个 token。**
拿它和 RQ-KMeans+（自带 index 的 3,686 个 item 里 3,670 个唯一、最大冲突组 3）对比，
就能理解"免训练"换来的是什么。

### 3.4 命令行与输出

| 参数 | 作用 |
|---|---|
| `--embeddings` | `[N, D]` 的 `.npy`，**不能带 padding 行** |
| `--item-meta` | `.item.json`，要求 key 恰好是 `0..N-1`（行 i 就是 item i） |
| `--item-ids` | 上面的替代品：JSON 列表，显式给出每行对应的 item id |
| `--format` | `minionerec`（默认，3 语义位 + 身份位）或 `raw`（默认 32 位，不给 MiniOneRec 用） |
| `--num-codebooks` / `--codebook-size` / `--seed` | 默认 3 / 256 / 2024 |
| `--suffix-size` | 单个冲突组最多容纳多少 item，默认 256，超了直接报错 |
| `--model-in` | 复用已保存的投影（换新数据编码时保持一致） |
| `--overwrite` | 显式覆盖同名输出 |

固定产出 5–6 个文件：`.index.json`、`.codes.npy`、`.item_ids.json`、`.simhash.npz`、
`.metrics.json`、`.manifest.json`。**只有 `.index.json` 是训练用的**，其余是复现/审计用的。

---

## 4. PRQ-KMeans：把"残差"换成"投影残差"

### 4.1 与 RQ-KMeans 的三个差异

| | RQ-KMeans（03 章） | PRQ-KMeans（本章） |
|---|---|---|
| 每层输入 | 上一层减掉质心后的**欧氏残差** | 上一层减掉"在所选质心方向上的投影"后的**归一化残差** |
| 分配方式 | 硬分配（最近质心） | 训练时 **top-k 软分配**，编码时硬分配（最大余弦） |
| 相似度 | L2 距离 | 余弦（先 L2 归一化） |

去掉"投影"而非"整个向量"，是为了只消除该质心方向上的分量、保留与之正交的语义方向；
top-k 软更新（权重 $\mathrm{softmax}(\beta \cdot \text{sim})$）让边界样本同时拉动多个质心。

第一层之前还有一步特殊操作：先把所有行 L2 归一化，再**减掉全局均值方向上的投影**。
这一步相当于去掉"所有商品共有的语义偏置"，避免第 1 层的质心都被这个偏置占满。

### 4.2 代码走读：`sid_generation/prq_kmeans.py`

| 函数 | 作用 | 备注 |
|---|---|---|
| `sid_common.normalize_rows` | 行归一化 | 用 float64 算，零范数行保持 0 |
| `sid_common.projection_residual` | $r - \frac{r^\top c}{\lVert c\rVert^2}c$ 再归一化 | $c$ 接近 0 时不做投影（避免除零） |
| `PRQKMeans._cosine_scores` | 余弦打分 | `np.einsum(..., optimize=False)`：**固定归约顺序**，否则不同 batch 形状会改变边界样本的分配，导致结果不可复现 |
| `PRQKMeans.fit` | 逐层拟合 | 每层：软分配累加 `mass` / `sums` → `centroids = sums/mass`；空质心保留旧值（不随机重置） |
| `PRQKMeans._assign` | 硬分配 | `argmax` 余弦；平局取最小下标 |
| `PRQKMeans.encode` | 编码 | 复用同一套 normalize → 去均值 → 逐层投影残差 |
| `PRQKMeans.save/load` | NPZ | 存 `global_mean` + `codebook_i`，带 `encoding_revision` 版本号 |

### 4.3 实测

```bash
python sid_generation/prq_kmeans.py fit \
  --data-path data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy \
  --dataset-name Industrial_and_Scientific \
  --output-dir /tmp/prq_out \
  --codebook-sizes 256 256 256 --top-k 2 --beta 15 --n-iter 30 --seed 42
```

同一份 3,686 × 2,560 数据（Mac CPU，单线程 NumPy）：

| 指标 | 数值 |
|---|---|
| 拟合耗时 | 91.9 s（3 层 × 256 质心 × 30 轮） |
| 编码耗时 | 2.8 s |
| 原始唯一 SID | 2,999 / 3,686 |
| 原始冲突率 | **18.64%**（最大冲突组 10） |
| 导出后 | 3,686 个 **4-token** SID，唯一率 100%，词表 777 |

对比 FLASH 的 63.6%，PRQ 的语义位利用率高得多；对比自带 RQ-KMeans+ 的 ~0.4%，
它仍然明显更差。**这是"不需要 GPU、不需要调参"的合理代价，不是 bug。**

### 4.4 复用已拟合的量化器

```bash
python sid_generation/prq_kmeans.py encode \
  --data-path <新的 npy> --dataset-name <名字> \
  --model-path /tmp/prq_out/Industrial_and_Scientific.tokenizer.npz \
  --output-dir /tmp/prq_new
```

`encode` 不会重新拟合均值 / 质心，只用 frozen 模型；`--item-ids` 必须与拟合时一致，
否则行号与 item 的对应关系就错了。

---

## 5. RQ-OPQ：串行残差 + 并列子码

### 5.1 结构

```
x ──RQ 第1层──> r1 ──RQ 第2层──> r2 ──RQ 第3层──> r3
                                                 │
                                          OPQ: 旋转 R
                                                 │
                                       切成 S 段 → 每段 8-bit 子码
                                                 │
输出： <a_i><b_j><c_k><d_s1><e_s2>
        └─ 顺序残差（3） ─┘└ 并列子空间（2） ┘
```

- 前 3 个 token 是**顺序残差**：每层都在修正上一层。
- 后 2 个 token 是**并列**的：把残差旋转后切成 2 个子空间，各自做乘积量化，二者没有先后关系。
- 所以它不是"5 层残差"，而是"3 层残差 + 2 个正交子空间"。生成时仍然要按
  `<a><b><c><d><e>` 顺序 emit，因为自回归模型只会从左往右写。

### 5.2 代码走读：`sid_generation/rq_opq.py`

| 函数 | 作用 | 为什么这么写 |
|---|---|---|
| `RQOPQ._prefix` | 逐层 L2 最近质心 + 直接相减 | RQ 部分和 03 章一致，`block -= centroids[chosen]` |
| `RQOPQ.fit` | 训练 RQ + OPQ/PQ | 每层喂 FAISS 前先**在 float64 里中心化并缩放**：否则一个很大的公共偏移会让 float32 的 L2 训练失真 |
| `_rotate_stably` | 残差 × 旋转矩阵 | 固定 float64 归约顺序，避免不同 batch 大小跨过 PQ 判决边界 |
| `_check_faiss_magnitude` | 溢出检查 | float32 中间量可能溢出，超限直接报错要求缩放 embedding |
| `unpack_pq_codes` | 解 FAISS 的紧凑位打包 | 子码可能**跨字节**（例如 nbits=5），要手动移位 |
| `_validate_opq_index` | 校验载入的模型 | 旋转矩阵必须正交（分块算 Gram 矩阵），PQ 结构必须匹配元数据 |

### 5.3 命令行

```bash
pip install -r sid_generation/requirements-faiss.txt   # faiss-cpu==1.12.0

python sid_generation/rq_opq.py fit \
  --data-path data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy \
  --dataset-name Industrial_and_Scientific \
  --output-dir /tmp/opq_out \
  --rq-levels 3 --codebook-size 256 \
  --opq-subspaces 2 --nbits 8 \
  --rq-niter 25 --opq-niter 20 --pq-niter 25 --threads 1 --seed 42
```

| 参数 | 默认 | 约束 |
|---|---|---|
| `--rq-levels` | 3 | 决定顺序 token 数 |
| `--codebook-size` | 256 | ≥ 2 |
| `--opq-subspaces` | 2 | **embedding 维度必须能被整除**（2560 ÷ 2 = 1280 ✓，不整除会报错，代码不做 padding） |
| `--nbits` | 8 | 1–8；决定子码词表大小 $2^{\text{nbits}}$ |
| `--threads` | 1 | FAISS OpenMP 线程数；设为 1 才可复现 |

想和 3-token 方法对齐比较时，用 `--rq-levels 1 --opq-subspaces 2` 得到 3 个 token，
但上游明确说明这只是**预算对齐**，不是 OneSearch 的原始配置。

> 这一段命令我没有在本机跑（这台 Mac 没装 faiss）；
> 上游测试套件里 faiss 相关的 15 个用例同样被 skip。
> 首次使用时请先跑 `python -m unittest discover -s sid_generation/tests -v` 确认环境。

---

## 6. 身份 token：它不是语义

三个新工具都会在冲突时补一个身份 token，但规则不同：

| 工具 | 何时补 | token 形态 | 容量 |
|---|---|---|---|
| FLASH | **每个 item 都补**（无论如何） | `<d_rank>`，组内按 item id 字符串字典序排名 | `--suffix-size`，默认 256，超出报错 |
| PRQ / RQ-OPQ（`append`） | 全库只要**有一处**冲突，就给**所有** item 补 | `<d_rank>`（3 语义位后）/ `<f_rank>`（5 位后） | 由最大的冲突组决定 |
| PRQ / RQ-OPQ（`error`） | 有冲突就**拒绝导出** | — | — |
| PRQ / RQ-OPQ（`allow`） | 不补，允许重复 | — | 需要额外的 item 级消歧协议 |

为什么 `append` 要"全都补"而不是"只补冲突的那些"？
因为**长度一致**是约束解码的前提：`prefix_allowed_tokens_fn` 用 trie 按前缀过滤，
如果同一个数据集里既有 3-token 又有 4-token 的 item，第 4 层就要区分
"这里该结束了"和"这里还能继续"，实现复杂度陡增。03 章的老做法（只给冲突项加 rank）
就留下了这个不一致——04 章 §8.1 描述的正是它。

三个必须记住的性质：

1. **rank 是 catalog-local 的**。同一批 item 换行序不影响 rank（实现里用排序），
   但**增删任何 item 都可能改变 rank**。
2. **不能分片拼接**。分别对两个分片生成 index 再合并，rank 会撞车。
3. **必须整库重算**。这也是"换 SID = 换数据集版本"的原因。

---

## 7. 换一套 SID 会波及什么（上游 diff 解读）

这一节是"版本对齐"的重点：上游在合并 SID 生成器时，顺手改了三处下游代码。

| 文件 | 改动前 | 改动后 | 为什么 |
|---|---|---|---|
| `data.py`（7 个 Dataset） | `if len(sids) >= 3: sids[0]+sids[1]+sids[2]` | `if sids: ''.join(sids)` | 辅助任务原来只取前三层，第 4 个身份 token 被丢掉 → "SID→标题"的 label 和主任务的 `item_sid` 不一致 |
| `ts_rec_data.py`（2 个 Dataset） | 同上 | `if sids: "".join(sids)` | TS-Rec 分支同步 |
| `minionerec_trainer.py` | `self.test_lp_list = LogitsProcessorList([ccc])`（复用训练的那个实例） | 新建独立的 `ConstrainedLogitsProcessor(num_beams=self.test_beam)` | `ConstrainedLogitsProcessor` 的 `count` 是**实例状态**，训练/测试共用一个实例会串掉生成步数；且测试的 beam 数（`test_beam=20`）和训练的 `num_generations` 不同 |

第一处改动正好修掉了 04 章 §8.1 记的那个"两个任务标签不一致"的问题。
反过来看，这意味着：**如果你的 SID 只有 3 个 token，改动前后行为完全一样**（`''.join` 等价于前三层拼接）。

换 SID 之后必须重做的清单：

```text
[ ] .index.json / .item.json     新 SID 与 item 的映射
[ ] convert_dataset.py           重新生成 train/valid/test.csv + info.txt
[ ] 约束前缀 trie                  evaluate.py 从 index.json 现建，无需改文件，但要和新 index 一起用
[ ] SFT 训练                      词表里新增的 SID token 变了，必须重训
[ ] RL 训练                       必须从新 SFT checkpoint 出发
[ ] 评估 info 映射                 第 3 列 item_id 必须与新 index 对齐
[ ] 旧 checkpoint                 全部作废，不能混用
```

---

## 8. 实操：三条最短路径

环境（只跑 SID，不需要 torch / GPU）：

```bash
python3 -m venv .venv-sid && source .venv-sid/bin/activate
python -m pip install -r sid_generation/requirements.txt          # 只跑 PRQ
python -m pip install -r sid_generation/requirements-faiss.txt    # 跑 RQ-OPQ
# FLASH 只需要 numpy，仓库主环境的 requirements-macos.txt 也够
```

| 目标 | 命令 | 本机实测耗时 |
|---|---|---|
| 最快拿到一套 4-token SID | §3.3 的 FLASH 命令 | 0.08 s 编码 + 写文件 |
| 拿到质量更好的 CPU-only SID | §4.3 的 PRQ 命令 | 92 s 拟合 + 3 s 编码 |
| 复现 OneSearch 结构 | §5.3 的 RQ-OPQ 命令 | 需要 faiss，未实测 |

**没有自带 embedding 时，用合成数据也能跑通全流程**：

```python
import numpy as np
rng = np.random.default_rng(0)
centers = rng.normal(size=(50, 64))           # 50 个"语义簇"
rows = centers[rng.integers(0, 50, 2000)] + 0.1 * rng.normal(size=(2000, 64))
np.save("/tmp/synth.npy", rows.astype("float32"))
```

```bash
python rq/generate_indices_flash.py --embeddings /tmp/synth.npy \
  --output-dir /tmp/synth_flash --dataset Synth          # 默认行号即 item id
python sid_generation/prq_kmeans.py fit --data-path /tmp/synth.npy \
  --dataset-name Synth --output-dir /tmp/synth_prq --n-iter 10
```

验证生成器本身没坏：

```bash
python -B -m unittest discover -s tests -p "test_flash_sid*.py" -v      # 31 个用例
python -m unittest discover -s sid_generation/tests -v                  # 50 个用例（15 个 faiss 相关会 skip）
```

接回主流程（`convert_dataset.py` 需要同一个目录里同时有 `.item.json` 和 `*.inter`，
所以生成时建议带上 `--dataset-dir`，它会校验并把这两类文件一并复制到输出目录）：

```bash
python sid_generation/prq_kmeans.py fit \
  --data-path <emb.npy> --dataset-name Industrial_and_Scientific \
  --dataset-dir <含 .item.json 与 .inter 的处理后目录> \
  --output-dir /tmp/prq_out

python convert_dataset.py --dataset_name Industrial_and_Scientific \
  --data_dir /tmp/prq_out --output_dir /tmp/converted --category Industrial_and_Scientific --seed 42
```

---

## 9. 常见坑

| 现象 | 原因 | 处理 |
|---|---|---|
| FLASH 生成时 `--item-meta` 报 "requires exactly IDs 0..N-1" | `.item.json` 的 key 不是连续从 0 开始 | 改用 `--item-ids` 显式给出行顺序 |
| `Largest collision group has N items, exceeding suffix_size=256` | 哈希太集中 | 调大 `--suffix-size`，或换 PRQ / 更好的 embedding |
| 想让 FLASH 输出 32 位码却被拒 | `minionerec` 格式强制 3 位 | 用 `--format raw`，但它不产出 `.index.json` |
| PRQ 结果每次不一样 | 改了 `--seed` 或 `--batch-size`（**batch 会改变归约顺序**） | 固定 `seed` + `batch_size` + 单线程 |
| RQ-OPQ 报维度不能整除 | `D % opq_subspaces != 0` | 改 `--opq-subspaces`（如 2/4/5/8），代码不做自动 padding |
| RQ-OPQ 报 `values exceed the safe FAISS float32 range` | embedding 数值过大 | 先缩放 embedding，别指望它内部兜底 |
| 目录里已有旧输出 | 生成器默认拒绝覆盖 | `--overwrite`（会先校验再原子替换） |
| 把 `.raw.index.json` 喂给 `TokenExtender` | 那是分析用的原始码，没有身份位 | 只认 `.index.json` |
| 混用两套 SID 的中间产物 | token 语义完全不同，词表也不一样 | 一个数据集只保留一套 SID 目录 |

---

## 10. 自检问题

1. 为什么 SimHash 的 `codebook_size` 必须是 2 的幂？
2. FLASH 的三层码为什么冲突率高达 63%，而 PRQ 只有 18%？差别在"有没有码本"还是"有没有残差"？
3. PRQ 的 `projection_residual` 与 RQ 的"减质心"在几何上差在哪？什么时候两者等价？
4. RQ-OPQ 的 5 个 token 里，哪几个是顺序的、哪几个是并列的？为什么并列的仍然要按顺序 emit？
5. `append` 策略为什么给**所有** item 加身份位，而不是只加给冲突的？
6. 上游把 `len(sids) >= 3` 改成 `''.join(sids)` 之后，对 3-token 的旧数据集有影响吗？
7. 如果你要把 `prefix_allowed_tokens_fn` 接到一套 4-token SID 上，`prefix_index` 需要改吗？（提示：它指的是 prompt 末尾的 marker 长度，不是 SID 长度，见 05 章）

---

## 11. 下一步

- 想弄清"一条样本怎么变成梯度" → [11 实操手册](./11-hands-on.md) → [04 SFT](./04-sft.md)
- 想看约束解码如何吃下不同长度的 SID → [06 评估](./06-evaluation.md)
- 想从论文角度对照 → `rq/FLASH_SID.md`、`sid_generation/README.md`、`sid_generation/VALIDATION.md`
