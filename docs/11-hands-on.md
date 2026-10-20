# 11 · 实操手册：从零跑通一条最小流水线

| | |
|---|---|
| **读这篇前** | [02 数据管线](./02-data-pipeline.md)（至少知道 `.npy / .item.json / .index.json / .csv` 分别是什么） |
| **读完能做到** | 在自有硬件上把「数据 → SID → 训练 CSV → SFT → RL → 评估」串起来；知道每一步在哪个文件、失败时先看哪里 |
| **对应源码/脚本** | `sft.sh`、`rl.sh`、`evaluate.sh`、`convert_dataset.sh`、`rq/`、`sid_generation/` |
| **预计耗时** | 只跑到"生成 SID + 校验数据"约 30 分钟；跑通 SFT 另需数小时（取决于硬件） |

> 这篇是**操作手册**，不解释原理。原理见 01–10；查函数见 [08 源码地图](./08-code-map.md)。
> 所有命令默认在仓库根目录执行。

---

## 1. 先确认你在哪一档硬件

| 档位 | 能做什么 | 不能做什么 | 建议路线 |
|---|---|---|---|
| **只有 CPU（含 Apple Silicon）** | 读代码、跑数据管线、生成 SID、跑指标计算、跑单测 | SFT / RL（`sft.py`、`rl.py` 都硬编码 `bf16=True`） | §4 → §5 → §10 → §12 |
| **单卡 16–24 GB** | 小模型 SFT（`freeze_LLM=True`）、极小的 RL | 8 卡默认配置照抄 | §5 → §6 → §7 → §9 |
| **8 卡（论文配置）** | 全部 | — | §6 的默认脚本 |

先把这张表记住：**这个仓库默认是为 8 卡 + bf16 写的**。
`torchrun --nproc_per_node 8`、`accelerate --num_processes 8`、DeepSpeed ZeRO-2 都写死在脚本里。

---

## 2. 环境准备

### 2.1 Linux + NVIDIA（完整训练）

```bash
conda create -n MiniOneRec python=3.10 -y
conda activate MiniOneRec
pip install -r requirements.txt      # 冻结的 Linux + CUDA 11.8 环境
```

### 2.2 macOS / Apple Silicon（只读代码、跑数据与 SID）

`requirements.txt` 在 macOS 上**装不上**（`torchrec`、`fbgemm_gpu`、`triton`、`nvidia-*`
只有 Linux wheel；`bitsandbytes` 0.49.0 之前没有 macOS arm64 wheel）。用精简集合：

```bash
conda create -n MiniOneRec-mac python=3.11 -y
conda activate MiniOneRec-mac
pip install -r requirements-macos.txt
```

### 2.3 只跑 SID 生成器（不需要 torch）

```bash
python3 -m venv .venv-sid && source .venv-sid/bin/activate
python -m pip install -r sid_generation/requirements.txt          # PRQ-KMeans
python -m pip install -r sid_generation/requirements-faiss.txt    # 追加 RQ-OPQ
```

FLASH-SimHash 只需要 `numpy`，上面任一环境都能跑。

---

## 3. 仓库自带的数据长什么样

`data/Amazon/` 里已经放好了**转换后的成品**（不需要下载 Amazon 原始数据就能开始学习）：

| 文件 | 内容 | 实测规模（Industrial_and_Scientific） |
|---|---|---|
| `train/*.csv` / `valid/*.csv` / `test/*.csv` | 7 列训练样本 | 36,259 / 4,532 / 4,533 行 |
| `info/*.txt` | `SID \t title \t item_id` | 3,686 行 |
| `index/*.index.json` | item_id → SID token 列表 | 3,686 个 item，全部 3 token；3,670 个唯一 SID（最大冲突组 3） |
| `index/*.item.json` | item_id → title/description/brand/categories | 3,686 个 item |
| `index/*.emb-qwen-td.npy` | 文本 embedding | `(3686, 2560)`，float16 |

CSV 的表头是：

```text
user_id,history_item_title,item_title,history_item_id,item_id,history_item_sid,item_sid
```

先花两分钟确认字段对齐（这是后面所有报错的根源）：

```bash
head -2 data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv
head -2 data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt
python -c "import json;d=json.load(open('data/Amazon/index/Industrial_and_Scientific.index.json'));print(len(d), list(d.items())[:2])"
```

---

## 4. 阶段 1：文本 embedding（可选，耗时数小时）

只有当你**换了数据集**才需要。脚本：`rq/text2emb/amazon_text2emb.py`（用 Accelerate 多卡）。

```bash
cd rq/text2emb
accelerate launch --num_processes 8 amazon_text2emb.py \
  --dataset Industrial_and_Scientific \
  --root ../../data/Amazon18/Industrial_and_Scientific \
  --plm_checkpoint <你的文本编码模型路径>
cd ../..
```

产出 `<Dataset>.emb-qwen-td.npy`。要求：**行顺序 = item id 顺序（0..N-1）**，
且不要包含 padding 行。SID 生成器会按这个约定对齐 item。

---

## 5. 阶段 2：生成 SID（4 分钟到 2 分钟）

三条路，选一条即可：

```bash
# A. 复用仓库自带的 SID（跳过本阶段）
ls data/Amazon/index/Industrial_and_Scientific.index.json

# B. FLASH-SimHash：零训练，最快，冲突率最高（4 token）
python rq/generate_indices_flash.py \
  --embeddings data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy \
  --item-meta data/Amazon/index/Industrial_and_Scientific.item.json \
  --output-dir /tmp/sid/flash --dataset Industrial_and_Scientific

# C. PRQ-KMeans：CPU 拟合约 1.5 分钟，冲突率明显更低（4 token）
python sid_generation/prq_kmeans.py fit \
  --data-path data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy \
  --dataset-name Industrial_and_Scientific \
  --output-dir /tmp/sid/prq --seed 42
```

实测结果（本机 Mac CPU，3,686 items）：

| 方法 | 编码/拟合耗时 | 原始冲突率 | 导出后 token 数 |
|---|---|---|---|
| FLASH-SimHash | 0.08 s | 63.56% | 4 |
| PRQ-KMeans | 92 s + 2.8 s | 18.64% | 4 |
| 自带 index（RQ-KMeans+） | — | ≈0.4% | 3 |

细节与原理见 [10 新 SID 构造](./10-sid-generators.md)。

---

## 6. 阶段 3：转换成训练 CSV

`convert_dataset.py` 需要的输入是 **同一目录下**的 `.item.json` + `.index.json` + 三个 `.inter`：

```bash
# 目录结构要求
# <data_dir>/<Dataset>.item.json
# <data_dir>/<Dataset>.index.json          <- 第 5 阶段生成
# <data_dir>/<Dataset>.{train,valid,test}.inter   <- amazon18_data_process.py 生成

python convert_dataset.py \
  --dataset_name Industrial_and_Scientific \
  --data_dir /tmp/sid/prq \
  --output_dir /tmp/converted \
  --category Industrial_and_Scientific \
  --seed 42
```

两个提示：

- 用 FLASH 生成时它只写 index 系列文件，**不会**复制 `.item.json` / `.inter`；需要自己拷进同一目录。
- 用 `sid_generation` 的 `fit` 时加 `--dataset-dir <处理后目录>`，它会做一致性校验并把
  `.item.json` 与可用的 `.inter` 一并复制到输出目录，省掉手工拷贝。

产出 `<output_dir>/{train,valid,test,info}/...`。
**注意：仓库自带的 `data/Amazon/` 只有 CSV，没有 `.inter`**，想重跑这一步必须先用
`data/amazon18_data_process.py` 从原始 review/metadata 生成 `.inter`：

```bash
cd data && python amazon18_data_process.py --dataset Industrial_and_Scientific \
  --user_k 5 --item_k 5 --st_year 1996 --st_month 10 --ed_year 2018 --ed_month 10 \
  --output_path ./Amazon18 && cd ..
```

`.inter` 的格式（三列 TSV，表头必须在）：

```text
user_id:token	item_id_list:token_seq	item_id:token
42	17 5 91 3	88
```

> 只想验证格式时，可以手写一个 3 行 `.inter` + 一个小 `.item.json` 跑通 `convert_dataset.py`，
> 不必下载完整 Amazon 数据。

---

## 7. 阶段 4：SFT 最小跑通

```bash
bash sft.sh     # 8 卡默认配置：torchrun --nproc_per_node 8
```

单卡/小显存的最小改动（只动这几个参数）：

| 参数 | 默认 | 单卡建议 | 说明 |
|---|---|---|---|
| `--nproc_per_node` | 8 | 1 | 进程数 |
| `--micro_batch_size` | 16 | 1–4 | 每卡每步样本数，直接吃显存 |
| `--batch_size` | 1024 | `micro_batch_size × 累积步数` | **不直接等于全局 batch**，它只用来算 `gradient_accumulation_steps = batch_size // micro_batch_size` |
| `--freeze_LLM` | False | True（显存不足时） | 只训新增 SID token 的 embedding，显存需求骤降 |
| `--num_epochs` | 10 | 1–2（试跑） | |

真正的全局 batch = `进程数 × micro_batch_size × gradient_accumulation_steps`。
按默认 8 × 16 × 64 = 8192，**不是** 1024——这点最容易误判学习率。

SFT 需要的关键输入（其它参数见 `sft.py` 顶部的 `train()` 签名）：

```text
--base_model        Qwen2.5 系列 base 模型路径
--train_file        data/Amazon/train/<Dataset>*.csv
--eval_file         data/Amazon/valid/<Dataset>*11.csv
--info_file         不需要（SFT 不用）
--sid_index_path    <Dataset>.index.json
--item_meta_path    <Dataset>.item.json
--category          Industrial_and_Scientific | Office_Products | ...
```

**判断跑没跑通**：日志里出现 `LOAD DATA FINISHED` → 开始有 `loss` 且 eval 的 `eval_loss` 下降。
如果 loss 一直是常数，先查 `--category` 是否在 `category_dict` 里（只有 5 个合法值）。

---

## 8. 阶段 5：RL 最小跑通

```bash
bash rl.sh     # accelerate + DeepSpeed ZeRO-2 + 8 进程
```

RL 比 SFT 多三个"吃显存"的地方：参考模型、16 个 `num_generations` 的 beam、`max_completion_length=128`。
最小化配置（改 `rl.sh` 里的数字）：

| 参数 | 默认 | 试跑建议 |
|---|---|---|
| `--num_processes` | 8 | 1–2 |
| `--train_batch_size` | 64 | 必须能被 `num_generations` 整除 |
| `--num_generations` | 16 | 4（reward 组内归一化仍成立） |
| `--beam_search` | True | 保持 True，否则 reward 里的排序惩罚失去意义 |
| `--test_during_training` | False | 试跑时保持 False（省时间） |
| `--reward_type` | ranking | `rule` 也可；`sasrec`/`semantic` 需要额外 `--cf_path`/`--ada_path` |

需要注意的两点：

- `rl.sh` 用的是 `--model_path <SFT checkpoint>`，**不是** base model；
- `rl.py` 里没有 vLLM 开关，`GRPOConfig` 的 `use_vllm` 默认 `False`，走的是 HF `generate`。

---

## 9. 阶段 6：评估

`evaluate.sh` 的流程是：`split.py` 把测试集切成 N 份 → 每卡一个 `evaluate.py` → `merge.py` 合并 → `calc.py` 算指标。

```bash
bash evaluate.sh   # 8 卡，num_beams=50
```

单卡等价流程（把 8 改成 1）：

```bash
python split.py --input_path data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv \
                --output_path ./temp/eval --cuda_list "0"
CUDA_VISIBLE_DEVICES=0 python -u evaluate.py \
  --base_model <你的模型路径> \
  --info_file data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
  --category Industrial_and_Scientific \
  --test_data_path ./temp/eval/0.csv \
  --result_json_data ./temp/eval/0.json \
  --batch_size 8 --num_beams 50 --max_new_tokens 256 --length_penalty 0.0
python merge.py --input_path ./temp/eval --output_path ./results/final.json --cuda_list "0"
python calc.py --path ./results/final.json --item_path data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt
```

**先看 `CC`**：它统计生成了非法 SID 的次数。`CC` 不为 0 说明约束解码没生效，
此时 HR/NDCG 没有参考价值（常见于 Qwen Instruct + 新版 transformers）。
省资源调试时把 `--num_beams` 降到 5–10、`--batch_size` 降到 1。

---

## 10. 纯 CPU 能做什么

| 能做 | 怎么做 |
|---|---|
| 数据管线全链路 | `amazon18_data_process.py` → `text2emb`（CPU 可跑但慢） → SID → `convert_dataset.py` |
| 生成 / 评估 SID 质量 | FLASH、PRQ 的 `metrics.json` / `sid_stats.json` |
| 跑单测 | §12 的两条 `unittest` 命令 |
| 手工还原一条训练样本 | [08 §4.1](./08-code-map.md) |
| 手算 HR/NDCG/advantage | [08 §4.3、§4.4](./08-code-map.md) |
| 跑 `calc.py`（指标计算） | 只需要已有 `result.json` |

| 不能做 | 原因 |
|---|---|
| SFT | `bf16=True` 硬编码，且需要把 0.5B+ 模型放进显存 |
| RL | 同上 + 需要参考模型与多 beam |
| `evaluate.py` | 需要真实模型前向 |

---

## 11. 排错速查

| 症状 | 原因 | 处理 |
|---|---|---|
| `KeyError` / `item not in item2id` | `.index.json` 与 `info.txt` 不是同一次构建 | 两者必须配套重生成 |
| 训练 loss 不降 | `--category` 拼错（不在 `category_dict` 的 5 个值里） | 用 `Industrial_and_Scientific` 这类精确值 |
| `eval()` 报 `NameError` | CSV 里的 list 字符串被改动过 | 检查 CSV 生成流程，不要手改引号 |
| `CUDA out of memory`（SFT） | `micro_batch_size` 太大 | 降到 1–4，或 `--freeze_LLM True` |
| `CUDA out of memory`（RL） | `num_generations` / 长 completion | 降到 4；关 `test_during_training` |
| `convert_dataset.py` 找不到 `*.inter` | 自带数据只有 CSV | 先跑 `amazon18_data_process.py`，或自造小 `.inter` |
| `FileExistsError: Outputs already exist` | SID 生成器保护已有产物 | 换目录，或显式 `--overwrite` |
| `Largest collision group ... exceeding suffix_size` | SID 冲突组太大 | 调大 `--suffix-size`，或换 PRQ |
| `ModuleNotFoundError: faiss` | RQ-OPQ 需要 FAISS | `pip install -r sid_generation/requirements-faiss.txt` |
| `HTTP 401 / 404` 拉模型 | HF 需要 token 或镜像 | `export HF_ENDPOINT=https://hf-mirror.com` + `huggingface-cli login` |
| 评估 `CC` 很高 | 约束解码失效 | 换 base 模型；核对 `info.txt` 与 index 配套 |

---

## 12. 命令速查卡

```bash
# ---- 环境 ----
pip install -r requirements.txt                 # Linux + CUDA
pip install -r requirements-macos.txt           # macOS 只读/调试
python -m venv .venv-sid && source .venv-sid/bin/activate
pip install -r sid_generation/requirements-faiss.txt

# ---- 数据 ----
python data/amazon18_data_process.py --dataset Industrial_and_Scientific --output_path ./data/Amazon18
accelerate launch --num_processes 8 rq/text2emb/amazon_text2emb.py --dataset <DS> --root <data_dir> --plm_checkpoint <model>
python rq/generate_indices_flash.py --embeddings <emb.npy> --item-meta <item.json> --output-dir <out> --dataset <DS>
python sid_generation/prq_kmeans.py fit --data-path <emb.npy> --dataset-name <DS> --output-dir <out>
python convert_dataset.py --dataset_name <DS> --data_dir <sid_dir> --output_dir <csv_dir>

# ---- 训练 ----
bash sft.sh
bash rl.sh

# ---- 评估 ----
bash evaluate.sh
python calc.py --path results/final.json --item_path data/Amazon/info/<DS>*.txt

# ---- 自检 ----
python -B -m unittest discover -s tests -p "test_flash_sid*.py" -v
python -m unittest discover -s sid_generation/tests -v
```

---

## 13. 下一步

- 想搞清 SID 生成器为什么这么设计 → [10 新 SID 构造](./10-sid-generators.md)
- 想搞清训练目标 → [04 SFT](./04-sft.md)、[05 RL](./05-rl.md)
- 想按调用链读源码 → [08 源码地图](./08-code-map.md)
- 一句话版的完整流程 → [09 RL 生成全过程](./09-rl-generation-walkthrough.md)
