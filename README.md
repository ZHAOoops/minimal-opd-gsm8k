# Minimal OPD GSM8K

这是一个最小化的 On-Policy Distillation 实验，用来验证 Thinking Machines Lab 关于 OPD 的核心想法。

本项目不是官方实现，也不追求完整复现官方结果。目标是用尽量少的代码，在 GSM8K 上观察到 OPD 的有效训练信号。

## 核心思路

训练流程：

1. student 根据当前策略自己生成答案。
2. teacher 不生成答案，只在 student 生成的轨迹上提供 token-level 分布。
3. teacher 和 student 都看同一段 prompt + student completion。
4. 只在 completion token 上计算 KL。
5. 更新 student。
6. 不使用 GSM8K 标准答案作为训练 label。
7. 不使用 teacher 生成的离线数据做 SFT。

当前默认 loss：

    KL(student || teacher)

也就是 reverse KL。

## 实验环境

本次实验环境：

    GPU: NVIDIA RTX 4090 48GB * 1
    CPU: 12 核
    内存: 64GB
    PyTorch: 2.6.0 + CUDA 12.6
    Teacher: Qwen/Qwen2.5-3B-Instruct
    Student: Qwen/Qwen2.5-0.5B
    Dataset: GSM8K
    Prompt: Qwen chat template

## 安装依赖

建议先设置 HuggingFace 缓存目录：

    export HF_HOME=/root/yijia-tmp/hf
    export HF_DATASETS_CACHE=/root/yijia-tmp/hf/datasets
    export TOKENIZERS_PARALLELISM=false

如果 HuggingFace 访问慢，可以使用镜像：

    export HF_ENDPOINT=https://hf-mirror.com

安装依赖：

    pip install -U -r requirements.txt

## 文件说明

    train_opd.py       OPD 训练脚本
    eval_gsm8k.py      GSM8K 评测脚本
    requirements.txt   Python 依赖
    README.md          说明文档

## 重要实现细节

### 1. 使用 chat template

Qwen2.5-3B-Instruct 如果不用 chat template，在 GSM8K 上表现很差。

因此本项目训练和评测都统一使用 Qwen chat template。

### 2. 只在 completion 上计算 KL

prompt 部分不参与 loss。

completion 起始位置为 prompt_len。CausalLM logits 需要做 shift：

    logits[:, prompt_len - 1 : -1]

### 3. student on-policy 生成

每个训练 step 都使用当前 student 生成 completion，再让 teacher 在这条 student 轨迹上提供分布信号。

这和普通 teacher 离线生成答案再 SFT 不一样。

### 4. 截断答案后的尾部噪声

GSM8K 要求 final answer 放在：

    #### number

模型有时已经生成 final answer，但后面继续啰嗦或重复。

当前代码支持：

    --truncate-after-answer

开启后，如果 student 自己生成了类似：

    #### 123

就截断 completion，并追加 Qwen 的 <|im_end|> token，减少尾部噪声。

### 5. 评测答案抽取

评测时优先抽取第一个完整的：

    #### number

这样可以避免模型重复输出 final answer 时解析失败。

## 实验结果

以下结果均为 GSM8K test 前 300 条。

最终公平对比使用同一个最终版 eval_gsm8k.py。

| 设置 | 正确数 | Acc |
|---|---:|---:|
| Qwen2.5-0.5B baseline | 119 / 300 | 0.3967 |
| OPD trunc-stop320 step250 | 139 / 300 | 0.4633 |

绝对提升：

    0.4633 - 0.3967 = +0.0666

也就是 300 条里多答对 20 题。

这说明在这个小规模实验中，我们观察到了 OPD 的有效训练信号。

## 历史探索记录

| 实验 | 结果 | 备注 |
|---|---:|---|
| Qwen2.5-3B-Instruct，plain prompt，完整 GSM8K test | 0.2980 | 未正确激活 instruct 能力，不采用 |
| Qwen2.5-3B-Instruct，chat template，前 50 条 | 0.7800 | teacher prompt 正常 |
| Qwen2.5-0.5B，chat template，前 100 条 | 0.3500 | student baseline sanity check |
| OPD smoke，20 step，前 100 条 | 0.4200 | 只用于检查训练链路 |
| OPD len192，1000 step，前 300 条 | 0.4167 | 有提升，但 completion 经常被截断 |
| 旧版 trunc320 step250，前 300 条 | 0.4933 | 有提升，但容易重复 final answer，不作为最终推荐版本 |
| OPD trunc-stop320 step250，前 300 条 | 0.4633 | 当前推荐版本 |

## 可执行命令

下面是本次实验用到的主要命令。

### 1. Teacher sanity check

    python eval_gsm8k.py \
      --model Qwen/Qwen2.5-3B-Instruct \
      --limit 50 \
      --batch-size 4 \
      --max-new-tokens 512 \
      --chat-template

本次结果：

    correct: 39
    total: 50
    acc: 0.7800

### 2. Student baseline，前 300 条

    python eval_gsm8k.py \
      --model Qwen/Qwen2.5-0.5B \
      --limit 300 \
      --batch-size 8 \
      --max-new-tokens 512 \
      --chat-template

本次最终结果：

    correct: 119
    total: 300
    acc: 0.3967

### 3. OPD smoke test

只跑 20 step，用来确认代码链路没问题。

    rm -rf /root/yijia-tmp/checkpoints/opd_smoke

    PYTHONUNBUFFERED=1 python -u train_opd.py \
      --teacher-model Qwen/Qwen2.5-3B-Instruct \
      --student-model Qwen/Qwen2.5-0.5B \
      --output-dir /root/yijia-tmp/checkpoints/opd_smoke \
      --train-limit 64 \
      --max-train-steps 20 \
      --grad-accum-steps 1 \
      --max-new-tokens 128 \
      --max-seq-len 512 \
      --lr 1e-5 \
      --kl-direction rkl \
      --log-every 1 \
      --sample-every 5

评测：

    python eval_gsm8k.py \
      --model /root/yijia-tmp/checkpoints/opd_smoke \
      --limit 100 \
      --batch-size 8 \
      --max-new-tokens 512 \
      --chat-template

本次结果：

    correct: 42
    total: 100
    acc: 0.4200

### 4. OPD len192，1000 step

这个版本不做答案截断，max_new_tokens=192。

    rm -rf /root/yijia-tmp/checkpoints/opd_1k

    PYTHONUNBUFFERED=1 python -u train_opd.py \
      --teacher-model Qwen/Qwen2.5-3B-Instruct \
      --student-model Qwen/Qwen2.5-0.5B \
      --output-dir /root/yijia-tmp/checkpoints/opd_1k \
      --train-limit 1000 \
      --max-train-steps 1000 \
      --grad-accum-steps 1 \
      --max-new-tokens 192 \
      --max-seq-len 768 \
      --lr 1e-5 \
      --kl-direction rkl \
      --log-every 10 \
      --sample-every 100 \
      2>&1 | tee /root/yijia-tmp/logs/opd_1k.log

评测：

    python eval_gsm8k.py \
      --model /root/yijia-tmp/checkpoints/opd_1k \
      --limit 300 \
      --batch-size 8 \
      --max-new-tokens 512 \
      --chat-template

本次结果：

    correct: 125
    total: 300
    acc: 0.4167

### 5. OPD trunc-stop320，推荐版本

这个版本使用：

    max_new_tokens = 320
    max_seq_len = 1024
    --truncate-after-answer

当检测到 student 自己生成了 #### number 后，截断并追加 <|im_end|>。

    rm -rf /root/yijia-tmp/checkpoints/opd_trunc_stop320

    PYTHONUNBUFFERED=1 python -u train_opd.py \
      --teacher-model Qwen/Qwen2.5-3B-Instruct \
      --student-model Qwen/Qwen2.5-0.5B \
      --output-dir /root/yijia-tmp/checkpoints/opd_trunc_stop320 \
      --train-limit 1000 \
      --max-train-steps 500 \
      --grad-accum-steps 1 \
      --max-new-tokens 320 \
      --max-seq-len 1024 \
      --lr 1e-5 \
      --kl-direction rkl \
      --truncate-after-answer \
      --log-every 10 \
      --sample-every 100 \
      --save-every 250 \
      2>&1 | tee /root/yijia-tmp/logs/opd_trunc_stop320.log

评测 step250：

    python eval_gsm8k.py \
      --model /root/yijia-tmp/checkpoints/opd_trunc_stop320/step_250 \
      --limit 300 \
      --batch-size 8 \
      --max-new-tokens 512 \
      --chat-template

本次结果：

    correct: 139
    total: 300
    acc: 0.4633

## 后续扩量实验

如果想进一步扩量，可以使用下面命令。

### 1. 全量 GSM8K train，1 epoch

    rm -rf /root/yijia-tmp/checkpoints/opd_full_trunc_stop320

    PYTHONUNBUFFERED=1 python -u train_opd.py \
      --teacher-model Qwen/Qwen2.5-3B-Instruct \
      --student-model Qwen/Qwen2.5-0.5B \
      --output-dir /root/yijia-tmp/checkpoints/opd_full_trunc_stop320 \
      --num-epochs 1 \
      --grad-accum-steps 1 \
      --max-new-tokens 320 \
      --max-seq-len 1024 \
      --lr 1e-5 \
      --kl-direction rkl \
      --truncate-after-answer \
      --log-every 20 \
      --sample-every 200 \
      --save-every 1000 \
      2>&1 | tee /root/yijia-tmp/logs/opd_full_trunc_stop320.log

完整 GSM8K test 评测：

    python eval_gsm8k.py \
      --model /root/yijia-tmp/checkpoints/opd_full_trunc_stop320 \
      --batch-size 8 \
      --max-new-tokens 512 \
      --chat-template

### 2. 尝试 forward KL

默认是 reverse KL：

    --kl-direction rkl

也可以尝试 forward KL：

    rm -rf /root/yijia-tmp/checkpoints/opd_1k_fkl

    PYTHONUNBUFFERED=1 python -u train_opd.py \
      --teacher-model Qwen/Qwen2.5-3B-Instruct \
      --student-model Qwen/Qwen2.5-0.5B \
      --output-dir /root/yijia-tmp/checkpoints/opd_1k_fkl \
      --train-limit 1000 \
      --max-train-steps 1000 \
      --grad-accum-steps 1 \
      --max-new-tokens 320 \
      --max-seq-len 1024 \
      --lr 1e-5 \
      --kl-direction fkl \
      --truncate-after-answer \
      --log-every 10 \
      --sample-every 100 \
      --save-every 250 \
      2>&1 | tee /root/yijia-tmp/logs/opd_1k_fkl.log

评测：

    python eval_gsm8k.py \
      --model /root/yijia-tmp/checkpoints/opd_1k_fkl/step_250 \
      --limit 300 \
      --batch-size 8 \
      --max-new-tokens 512 \
      --chat-template

### 3. 尝试更小学习率

如果发现模型重复 final answer 或格式变差，可以尝试 lr=5e-6：

    rm -rf /root/yijia-tmp/checkpoints/opd_1k_lr5e6

    PYTHONUNBUFFERED=1 python -u train_opd.py \
      --teacher-model Qwen/Qwen2.5-3B-Instruct \
      --student-model Qwen/Qwen2.5-0.5B \
      --output-dir /root/yijia-tmp/checkpoints/opd_1k_lr5e6 \
      --train-limit 1000 \
      --max-train-steps 1000 \
      --grad-accum-steps 1 \
      --max-new-tokens 320 \
      --max-seq-len 1024 \
      --lr 5e-6 \
      --kl-direction rkl \
      --truncate-after-answer \
      --log-every 10 \
      --sample-every 100 \
      --save-every 250 \
      2>&1 | tee /root/yijia-tmp/logs/opd_1k_lr5e6.log

## 注意事项

1. 本项目只是小规模 sanity check，不是官方复现。
2. 当前主要报告 GSM8K test 前 300 条，不代表完整 benchmark。
3. 结果会受 prompt、seed、答案抽取、生成长度影响。
4. 模型有时仍可能重复 final answer，本项目重点是验证 OPD 训练信号，不是打磨最终模型。
5. 不建议把 checkpoints 直接提交到 GitHub。
