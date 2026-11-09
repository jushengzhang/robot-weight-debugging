# 大语言模型能通过修改权重调试机器人吗？

[论文](https://arxiv.org/abs/2611.09376) · [English](README.md) · [复现指南](docs/REPRODUCING.md) · [65 行稀疏修改](edits/pi05-language-c) · [逐回合证据](evidence) · [16 次实验记录](transcripts)

**GPT-6 Astra 读取机器人策略的权重，结合有限次数的模拟器反馈，直接修改 65 行词嵌入，修复一类词汇理解失败。编辑过程不做梯度更新，也不使用新的示范数据。**

Jusheng Zhang¹ · Keze Wang² · Liang Lin²  
¹ Stanford University · ² X-Era AI

![预留语言变体上的修复结果](assets/pi05-teaser.svg)

最好的 π₀.₅ 编辑在 765 条预留的 LIBERO-Plus 改写指令上，将成功率从 **86.3% 提高到 92.0%**，
修好 48 个回合，同时使 4 个原本成功的回合失败。精确配对检验 *p* = 1.31 × 10⁻¹⁰。
含有被改 token 的 335 条指令从 **74.3% 提高到 87.5%**；其余 **430 个回合的成败结果完全不变**。
这里说的是成败结果，不是整条轨迹逐值相同。

| 独立编辑实验 | 修改前 | 修改后 | 提升 |
| --- | ---: | ---: | ---: |
| A | 660 / 765 | 684 / 765 | +3.1 个百分点 |
| B | 660 / 765 | 673 / 765 | +1.7 个百分点 |
| C | 660 / 765 | 704 / 765 | +5.8 个百分点 |

三次实验使用同一组预留样本。随机方向对照没有带来收益；按编辑器自己的词表做整行替换能解释大部分增益，
但产生更多退化。整行替换与编辑器的直接比较在 0.05 水平上不显著，论文保留了两种错误计数口径。

另外 13 次编辑实验中，编辑器主动放弃了 12 个候选；唯一保留的编辑没有显著的预留评测收益。
独立的后缀实验还发现：评测元数据让 FastWAM 的成功率从 70.0% 降到 48.6%，
而被测 VLA 的总体成功率不变。这些结果支持的是精确的词汇修复，不是普遍获得新的机器人能力。

## 不用 GPU，先核对论文数字

```bash
git clone https://github.com/jushengzhang/robot-weight-debugging.git
cd robot-weight-debugging
python3 evidence/audit_new_results.py
python3 -m pip install 'numpy>=1.26,<3'
python3 evidence/extended/audit_extended.py
```

第一个脚本只依赖 Python 标准库，重新计算主结果、精确配对检验、命中 token 的子集、对照实验和后缀影响。
第二个脚本核对补充材料中的种子、局部性及范数／方向证据。

## 应用实际评测过的修改

`edits/pi05-language-c/` 提供实际评测过的 65 行原值、修改后数值、token 配方及应用脚本。
脚本会先核对原行值，拒绝不兼容的基座，不改写原检查点。生成完整替换矩阵约需 2.11 GB 空间。

```bash
python3 edits/pi05-language-c/apply_sparse_rows.py \
  /path/to/pi05_libero_flat.safetensors --dry-run
python3 edits/pi05-language-c/apply_sparse_rows.py \
  /path/to/pi05_libero_flat.safetensors --output /path/to/embed_override.npy
```

完整模型、模拟器和第三方资源需按上游说明获取。
[复现指南](docs/REPRODUCING.md) 说明了检查点转换、路径配置及运行入口；
`experiments/` 保存的是研究使用的历史源码，不是一键安装的仿真容器。
配置工具可以将其中的机器路径映射到新目录，并补齐共享 harness 文件。

## 如何理解归档文件

第一轮 FastWAM 实验中，收尾代码没有执行，导致部分编辑虽然被编辑器要求撤回，仍留在归档中。
**有文件不等于最终保留。** 当前 driver 已包含收尾修复，论文和记录区分了归档候选与最终决定。
后缀跟进实验的负结果和 LIBERO-PRO 的不确定结果同样保留。

本次开源对应 [arXiv:2611.09376v1](https://arxiv.org/abs/2611.09376v1)，不包含 RoboDojo 提升结论。
引用格式见 [英文 README](README.md#citation)。作者原创代码采用 [Apache-2.0](LICENSE)；
基座模型、派生权重、数据和论文沿用各自适用条款，见 [第三方说明](THIRD_PARTY_NOTICES.md)。
