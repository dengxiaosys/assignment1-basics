# CS336 Assignment 1 笔记阅读顺序

## 编号规则

文件名采用 `章_节_主题.md`：

| 章号 | 主题 |
|---|---|
| `00` | 环境、测试与代码组织 |
| `01` | BPE 与语料编码 |
| `02` | Transformer 模型 |
| `03` | 损失、优化与训练循环 |
| `04` | checkpoint 推理与文本生成 |
| `90` | 独立运维附录 |

推荐按章号、节号依次阅读。handout 提取稿用于查题，不必从头精读。

## 00 准备与参考

1. [00_01 作业 handout 提取稿](./00_01_cs336_assignment1_basics_extracted.md)：题目原文与接口要求，作为索引和参考。
2. [00_02 pyproject.toml 详解](./00_02_pyproject_toml_explained.md)：项目、依赖、uv、pytest、ruff 配置。
3. [00_03 pytest fixtures 详解](./00_03_pytest_fixtures_explained.md)：理解测试如何向 adapters 注入数据与参考结果。
4. [00_04 代码组织与模块职责](./00_04_code_organization_notes.md)：当前 `cs336_basics` 的模块划分和依赖方向。

## 01 BPE 与语料编码

BPE 笔记集中在 [`bpe/`](./bpe/) 子目录：

1. [01_01 BPE 分词器原理](./bpe/01_01_bpe_tokenizer.md)
2. [01_02 Byte-level BPE 手工复算](./bpe/01_02_byte_level_bpe_worked_example.md)
3. [01_03 BPE 非编码问题与背景](./bpe/01_03_bpe_conceptual_questions_notes.md)
4. [01_04 train_bpe 实现](./bpe/01_04_train_bpe_implementation_notes.md)
5. [01_05 TinyStories BPE 实验](./bpe/01_05_train_bpe_tinystories_experiment_notes.md)
6. [01_06 Tokenizer 编解码实现](./bpe/01_06_tokenizer_implementation_notes.md)
7. [01_07 语料预编码](./bpe/01_07_encode_corpus_implementation_notes.md)

## 02 Transformer 模型

1. [02_01 nn.Module 与 Linear 背景](./02_01_nn_module_and_linear_explained.md)
2. [02_02 Linear 实现](./02_02_linear_implementation_notes.md)
3. [02_03 Embedding 实现](./02_03_embedding_implementation_notes.md)
4. [02_04 RMSNorm 实现](./02_04_rmsnorm_implementation_notes.md)
5. [02_05 GLU 家族原理](./02_05_glu_explained.md)
6. [02_06 SwiGLU 实现](./02_06_swiglu_implementation_notes.md)
7. [02_07 Pre-Norm 与 Post-Norm](./02_07_prenorm_vs_postnorm_explained.md)
8. [02_08 位置编码与 RoPE 原理](./02_08_rope_explained.md)
9. [02_09 RoPE 实现](./02_09_rope_implementation_notes.md)
10. [02_10 Softmax 实现](./02_10_softmax_implementation_notes.md)
11. [02_11 Scaled Dot-Product Attention 实现](./02_11_attention_implementation_notes.md)
12. [02_12 Multi-Head Attention 实现](./02_12_multihead_attention_implementation_notes.md)
13. [02_13 Transformer Block 实现](./02_13_transformer_block_implementation_notes.md)
14. [02_14 TransformerLM 实现](./02_14_transformer_lm_implementation_notes.md)
15. [02_15 Transformer 资源核算](./02_15_transformer_resource_accounting_notes.md)

## 03 损失、优化与训练

1. [03_01 Cross Entropy 实现](./03_01_cross_entropy_implementation_notes.md)
2. [03_02 PyTorch Optimizer 接口](./03_02_pytorch_optimizer_api_notes.md)
3. [03_03 学习率调优基础实验](./03_03_learning_rate_tuning_notes.md)
4. [03_04 AdamW 实现](./03_04_adamw_implementation_notes.md)
5. [03_05 AdamW 资源核算](./03_05_adamw_accounting_notes.md)
6. [03_06 Warmup + Cosine LR](./03_06_lr_schedule_implementation_notes.md)
7. [03_07 梯度裁剪](./03_07_gradient_clipping_implementation_notes.md)
8. [03_08 数据批采样](./03_08_data_loading_implementation_notes.md)
9. [03_09 Checkpointing](./03_09_checkpointing_implementation_notes.md)
10. [03_10 完整训练循环](./03_10_training_loop_implementation_notes.md)

## 04 推理

1. [04_01 基于 checkpoint 的推理与自回归生成](./04_01_checkpoint_inference_generation_notes.md)

## 90 运维附录

1. [90_01 跨机开发：B 上开发、C 上使用 GPU](./90_01_cross_machine_dev_B_to_C_gpu_guide.md)

## 建议路线

- **先跑通课程主线**：`00_02 → 00_03 → 01 → 02 → 03 → 04`
- **只学习 Transformer 模型**：`02_01 → 02_15`
- **只学习训练系统**：`03_01 → 03_10`
- **查具体实现**：直接按编号定位对应实现笔记，再跟随文档中的代码链接。
