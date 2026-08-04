# CS336 Assignment 1 Basics Writeup

This file contains the written and report-style questions from `cs336_assignment1_basics.pdf`.

## Problem (unicode1): Understanding Unicode (1 point)

### (a)

What Unicode character does `chr(0)` return?

Deliverable: A one-sentence response.

Answer: 
``` python
>>> chr(0)
'\x00'
```

### (b)

How does this character's string representation (`__repr__()`) differ from its printed representation?

Deliverable: A one-sentence response.

Verification code:

```python
c = chr(0)
print(f"repr={repr(c)}")
print("printed_start")
print(c)
print("printed_end")
print(f"len={len(c)}")
print(f"ord={ord(c)}")
```

Observed output:

```text
repr='\x00'
printed_start

printed_end
len=1
ord=0
```

Answer: `repr(chr(0))` 会把该字符显示为转义字符串 `'\x00'`，而 `print(chr(0))` 会输出实际的 NUL 控制字符；该字符确实存在，但在终端中不可见。

### (c)

What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:

```python
chr(0)
print(chr(0))
"this is a test" + chr(0) + "string"
print("this is a test" + chr(0) + "string")
```

Deliverable: A one-sentence response.

Verification code:

```python
s = "this is a test" + chr(0) + "string"
print(f"repr={repr(s)}")
print("printed_start")
print(s)
print("printed_end")
print(f"len={len(s)}")
print(f"nul_index={s.index(chr(0))}")
print(f"codepoints={[ord(c) for c in s]}")
```

Observed output:

```text
repr='this is a test\x00string'
printed_start
this is a teststring
printed_end
len=21
nul_index=14
codepoints=[116, 104, 105, 115, 32, 105, 115, 32, 97, 32, 116, 101, 115, 116, 0, 115, 116, 114, 105, 110, 103]
```

Answer: 当 `chr(0)` 出现在文本中时，它会作为一个真实字符保留在字符串中并占据一个索引位置，但由于 NUL 是不可见控制字符，打印结果会让前后文本看起来像是直接拼接在一起。

Conclusion: 这一组问题的核心结论是：Python 字符串可以包含不可见控制字符，`repr` 展示的是便于调试的转义表示，`print` 展示的是字符实际写入输出流后的效果；因此，不能仅凭终端中的可见文本判断字符串中是否存在某个字符。

## Problem (unicode2): Unicode Encodings (3 points)

### (a)

What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.

Deliverable: A one-to-two sentence response.

Verification code:

```python
samples = [
    "hello!",
    "hello! こんにちは!",
    "你好，world!",
    "🙂",
]
encodings = ["utf-8", "utf-16", "utf-32"]

for s in samples:
    print(f"sample={s!r}, chars={len(s)}")
    for enc in encodings:
        b = s.encode(enc)
        print(f"  {enc}: bytes={len(b)}, prefix={list(b[:24])}")
    print()
```

Observed output:

```text
sample='hello!', chars=6
  utf-8: bytes=6, prefix=[104, 101, 108, 108, 111, 33]
  utf-16: bytes=14, prefix=[255, 254, 104, 0, 101, 0, 108, 0, 108, 0, 111, 0, 33, 0]
  utf-32: bytes=28, prefix=[255, 254, 0, 0, 104, 0, 0, 0, 101, 0, 0, 0, 108, 0, 0, 0, 108, 0, 0, 0, 111, 0, 0, 0]

sample='hello! こんにちは!', chars=13
  utf-8: bytes=23, prefix=[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
  utf-16: bytes=28, prefix=[255, 254, 104, 0, 101, 0, 108, 0, 108, 0, 111, 0, 33, 0, 32, 0, 83, 48, 147, 48, 107, 48, 97, 48]
  utf-32: bytes=56, prefix=[255, 254, 0, 0, 104, 0, 0, 0, 101, 0, 0, 0, 108, 0, 0, 0, 108, 0, 0, 0, 111, 0, 0, 0]

sample='你好，world!', chars=9
  utf-8: bytes=15, prefix=[228, 189, 160, 229, 165, 189, 239, 188, 140, 119, 111, 114, 108, 100, 33]
  utf-16: bytes=20, prefix=[255, 254, 96, 79, 125, 89, 12, 255, 119, 0, 111, 0, 114, 0, 108, 0, 100, 0, 33, 0]
  utf-32: bytes=40, prefix=[255, 254, 0, 0, 96, 79, 0, 0, 125, 89, 0, 0, 12, 255, 0, 0, 119, 0, 0, 0, 111, 0, 0, 0]

sample='🙂', chars=1
  utf-8: bytes=4, prefix=[240, 159, 153, 130]
  utf-16: bytes=6, prefix=[255, 254, 61, 216, 66, 222]
  utf-32: bytes=8, prefix=[255, 254, 0, 0, 66, 246, 1, 0]
```

Answer: UTF-8 更适合作为 byte-level tokenizer 的训练编码，因为它对 ASCII 和常见网页混合文本通常更紧凑，例如 `"hello!"` 分别需要 6、14、28 字节；同时 UTF-16/UTF-32 会引入 BOM、大量零字节、端序信息以及 UTF-16 代理对等额外结构，使 BPE 更容易把编码格式的填充模式学进去，而不是学习文本本身的统计规律。

### (b)

Consider the following incorrect function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
```

Deliverable: An example input byte string for which `decode_utf8_bytes_to_str_wrong` produces incorrect output, with a one-sentence explanation of why the function is incorrect.

Verification code:

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

samples = ["hello", "é", "こんにちは", "🙂"]
for s in samples:
    b = s.encode("utf-8")
    print(f"sample={s!r}, utf8_bytes={list(b)}")
    print(f"  correct={b.decode('utf-8')!r}")
    try:
        wrong = decode_utf8_bytes_to_str_wrong(b)
        print(f"  wrong={wrong!r}")
    except UnicodeDecodeError as e:
        print(f"  wrong_error={type(e).__name__}: {e}")
```

Observed output:

```text
sample='hello', utf8_bytes=[104, 101, 108, 108, 111]
  correct='hello'
  wrong='hello'
sample='é', utf8_bytes=[195, 169]
  correct='é'
  wrong_error=UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc3 in position 0: unexpected end of data
sample='こんにちは', utf8_bytes=[227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175]
  correct='こんにちは'
  wrong_error=UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe3 in position 0: unexpected end of data
sample='🙂', utf8_bytes=[240, 159, 153, 130]
  correct='🙂'
  wrong_error=UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf0 in position 0: unexpected end of data
```

Answer: 反例可以取 `"é".encode("utf-8")`，即字节序列 `[195, 169]`；该函数错误地逐字节调用 `decode("utf-8")`，但 UTF-8 中一个 Unicode 字符可能由多个字节共同表示，所以首字节 `0xc3` 单独解码时会因为缺少后续字节而失败。

### (c)

Give a two-byte sequence that does not decode to any Unicode character(s).

Deliverable: An example, with a one-sentence explanation.

Verification code:

```python
samples = [
    bytes([0x80, 0x80]),
    bytes([0xC0, 0xAF]),
    bytes([0xE3, 0x81]),
    "é".encode("utf-8"),
]

for b in samples:
    print(f"bytes={list(b)} hex={b.hex()}")
    try:
        print(f"  decoded={b.decode('utf-8')!r}")
    except UnicodeDecodeError as e:
        print(f"  error={type(e).__name__}: {e}")
```

Observed output:

```text
bytes=[128, 128] hex=8080
  error=UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
bytes=[192, 175] hex=c0af
  error=UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc0 in position 0: invalid start byte
bytes=[227, 129] hex=e381
  error=UnicodeDecodeError: 'utf-8' codec can't decode bytes in position 0-1: unexpected end of data
bytes=[195, 169] hex=c3a9
  decoded='é'
```

Answer: 一个例子是字节序列 `bytes([0x80, 0x80])`，即十六进制 `8080`；它不能解码为任何 Unicode 字符，因为 UTF-8 中 `0x80` 是 continuation byte，只能跟在合法起始字节之后，不能作为字符的起始字节出现。

## Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)

### (a)

Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories `<|endoftext|>` special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How much time and memory did training take? What is the longest token in the vocabulary? Does it make sense?

Resource requirements: <= 30 minutes, no GPUs, <= 30 GB RAM.

Hint: You should be able to get under 2 minutes for BPE training using multiprocessing during pre-tokenization and the following two facts:

- The `<|endoftext|>` token delimits documents in the data files.
- The `<|endoftext|>` token is handled as a special case before the BPE merges are applied.

Deliverable: A one-to-two sentence response.

Answer: TODO

### (b)

Profile your code. What part of the tokenizer training process takes the most time?

Deliverable: A one-to-two sentence response.

Answer: TODO

## Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)

### (a)

Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What is the longest token in the vocabulary? Does it make sense?

Resource requirements: <= 12 hours, no GPUs, <= 100 GB RAM.

Deliverable: A one-to-two sentence response.

Answer: TODO

### (b)

Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.

Deliverable: A one-to-two sentence response.

Answer: TODO

## Problem (tokenizer_experiments): Experiments with tokenizers (4 points)

### (a)

Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyStories and OpenWebText tokenizers, with 10K and 32K vocabulary size respectively, encode these sampled documents into integer IDs. What is each tokenizer's compression ratio in bytes/token?

Deliverable: A one-to-two sentence response.

Answer: TODO

### (b)

What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens.

Deliverable: A one-to-two sentence response.

Answer: TODO

### (c)

Estimate the throughput of your tokenizer, for example in bytes/second. How long would it take to tokenize the Pile dataset, which contains 825 GB of text?

Deliverable: A one-to-two sentence response.

Answer: TODO

### (d)

Using your TinyStories and OpenWebText tokenizers, encode the respective training and development datasets into a sequence of integer token IDs. The assignment recommends serializing the token IDs as a NumPy array of datatype `uint16`. Why is `uint16` an appropriate choice?

Deliverable: A one-to-two sentence response.

Answer: TODO

## Problem (transformer_accounting): Transformer LM resource accounting (5 points)

### (a)

Consider a GPT-2 XL-sized model using the assignment architecture, with the following configuration:

- `vocab_size`: 50,257
- `context_length`: 1,024
- `num_layers`: 48
- `d_model`: 1,600
- `num_heads`: 25
- `d_ff`: 4,288, the nearest multiple of 64 to $8/3 \times 1,600$

Suppose the model is constructed using this configuration. How many trainable parameters would the model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?

Deliverable: A one-to-two sentence response.

Answer: TODO

### (b)

Identify the matrix multiplies required to complete a forward pass of the GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that the input sequence has `context_length` tokens.

Deliverable: A list of matrix multiplies with descriptions, and the total number of FLOPs required.

Answer: TODO

### (c)

Based on the analysis above, which parts of the model require the most FLOPs?

Deliverable: A one-to-two sentence response.

Answer: TODO

### (d)

Repeat the analysis with GPT-2 small, medium, and large:

- GPT-2 small: 12 layers, 768 `d_model`, 12 heads.
- GPT-2 medium: 24 layers, 1024 `d_model`, 16 heads.
- GPT-2 large: 36 layers, 1280 `d_model`, 20 heads.

As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?

Deliverable: For each model, provide a breakdown of model components and their associated FLOPs as a proportion of the total FLOPs required for a forward pass. In addition, provide a one-to-two sentence description of how varying the model size changes the proportional FLOPs of each component.

Answer: TODO

### (e)

Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How does the relative contribution of FLOPs of the model components change?

Deliverable: A one-to-two sentence response.

Answer: TODO

## Problem (learning_rate_tuning): Tuning the learning rate (1 point)

Run the SGD example from the assignment with three other values for the learning rate: `1e1`, `1e2`, and `1e3`, for just 10 training iterations. What happens with the loss for each of these learning rates? Does it decay faster, slower, or does it diverge, meaning increase over the course of training?

Deliverable: A one-to-two sentence response with the behaviors observed.

Answer: TODO

## Problem (adamw_accounting): Resource accounting for training with AdamW (2 points)

Assume float32 is used for every tensor.

### (a)

How much peak memory does running AdamW require? Decompose the answer based on the memory usage of the parameters, activations, gradients, and optimizer state. Express the answer in terms of the `batch_size` and the model hyperparameters `vocab_size`, `context_length`, `num_layers`, `d_model`, and `num_heads`. Assume $d_{ff} = 8/3 \times d_{model}$.

For simplicity, when calculating memory usage of activations, consider only the following components:

- Transformer block.
- RMSNorm(s).
- Multi-head self-attention sublayer: QKV projections, $QK^T$ matrix multiply, softmax, weighted sum of values, output projection.
- Position-wise feed-forward, SwiGLU: $W_1$, $W_2$, SiLU on the gate branch, element-wise product, $W_3$.
- Final RMSNorm.
- Output embedding.
- Cross-entropy on logits.

Deliverable: An algebraic expression for each of parameters, activations, gradients, and optimizer state, as well as the total.

Answer: TODO

### (b)

Instantiate the answer for a GPT-2 XL-shaped model to get an expression that only depends on `batch_size`. What is the maximum batch size that can still fit within 80 GB memory?

Deliverable: An expression that looks like $a \cdot batch\_size + b$ for numerical values $a$ and $b$, and a number representing the maximum batch size.

Answer: TODO

### (c)

How many FLOPs does running one step of AdamW take?

Deliverable: An algebraic expression, with a brief justification.

Answer: TODO

### (d)

Model FLOPs utilization, MFU, is defined as the ratio of observed throughput in tokens per second relative to the hardware's theoretical peak FLOP throughput. An NVIDIA H100 GPU has a theoretical peak of 495 teraFLOP/s for float32, actually TensorFloat-32, operations. Assuming 50% MFU, how long would it take to train a GPT-2 XL for 400K steps and a batch size of 1024 on a single H100? Following Kaplan et al. and Hoffmann et al., assume that the backward pass has twice the FLOPs of the forward pass.

Deliverable: The number of hours training would take, with a brief justification.

Answer: TODO

## Problem (experiment_log): Experiment logging (3 points)

For the training and evaluation code, create experiment tracking infrastructure that allows experiments and loss curves to be tracked with respect to gradient steps and wall-clock time.

Deliverable: Logging infrastructure code for the experiments and an experiment log, meaning a document of all the things tried, for the assignment problems in the experiments section.

Answer: TODO

## Problem (learning_rate): Tune the learning rate (2 B200 hrs) (3 points)

The learning rate is one of the most important hyperparameters to tune. Taking the base model that has been trained, answer the following questions.

### (a)

Perform a hyperparameter sweep over the learning rates and report the final losses, or note divergence if the optimizer diverges.

Deliverable: Learning curves associated with multiple learning rates. Explain the hyperparameter search strategy.

Deliverable: A model with validation loss, per-token, on TinyStories of at most 1.45.

Answer: TODO

### (b)

Folk wisdom is that the best learning rate is "at the edge of stability." Investigate how the point at which learning rates diverge is related to the best learning rate.

Deliverable: Learning curves of increasing learning rate which include at least one divergent run and an analysis of how this relates to convergence rates.

Answer: TODO

## Problem (batch_size_experiment): Batch size variations (1 B200 hr) (1 point)

Vary the batch size all the way from 1 to the GPU memory limit. Try at least a few batch sizes in between, including typical sizes like 64 and 128.

Deliverable: Learning curves for runs with different batch sizes. The learning rates should be optimized again if necessary.

Deliverable: A few sentences discussing the findings on batch sizes and their impacts on training.

Answer: TODO

## Problem (generate): Generate text (1 point)

Using the decoder and trained checkpoint, report the text generated by the model. Decoder parameters such as temperature and top-p may need to be manipulated to get fluent outputs.

Deliverable: Text dump of at least 256 tokens of text, or until the first `<|endoftext|>` token, and a brief comment on the fluency of this output and at least two factors which affect how good or bad this output is.

Answer: TODO

## Problem (layer_norm_ablation): Remove RMSNorm and train (0.5 B200 hrs) (1 point)

Remove all RMSNorms from the Transformer and train. What happens at the previous optimal learning rate? Can stability be obtained by using a lower learning rate?

Deliverable: A learning curve for when RMSNorms are removed and training is run, as well as a learning curve for the best learning rate.

Deliverable: A few sentences of commentary on the impact of RMSNorm.

Answer: TODO

## Problem (pre_norm_ablation): Implement post-norm and train (0.5 B200 hrs) (1 point)

Modify the pre-norm Transformer implementation into a post-norm one. Train with the post-norm model and see what happens.

Deliverable: A learning curve for a post-norm Transformer, compared to the pre-norm one.

Answer: TODO

## Problem (no_pos_emb): Implement NoPE (0.5 B200 hrs) (1 point)

Modify the Transformer implementation with RoPE to remove the position embedding information entirely, and see what happens.

Deliverable: A learning curve comparing the performance of RoPE and NoPE.

Answer: TODO

## Problem (swiglu_ablation): SwiGLU vs. SiLU (0.5 B200 hrs) (1 point)

Compare the performance of SwiGLU feed-forward networks versus feed-forward networks using SiLU activations but no gated linear unit. For the SiLU ablation baseline, use $d_{ff} = 4 \times d_{model}$ to approximately match the parameter count of the default SwiGLU feed-forward network.

Deliverable: A learning curve comparing the performance of SwiGLU and SiLU feed-forward networks, with approximately matched parameter counts.

Deliverable: A few sentences discussing the findings.

Answer: TODO

## Problem (main_experiment): Experiment on OWT (2 B200 hrs) (2 points)

Train the language model on OpenWebText with the same model architecture and total training iterations as TinyStories. How well does this model do?

Deliverable: A learning curve of the language model on OpenWebText. Describe the difference in losses from TinyStories. How should these losses be interpreted?

Deliverable: Generated text from the OpenWebText LM, in the same format as the TinyStories outputs. How is the fluency of this text? Why is the output quality worse even though the same model and compute budget are used as TinyStories?

Answer: TODO

## Problem (leaderboard): Leaderboard (10 B200 hrs) (6 points)

Train a model under the leaderboard rules with the goal of minimizing the validation loss of the language model within 0.75 B200-hours.

Rules:

- Runtime: the submission can run for at most 45 minutes on a B200.
- Data: only the provided OpenWebText training dataset may be used.

Deliverable: The final validation loss that was recorded, an associated learning curve that clearly shows a wall-clock-time x-axis that is less than 45 minutes, and a description of what was done. The leaderboard submission is expected to beat at least the naive baseline of a 5.0 loss. Submit to the leaderboard here: github.com/stanford-cs336/assignment1-basics-leaderboard.

Answer: TODO
