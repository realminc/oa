# OA NLP Training Benchmark

OA's canonical NLP matrix trains five neural-network architectures with three
tokenizers through the same Vulkan runtime, autograd engine, AdamW optimizer,
metrics, generation and checkpoint path.

This document is updated at preview releases so performance work remains
measurable rather than anecdotal.

## Workload

- 300 optimizer steps;
- batch 64, sequence length 16;
- 1,024 predicted positions per step;
- model width 32 and recurrent/FFN width 64;
- sparse MoE uses four `DFF=16` experts and top-2 routing;
- identical 576-byte teaching corpus and deterministic sampling;
- complete wall time includes forward, loss, backward, AdamW, submission,
  synchronization, scalar metrics and callbacks;
- evaluation, generation and checkpoint reload are mandatory pass conditions.

These are deliberately small educational models. They measure implementation
correctness and runtime overhead; they are not claims about production language
quality.

## Reference system

| Component | Configuration |
|---|---|
| System | Lenovo ThinkPad X1 Carbon Gen 9 |
| CPU | Intel Core i5-1145G7, 4 cores / 8 threads |
| GPU | Intel Iris Xe TGL GT2 integrated GPU |
| Driver | Mesa Intel 26.1.4, Xe KMD, Vulkan 1.4.354 |
| Precision | FP32 |
| Build | CMake Release, Clang 22.1.6 |

Protocol: one excluded warm-up process followed by three sequential measured
processes for every executable. All 60 processes passed training, evaluation,
generation and checkpoint assertions. Values are arithmetic mean ± population
standard deviation.

## OA 0.7.4 results

| Architecture | Tokenizer | Parameters | Wall ms/step | token/s | source byte/s |
|---|---|---:|---:|---:|---:|
| RNN | Byte | 31,104 | 5.00 ± 0.08 | 204.69 ± 3.09K | 204.68 ± 3.09K |
| RNN | BPE | 37,312 | 5.39 ± 0.17 | 189.99 ± 5.75K | 527.31 ± 15.97K |
| RNN | Char | 8,891 | 4.75 ± 0.09 | 215.66 ± 4.04K | — |
| GRU | Byte | 43,648 | 11.17 ± 1.26 | 92.82 ± 9.74K | 92.81 ± 9.74K |
| GRU | BPE | 49,856 | 10.85 ± 0.40 | 94.55 ± 3.39K | 262.43 ± 9.42K |
| GRU | Char | 21,435 | 10.92 ± 0.45 | 93.97 ± 3.84K | — |
| Transformer | Byte | 25,760 | 7.99 ± 0.63 | 128.94 ± 9.54K | 128.93 ± 9.53K |
| Transformer | BPE | 29,920 | 7.93 ± 0.45 | 129.50 ± 6.97K | 359.41 ± 19.36K |
| Transformer | Char | 10,875 | 6.89 ± 0.17 | 148.63 ± 3.63K | — |
| Sparse MoE Transformer | Byte | 28,068 | 8.62 ± 0.10 | 118.80 ± 1.39K | 118.79 ± 1.38K |
| Sparse MoE Transformer | BPE | 32,228 | 8.84 ± 0.14 | 115.78 ± 1.80K | 321.36 ± 4.99K |
| Sparse MoE Transformer | Char | 13,183 | 8.36 ± 0.34 | 122.77 ± 4.88K | — |
| Mamba-3 | Byte | 25,800 | 33.34 ± 0.46 | 30.72 ± 0.42K | 30.72 ± 0.42K |
| Mamba-3 | BPE | 29,960 | 33.39 ± 0.19 | 30.67 ± 0.17K | 85.12 ± 0.49K |
| Mamba-3 | Char | 10,915 | 32.45 ± 0.04 | 31.55 ± 0.03K | — |

## GPU execution

| Architecture | Tokenizer | GPU mean ms/step | p50 | p95 | Wall-GPU gap |
|---|---|---:|---:|---:|---:|
| RNN | Byte | 4.482 ± 0.063 | 4.224 ± 0.014 | 5.819 ± 0.391 | 10.7 ± 0.5% |
| RNN | BPE | 4.843 ± 0.167 | 4.652 ± 0.026 | 5.659 ± 0.729 | 10.3 ± 0.5% |
| RNN | Char | 4.213 ± 0.061 | 3.838 ± 0.007 | 5.996 ± 0.287 | 11.3 ± 0.5% |
| GRU | Byte | 10.498 ± 1.190 | 10.243 ± 1.008 | 12.398 ± 2.502 | 6.0 ± 0.0% |
| GRU | BPE | 10.225 ± 0.387 | 9.928 ± 0.067 | 11.704 ± 1.357 | 6.0 ± 0.0% |
| GRU | Char | 10.282 ± 0.433 | 9.772 ± 0.204 | 12.683 ± 1.461 | 6.0 ± 0.0% |
| Transformer | Byte | 7.033 ± 0.644 | 6.640 ± 0.367 | 9.060 ± 2.087 | 12.0 ± 1.4% |
| Transformer | BPE | 6.952 ± 0.454 | 6.559 ± 0.053 | 8.307 ± 1.714 | 12.3 ± 0.9% |
| Transformer | Char | 5.945 ± 0.150 | 5.706 ± 0.013 | 7.133 ± 0.965 | 13.7 ± 0.5% |
| Sparse MoE Transformer | Byte | 7.514 ± 0.071 | 7.405 ± 0.033 | 8.163 ± 0.304 | 13.0 ± 0.0% |
| Sparse MoE Transformer | BPE | 7.784 ± 0.146 | 7.616 ± 0.058 | 8.618 ± 0.814 | 12.3 ± 0.5% |
| Sparse MoE Transformer | Char | 7.228 ± 0.345 | 7.006 ± 0.174 | 8.165 ± 1.092 | 13.7 ± 0.5% |
| Mamba-3 | Byte | 32.319 ± 0.468 | 31.863 ± 0.110 | 37.012 ± 2.570 | 3.0 ± 0.0% |
| Mamba-3 | BPE | 32.386 ± 0.185 | 32.122 ± 0.002 | 34.737 ± 1.189 | 3.0 ± 0.0% |
| Mamba-3 | Char | 31.445 ± 0.040 | 31.358 ± 0.012 | 33.176 ± 0.405 | 3.0 ± 0.0% |

## Learning gate

| Architecture | Byte final CE / accuracy | BPE final CE / accuracy | Char final CE / accuracy |
|---|---:|---:|---:|
| RNN | 0.1877 / 92.30% | 0.0208 / 98.80% | 0.1833 / 92.83% |
| GRU | 0.1758 / 92.30% | 0.0208 / 98.80% | 0.1778 / 92.63% |
| Transformer | 0.1997 / 92.50% | 0.0205 / 98.87% | 0.1940 / 92.87% |
| Sparse MoE Transformer | 0.1947 / 92.20% | 0.0203 / 98.90% | 0.1955 / 92.87% |
| Mamba-3 | 0.2166 / 93.00% | 0.0221 / 98.87% | 0.1971 / 92.83% |

BPE accuracy is not directly comparable with Byte or Char. Its learned tokens
cover more source bytes and are trained on this same small teaching corpus.
Held-out bits per source byte is the appropriate production tokenizer metric.

## Preview-to-preview progress

The most important 0.7.4 change is sparse execution efficiency. GPU-written
dispatch arguments, deterministic route compaction/scatter, grouped projections,
packed Transformer projections and stable compiled replay close most of the tiny
MoE execution gap without changing the public model API.

| Tokenizer | 0.7.3 MoE/dense wall ratio | 0.7.4 ratio | Relative MoE overhead |
|---|---:|---:|---:|
| Byte | 1.46× | **1.08×** | 46% → 8% |
| BPE | 1.46× | **1.12×** | 46% → 12% |
| Char | 1.57× | **1.21×** | 57% → 21% |

Dense Transformer remains the default because this teaching corpus cannot prove
an expert-capacity quality advantage. MoE is now a credible optional capacity
path: the next gate is held-out quality at equal wall time, not basic executor
correctness.

## Reproduce

Build the 15 Release tutorials named `TutorialNlp{Byte,Bpe,Char}` ×
`{Rnn,Gru,Transformer,Moe,Mamba3}Ag`. Run one warm-up process per executable,
then three sequential measured passes with `OA_LOG_TRAINING_PHASES=1`.

Every process must pass:

- 300 optimizer updates;
- finite and converged training loss;
- evaluation accuracy;
- deterministic generation checks;
- `.oam` save/load and optimizer-step round trip;
- the executable's GoogleTest assertions.
