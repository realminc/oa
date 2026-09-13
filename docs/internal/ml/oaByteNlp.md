# OARS byte and BPE NLP substrate

**Status:** Experimental byte/BPE primitives and complete canonical 5×2 rows

**Updated:** 2026-09-12

**Donor authority:** OA C++ `tokenizer.{h,cpp}`, `byte.h`,
`byte/encoder.cpp`, `nn/byte.cpp`, `ops/gather.slang`,
`nn/gatherBwd.slang`, `testTokenizer.cpp`, and `testByteEmbedding.cpp`

## Public ownership

`oa::ml::tokenizer::BpeTokenizer` owns learned byte-pair ranks. The explicit
`oa::ml::BpeTokenizer` re-export is the same type, not a wrapper. Tokenization
is host preprocessing and does not borrow or create an Engine.

`oa::ml::byte` owns raw-byte Matrix upload and byte-logit decoding operations.
`oa::ml::nn::{ByteEmbedding, ByteHead}` own the trainable 256-row lookup and
256-class affine head. They compose the existing Embedding and Linear owners;
there is no second byte-specific parameter registry, autograd tape, execution
session, or optimizer path.

## BPE contract

The tokenizer begins with tokens `0..=255`. Merge rank `i` creates token
`256 + i`, and later ranks may reference earlier learned tokens. Training
counts adjacent pairs after every applied rank, selects the greatest count,
and resolves equal counts by the numerically smallest `(left, right)` pair.
Encoding applies ranks in learned order. Decoding expands learned tokens back
to exact bytes; UTF-8 conversion is a separate fallible operation.

Persistence uses the donor's deterministic `oa_bpe_v1` text format. Loading is
transactional, bounds the merge count, parses checked unsigned values, and
rejects self or forward references. Tests prove deterministic independent
training, nested learned tokens, exact byte/text round trips, right-aligned
prompt padding, exact save/load identity, corruption rejection, and preservation
of the previously loaded vocabulary after a failed load.

## Byte execution contract

The byte vocabulary is always 256. `encode`, `encode_batched`, and
`encode_text` upload exact U8 values without normalization. `decode` and
`sample` require FP32 logits ending in exactly 256 classes, execute the existing
schema-owned `matrix::sample_logits` operation, wait only at the explicit
compact ID readback boundary, and return exact bytes.

The Embedding semantic operation admits U8 and U32 class-index values. U32
tokens retain the existing physical kernels. Packed-U8 byte indices select
separate schema-owned physical candidates, IDs 438 and 439, adapted from the
donor gather and table-adjoint algorithms. They use the portable packed-byte
storage helper instead of native Int8 SPIR-V, so U32 embedding does not acquire
an optional 8-bit device requirement. Repeated-index adjoints accumulate in
deterministic position order, and out-of-range U32 rows retain the existing NaN
contract.

Intel Iris Xe, Mesa 26.2.2, Vulkan 1.4.354 tests prove exact byte upload,
all-position gather values, repeated-row MSE adjoints, exact greedy byte-logit
decoding, and invalid vocabulary rejection. This is one-device functional
evidence, not general device qualification or a performance claim.

## Implemented Byte and BPE rows

All ten Byte/BPE rows use the exact donor corpus, `[64, 16]` all-position
sampler, widths 32/64, 300 AdamW steps, greedy 80-source-byte continuation
from `to be`, and native `.oam` model/optimizer reload. Mamba-3 uses learning
rate 0.003; every other row uses 0.01. Byte inputs use packed U8 storage with
U32 targets. BPE inputs and targets use U32 token IDs and persist the exact
`oa_bpe_v1` vocabulary beside the `.oam` checkpoint.

Deterministic fresh-process Iris Xe gates reproduced these accepted results:

| Tokenization | Architecture | Parameters | Post-update loss | Accuracy |
| --- | --- | ---: | ---: | ---: |
| Byte | RNN | 31,104 | 0.186080 | 92.1875% |
| Byte | GRU | 43,648 | 0.499343 | 85.5469% |
| Byte | Transformer | 25,760 | 0.190565 | 92.8711% |
| Byte | sparse MoE Transformer | 28,068 | 0.193275 | 92.5781% |
| Byte | Mamba-3 | 25,800 | 0.207903 | 92.9688% |
| BPE | RNN | 37,312 | 0.020861 | 98.8281% |
| BPE | GRU | 49,856 | 0.021224 | 98.8281% |
| BPE | Transformer | 29,920 | 0.020028 | 98.9258% |
| BPE | sparse MoE Transformer | 32,228 | 0.019962 | 98.8281% |
| BPE | Mamba-3 | 29,960 | 0.019415 | 98.9258% |

Every untimed integration gate proves one command recording, 299 cache hits,
598 input uploads, and 300 submissions. The metric-enabled tutorials use one
timestamp-wrapped recording per step because a live query pair cannot be reset
while an earlier asynchronous submission may own it. That distinction is
intentional and reported rather than hidden.

All ten release SDK executables use `ItTraining`, `LossMetric`, `ProgressBar`,
`TrainingSummary`, device/wall timing, evaluation, generation, and checkpoint
reload. These are correctness runs on one device, not fixed-clock performance
qualification.

## Remaining suite work

The canonical Byte/BPE comparison matrix is complete. The additional donor
Empyrealm-Core fidelity tutorial is also connected through
`nn::EmpyrealmCore`: it preserves the `embed`/`mixer` parameter tree and
residual ownership while deliberately sharing the Mamba-3 operation providers.
OA C++'s Empyrealm shaders are renamed copies with identical SPIR-V today;
OARS will introduce a second lowering only when its algorithm actually
diverges. Fixed-clock fresh-process cross-implementation performance
qualification and broader device coverage remain pending.
