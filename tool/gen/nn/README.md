# Rust NN generation

Generate complete layer implementations from `schema/ml/` and shared Rust
`template/` bodies. The current slice owns forty-two public layers and containers plus
two policy enums, MoE telemetry, VQ config/results, Mamba3 config/state and two
Flow configs. Families are activation, softmax, utility, dropout, linear, embedding,
layer_norm, rms_norm, conv, swiglu, pool, upsample, rope, batch_norm, sequential,
attention, ffn, transformer, moe, rnn, gru, byte, vq, mamba3, flow and empyrealm.
Matrix operations retain their existing fn owner.

```bash
python3 tool/gen/nn/generate.py --live
python3 tool/gen/nn/generate.py --check
python3 tool/gen/nn/generate.py --output-dir target/gen/nn
python3 tool/gen/checkDrift.py
python3 -m unittest discover -s test/py/tool/gen/nn -v
```

Live generation consumes the full schema set, uses repository rustfmt, preserves
unchanged timestamps and removes only this generator's ownership-marked orphans.
Check mode is read-only. Preview mode writes a disposable source tree. Output
paths cannot traverse symlinks, and foreign-owned artifacts are never overwritten.

Donor: `oacpp/tools/gen/nn/`. Proven layer behavior is preserved with Rust
ownership, checked validation and ModuleRegistry adaptation. Trainable templates
preserve parameter registration and versioned adjoint recording. Pooling,
Upsample and Rope preserve their existing parameterless adapters.
BatchNorm2d preserves persistent running state and versioned affine adjoints;
Sequential preserves checked ordered child composition through ModuleRegistry.
Attention retains the existing explicit Standard/Flash policy, registered
projections, replay-safe dropout and private mask cache. All currently implemented NN Module bodies are generated. Ffn preserves its pre-normalized SwiGLU residual
composition and registry-owned children. Transformer retains dense/MoE recipes,
checkpoint registration, masked forward and adaptive conditioning in one complete
generated category. MoE generation preserves its single parameter owner, explicit dense oracle,
balancing objectives, shared experts and reduced host telemetry. See
the generated source ownership banners and schema rows for actual coverage.
Generation does not establish numerical or hardware qualification.


### Recurrent generation ownership

Rnn/RnnCell and Gru/GruCell are complete generated categories under the
existing thin NN facade. Each family schema owns both public type names,
exact initializer/projection/scan/cell/adjoint provider roles, the four
parameter names and the layer checkpoint prefix. Complete templates preserve
checked geometry, deterministic seed offsets, optional biases, versioned
parameter snapshots and one hoisted input projection per sequence layer.
The GRU Xavier helper remains an actual shared initialization algorithm.

The recurrent checkpoint reached thirty-three layers/containers in twenty-one files,
plus the two attention policy enums and MoE telemetry value. Matrix operation
schemas still own numerical dispatch and BPTT. This mechanical ownership
checkpoint adds no fresh GPU qualification or performance claim.


## Byte composition generation

`ml_nn_byte.json` owns complete `ByteEmbedding` and `ByteHead` bodies in
`nn/byte.gen.rs`. Separate embedding/head templates preserve the fixed
`ml::byte::VOCAB_SIZE` contract, seeded constructors, table restoration,
accessors and object-safe Module implementations. The thin NN facade exports
both values from their single implementation owner.

The embedded Embedding/Linear registry remains authoritative: byte layers add
no child prefix, copied parameter, dispatch, wait or backward route. Existing
`weight`/`bias` checkpoint paths and parameter identities stay unchanged. U8/U32
embedding and repeated-index gradients use the canonical Embedding provider;
the affine head uses Linear's versioned parameter adjoints and mandatory bias.
Provider paths are validated before generation, including the vocabulary
constant. Imported Embedding/Linear names cannot collide with byte layer names.

Donor behavior was audited in `source/cpp/lib/oa/ml/nn/byte.cpp`, its public
`oa/ml/byte.h` constructors and `test/cpp/ml/byte/testByteEmbedding.cpp`.
The existing Rust deterministic seeded initialization is retained; it differs
from the donor's random-normal embedding and uniform head initialization.
This is mechanical source generation, not a new numerical qualification.
The Byte checkpoint reached thirty-five layers/containers in twenty-two files,
plus two attention policy enums and one MoE telemetry value.


## VQ generation ownership

`ml_nn_vq.json` and the complete `vq.rs.in` template own `nn/vq.gen.rs`:
VectorQuantizer, ResidualVectorQuantizer, their config and both result values.
One family owns shared configuration validation and the residual child recipe.
Schemas own all five public type names, numerical provider roles, persistent
buffer/counter names and the residual level prefix. Public names cannot collide
with imported support types. Configuration defaults and
complete validation remain template-owned. The thin NN facade retains the
same public re-exports.

EMA keeps fresh semantic outputs behind stable NamedBuffer identities, with
`ema_step` stored as registered NamedStateU32 and advanced after state replacement.
Codebook state is not a gradient parameter. Quantization, lookup, EMA and residual
composition remain deferred; `seed` retains its explicit exact checkpoint wait.
The straight-through estimator, commitment-loss reduction, independent initial
EMA buffers, ordered residuals and highest-norm TopK seeding are unchanged.

Donor authority is `source/cpp/lib/oa/ml/nn/vq/vq.cpp`, `oa/ml/nn/vq/vq.h`,
`mlFnMatrixVq.toml` and `test/cpp/ml/nn/testVqAssign.cpp`. Rust's seeded
initialization, out-of-place semantic EMA state and serialized revival counter
are existing adaptations; generation introduces no numerical algorithm.
The existing Rust assignment/gradient/EMA/checkpoint tests remain the runtime
oracle. Generator proofs do not constitute fresh hardware qualification.
Current coverage is forty-two public Module layers/containers in twenty-six
files, plus two attention policy enums, MoE telemetry, VQ configuration and
two VQ result values, Mamba3 configuration/state and two Flow configurations.
Every currently implemented public NN Module now has a complete generated owner, including Empyrealm.

## Mamba3 generation ownership

`tool/gen/nn/schema/ml/ml_nn_mamba3.json` and the complete
`mamba3.rs.in` template own `nn/mamba3.gen.rs`, including `Mamba3`,
`Mamba3Config` and `Mamba3State`. The schema owns public type names, exact
operation/initializer/adjoint providers and all ten checkpoint parameter names.
The template owns configuration fields/defaults, numerical initialization,
checked geometry, parameter registration/order, full forward and inference step.
Case-preserving parameter names retain donor `B_bias`, `C_bias` and `D`;
validation rejects invalid, duplicate and imported-type-colliding names.

Donor evidence is `source/cpp/lib/oa/ml/nn/mamba3/mamba3.cpp`, its header,
`testMamba3.cpp`, `mlFnMatrixSsm.toml` and `oaMamba3.md`. The generation change
preserves the existing Rust FP32 body: deterministic host initialization,
stable inverse softplus using `exp_m1`, rank-selected SISO/MIMO, versioned
projection adjoints, parameter leaf registration and optional gated RMSNorm.
Rust retains a one-dimensional dt bias; the donor uses `[1, heads]`. The
existing preprocess provider admits the Rust shape. Mixed-precision donor
projection islands remain outside this FP32 module's qualification.

The four-buffer inference cache remains caller-owned, initialized to zero and
bound to its input-projection value identity and fixed batch size. Parameter
replacement can invalidate that identity; generation does not expand the cache
lifetime contract. Recurrent steps reject active tapes and another owner's cache
before recording. Forward/step do not read back, submit or wait. The shared
parameter projection helper remains justified by its versioned adjoint contract;
no new lowering chain, shader, runtime owner or execution provider is introduced.

Generator validation and body-equivalence proofs establish source ownership.
Historical numerical and hardware evidence remains historical; this migration
does not imply a fresh GPU, throughput or broader dtype qualification.

## Flat Flow generation ownership

`tool/gen/nn/schema/ml/ml_nn_flow.json` and one complete `flow.rs.in` template
own `src/rs/ml/nn/flow.gen.rs`: `FlowTimeEmbedding`, `FlowTransformer`,
`FlowTransformerConfig`, `FlowDenoiser` and `FlowDenoiserConfig`. The private
`flow` module is included directly by `ml/nn.rs`, which retains its explicit
public re-exports. No nested source modules or secondary implementations remain.
This is one coherent Transformer-based generative model family; another future
backbone does not require creating an extra hierarchy now.

The schema owns all five public type names, admitted operation/constructor
providers, child/parameter/buffer names and the block prefix. The complete Rust
template owns config fields/defaults, validated constructors, time embedding,
dense/MoE and conditioned/masked backbone paths, denoising, condition dropout,
CFG and shared private validation. These helpers express actual shared behavior,
rather than separating each dispatch into another lowering chain.

Donor authority is the three `source/cpp/lib/oa/ml/nn/flow/flow*.cpp` files,
their headers and `test/cpp/ml/generative/testFlow.cpp`. Existing Rust adaptations
remain explicit: FP32 inputs and same-engine checks, deterministic constructor
seeds, `block_0` checkpoint names instead of donor `block0`, nonpersistent
frequency storage and a registered learned position parameter. Per-sample binary
conditioning dropout still uses replay-safe inverted Dropout with its scale
cancelled; CFG still uses an RAII evaluation scope. The mask remains the donor
`(mask - 1) * 1e4` additive key mask. Underlying Matrix, TransformerBlock,
Linear, LayerNorm, MoE and autograd owners are unchanged.

Generator tests and exact body comparison establish source ownership and the
mechanical consolidation. Existing forward/reverse, padding and CFG oracles
remain the numerical authority; historical GPU evidence is not fresh qualification.
`oa::ml::flow` stateless operations retain their separate operation-schema owner.
