# OA SDK

Runnable source-companion validation, workloads, and language previews for
the OA runtime. The SDK is a separate `sdk` Cargo crate that consumes `oa`'s
public API only; it does not reach into private runtime modules.

## Structure

```
sdk/
├── Cargo.toml          # the `sdk` crate manifest
├── build.rs            # SDK shader compilation (slang -> SPIR-V)
├── rs/
│   ├── lib/            # SDK library modules (ml, data, workloads)
│   │   └── ml/rl/      # CartPole and LunarLander environments
│   ├── tutorials/      # runnable tu_* Rust workflows
│   ├── examples/       # runnable ex_* Rust demos
│   ├── benchmarks/     # correctness-gated measurement workloads
│   └── apps/           # larger applications (ALM training/generation)
├── slang/              # SDK-owned compute shaders
│   └── ml/rl/          # environment kernels (extension dispatch)
├── py/                 # Python examples and tutorials (.py)
└── asset/              # shared SDK assets
```

## Extension kernels

SDK-owned shaders under `slang/` compile to SPIR-V at SDK build time and run
through the public `oa::compute` extension-kernel API: `Kernel::compile`
validates the SPIR-V ABI, `Engine::dispatch_kernel` records the dispatch with
explicit `KernelBinding::{Read,Write,ReadWrite}` accesses through OA's ordinary
session graph, barriers, batching, and completion. The built-in `KernelId`
registry stays closed; SDK kernels are never schema operations.

The vectorized CartPole (`sdk::ml::rl::CartPole`) and LunarLander
(`sdk::ml::rl::LunarLander3dVector`) environments are the reference consumers:
each compiles its reset/step kernels once per environment object and dispatches
them through the same `Engine`, transaction, and Event lifecycle as every other
OA operation. Hardware tests compare reset, seeded multi-lane traces, and
complete contact/terminal episodes against independent scalar host oracles.

## Rust tutorials and examples

```sh
cargo run -p sdk --release --example tu_core_mat_mul_intro
cargo run -p sdk --release --example tu_nlp_char_transformer
cargo run -p sdk --release --example tu_rl_cart_pole_ppo
```

Tutorial targets are prefixed `tu_*`, examples `ex_*`, benchmarks `bench_*`/`*_bench`,
and profilers `*_profile`. The complete canonical 5x3 NLP comparison matrix
(RNN, GRU, Transformer, sparse MoE, Mamba-3 over Char, packed-U8 Byte, and BPE
tokenizers) plus the `tu_nlp_byte_empyrealm` fidelity variant is under
`rs/tutorials/ml/nlp/`.

## Python preview

`oapython` is the native Python distribution for the Rust OA runtime. It
installs the `oa` import package and mirrors Rust's public module structure:

```python
import oa

engine = oa.Engine()
one = oa.matrix.ones(engine, [2, 3])
total = oa.matrix.add(one, oa.matrix.full(engine, [2, 3], 2.0))
assert total.to_list() == [3.0] * 6
```

Python mirrors Rust module ownership: stateless operations live in
`oa.matrix`, and host observation through `read_f32()` or `to_list()` is the
synchronization boundary. The canonical PyO3 crate lives at `src/py/`; root
`pyproject.toml` and `uv.lock` own packaging. Python examples and tutorials
live under `sdk/py/{examples,tutorials}` and run against the built extension:

```sh
python3 tool/build/workflow.py --profile debug --python-only
.venv/bin/python -m unittest discover -s test/py -v
```
