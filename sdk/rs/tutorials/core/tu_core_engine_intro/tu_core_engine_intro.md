# tu_core_engine_intro

The three-step eager authoring pattern: construct an `Engine`, record domain operations with no submit/wait ceremony, and read results back through explicit host observation.

## Run

```bash
cargo run -p sdk --release --example tu_core_engine_intro
```

## Source

The complete runnable implementation is [`tu_core_engine_intro.rs`](./tu_core_engine_intro.rs).

> The full walkthrough for this tutorial is being written — the source
> file is the authoritative reference and is kept compiling.
