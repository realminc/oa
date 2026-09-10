# OA C++ to Rust API translation

**Status:** Porting contract

**Updated:** 2026-09-10

**Authority:** [OA Rust Architecture](../architecture/oaArchitecture.md)

**Ledger:** [OA compatibility ledger](oaCompatibility.md)

This document defines the mechanical part of public API translation. It does
not authorize a port: the donor source, tests, subsystem contract, dependency
stage, and acceptance gate still decide whether a capability is admitted.

## Stable concepts and Rust spelling

| OA C++ | OA Rust | Decision |
|---|---|---|
| `oa::Matrix` | `oa::Matrix` | Preserve the semantic value name. |
| `oa::Image` | `oa::Image` | Preserve the semantic value name. |
| `oa::Audio` | `oa::Audio` and `oa::audio::Audio` | One domain-owned definition with an admitted root identity alias. |
| `oa::VideoFrame` | `oa::VideoFrame` and `oa::video::VideoFrame` | One definition with a root identity alias when ported. |
| `oa::Texture` | `oa::Texture` and `oa::render::Texture` | One render-owned definition with a root identity alias when ported. |
| Core `oa::FnMatrix` operations | `oa::matrix` | Lowercase Core numerical-operation module. |
| ML extension `oa::FnMatrix` operations | `oa::ml::matrix` | Preserve ML ownership instead of emulating C++ namespace reopening. |
| `oa::FnLoss` | `oa::ml::loss` | Loss operation module remains distinct from Matrix primitives. |
| `oa::FnOptim` | `oa::ml::optim` | Stateless optimizer primitives; stateful optimizers remain types owned by `oa::ml`. |
| `oa::FnFlow` | `oa::ml::flow` | Stateless flow-model operations. |
| `oa::FnPolicy`, `oa::FnAdvantage`, `oa::FnEnvironment` | `oa::ml::policy`, `oa::ml::advantage`, `oa::ml::environment` | RL operations stay within ML without creating another framework root. |
| `oa::FnImage` | `oa::image` | Lowercase stateless operation module. |
| `oa::FnAudio` | `oa::audio` | Lowercase stateless operation module. |
| `oa::FnHash` | `oa::cryptography::hash` | Device batch hashing remains distinct from CPU one-shot primitives at `oa::cryptography`. |
| `oa::SecureBuffer` | `oa::SecureBuffer` and `oa::cryptography::SecureBuffer` | Preserve one non-owning host type, best-effort page locking, and erase-on-drop behavior. |
| `oa::{PublicKey, SecretKey, Signature, Keypair}` | same root identities, owned by `oa::cryptography::pqc` | Preserve C++ spelling through identity aliases while the PQC subfamily remains the structural owner; secret keys are never generic Matrix values. |
| `oa::FnVideo` | `oa::video` | Lowercase stateless operation module. |
| `oa::FnDetection` | `oa::vision` | Interpretation operation module; narrower children may be introduced by schema ownership. |
| `oa::FnCamera`, `oa::FnMesh`, `oa::FnScene` | `oa::render` | Stateless render-domain operations. |
| `oa::Status` | `oa::Error` / `oa::Result<T>` | Preserve failure meaning and source context. |
| `oa::Result<T>` | `oa::Result<T>` | Preserve fallibility with Rust propagation. |
| `oa::ExecutionPlan` | `oa::ExecutionPlan` | Preserve the public immutable execution concept. |
| `oa::OaEvent` / event handle | `oa::Event` | Rust type name without redundant project prefix. |
| `oa::ModelFile` / `.oam` | native `.oam` through `oa::ml::{save_checkpoint, load_checkpoint}`; `oa::ml::ModelFile` Planned | Preserve ModelFile as the semantic artifact name. Do not call it an archive or make `io` own a second wire representation. |

Examples:

```text
oa::FnMatrix::matMulNt(a, b)       -> oa::matrix::mat_mul_nt(&a, &b)
oa::FnMatrix::gelu(x)              -> oa::ml::matrix::gelu(&x)
oa::FnLoss::crossEntropy(x, y)     -> oa::ml::loss::cross_entropy(&x, &y)
oa::FnOptim::clipGradNorm(gs, max) -> oa::ml::optim::clip_grad_norm(&gs, max)
oa::FnImage::resize(image, size)   -> oa::image::resize(&image, size)
oa::FnAudio::decodeFile(path)      -> oa::audio::decode_file(&engine, path)
oa::FnHash::shake256(bytes, n)     -> oa::cryptography::hash::shake256(&bytes, n)
oa::FnVideo::parseNalAnnexB(bytes) -> oa::video::parse_nal_annex_b(bytes)
oa::FnDetection::boxIou(a, b)      -> oa::vision::box_iou(&a, &b)
decoder.Decode(packet)             -> decoder.decode(packet)
```

An explicit Engine argument is added when Rust cannot derive the one owning
engine from existing input values, such as construction or host-to-device
decode. This is ownership information, not a second operation identity.

## Value, operation, and session mapping

### Values

C++ value classes become Rust structs or enums with private fields and checked
constructors. Copy constructors do not translate mechanically: use `Copy` only
for small independent values and `Clone` for retained handles whose clone
semantics are explicit.

### Operations

C++ `Fn*` namespaces become lowercase modules of free functions. Do not create
`FnMatrix`, `FnAudio`, or zero-sized service structs in Rust. Convenience
methods may delegate to the same operation authority when they add no second
validation, graph, or lowering path.

C++ can reopen `oa::FnMatrix` across Core and ML headers. Rust must not recreate
that include-time extension with forwarding functions or one giant mixed
module. Core operations live in `oa::matrix`; ML-owned Matrix operations live
in `oa::ml::matrix`. File names use the semantic family, never literal `fn` or
`fns` directories.

### Sessions

Stateful C++ classes remain named Rust session structs when the lifecycle is
real: `VideoDecoder`, `AudioCapture`, `AudioEncoder`, `AudioPlayer`, `MediaPlayer`, `Presenter`, or
`TrainingSession`. Constructors return `Result`; methods use borrowing to
express access; terminal `close`, `flush`, `drain`, or `abort` operations are
explicit and failure-bearing. `Drop` never performs them implicitly.

Principal public sessions may be explicitly re-exported at the crate root for
C++/Python continuity:

```text
oa::audio::AudioCapture   == oa::AudioCapture
oa::audio::AudioEncoder   == oa::AudioEncoder
oa::audio::AudioPlayer    == oa::AudioPlayer
oa::video::VideoDecoder   == oa::VideoDecoder
oa::media::MediaPlayer    == oa::MediaPlayer
oa::render::Renderer      == oa::Renderer
```

The equality denotes the same Rust item, not wrapper types or forwarding
implementations. The owning module is the structural authority; the root is an
admitted ergonomic identity alias.

One-shot `decode_file` or `encode_wav_f32` functions are operations, not
decoder/encoder session classes. OARS therefore has no public `AudioDecoder`:
the donor's incremental decoder state is private to `AudioPlayer`. A
`decoder.rs` or `encoder.rs` implementation file is introduced only when a
stateful protocol owner exists.

Supporting domain metadata stays at its owning path by default. Current
examples are `oa::audio::AudioChannelLayout`, `oa::video::NalUnit`, and
`oa::video::VideoFrameTiming`; they do not receive root aliases merely because
their principal `Audio` or `VideoFrame` value does.

## Namespace ownership changes

The donor C++ include tree is not the Rust public hierarchy:

- Image codecs and transformations move from the broad C++ Vision umbrella to
  `image`.
- Video values, codecs, containers, and capture move to `video`.
- Detection, tracking, segmentation, and other interpretation remain in
  `vision`.
- Cross-track clocks, sources, transport, and synchronized playback belong to
  sibling module `media`; Audio and Video are not nested beneath it.
- Texture, Mesh, Material, Scene, Renderer, and Presenter belong to `render`.
- UI and Plot consume renderable values and do not own Image, Texture, Video,
  Audio, or Scene.

Compatibility aliases may exist only when explicitly admitted and generated
from the same authority. Type identity aliases are permitted; duplicate
functions, wrappers, validation, lowering, or state machines are not.

Short OpenMaya-style names remain caller policy:

```rust,ignore
use oa::{audio as oaa, core as oac, cryptography as oacr, ml as oaml, vision as oacv};
```

Do not publish `oaa`, `oac`, `oacr`, `oaml`, or `oacv` as crates or permanent alias
modules. Python callers may likewise write `import oa.audio as oaa` without OA
creating a second package.

## Mechanical language changes

- PascalCase types and variants remain PascalCase where Rust conventions agree.
- camelCase functions, methods, fields, and parameters become snake_case.
- C++ `in`, `out`, and `inOut` prefixes become owned values, shared borrows,
  mutable borrows, or returned results.
- `oa::U32`, `oa::F32`, and similar aliases become Rust primitives unless a
  semantic newtype prevents a real category error.
- `UniquePtr`, `SharedPtr`, Pimpl, and access-friend structures are redesigned
  with direct ownership, private fields, `Box`, `Rc`, or `Arc` only as required
  by the proved lifetime and thread contract.
- C++ overloads become distinct names, trait conversions, or generic bounds
  only when error behavior stays unambiguous.
- Raw Vulkan and third-party codec types remain private adapter details.

## Generated operations

The operation schema—not this table—owns signatures, validation, attributes,
effects, autograd, kernel identity, registry rows, documentation, and generated
tests. Port the donor operation record and algorithm before adapting its Rust
spelling. Direct edits to generated output are not API migration.

Checked-in generated Rust fragments live beside their owning facade and use
`<family>.gen.rs`: for example `matrix/blas.gen.rs`,
`ml/matrix/activation.gen.rs`, `ml/loss/core.gen.rs`, or
`audio/dsp.gen.rs`. A generic `generated.rs` is reserved for a genuinely single
generated authority such as the backend-neutral operation-contract catalog; it
must not become a dumping ground for unrelated domain functions. Runtime shader
identity remains generated separately from public operation functions.

Autograd remains owned by the same operation row. Forward and explicit
backward operation wrappers are generated into the owning operation family;
private tape attachments are generated into the matching
`ml/autograd/matrix/<family>.gen.rs` or `ml/autograd/loss/<family>.gen.rs`
family. Handwritten saved-state behavior follows the same family split. Rust
does not expose the C++ concrete `Grad*` classes or manufacture public
`grad_fns` modules.

## Review checklist

- Is the donor contract classified as value, operation, or session?
- Is the Rust owning module based on responsibility rather than the C++ folder?
- Does the public name follow this table without exposing a second facade?
- Is every root type name an identity re-export from exactly one owning module?
- Have stateless functions remained module-only?
- Are storage sharing, aliasing, readiness, and host boundaries explicit?
- Does a stateful type define transitions and an explicit terminal operation?
- Are backend and third-party errors translated into `oa::Error`?
- Is every mechanically derivable surface generated from one schema?
- Do donor differential tests and Rust-specific ownership tests both pass?
