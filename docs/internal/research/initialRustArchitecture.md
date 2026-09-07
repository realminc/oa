# Initial Rust Architecture Notebook

**Status:** Research

**Superseded as architecture authority:** 2026-09-07

**Canonical replacement:**
[OA Rust Architecture](../architecture/oaArchitecture.md)

This document preserves the initial Rust-port design exploration. It contains
useful proposals and implementation notes, but it also mixes alternatives,
future APIs, and assumptions that have not been verified. It must not be used
as evidence of shipped behavior or to override the canonical architecture.

---

# oa (Rust port / prototype)

Experimental Rust port / successor prototype for [OA](https://github.com/realminc/oa).

The Rust crate is deliberately named `oa`, so the public namespace remains `oa::...` and normal application code can simply use `use oa::*;`.

The goal is not to translate OA line-for-line. The goal is to preserve the GPU-first semantic architecture while simplifying the implementation, keeping Vulkan explicit, preserving Slang/SPIR-V, and making one-GPU, multi-GPU, and eventually distributed execution fit the same model.

This README assumes you already know C/C++, Vulkan, and Python.


## 0. Architectural decision

The Rust port should simplify OA's public shape rather than preserve the old C++ layering.

The public crate itself is the semantic API:

```text
oa
├── Matrix
├── Image
├── Audio
├── Video
├── Model
├── Event
├── Runtime
├── Engine
├── Device
│
├── matrix::
├── image::
├── audio::
├── video::
├── vision::
├── render::
├── ml::
└── crypto::
```

There is no public `core` namespace.

What used to conceptually live in `core` becomes the root OA language:

```rust
use oa::*;

let x = Matrix::ones([2, 3])?;
let y = matrix::relu(&x)?;
let z = ml::softmax(&y, -1)?;
```

This should map naturally to Python:

```python
import oa

x = oa.Matrix.ones((2, 3))
y = oa.matrix.relu(x)
z = oa.ml.softmax(y, -1)
```

The goal is for Rust and Python to expose nearly the same semantic structure.

### Runtime / Engine / Device

Use the following concepts:

```text
Runtime
    process-local OA runtime
    ├── Vulkan Entry / Instance
    ├── all discovered local devices
    ├── global caches
    ├── scheduler infrastructure
    └── default Engine

Device
    one local Vulkan execution domain
    ├── VkPhysicalDevice / VkDevice
    ├── queues
    ├── VMA allocator
    ├── device-local capabilities
    └── device-local caches

Engine
    execution / lowering context
    ├── DeviceSet
    ├── graph / capture state
    ├── placement policy
    └── execution policy

Cluster
    future distributed collection of remote runtimes / nodes
```

`Runtime` discovers all usable local GPUs by default.

```rust
let runtime = Runtime::init()?;

for device in runtime.devices() {
    println!("{}", device.name());
}
```

A normal application uses the default engine:

```rust
let engine = runtime.default_engine();
```

or targets a subset:

```rust
let engine = runtime
    .engine()
    .devices([gpu0, gpu1, gpu2, gpu3])
    .build()?;
```

For language-like code, OA may expose a default process runtime/context:

```rust
oa::init()?;

let a = Matrix::ones([4096, 4096])?;
let b = Matrix::ones([4096, 4096])?;
let c = &a + &b;
```

Advanced execution can target devices explicitly:

```rust
runtime::execute_on(&[gpu0, gpu1], || {
    let x = Matrix::ones([8192, 8192])?;
    let y = matrix::matmul(&x, &x)?;
    Ok(y)
})?;
```

The exact convenience API can evolve, but the architectural rule is:

> Runtime owns the local hardware universe. Engine is the semantic execution context over a set of devices.

Do not make `Engine == VkDevice`.

Do not make remote machines pretend to be Vulkan devices.

---

## 1. Public API philosophy

OA should feel like a small language, not a framework hierarchy.

Prefer:

```rust
use oa::*;

let image = Image::zeros([1920, 1080], Format::Rgba8)?;
let small = vision::resize(&image, [224, 224])?;
let logits = ml::run(&model, &small)?;
let frame = render::overlay(&image, &logits)?;
```

The public vocabulary should remain small:

```text
Matrix
Image
Audio
Video
Model
Event
Runtime
Engine
Device
Presenter
```

Domain behavior lives in modules:

```text
matrix::
image::
audio::
video::
vision::
render::
ml::
crypto::
```

Stateful hardware/protocol objects remain structs:

```text
Presenter
VideoDecoder
AudioStream
Camera
TrainingSession
```

Use modules for stateless operations and concrete structs for things that genuinely have state/lifetime.

---

## 2. Suggested repository layout

Keep one crate initially.

```text
oa/
├── Cargo.toml
├── README.md
├── shaders/
├── tests/
└── src/
    ├── lib.rs
    ├── error.rs
    │
    ├── matrix.rs
    ├── matrix/
    │   ├── creation.rs
    │   ├── arithmetic.rs
    │   ├── reduction.rs
    │   └── linalg.rs
    │
    ├── image.rs
    ├── image/
    │   ├── creation.rs
    │   ├── convert.rs
    │   └── composite.rs
    │
    ├── audio.rs
    ├── audio/
    │   ├── ops.rs
    │   ├── stream.rs
    │   └── device.rs
    │
    ├── video.rs
    ├── video/
    │   ├── decode.rs
    │   ├── encode.rs
    │   ├── stream.rs
    │   └── frame.rs
    │
    ├── vision.rs
    ├── vision/
    │   ├── resize.rs
    │   ├── color.rs
    │   ├── geometry.rs
    │   └── normalize.rs
    │
    ├── render.rs
    ├── render/
    │   ├── presenter.rs
    │   ├── compositor.rs
    │   └── surface.rs
    │
    ├── ml.rs
    ├── ml/
    │   ├── ops.rs
    │   ├── dnn.rs
    │   ├── dna.rs
    │   ├── training.rs
    │   ├── optimizer.rs
    │   └── model.rs
    │
    ├── crypto.rs
    ├── crypto/
    │   └── ops.rs
    │
    ├── shader.rs
    ├── shader/
    │   ├── compiler.rs
    │   ├── metadata.rs
    │   ├── reflection.rs
    │   ├── cache.rs
    │   ├── generated.rs
    │   └── slang/
    │       ├── common/
    │       │   ├── types.slang
    │       │   ├── math.slang
    │       │   └── tensor.slang
    │       ├── matrix/
    │       ├── vision/
    │       ├── ml/
    │       │   ├── matmul/
    │       │   ├── conv/
    │       │   ├── attention/
    │       │   └── norm/
    │       ├── audio/
    │       ├── video/
    │       └── render/
    │
    ├── runtime.rs
    └── runtime/
        ├── engine.rs
        ├── device.rs
        ├── scheduler.rs
        ├── graph.rs
        ├── placement.rs
        ├── transfer.rs
        ├── memory.rs
        └── vk/
            ├── instance.rs
            ├── physical.rs
            ├── device.rs
            ├── queue.rs
            ├── command.rs
            ├── sync.rs
            ├── memory.rs
            ├── video.rs
            └── present.rs
```

This deliberately removes the old public `core` namespace.

The semantic types live directly in their root modules and are re-exported from `lib.rs`.

The actual machinery lives under `runtime/`.

---

## 3. How Rust files map to C/C++

Rust does not have a required `.h` / `.cpp` split.

Usually the type definition and implementation live together:

```rust
pub struct Matrix {
    storage: StorageId,
    shape: Shape,
    dtype: DType,
}

impl Matrix {
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}
```

Think of a Rust module as the unit that replaces a C++ header/source pair.

A module can be a file:

```text
matrix.rs
```

or a file plus submodules:

```text
matrix.rs
matrix/
    arithmetic.rs
    reduction.rs
```

`matrix.rs` is the module root:

```rust
mod creation;
mod arithmetic;
mod reduction;
mod linalg;

pub use creation::*;
pub use arithmetic::*;
pub use reduction::*;
pub use linalg::*;
```

There is no need for `mod.rs` unless you prefer that layout.

For OA, prefer:

```text
matrix.rs
matrix/...
```

over:

```text
matrix/mod.rs
matrix/...
```

because it is easier to navigate in a large crate.

---

## 4. Root namespace / Python alignment

`lib.rs` is the public API map.

```rust
mod error;
mod runtime;

pub mod matrix;
pub mod image;
pub mod audio;
pub mod video;
pub mod vision;
pub mod render;
pub mod ml;
pub mod crypto;

pub use error::{Error, Result};

pub use matrix::Matrix;
pub use image::Image;
pub use audio::Audio;
pub use video::Video;

pub use runtime::{
    Runtime,
    Engine,
    Device,
    DeviceId,
    Event,
};
```

Normal Rust:

```rust
use oa::*;

let runtime = Runtime::init()?;

let a = Matrix::ones([2, 3])?;
let b = Matrix::full([2, 3], 2.0)?;
let c = matrix::add(&a, &b)?;
```

Normal Python:

```python
import oa

oa.init()

a = oa.Matrix.ones((2, 3))
b = oa.Matrix.full((2, 3), 2.0)
c = oa.matrix.add(a, b)
```

Do not expose internal runtime/Vulkan details through the root namespace.

---

## 5. `FnMatrix` / `FnImage` replacement

Do not port `FnMatrix`, `FnImage`, `FnAudio`, etc. literally.

C++:

```cpp
oa::FnMatrix::add(a, b);
oa::FnImage::resize(image, size);
oa::FnAudio::resample(audio, rate);
```

Rust:

```rust
matrix::add(&a, &b)?;
image::resize(&image, size)?;
audio::resample(&audio, rate)?;
```

ML-specific matrix operations live in `ml`:

```rust
ml::softmax(&x, -1)?;
ml::layer_norm(&x, ...)?;
```

Vision-specific image operations live in `vision`:

```rust
vision::resize(&image, [224, 224])?;
vision::normalize(&image, ...)?;
```

The same semantic type can participate in multiple domains without duplicating the type.

`Matrix` remains one type.

`Image` remains one type.

The module says what semantic domain an operation belongs to.

---

## 6. Operator overloading

Rust supports operator overloading through `std::ops`.

```rust
use std::ops::Add;

impl<'a, 'b> Add<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn add(self, rhs: &'b Matrix) -> Matrix {
        matrix::add(self, rhs)
            .expect("matrix add failed")
    }
}
```

Then:

```rust
let c = &a + &b;
```

For OA this is especially attractive if operations are graph-recording/lazy.

Instead of performing Vulkan work immediately:

```text
a + b
```

can create an `Add` node in the current Engine graph.

Errors that depend on lowering/device capabilities can be reported when the graph is resolved/submitted.

Keep explicit functions too:

```rust
let c = matrix::add(&a, &b)?;
```

Operator syntax is ergonomic sugar, not the only API.

---

## 7. Runtime initialization

Recommended:

```rust
let runtime = Runtime::init()?;
```

This should:

```text
load Vulkan
create VkInstance
enumerate all physical devices
query capabilities
open usable VkDevices
create per-device VMA allocators
initialize queues
initialize global caches
initialize default Engine
```

The process normally needs one Runtime.

A global convenience API may be layered on top:

```rust
oa::init()?;
```

backed by something like:

```rust
OnceLock<Runtime>
```

Do not use mutable global state directly.

---

## 8. Multi-GPU

A workstation with eight GPUs should look like:

```text
Runtime
├── Device 0
├── Device 1
├── Device 2
├── Device 3
├── Device 4
├── Device 5
├── Device 6
└── Device 7

Default Engine
└── DeviceSet::All
```

Explicit subset:

```rust
let engine = runtime
    .engine()
    .devices([0, 1, 2, 3])
    .build()?;
```

Short execution scope:

```rust
runtime::execute_on(&[gpu0, gpu1], || {
    let x = Matrix::ones([4096, 4096])?;
    let y = matrix::matmul(&x, &x)?;
    Ok(y)
})?;
```

The exact syntax can change; the architectural rule should not:

```text
Runtime owns all local devices.
Engine executes/lowers semantics over a selected DeviceSet.
```

Cross-device copies are explicit planner edges, not invisible side effects.

---

## 9. Distributed execution

Do not make remote GPUs look like local Vulkan devices.

```text
Cluster
├── Node 0
│   └── Runtime
│       ├── Device 0
│       └── Device 1
├── Node 1
│   └── Runtime
│       ├── Device 0
│       └── Device 1
└── Node 2
    └── Runtime
        └── Device 0
```

Remote execution sends:

```text
commands
graphs
artifacts
resource payloads
```

not Vulkan handles.

Keep `Cluster` out of the first implementation.

Solve local multi-device execution first.

---

## 10. Domain split

The public split should be semantic and Python-friendly.

### `matrix`

General numerical operations:

```text
creation
arithmetic
linear algebra
reduction
broadcasting
elementwise
```

### `image`

General image representation and generic image operations:

```text
creation
format conversion
basic composition
storage/views
```

### `vision`

Computer-vision semantics:

```text
resize
warp
normalize
color transforms
preprocessing
```

### `audio`

Audio resources and processing:

```text
mix
resample
filter
stream/output
```

### `video`

Video resource/session semantics:

```text
decode
encode
stream
frame
```

Vulkan Video implementation details remain under:

```text
runtime/vk/video.rs
```

### `render`

Presentation/rendering/composition:

```text
Presenter
surface
swapchain-facing presentation
compositor
```

### `ml`

Machine-learning semantics:

```text
DNN ops
Dna compiler/planner
training
optimizers
models
autodiff
```

### `crypto`

Accelerated cryptographic operations.

This split is intentionally language-like and close to the Python surface.

---

## 11. Stateful vs stateless

Stateless/domain operations:

```rust
vision::resize(&image, [224, 224])?;
matrix::add(&a, &b)?;
ml::softmax(&x, -1)?;
```

Stateful objects:

```rust
let mut presenter = render::Presenter::new(...)?
let mut decoder = video::Decoder::new(...)?
let mut audio = audio::Stream::new(...)?
```

Do not force a fake class hierarchy.

Prefer:

```text
structs
enums
modules/functions
```

Traits only when there is a genuine open/backend boundary.

---

## 12. Visibility

A public type does not require public fields.

```rust
pub struct Matrix {
    pub(crate) storage: StorageId,
    pub(crate) shape: Shape,
    pub(crate) dtype: DType,
}
```

External users can construct/use `Matrix` through the API.

OA internals can access fields.

Useful levels:

```rust
pub
pub(crate)
pub(super)
private
```

---

## 13. Low-level stack

Keep the low-level decisions already made:

```text
ash
    Vulkan API/function loading

vk-mem / VMA
    general Vulkan memory allocation

OA runtime memory layer
    upload ring
    readback ring
    transient heap
    resource pools

Slang
    shader/kernel source

SPIR-V
    Vulkan execution
```

No Volk.

No Vulkano.

No wgpu.

No giant async framework initially.

---

## 14. First implementation milestone

Build this path first:

```text
Runtime::init
  ↓
discover all GPUs
  ↓
open one Device
  ↓
VMA
  ↓
buffer
  ↓
Slang kernel
  ↓
Matrix::ones
  ↓
matrix::add
  ↓
timeline
  ↓
readback
```

Then prove two devices:

```text
Device 0
  ↓
Matrix
  ↓
explicit transfer
  ↓
Device 1
  ↓
matrix operation
```

Then prove the unified pipeline:

```text
video decode
  ↓
Image
  ↓
vision
  ↓
ml
  ↓
render::Presenter
```

If those slices are substantially simpler than the C/C++ implementation, continue the port.

---

## 15. Porting rule

The Rust tree should remain simpler than the implementation it replaces.

For every C++ abstraction ask:

```text
Do we need the behavior?
    yes

Do we need this implementation?
    probably not

Do we need this abstraction?
    prove it
```

Prefer plain data and explicit execution.

If the Rust version starts growing things like:

```text
Arc<Mutex<Box<dyn AbstractDeviceFactory>>>
```

the port has gone in the wrong direction.




---

## 16. Rust coding and formatting standard

OA should follow normal Rust conventions rather than preserve C++ naming habits.

The coding standard should be intentionally small.

### Naming

Use:

```text
Types / structs / enums / traits / enum variants
    PascalCase

Functions / methods / variables / fields
    snake_case

Modules / files
    snake_case

Constants / statics
    SCREAMING_SNAKE_CASE
```

Examples:

```rust
pub struct VideoDecoder {
    device_id: DeviceId,
    frame_count: u64,
}

pub enum StreamState {
    Stopped,
    Playing,
    Paused,
}

pub const MAX_DEVICES: usize = 16;

pub fn decode_frame(...) -> Result<VideoFrame> {
    ...
}
```

Do not preserve C++ camelCase:

```rust
decodeFrame()
numElements()
outputMatrix
deviceId
```

Use:

```rust
decode_frame()
num_elements()
output_matrix
device_id
```

### Acronyms

Follow normal Rust casing:

```text
GPU -> Gpu
CPU -> Cpu
API -> Api
ID  -> Id
DNN -> Dnn
```

Examples:

```rust
GpuAllocator
CpuBuffer
ApiVersion
DeviceId
DnnGraph
```

OA-specific names such as `Dna` can keep their intended spelling.

### Primitive types

Use Rust primitive types directly:

```rust
u8
u16
u32
u64
u128
usize

i8
i16
i32
i64
i128
isize

f32
f64

bool
```

Do not recreate C++ aliases such as:

```text
U32
U64
I32
F32
F64
Size
```

unless a semantic type is actually useful.

Prefer:

```rust
pub struct ImageDesc {
    width: u32,
    height: u32,
    channels: u32,
}
```

not:

```rust
pub struct ImageDesc {
    width: U32,
    height: U32,
    channels: U32,
}
```

### Semantic newtypes

Use zero-cost transparent newtypes where type safety matters.

Good:

```rust
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(u32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferId(u32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageId(u32);
```

This prevents accidental mixing of unrelated IDs without runtime overhead.

Do not create newtypes for every ordinary number unless there is a real semantic or safety benefit.

### Public type, private fields

`pub struct` makes the type public, not its fields.

Preferred:

```rust
pub struct Matrix {
    pub(crate) storage: StorageId,
    pub(crate) shape: Shape,
    pub(crate) dtype: DType,
}
```

External users cannot mutate implementation fields directly.

OA internals can access them.

Use visibility deliberately:

```text
pub
pub(crate)
pub(super)
private
```

### Borrowing replaces `in` / `out` / `inOut`

Do not use C++-style argument-direction prefixes.

Rust already expresses access mode in the type system.

Read-only:

```rust
fn process(input: &Matrix)
```

Mutable:

```rust
fn process(output: &mut Matrix)
```

Ownership transfer:

```rust
fn consume(matrix: Matrix)
```

Do not write:

```rust
fn process(in_matrix: &Matrix, out_matrix: &mut Matrix)
```

unless the words are semantically meaningful beyond direction.

### Field names

Do not use C++ member suffixes such as:

```text
device_
shape_
storage_
```

Use:

```rust
pub struct Matrix {
    device: DeviceId,
    shape: Shape,
    storage: StorageId,
}
```

Rust constructor shorthand keeps this clean:

```rust
Self {
    device,
    shape,
    storage,
}
```

### Functions and methods

Prefer explicit, unsurprising names.

Good:

```rust
matrix::mat_mul(...)
matrix::mat_mul_nt(...)
vision::resize(...)
video::decode(...)
```

Avoid abbreviation-heavy names unless already part of OA vocabulary.

Common type-local behavior may also be exposed as methods:

```rust
let y = x.relu()?;
let z = x.add(&y)?;
```

The canonical implementation should remain in the domain module where practical:

```rust
impl Matrix {
    pub fn add(&self, rhs: &Matrix) -> Result<Matrix> {
        matrix::add(self, rhs)
    }
}
```

### Files and modules

Use `snake_case.rs`.

Examples:

```text
video_decoder.rs
execution_plan.rs
device_memory.rs
transfer_planner.rs
```

Prefer:

```text
matrix.rs
matrix/
    arithmetic.rs
    reduction.rs
```

over camelCase filenames.

### Formatting

Use `rustfmt`.

```bash
cargo fmt
```

Formatting policy should not be manually reinvented.

CI should check:

```bash
cargo fmt --check
```

### Linting

Use Clippy:

```bash
cargo clippy --all-targets --all-features
```

Treat meaningful warnings as defects.

Do not blindly enable every pedantic lint on day one.

Start with the defaults and add stricter lints only when they consistently improve OA.

### Unsafe

`unsafe` is allowed where the architecture requires it.

Keep it concentrated around:

```text
runtime/vk/
FFI
allocator internals
external-memory import/export
carefully reviewed zero-copy code
```

Semantic modules should normally remain safe:

```text
matrix/
image/
vision/
audio/
video/
ml/
crypto/
render/
```

The intended layering is:

```text
unsafe Vulkan / FFI implementation
              ↓
safe OA runtime interface
              ↓
safe semantic OA API
```

### Ownership preference

Use this order of preference:

```text
borrow
    ↓
owned value
    ↓
Box
    ↓
Arc
    ↓
Arc + lock only when genuinely required
```

Do not translate every C++ `shared_ptr` to `Arc`.

Do not use `Arc<Mutex<...>>` as a default architecture.

### Collections

Use Rust standard collections first:

```rust
Vec<T>
VecDeque<T>
HashMap<K, V>
```

Use specialized containers where profiling or semantics justify them:

```rust
SmallVec
typed slab
arena
ring buffer
packed command stream
```

Port measured OA optimizations, not C++ container folklore.

### Error handling

Prefer:

```rust
Result<T, Error>
```

and `?`.

Example:

```rust
let buffer = device.create_buffer(desc)?;
let plan = engine.compile(graph)?;
engine.submit(plan)?;
```

Do not use integer status codes inside normal Rust APIs.

C ABI wrappers can translate `Result` into stable status codes at the boundary.

### Constants and enums

Enum variants:

```rust
pub enum DType {
    Float16,
    Float32,
    Float64,
}
```

Constants:

```rust
pub const DEFAULT_TILE_SIZE: u32 = 16;
```

Do not confuse enum-style PascalCase with constant naming.

### Operator overloading

Use operators only where the meaning is obvious.

Reasonable:

```rust
let c = &a + &b;
let d = &a - &b;
let e = &a * &b;
```

Keep named operations for non-obvious semantics:

```rust
matrix::mat_mul(&a, &b)?;
matrix::mat_mul_nt(&a, &b)?;
```

Do not overload operators with surprising behavior.

If OA operations are lazy/graph-recording, operator overloads may record semantic nodes and defer execution errors until lowering/submission.

### Public API rule

The root crate namespace should stay small and stable.

Good:

```rust
use oa::*;

Matrix
Image
Audio
Video
Model
Runtime
Engine
Device
Event
```

Domain APIs:

```rust
matrix::
image::
audio::
video::
vision::
render::
ml::
crypto::
runtime::
```

Do not expose implementation machinery merely because it is public inside the crate.

Keep internals `pub(crate)` wherever possible.

### One-line standard

If a rule is not listed here, follow normal Rust ecosystem conventions and let `rustfmt` + Clippy decide the boring parts.


---

## 17. Shader and kernel source organization

Keep Slang shaders as real `.slang` files.

Do not inline serious GPU programs into Rust raw strings.

Inline source is acceptable for:

```text
tiny tests
bootstrap shaders
minimal examples
```

Production kernels should remain first-class shader files so they can be:

```text
edited independently
compiled independently
debugged by filename / entry point
profiled clearly
shared through Slang modules/imports
inspected without opening host code
```

OA should treat the Slang source tree as part of the source code, not as opaque runtime data.

Recommended layout:

```text
src/
├── shader.rs
└── shader/
    ├── compiler.rs
    ├── metadata.rs
    ├── reflection.rs
    ├── cache.rs
    ├── generated.rs
    └── slang/
        ├── common/
        │   ├── types.slang
        │   ├── math.slang
        │   ├── subgroup.slang
        │   └── tensor.slang
        │
        ├── matrix/
        │   ├── arithmetic.slang
        │   ├── reduction.slang
        │   └── matmul/
        │       ├── generic.slang
        │       ├── tiled.slang
        │       └── cooperative.slang
        │
        ├── vision/
        │   ├── resize.slang
        │   ├── color.slang
        │   └── warp.slang
        │
        ├── ml/
        │   ├── softmax.slang
        │   ├── norm.slang
        │   ├── matmul/
        │   ├── conv/
        │   └── attention/
        │
        ├── audio/
        ├── video/
        └── render/
```

This keeps the shader corpus close to the crate while preserving a clear language boundary:

```text
Rust
    semantic API / runtime / scheduling

Slang
    GPU implementation / specialization / hardware-facing kernels
```

Do not create a matching Rust file for every shader unless there is actual host-side semantic logic to put there.

---

## 18. No handwritten global kernel registry

Do not rebuild the existing large handwritten kernel registry in Rust.

The target architecture is:

```text
.sl​ang source
    ↓
Slang attributes + reflection
    ↓
generator
    ↓
generated Rust metadata
    ↓
KernelRef / KernelCandidate
    ↓
pipeline cache
    ↓
ash
```

The shader source should be the primary source of truth.

Rust should not manually repeat:

```text
file path
entry point
stage
bindings
thread-group dimensions
operation family
dtype support
capability requirements
specialization schema
```

when Slang reflection or OA-specific attributes can describe them.

The central runtime should maintain a **pipeline/cache**, not a giant manually maintained semantic registry.

---

## 19. OA Slang attributes

Slang supports user-defined attributes and exposes them through reflection.

OA should use this to attach machine-readable metadata directly to kernels.

Define OA attributes in a shared Slang module, conceptually:

```slang
[__AttributeUsage(_AttributeTargets.Function)]
struct OaKernelAttribute
{
    string op;
};

[__AttributeUsage(_AttributeTargets.Function)]
struct OaDomainAttribute
{
    string domain;
};

[__AttributeUsage(_AttributeTargets.Function)]
struct OaRequiresAttribute
{
    string capability;
};

[__AttributeUsage(_AttributeTargets.Function)]
struct OaDTypeAttribute
{
    string dtype;
};

[__AttributeUsage(_AttributeTargets.Function)]
struct OaVariantAttribute
{
    string name;
};
```

Then a kernel can describe itself:

```slang
[OaKernel("matmul")]
[OaDomain("ml")]
[OaVariant("cooperative")]
[OaRequires("cooperative_matrix")]
[OaDType("f16")]
[shader("compute")]
[numthreads(128, 1, 1)]
void matmulCooperative(...)
{
    ...
}
```

The exact attribute schema should evolve with OA.

Potential metadata:

```text
operation
domain
variant
dtype
layout
capability requirements
minimum subgroup properties
cooperative-matrix requirements
quantization format
workspace requirements
fusion compatibility
priority
experimental/stable status
autotune family
```

The more useful metadata can be reflected reliably, the less glue code OA needs.

Keep attributes descriptive.

Do not turn them into an entire programming language.

---

## 20. Generated shader metadata

Add a build/generation step that scans and compiles the Slang source tree through the Slang compilation API.

The generator should use reflection to produce Rust metadata.

Conceptual generated output:

```rust
pub static ML_MATMUL_COOPERATIVE: KernelDesc = KernelDesc {
    module: "ml/matmul/cooperative",
    entry: "matmulCooperative",
    domain: Domain::Ml,
    op: Op::MatMul,
    variant: "cooperative",
    requirements: CapabilitySet::COOPERATIVE_MATRIX,
    dtypes: &[DType::Float16],
};
```

This file should be generated:

```text
src/shader/generated.rs
```

or into Cargo's build output directory and included from Rust.

Do not hand-edit it.

The generator can also validate shader metadata at build time:

```text
duplicate operation/variant IDs
invalid attribute combinations
missing entry points
unsupported declared dtype names
missing capability definitions
reflection/binding mismatches
```

This makes the shader corpus self-checking.

---

## 21. Kernel references

Normal semantic operations should refer to generated kernel descriptors, not strings.

Prefer:

```rust
let kernel = shader::ML_SOFTMAX;
runtime.dispatch(kernel, bindings, groups)?;
```

over:

```rust
runtime.dispatch(
    "src/shader/slang/ml/softmax.slang",
    "softmax",
    ...
)?;
```

A basic reference can stay small:

```rust
#[derive(Clone, Copy)]
pub struct KernelRef {
    id: KernelId,
}
```

Generated IDs may be:

```rust
pub enum KernelId {
    MatrixAdd,
    MatrixReduceSum,
    MlSoftmax,
    MlMatMulGeneric,
    MlMatMulCooperative,
    VisionResize,
}
```

or another compact generated representation.

Do not require string hashing / lookup in the hot execution path.

---

## 22. Dna kernel candidates

For ordinary OA operations, one semantic operation may map to one kernel.

For `Dna`, one operation may map to many candidates:

```text
matmul
├── generic
├── tiled
├── subgroup
├── cooperative matrix
├── fp16
├── fp8
└── quantized
```

Generated metadata should make candidate discovery automatic.

Conceptually:

```rust
pub struct KernelCandidate {
    kernel: KernelRef,
    requirements: CapabilitySet,
    dtypes: &'static [DType],
    layouts: &'static [Layout],
    tune_family: TuneFamily,
}
```

Then Dna:

```text
semantic op
    ↓
generated candidate set
    ↓
capability filtering
    ↓
shape/layout filtering
    ↓
heuristic ranking
    ↓
autotune if needed
    ↓
cached winner
```

Adding a new optimized kernel should ideally mean:

```text
1. add .slang file / entry point
2. add OA attributes
3. run generator
```

not:

```text
1. add shader
2. edit kernel registry
3. edit DNN registry
4. edit dispatch switch
5. edit capability table
6. edit another enum
7. edit Python bindings
```

The goal is to make kernel addition increasingly declarative.

---

## 23. Shader reflection

Use Slang reflection for information the compiler already knows.

Examples:

```text
entry points
parameter types
resource bindings
layouts
specialization parameters
thread-group information where exposed
user-defined OA attributes
```

Do not manually duplicate reflection data in Rust.

Reflection should feed the generator and runtime validation.

The generated Rust API should contain only the information OA needs for fast execution and planning.

---

## 24. Development vs release shader compilation

Support both workflows.

### Development

Prefer runtime/development compilation:

```text
.sl​ang source
    ↓
Slang compiler
    ↓
SPIR-V
```

Benefits:

```text
fast shader iteration
good diagnostics
clear source filenames
easy experimentation
```

### Release

Allow generated/precompiled artifacts:

```text
.sl​ang
    ↓ build step
SPIR-V + metadata
    ↓
embedded/package cache
```

The runtime should consume a common representation:

```rust
pub struct ShaderArtifact {
    code: Spirv,
    metadata: ShaderMetadata,
}
```

It should not care whether the artifact came from:

```text
live Slang compilation
disk cache
embedded SPIR-V
precompiled package
```

---

## 25. Shader cache keys

A compiled shader/pipeline cache should be deterministic.

A useful key may include:

```text
source/module hash
entry point
specialization constants
target profile
relevant device capabilities
driver/device fingerprint where required
```

Dna's autotune cache sits above this and additionally includes:

```text
operation signature
shape
dtype
layout
device
```

Keep:

```text
shader compilation cache
```

and:

```text
Dna autotune/selection cache
```

as separate concepts.

---

## 26. Shared Slang modules

Use `src/shader/slang/common/` for reusable GPU-side infrastructure.

Examples:

```text
types.slang
math.slang
tensor.slang
subgroup.slang
memory.slang
quantization.slang
cooperative_matrix.slang
```

Avoid copy/pasting helper code between domain kernels.

Domain kernels should import common Slang modules.

Keep the common layer small enough that unrelated kernels do not acquire giant implicit dependencies.

---

## 27. Shader naming

Follow the Rust/OA semantic naming where practical.

Files:

```text
snake_case.slang
```

Examples:

```text
mat_mul.slang
layer_norm.slang
color_convert.slang
video_convert.slang
```

Slang function naming may follow the existing shader convention if changing the shader corpus would create unnecessary churn, but new OA shader entry points should preferably use a consistent style.

Generated Rust names follow Rust conventions:

```rust
KernelId::MlMatMulCooperative
KernelId::VisionResize
```

Do not encode full filesystem paths into public semantic names.

---

## 28. Shader system rule

The shader subsystem should make the following statement increasingly true:

> Adding metadata to the shader removes host-side glue.

Attributes + reflection + generation should automate as much as is reliable:

```text
kernel discovery
operation grouping
capability filtering
dtype filtering
candidate generation
binding metadata
debug naming
Dna candidate sets
validation
precompile manifests
```

Do not automate decisions that require runtime measurements or workload knowledge.

Those remain in:

```text
Engine
scheduler
Dna
autotuner
```

The division is:

```text
shader metadata
    what this kernel is and what it requires

runtime planner
    whether this kernel should run here and now
```

---

# Appendix: low-level Vulkan / allocator notes

# 41. Why ash instead of Vulkano

Use **ash** as the Vulkan layer.

Vulkano is a good project and provides substantially more safety around Vulkan objects, command recording, validation, synchronization, and ownership. That is attractive for a normal Vulkan application.

OA is different.

OA itself already wants to own:

```text
resource lifetime
allocation strategy
queue selection
submission batching
dependency tracking
timeline synchronization
cross-device placement
Vulkan Video sessions
specialized zero-copy paths
```

Using Vulkano would put another resource/synchronization model between OA and Vulkan:

```text
OA scheduler
   ↓
Vulkano resource / synchronization model
   ↓
Vulkan
```

The intended OA architecture is instead:

```text
OA scheduler
   ↓
small OA Vulkan backend
   ↓
ash
   ↓
Vulkan
```

ash is intentionally thin. It gives us:

```text
generated Vulkan types
core Vulkan entry points
KHR / EXT / vendor extensions
instance/device function loading
Vulkan Video bindings
raw function pointers when needed
```

and does not try to schedule or synchronize work for us.

The design target is:

```text
unsafe ash implementation
        ↓
safe OA internal API
        ↓
semantic OA operations
```

Most of OA should still be safe Rust.

The unsafe code should be concentrated in:

```text
vk/
ffi/
allocator internals
carefully reviewed zero-copy code
```

Do not spread `unsafe` throughout matrix/image/audio/vision code.

---

# 42. ash already replaces Volk

Do not add Volk to the Rust implementation unless a concrete unsupported case appears.

ash already performs the function loading we need.

Typical startup:

```rust
let entry = unsafe {
    ash::Entry::load()?
};
```

Then:

```rust
let instance = unsafe {
    entry.create_instance(&instance_info, None)?
};

let device = unsafe {
    instance.create_device(
        physical_device,
        &device_info,
        None,
    )?
};
```

ash stores loaded entry/instance/device function tables on the corresponding Rust objects.

Conceptually:

```text
ash::Entry
    global Vulkan entry points

ash::Instance
    instance-level dispatch

ash::Device
    device-local dispatch
```

So a call such as:

```rust
unsafe {
    device.cmd_dispatch(cmd, x, y, z);
}
```

uses an already-loaded device function pointer.

There is no `vkGetDeviceProcAddr()` lookup on every dispatch.

This is broadly the role Volk serves in the C/C++ implementation, but without putting the entire dispatch model into global statics.

For OA this is cleaner because:

```text
Device 0 -> ash::Device 0 -> dispatch table 0
Device 1 -> ash::Device 1 -> dispatch table 1
```

naturally matches the multi-device architecture.

Extensions have their own loaded wrappers, for example:

```rust
let swapchain =
    ash::khr::swapchain::Device::new(
        runtime.instance(),
        device.raw(),
    );
```

and Vulkan Video follows the same pattern.

---

# 43. Vulkan memory allocation: use VMA first

Use AMD's Vulkan Memory Allocator through the Rust `vk-mem` crate.

Initial stack:

```text
OA
 ↓
OA memory policy
 ↓
vk-mem
 ↓
VMA
 ↓
ash / Vulkan
```

VMA is a very different proposition from a high-level Vulkan framework.

It does **not** sit in the command submission / dispatch hot path.

It primarily solves:

```text
VkDeviceMemory allocation
memory-type selection
suballocation
fragmentation
mapping
buffer/image allocation
memory budgets
defragmentation
```

Once a buffer/image exists, dispatch and submission still go directly through ash.

So for the first implementation:

```rust
pub struct DeviceInner {
    physical: vk::PhysicalDevice,
    logical: ash::Device,

    memory: DeviceMemory,

    queues: Queues,
    capabilities: DeviceCapabilities,
}
```

and:

```rust
pub struct DeviceMemory {
    general: GpuAllocator,

    // specialized OA paths
    upload: UploadRing,
    readback: ReadbackRing,
    transient: TransientHeap,
}
```

where:

```rust
pub struct GpuAllocator {
    vma: vk_mem::Allocator,
}
```

---

# 44. Do not leak VMA types through OA

VMA is an implementation detail.

Avoid:

```rust
pub struct Matrix {
    pub allocation: vk_mem::Allocation,
}
```

Prefer:

```rust
pub struct Allocation {
    raw: vk_mem::Allocation,
    size: u64,
    class: MemoryClass,
}
```

and:

```rust
pub enum MemoryClass {
    Device,
    Upload,
    Readback,
    Transient,
}
```

Then expose OA-owned allocation results:

```rust
pub struct BufferAllocation {
    pub buffer: vk::Buffer,
    allocation: Allocation,
}
```

The rest of OA should talk to:

```text
GpuAllocator
Allocation
Buffer
Image
```

not directly to VMA.

That preserves the ability to replace VMA later without rewriting the semantic runtime.

---

# 45. General allocation vs specialized OA allocation

Do not make every allocation go through the same slow/general path.

Recommended hierarchy:

```text
DeviceMemory
├── General allocator
│     └── VMA
│
├── Upload ring
│     └── one/few persistent mapped VMA allocations
│
├── Readback ring
│     └── one/few persistent mapped VMA allocations
│
└── Transient heap
      └── large VMA backing allocations
           + OA lifetime-aware suballocation
```

Use VMA for the physical memory problem.

Use OA-specific allocators for workload knowledge VMA cannot have.

Examples:

```text
temporary tensor A lives commands 1..8
temporary tensor B lives commands 9..14
```

Those allocations may safely alias.

An execution-plan compiler knows this.

VMA does not.

That makes transient graph memory a good candidate for:

```text
large backing block
      ↓
OA arena / virtual allocator
      ↓
timeline retirement
```

---

# 46. Upload ring

Avoid:

```text
allocate staging buffer
map
copy
unmap
submit
destroy
```

for every small upload.

Prefer:

```rust
pub struct UploadRing {
    buffer: vk::Buffer,
    allocation: Allocation,
    mapped: std::ptr::NonNull<u8>,

    head: u64,
    tail: u64,
    retirements: std::collections::VecDeque<Retirement>,
}
```

Conceptually:

```text
persistently mapped upload buffer

[ free | used | submitted | free ... ]

                    ↓ timeline N completes

submitted region becomes reusable
```

API:

```rust
let upload = device.memory().upload(bytes, alignment)?;

upload.copy_from_slice(data);

cmd.copy_buffer(
    upload.buffer(),
    destination,
    ...
);
```

The allocation remains owned by the device memory system.

The ring handles reuse.

---

# 47. Readback ring

Same concept in reverse.

```text
GPU result
   ↓
copy into persistent readback allocation
   ↓
timeline completes
   ↓
CPU reads completed region
   ↓
region recycled
```

This is appropriate for:

```text
Matrix::to_vec()
profiling/query data
small CPU-visible results
media metadata
debugging
```

Do not allocate/free a new staging object for every readback unless the workload actually calls for it.

---

# 48. Transient graph memory

The execution planner should eventually be allowed to allocate temporary resources from a lifetime-aware transient heap.

Example:

```text
A  [----------------]
B      [----]
C             [----]
D        [-------------]
```

If:

```text
B lifetime
```

does not overlap:

```text
C lifetime
```

they may use the same underlying region.

Potential representation:

```rust
pub struct TransientHeap {
    blocks: Vec<TransientBlock>,
}

pub struct TransientAllocation {
    block: u32,
    offset: u64,
    size: u64,
}
```

This is one of the areas where OA can outperform a completely generic memory allocator because the execution plan knows resource lifetimes ahead of time.

---

# 49. Pools are above the allocator

For repeated media resources, do not confuse:

```text
memory allocator
```

with:

```text
resource pool
```

For example Vulkan Video:

```rust
pub struct VideoFramePool {
    free: Vec<VideoFrameId>,
    in_flight: Vec<RetiredVideoFrame>,
}
```

The images may have been physically allocated through VMA.

The pool decides when an existing image is reused.

Same for:

```text
descriptor arenas
command pools
temporary images
audio blocks
capture buffers
```

Rule:

```text
VMA owns physical memory allocation strategy.

OA owns semantic resource reuse and lifetime policy.
```

---

# 50. Multi-GPU memory

Every local `Device` owns its own allocator.

```text
Engine
├── Device 0
│    └── DeviceMemory
│         └── VMA allocator 0
│
├── Device 1
│    └── DeviceMemory
│         └── VMA allocator 1
│
└── Device N
     └── DeviceMemory
          └── VMA allocator N
```

Do not make one VMA allocator span unrelated Vulkan logical devices.

Cross-device transfers are a higher-level Engine concern.

They may use:

```text
same-device no-op
device-group peer path
external memory
external semaphore
host staging
remote transport
```

depending on capabilities.

---

# 51. External memory stays explicit

OA may care heavily about:

```text
camera -> GPU
decoder -> compute
GPU -> compositor
GPU A -> GPU B
```

without unnecessary copies.

Keep imported/exported memory as an explicit allocation path.

For example:

```rust
pub enum AllocationKind {
    General,
    Upload,
    Readback,
    Transient,

    Exportable {
        handle_type: ExternalHandleType,
    },

    Imported {
        handle: ExternalMemoryHandle,
    },
}
```

Then the allocator layer constructs the required Vulkan `pNext` chains and VMA/Vulkan flags.

Do not hide this behind a vague:

```text
MemoryUsage::Auto
```

because imported/exported allocations have external synchronization and ownership semantics that matter to OA.

---

# 52. VMA is the default, not a permanent architectural dependency

Start with:

```text
ash + vk-mem
```

because we are already changing:

```text
language
ownership model
build system
public API
device model
```

Do not simultaneously rewrite a mature Vulkan memory allocator unless necessary.

Later benchmark:

```text
vk-mem / VMA
vs
gpu-allocator
vs
OA custom allocator
```

on OA's actual workloads.

If a specialized allocator wins for one path, replace that path.

The OA-facing memory API should make this possible without affecting Matrix/Image/Video code.

---
