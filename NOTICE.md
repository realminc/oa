# NOTICE — Third-Party Software and Attributions

OA is licensed under the Business Source License 1.1; see `LICENSE`. Third-party
works retain their own licenses. Listing a project or mark does not imply
endorsement or affiliation.

The Rust build resolves these direct dependencies through Cargo:

| Component | OA use | License |
| --- | --- | --- |
| Ash / ash-vp9 | Vulkan and VP9 C-ABI bindings | Apache-2.0 OR MIT |
| vk-mem | Vulkan Memory Allocator binding | MIT |
| bitflags | typed flags | MIT OR Apache-2.0 |
| chrono | host timestamps | MIT OR Apache-2.0 |
| smallvec | compact retained collections | MIT OR Apache-2.0 |
| serde_json | metadata and evidence serialization | MIT OR Apache-2.0 |
| unicode-normalization / unicode-general-category | text processing | MIT OR Apache-2.0 |
| CPAL / rtrb | audio sessions and ring buffers | Apache-2.0 / MIT |
| Symphonia | audio codecs | MPL-2.0 |
| image / webp | image codecs | MIT OR Apache-2.0 / MIT |
| libc | platform ABI | MIT OR Apache-2.0 |
| ml-dsa | post-quantum signatures | Apache-2.0 OR MIT |
| PyO3 | optional Python extension binding | Apache-2.0 OR MIT |

Slang (Apache-2.0 with LLVM exception) and SPIR-V Tools (Apache-2.0) compile and
validate shaders during the build but are not linked into the OA runtime. The
system Vulkan loader and device driver remain host-provided components.

The OA C++ repository is used as behavioral donor evidence and as a differential
test reference. Its source is not copied into Rust release binaries. PyTorch,
TensorFlow, NumPy, GLM, OpenCV, and FFmpeg may appear in documentation or tests
as interoperability or differential references; they are not linked into the
core Rust library unless a dependency manifest explicitly says otherwise.
