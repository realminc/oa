# NOTICE — Third-Party Software and Attributions

OA is licensed under the Business Source License 1.1; see `LICENSE`. Third-party
works retain their own licenses. Listing a project or mark does not imply
endorsement or affiliation.

VP9 ABI definitions are retained privately from Ash commit
`33d758c81475c4852166d30655c85834e11b0731`; the MIT text is retained below.
No second Vulkan loader is included.

The Rust build resolves these direct dependencies through Cargo:

| Component | OA use | License |
| --- | --- | --- |
| SDL3 / sdl3 | window, input and camera integration | Zlib / MIT |
| Ash | Vulkan loader and C-ABI bindings | Apache-2.0 OR MIT |
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

## Retained Ash VP9 ABI — MIT license

Copyright (c) 2016 ASH

Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

## Bundled SDL3 — Zlib license

Copyright (C) 1997-2026 Sam Lantinga <slouken@libsdl.org>

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.


## SDL3 Rust bindings — MIT license

The MIT License (MIT)

Copyright (c) 2013 Mozilla Foundation

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
