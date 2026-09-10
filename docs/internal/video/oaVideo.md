# OA Rust Video

**Status:** Experimental packed-frame value, Annex-B operations, Vulkan Video
device and exact decode-profile queries, and bounded MP4 demux session; codec
sessions Planned

**Updated:** 2026-09-09

**Authority:** [OA Rust Architecture](../architecture/oaArchitecture.md)

**Media boundary:** [OA Rust media boundary](../media/oaMedia.md)

**Donor references:** OA C++ `include/oa/vision/videoDecoder.h` and
`docs/internal/vision/oaVideo.md`

Rust owns Video directly under `oa::video`; it does not reproduce the donor's
`vision::video` source hierarchy. Vision consumes frames for interpretation,
while Video owns frame semantics, bitstreams, codecs, containers, and codec or
capture sessions.

## Current value slice

`oa::VideoFrame` and `oa::video::VideoFrame` are identity aliases for the same
type. The first admitted backing is one packed `Image`. Construction rejects
batched Images with a batch extent other than one and rejects zero spatial
extents. The frame retains:

- visible width and height through its backing Image;
- a non-negative presentation timestamp relative to the producer's epoch;
- an optional, strictly positive duration;
- source matrix-coefficient and component-range metadata;
- the backing Image, Matrix storage, and Matrix producer readiness.

Cloning retains the same semantic storage. It does not copy pixels or create a
second completion event. `as_image` and `into_image` return optional values so
future native multi-plane backing cannot be misrepresented as one dense Image.

The current color record is deliberately incomplete. `Bt601`, `Bt709`, and
`Bt2020` identify source matrix coefficients only. Color primaries, transfer
function, chromatic adaptation, mastering metadata, and HDR remain Planned and
must not be inferred from that enum.

## Current Annex-B operations

The donor's CPU-only `oa::FnVideo` NAL family maps directly onto free functions
in `oa::video`:

| OA C++ | OA Rust |
|---|---|
| `FnVideo::parseNalAnnexB` | `video::parse_nal_annex_b` |
| `FnVideo::emitNalAnnexB` | `video::emit_nal_annex_b` |
| `FnVideo::extractSps` / `extractPps` | `video::extract_sps` / `extract_pps` |
| H.265 VPS/SPS/PPS extractors | `video::extract_vps_h265` / `extract_sps_h265` / `extract_pps_h265` |

`NalUnit` is supporting borrowed metadata and remains under `oa::video`; it is
not root-re-exported. Parsed payloads alias the caller's bytes and exclude
three- or four-byte start codes. Emission canonicalizes every prefix to four
bytes. Parsed units retain the source start-code offset and width so codec
recorders can use the required Annex-B slice marker without recovering it by
pointer arithmetic. Raw-payload construction reports no source marker. H.264
and H.265 parameter extraction remain codec-explicit because their header
layouts differ.

There is no `FnVideo` compatibility type and no duplicate root operation. These
are bounded host bitstream operations, not GPU dispatches or decoder session
commands.

`parse_h264_sps` and SPS-resolved `parse_h264_pps` add the first codec-parameter
parser checkpoint. They return owned backend-neutral records containing the
fields needed for standard-video session parameters and picture parsing. PPS
parsing requires the referenced SPS so 4:4:4 streams use the correct six 8x8
scaling-list count rather than assuming the usual two. RBSP
emulation-prevention removal and unsigned/signed Exp-Golomb reads are shared
with stream-profile extraction. SPS POC cycles are bounded to 256 entries;
invalid identifiers, dimensions, POC types, truncated data, and reference-list
counts fail closed. PPS slice groups remain explicitly unsupported, matching
the donor decoder's current restriction.

`parse_h264_slice_header` resolves one coded slice against explicit SPS/PPS
records. It retains frame, field, coded POC-LSB, reference, IDR, and decoded
reference-picture marking state while bounding active lists, list modifications,
weighted-prediction walks, and MMCO commands. POC modes that require prior
picture state remain unset for the future decoder session to derive; the
stateless parser does not substitute frame number for POC.

`parse_h265_vps`, `parse_h265_sps`, and `parse_h265_pps` provide the matching
HEVC parameter records. Profile-tier-level, sub-layer ordering, conformance
crop, coding and transform geometry, bounded reference-picture-set walks,
tiles, deblocking, and session-relevant flags are retained. SPS/PPS scaling-list
syntax is no longer discarded: prediction references, default matrices, DC
coefficients, and diagonal coefficient walks resolve into owned raster-order
4x4, 8x8, 16x16, and 32x32 matrices suitable for a future standard-video
parameter object. Invalid backward references and coefficient ranges fail
closed. SPS short-term reference sets now retain direct syntax masks and delta
arrays; inter-predicted sets retain their predictor masks and resolve ordered
delta POCs for the next set instead of losing predecessor cardinality. SPS
long-term POC-LSB and current-picture flags are typed as well. Counts that would
drive unbounded parser work are rejected before iteration.

`parse_h265_slice_header` resolves coded slices against explicit SPS/PPS
records and retains slice address, type, temporal/reference identity, output
intent, coded POC-LSB, and bounded inline short-term reference deltas. Checked
CTB geometry prevents damaged dimensions from overflowing address derivation.
Dependent segments and inline inter-predicted RPS remain explicit
`MissingCapability` results until the decoder owns their required retained
state.

## Current device capability query

`video::query_device_capabilities(&engine)` queries the selected physical
device's queue-family flags and Vulkan Video extension advertisements. It
reports decode and encode queue-family indices plus H.264, H.265, AV1, and VP9
decode and H.264, H.265, and AV1 encode advertisements without exposing Vulkan
handles.

Hardware advertisement, enabled Engine queues, and OARS session availability
are separate fields. Engine construction now requests each unique advertised
video queue family and enables only the reported base, queue, and codec
extension chain. `video_queues_enabled()` therefore records logical-device
admission, while `decoder_sessions_available()` and
`encoder_sessions_available()` truthfully remain false. On the tested Intel
Iris Xe / Mesa 26.2.2 device, creation succeeds with its dedicated combined
decode/encode queue and advertised H.264, H.265, AV1, and VP9 decode support.
The Engine now lazily owns a command pool for the selected decode family;
commands retain their originating pool identity through timeline retirement,
and a live-device test submits, waits, and retires an empty decode-family
command. This proves queue plumbing only, not codec-session qualification.

The private runtime admission test now also creates, binds, and destroys one
session for every advertised typed H.264, H.265, and AV1 profile on that
device. Creation reuses the exact profile query's standard-header version,
chooses profile-compatible output and DPB formats, bounds the memory-requirement
count, constrains each VMA allocation to its reported memory-type mask, and
binds the exact required size. The H.264 path additionally converts the donor
MP4's parsed SPS/PPS into Khronos `StdVideoH264` records and successfully
creates and destroys the dependent `VkVideoSessionParametersKHR` object on the
same device. The conversion preserves IDs, profile/level, coded geometry, POC
fields, reference limits, signed QP offsets, and represented flags; no Vulkan
type crosses the runtime boundary. Typed VUI parsing retains aspect ratio,
overscan, signal/colour, chroma location, timing, NAL/VCL HRD, picture-structure,
and bitstream-restriction syntax. The Vulkan conversion supplies the exact
`StdVideoH264SequenceParameterSetVui` and optional HRD table; distinct NAL and
VCL HRD tables fail closed because the standard-video record has one pointer.
Sequence- and picture-level scaling lists retain their presence/default masks
and all 4x4/8x8 scan-order values, then lower without reordering into the
corresponding `StdVideoH264ScalingLists`. This proves bounded H.264
parameter-object ownership; the first-picture qualification below separately
proves one narrow decode command. `decoder_sessions_available()` remains false
until the other codec parameter objects, reusable DPB state, and public
decoded-frame synchronization are connected.

Session admission now allocates the first native decode image set as well. The
runtime queries formats for the exact profile and intended decode-output/DPB
usage, carries the required profile list into image creation, and preserves the
driver's format, tiling, component mapping, and applicable creation flags. It
strips `VIDEO_PROFILE_INDEPENDENT_KHR` while supplying the explicit profile
chain because the maintenance feature is not enabled. Profiles that
permit distinct resources receive one output image and one layered DPB image;
coincident-only profiles receive one layered image with both usages. Disjoint
multi-plane formats fail closed until per-plane memory binding is implemented.
The session owns view-before-image-before-allocation teardown. Live creation and
destruction pass for every advertised typed H.264, H.265, and AV1 profile on
the local Intel/Mesa device. Images begin in `UNDEFINED`; the narrow H.264
qualification transitions its first reconstructed/output layer explicitly,
while recyclable frame lifetime remains unimplemented.

The first private encoded-bitstream owner now creates a
`VIDEO_DECODE_SRC_KHR` buffer with the same exact profile-list chain. Packet
length is rounded up with checked arithmetic to the profile's reported minimum
bitstream-size alignment; offset zero satisfies the separately validated
offset alignment. The complete submitted range is initialized—packet bytes
followed by a zeroed alignment tail—and host writes are flushed before the
buffer is retained. Before upload, the narrow H.264 qualification extracts its
single VCL NAL from the demuxed access unit and emits the canonical three-byte
byte-stream prefix consumed by the Khronos/FFmpeg Vulkan Video paths. SPS, PPS,
and SEI NALs remain represented through session parameters or host metadata;
they are not smuggled into the submitted coded-slice range. Unit coverage locks
this normalization and rejects zero- or multi-slice access units until the
general decoder owns a slice-offset table. Reuse across in-flight pictures
waits for explicit epoch retirement rather than destructor synchronization.

The first executable codec command checkpoint records one progressive H.264
IDR I-slice from the donor packet. It begins the video coding scope with the
bound session and parameters, resets the new session, transitions the complete
DPB array and distinct output image from `UNDEFINED` using
`VIDEO_DECODE_READ/WRITE` scopes, publishes the host-written bitstream through
a HOST→VIDEO_DECODE buffer barrier, binds reconstructed slot zero, and ends the
scope. Submission uses the decode-family pool and queue while the engine
timeline supplies the completion/retirement edge. Queue-family result-status
support is queried explicitly; where supported, the session owns an
exact-profile `RESULT_STATUS_ONLY_KHR` pool whose query encloses the decode.
The live Intel/Mesa run returns `COMPLETE` after the engine event, proving the
codec operation succeeded rather than merely that the queue completed. It then
releases output layer zero from the decode family and acquires it on the
compute family with identical old/new layouts and subresource range, copies
NV12 planes into retained host-visible storage, normalizes them
to planar YUV420, and waits through the same engine timeline. The resulting
1,382,400 bytes match an independent FFmpeg first-frame SHAKE-256 oracle. This
proves one progressive High-profile IDR on the recorded Intel/Mesa provenance;
it does not imply a reusable or portable public decoder.

`video::query_decode_capabilities` performs the second, exact query for one
complete `VideoDecodeProfile`. H.264, H.265, and AV1 profiles carry their
codec-specific profile identity alongside chroma subsampling and component bit
depth. H.264 additionally carries the decoded-picture layout, and AV1 carries
the film-grain support request. The returned backend-neutral limits include
coded extents, picture granularity, bitstream alignments, DPB/reference limits,
the codec-specific maximum level, DPB/output relationship flags, protected
content, and separate-reference-image support.

`video::query_decode_formats` then enumerates the decode-output and DPB images
for that same exact profile. OARS currently recognizes planar 4:2:0 8/10-bit
layouts plus NV12, P010, and P012 and reports tiling and relevant image usages.
Unknown backend formats are not mislabeled or discarded silently: each result
retains an unrecognized-format count while keeping backend numeric format
identities private. This is the format-selection evidence needed before video
session and native-frame allocation.

The Vulkan boundary constructs both required halves of each query chain: the
codec-specific profile structure under `VkVideoProfileInfoKHR`, and generic
decode plus codec-specific capability structures under
`VkVideoCapabilitiesKHR`. Extension advertisement is checked before dispatch,
and Vulkan's profile-operation, profile-format, and profile-codec rejection
results become `MissingCapability` instead of being confused with runtime
failure. No `ash::vk` type crosses the public API. VP9 exact-profile querying is
still gated: the pinned Ash 1.3.281 bindings predate the typed KHR VP9 profile
and capability structures even though the live Vulkan registry and driver may
advertise the extension.

## Current MP4 demux session

`VideoDemuxer` is the first admitted stateful Video session. It opens one local,
unfragmented ISO-BMFF/MP4 source, chooses the first supported video track, and
builds a bounded validated sample index. The public values are
`VideoContainerInfo`, `VideoTimeBase`, `VideoPacket`, `VideoCodec`, and
`VideoContainerKind`.

For AVC, the demuxer derives profile, chroma precision, and progressive versus
interlaced picture layout from the first bounded SPS in `avcC`. For HEVC it
reads the profile, chroma, and component depths from `hvcC`; for AV1 it reads
the profile and format fields from `av1C`. `VideoContainerInfo::decode_profile`
therefore feeds the exact capability and format queries directly when the
stream profile is representable. It returns `None` for VP9 and standard-video
profiles outside the pinned binding rather than inventing a compatible profile.

Media payload remains in the file until `read_next_packet`; the demuxer bounds
movie metadata, sample-table expansion, and individual packet allocation.
H.264 and H.265 length-prefixed samples are converted to canonical four-byte
Annex-B start codes. AVC/HEVC parameter sets, or AV1 configuration OBUs, are
prepended to the first keyframe after open or seek. `seek` selects the closest
preceding random-access sample by presentation timestamp. EOS is explicit,
`close` is idempotent, and reads after close fail with `FailedPrecondition`.
Drop performs no seek, read, flush, or manufactured finalization.

The current session supports classic sample-table MP4 with H.264, H.265, AV1,
or VP9. Fragmented MP4, WebM/Matroska, MPEG-TS, network sources, audio tracks,
and muxing remain Planned rather than silently routed through an external
process or hidden fallback.

## Planned backing extension

Native decoder output needs one retained private backing variant rather than
raw public Vulkan handles. That variant must own or retain:

- plane format and subsampling, including at least qualified NV12 and P010;
- coded and visible extents plus row/plane strides;
- image layout and queue-family state inside the runtime boundary;
- exact producer completion and the decoder pool slot lifetime;
- a consumer-completion edge before a recyclable slot can be reused.

Adding native backing must preserve the current `VideoFrame` value contract.
It must not force native images into Matrix, pretend a Texture is an Image, or
expose `ash::vk` handles through `oa::video`.

## Session order

The remaining video checkpoints are dependency ordered:

1. connect demux codec configuration to an exact decode profile;
2. one decoder session with explicit open, decode, flush, and close behavior;
3. native frame-plane retention and exact video-to-compute synchronization;
4. YUV-to-packed-Image conversion checked against an independent oracle;
5. display-order handling and a complete-file FFmpeg differential stream;
6. fragmented/container breadth, encoding, capture, recording, and playback one qualified
   capability row at a time.

`VideoDecoder`, `VideoEncoder`, demuxers, muxers, capture devices, and players
remain absent from source until their state machine and one working path ship.
Drop will release safe ownership only; it will not drain, flush, submit, wait,
finalize a container, or conceal close failure.

## Verification

The current tests cover root/module type identity, timing and color metadata,
known-versus-unknown duration, packed Image retention and exact readback,
single-frame batch validation, zero-extent rejection, mixed Annex-B start-code
parsing, canonical emission, borrowed payload identity, H.264/H.265 parameter
extraction, malformed length-prefixed NAL rejection, hardware/device-session
capability separation, exact H.264/H.265/AV1 profile limits on advertised
implementations, bound session memory for each advertised typed codec profile,
profile-qualified output/DPB image allocation, live H.264 session-parameter
creation, single-VCL Vulkan bitstream normalization, aligned upload, result
status, cross-family image readback, exact FFmpeg pixel-oracle agreement, and
open/read/seek/close over H.264, H.265, AV1, and VP9 donor MP4 fixtures.
Hardware-backed frame tests use the same Matrix storage path as Image. The
codec proof remains deliberately private and one-picture-only.
