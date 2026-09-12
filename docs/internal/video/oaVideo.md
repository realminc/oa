# OA Rust Video

**Status:** Experimental packed/planar/Texture/native frame value, FnVideo
operations, Vulkan Video device and exact decode-profile queries, bounded MP4
demux, streaming MP4 mux, public H.264/H.265/AV1/VP9 decode, and composed local
playback sessions

**Updated:** 2026-09-12

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
type. Admitted backing is one packed `Image`, one retained packed RGBA8
`Texture`, host-retained planar 8-bit YUV420, or one private native decoded-image
slot. Packed Image construction
rejects batched Images with a batch extent other than one and rejects zero
spatial extents. Texture adaptation shares its render-owned storage and Matrix
producer readiness without copying pixels or erasing it into Image. Planar
construction requires nonzero even geometry and exactly
`width * height * 3 / 2` bytes. The frame retains:

- visible width and height through its semantic backing;
- a non-negative presentation timestamp relative to the producer's epoch;
- an optional, strictly positive duration;
- source matrix-coefficient and component-range metadata;
- backing storage and producer readiness.

Cloning retains the same semantic storage. It does not copy pixels or create a
second completion event. `as_image`, `into_image`, `as_texture`,
`into_texture`, and `as_yuv420p` are backing-specific optional observations,
so Texture, planar host bytes, and a native multi-plane image cannot be
misrepresented as one dense Image. Native decoded frames additionally expose
their backend-neutral plane format and exact producer `Event`. They never
expose raw Vulkan handles.

The current color record is deliberately incomplete. `Bt601`, `Bt709`, and
`Bt2020` identify source matrix coefficients only. Color primaries, transfer
function, chromatic adaptation, mastering metadata, and HDR remain Planned and
must not be inferred from that enum.

## Current FnVideo operations

The donor's `FnVideo::fromTexture` maps to `video::from_texture`. It retains a
buffer-backed Texture in a VideoFrame with explicit timing and color metadata;
the current Texture's Matrix readiness remains the producer-completion edge.
The adapter performs no copy, submission, wait, or Image conversion. Native
image-backed Texture remains later backing work; native decoder images use a
separate retained backing because they are recyclable codec resources rather
than render Textures.

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

There is no `FnVideo` compatibility type and no duplicate root operation. NAL
helpers are bounded host bitstream operations, while `from_texture` is a typed
zero-copy value adapter; neither is a decoder session command.

AV1 begins with a separate OBU boundary rather than pretending its stream is
Annex-B. `parse_av1_obus` returns borrowed payload views with exact header
offset/size, validates forbidden/reserved bits, bounds LEB128 sizes to eight
bytes, and rejects truncation or payload overrun. `inspect_av1_access_unit`
accepts raw MP4 access units or the first frame of an IVF stream and reports
sequence-header, frame, frame-header, and tile-group counts plus an optional
IVF timestamp. All packets in the 60-sample AV1 donor pass this structural
inventory, including access units whose picture count is greater than one.

`parse_av1_sequence_header` is the first semantic AV1 parameter checkpoint. It
returns owned backend-neutral profile, coded extent, operating-point
level/tier/delay, timing, frame-ID, coding-tool, bit-depth, monochrome, chroma,
range, color-description, and film-grain metadata. Reserved profiles, invalid
frame-ID widths, and every truncated field fail closed. The reduced
still-picture branch follows AV1 section 5.5.1: operating-point IDC zero is
implicit and only the level is read. This deliberately corrects the C++ donor
parser, which currently consumes a nonexistent 12-bit IDC in that branch and
would misalign an AVIF-style reduced header. Synthetic reduced-header evidence
locks the correction; the complete donor MP4 locks Main, 8-bit, 4:2:0, and
1280-by-720 extraction.

`parse_av1_frame_header` consumes the uncompressed-header syntax against an
explicit eight-slot `Av1ReferenceState`. It retains frame/show identity,
refresh and named-reference mapping, order hints, frame sizing, interpolation,
uniform tile geometry, quantization, segmentation, delta-Q/LF, loop-filter,
CDEF, restoration, transform, skip-mode, and supported motion/film-grain flags.
Reference refresh is an explicit caller operation; the stateless parser does
not hide stream state. CDEF secondary strength value three is normalized to
four as specified rather than preserving the donor parser's raw coded value.
`parse_av1_tile_group` then returns exact checked access-unit-relative tile
offsets and sizes for combined Frame or separate TileGroup OBUs. Every coded
and show-existing picture and every tile range in the complete 60-packet donor
passes. Frame IDs, decoder-model removal timing, short reference signaling,
non-uniform tile spacing, non-identity global motion, and applied film grain
fail explicitly. The Vulkan backend now lowers every retained picture field to
owned Khronos StdVideo tables, including `diff_uv_delta`, packed segmentation
and loop-filter masks, restoration sizes, and Q16 affine identity global
motion. A transactional DPB planner resolves the eight logical AV1 reference
map entries to bounded physical slots, deduplicates active references, handles
show-existing without allocating a decode destination, observes retained-frame
lease exclusions, and commits no state when planning fails. Command recording
now consumes that plan, binds each distinct active reference exactly once,
keeps the reconstruction association inactive until decode, supplies exact
frame-header/tile offsets into the uploaded OBU access unit, and uses the same
explicit DPB/output barriers, result-status query, and readback ownership round
trip as the qualified AVC/HEVC paths. On Intel Iris Xe with Mesa 26.2.2, the
first donor keyframe reports `COMPLETE` and all 1,382,400 YUV420 bytes match an
independent FFmpeg decode through a fixed SHAKE-256 oracle. The public decoder
now applies that route to every coded, hidden, and show-existing picture in the
complete 60-packet donor stream. Both host-planar output and retained-native
slot readback match all 60 FFmpeg display frames byte-for-byte while an early
native frame remains leased across later DPB recycling.

VP9 uses a separate transactional parser because it has neither Annex-B NALs
nor AV1 OBUs. Raw MP4 frames and complete IVF frames are accepted; a trailing
superframe index is validated and every component is parsed before stream state
commits. The parser retains profile/color configuration, reference-derived
geometry, loop-filter deltas, segmentation features and probabilities,
quantization, interpolation, tile layout, hidden-picture state, and the eight
logical reference-buffer extents. Malformed superframes, truncated fields, or
show-existing references to an uninitialized buffer leave prior state
unchanged. Public admission is VP9 Profile 0, 8-bit 4:2:0 with fixed coded and
render geometry.

The Vulkan VP9 planner maps those eight logical buffers onto bounded physical
slots, deduplicates the three named references, excludes retained native
leases, resolves show-existing without decode, and commits transactionally.
Standard-video lowering supplies complete color, loop-filter, segmentation,
picture-flag, quantizer, reference-name, and header/tile-offset records. VP9
has no session-parameter object. On Intel Iris Xe with Mesa 26.2.2, all 60
donor display frames match independent FFmpeg YUV420 output byte-for-byte
through both host-planar and retained-native paths.

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
tiles, deblocking, and session-relevant flags are retained. VPS decoded-picture
buffer limits, layer-set membership, timing, and HRD tables are typed rather
than skipped. SPS profile-tier-level, PCM, VUI aspect/signal/colour/chroma,
display-window, timing, HRD, and bitstream-restriction syntax are retained as
well. PPS non-uniform tile widths and heights are owned explicitly. SPS/PPS scaling-list
syntax is no longer discarded: prediction references, default matrices, DC
coefficients, and diagonal coefficient walks resolve into owned raster-order
4x4, 8x8, 16x16, and 32x32 matrices suitable for a future standard-video
parameter object. Invalid backward references and coefficient ranges fail
closed. SPS short-term reference sets now retain direct syntax masks and delta
arrays; inter-predicted sets retain their predictor masks and resolve ordered
delta POCs for the next set instead of losing predecessor cardinality. SPS
long-term POC-LSB and current-picture flags are typed as well. Counts that would
drive unbounded parser work are rejected before iteration. Range, multilayer,
3D, and screen-content extensions fail with `MissingCapability`; the parser
does not return an incomplete baseline-profile record for those streams.

`parse_h265_slice_header` resolves coded slices against explicit SPS/PPS
records and retains slice address, type, temporal/reference identity, output
intent, coded POC-LSB, and bounded inline short-term reference deltas. Checked
CTB geometry prevents damaged dimensions from overflowing address derivation.
The parser consumes `short_term_ref_pic_set_sps_flag` even when the SPS declares
zero reference-picture sets; treating that syntax bit as conditional shifted
the inline RPS and produced impossible reference POCs on the second donor
picture. Fixture coverage locks the corrected P-picture POC and `-5` reference
delta.
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
are separate fields. `decoder_sessions_available()` is true only when the
Engine owns a decode queue, the device advertises at least one shipped decoder
codec, and the selected queue supports the result-status queries required by
the shipped decoder.
`video::query_encode_capabilities` now performs the corresponding exact H.264
High or H.265 Main profile query. It reports common coded extents, access and
input granularity, bitstream alignments, DPB/reference limits, rate-control
modes, feedback support, bitrate/quality limits, and typed codec-specific
slice, reference, level, QP, tile, CTB, and transform-block limits.
`video::query_encode_formats` separately enumerates input and reconstructed-DPB
formats with the same recognized/unknown accounting as decode. The live
Intel/Mesa query accepts both donor profiles and exposes usable format rows.

These are discovery contracts, not a codec session:
`encoder_sessions_available()` remains false until create/encode/flush/close is
qualified. Engine construction requests each unique advertised video queue
family and enables only the reported base, queue, and codec extension chain.
`video_queues_enabled()` therefore records logical-device admission. On the tested Intel
Iris Xe / Mesa 26.2.2 device, creation succeeds with its dedicated combined
decode/encode queue and advertised H.264, H.265, AV1, and VP9 decode support.
The Engine now lazily owns a command pool for the selected decode family;
commands retain their originating pool identity through timeline retirement,
and a live-device test submits, waits, and retires an empty decode-family
command. This proves queue plumbing only, not codec-session qualification.

The private runtime admission test now also creates, binds, and destroys one
session for every advertised typed H.264, H.265, AV1, and VP9 profile on that
device. Creation reuses the exact profile query's standard-header version,
chooses profile-compatible output and DPB formats, bounds the memory-requirement
count, constrains each VMA allocation to its reported memory-type mask, and
binds the exact required size. The H.264 and H.265 paths additionally convert
the corresponding donor MP4 parameter sets into Khronos `StdVideoH264` or
`StdVideoH265` records and successfully create and destroy dependent
`VkVideoSessionParametersKHR` objects on the same device. The HEVC conversion
supplies VPS/SPS/PPS records, including profile/level, DPB, scaling,
short-/long-term references, PCM, VUI/HRD, and tile backing whose pointers
remain live through object creation. The conversion preserves IDs,
profile/level, coded geometry, POC
fields, reference limits, signed QP offsets, and represented flags; no Vulkan
type crosses the runtime boundary. Typed VUI parsing retains aspect ratio,
overscan, signal/colour, chroma location, timing, NAL/VCL HRD, picture-structure,
and bitstream-restriction syntax. The Vulkan conversion supplies the exact
`StdVideoH264SequenceParameterSetVui` and optional HRD table; distinct NAL and
VCL HRD tables fail closed because the standard-video record has one pointer.
Sequence- and picture-level scaling lists retain their presence/default masks
and all 4x4/8x8 scan-order values, then lower without reordering into the
corresponding `StdVideoH264ScalingLists`. This proves bounded H.264 and H.265
parameter-object ownership; the command qualifications below separately prove
the submitted codec paths. AV1 OBU structure and backend-neutral sequence
header/frame/tile parsing and standard-video sequence lowering are implemented.
The local Intel/Mesa device accepts a `VkVideoSessionParametersKHR` created from
the exact donor sequence metadata. AV1 and VP9 frame standard-video lowering,
transactional DPB planning, command submission, result queries, and complete
stream differentials are implemented and contribute to public decoder
admission.

Session admission now allocates the first native decode image set as well. The
runtime queries formats for the exact profile and intended decode-output/DPB
usage, carries the required profile list into image creation, and preserves the
driver's format, tiling, component mapping, and applicable creation flags. It
strips `VIDEO_PROFILE_INDEPENDENT_KHR` while supplying the explicit profile
chain because the maintenance feature is not enabled. Profiles that permit
coincident resources use combined output/DPB usage. When the profile also
advertises separate reference images, each DPB slot owns a one-layer image;
otherwise one layered image backs the slots. Distinct-only profiles receive
one layered output image and one layered DPB image. Disjoint
multi-plane formats fail closed until per-plane memory binding is implemented.
The session owns view-before-image-before-allocation teardown. Live creation and
destruction pass for every advertised typed H.264, H.265, AV1, and VP9 profile on
the local Intel/Mesa device. Images begin in `UNDEFINED`; the H.264 and H.265
recorders transition reconstructed/output layers explicitly. Recyclable output
slots are protected by native frame-clone lifetime and, when registered, the
latest same-Engine consumer-completion event.

The first private encoded-bitstream owner now creates a
`VIDEO_DECODE_SRC_KHR` buffer with the same exact profile-list chain. Packet
length is rounded up with checked arithmetic to the profile's reported minimum
bitstream-size alignment; offset zero satisfies the separately validated
offset alignment. Each source buffer has a dedicated allocation so replacement
packets remain bound at memory offset zero. This is required by the qualified
Intel/Mesa path, whose command recorder maps and parses the source BO on the
CPU; a VMA suballocation at a non-zero binding offset produced an inaccessible
mapping during the second picture. The complete submitted range is initialized—packet bytes
followed by a zeroed alignment tail—and host writes are flushed before the
buffer is retained. Before upload, each H.264/H.265 recorder extracts its
single VCL NAL from the demuxed access unit and emits the canonical three-byte
byte-stream prefix consumed by the Khronos/FFmpeg Vulkan Video paths. SPS, PPS,
and SEI NALs remain represented through session parameters or host metadata;
they are not smuggled into the submitted coded-slice range. Unit coverage locks
this normalization and rejects zero- or multi-slice access units until the
general decoder owns a slice-offset table. Reuse across in-flight pictures
waits for explicit epoch retirement rather than destructor synchronization.

The executable H.264 and H.265 checkpoints record every picture in their
60-sample donor streams, including P/B pictures and reordered POC. Each command
begins the video coding scope with the bound session and parameters, resets at
the planned random-access boundary, transitions the DPB and output resources,
publishes the host-written bitstream through a HOST→VIDEO_DECODE barrier, and
ends the scope. Submission uses the decode-family pool and queue while the
Engine timeline supplies the completion/retirement edge. An exact-profile
`RESULT_STATUS_ONLY_KHR` query encloses every decode, and all operations return
`COMPLETE` on the recorded Intel/Mesa device. Each decoded slot is then
released to the compute family, copied into retained host-visible storage,
normalized from NV12 to planar YUV420, and returned to the decode family before
reuse. Presentation ordering matches all 60 independent FFmpeg frames
byte-for-byte for AVC High and HEVC Main. This evidence applies to those exact
streams and device provenance; unsupported profiles and bitstream shapes still
fail closed.

The private HEVC session owns a transactional host DPB planner initialized
from the admitted session slot count. It derives wrap-aware picture order,
resets on IDR/BLA/no-output boundaries, retains current/following short-term
references, resolves current-before/current-after POC deltas to exact slots,
and recycles only free or non-reference slots. A failure leaves the prior DPB
state unchanged. The donor HEVC fixture plans every demuxed picture, exercises
non-monotonic decode-order POC, and performs exactly one reset with every
resolved slot inside the SPS bound. The reusable Vulkan recorder consumes each
plan transactionally, binds the current-before/current-after lists and active
slot POCs, uses an inactive current resource at begin-coding, resets only at
the planned random-access boundary, and orders decode writes before later DPB
reads. On Intel/Mesa 26.2.2 all 60 donor operations return `COMPLETE`. The
recorder remains serial and materializes host frames rather than retaining
native output values for consumers.

The H.264 planner and recorder admit progressive POC-type-zero pictures. They
derive wrap-aware full POC, reset at IDR, allocate the lowest free or oldest
non-reference slot, bind every active short-/long-term association, and apply
sliding-window eviction plus MMCO 1, 5, and 6 transactionally. Long-term MMCO
2–4 and multi-slice pictures fail closed. Every operation in the reordered AVC
High donor returns `COMPLETE`, and PTS ordering matches all 60 independent
FFmpeg frames byte-for-byte.

## Current H.264/H.265/AV1/VP9 decoder session

`oa::VideoDecoder` and `oa::video::VideoDecoder` are identity aliases for the
same lifecycle-bearing session. `VideoDecoder::create(&engine, info)` connects
one `VideoDemuxer` track description to its exact codec profile and the sole
Engine-owned Vulkan runtime. Admitted profiles are progressive H.264
Baseline/Main/High, H.265 Main, AV1 Main without film grain, and VP9 Profile 0,
all 8-bit 4:2:0. Other codecs, layouts, and profiles fail with `MissingCapability`
rather than falling back to FFmpeg or another hidden decoder.

The first random-access packet supplies SPS/PPS, VPS/SPS/PPS, an AV1 sequence
header, or an in-band VP9 keyframe. AVC/HEVC currently admit one complete slice
per picture. AV1 assembles
combined Frame OBUs or separate FrameHeader plus contiguous TileGroup OBUs,
decodes hidden pictures, and resolves show-existing pictures without issuing a
second decode. VP9 parses every complete superframe component, decodes hidden
pictures, and resolves show-existing through its logical reference map. The
session creates standard-video parameters where the codec requires them, retains the
parsed parameter records and transactional DPB planner, submits packets in
decode order, verifies each result query, performs the explicit decode/compute
ownership round trip, and buffers a host-retained planar YUV420 `VideoFrame`.
Frame timing comes from demuxed packet PTS/duration and the exact track time
base; source range and known BT.601/709/2020 coefficients come from AVC/HEVC
VUI, AV1 color metadata, or VP9 color-space configuration.

`decode` performs synchronous host materialization, then uses the codec SPS
reorder bound (or the conservative 16-picture AVC bound when VUI omits one) to
emit the lowest-PTS frame only when safe;
it returns `None` while the display queue must retain more pictures. `flush`
returns the delayed tail in presentation order, establishes a new random-access
boundary, and makes the next packet require a keyframe. `close` is explicit and
idempotent. Drop only releases already-idle resources: it does not submit,
wait, flush, or fabricate frames. Complete public-session tests consume all 60
display frames from each AVC, HEVC, AV1, and VP9 donor directly in decoder-emitted
order and match each FFmpeg display-order YUV420 stream byte-for-byte. The
AVC/HEVC tests additionally flush, seek, and redecode the first keyframe
through the same session.

`decode_native` follows the same parser, DPB, result-status, and display-order
path but omits the decode-to-host pixel copy. It currently waits for and
verifies the exact decode submission before publishing a frame. The returned
`VideoFrame` retains the qualified NV12 or planar 8-bit 4:2:0 image set and its
DPB/output slot. The planner excludes slots held by any live frame clone or a
registered consumer event. `mark_consumed` accepts only an event from the same
Engine whose timeline point does not precede frame readiness; the retirement
service retains the slot pool through that event even if the frame, decoder,
and original Engine handle are dropped. Dropping the last clone without a
consumer event declares that no asynchronous consumer remains. No Drop path
submits or waits. Host and native output modes cannot mix until `flush`
establishes a new random-access boundary.

`VideoDecoder::read_native_yuv420p` is the first decoder-owned native-frame
observation. It validates that the frame belongs to that exact decoder, waits
for producer readiness, releases the retained image layer from the decode queue,
copies its planes on the compute queue, and restores codec layout and decode
ownership before returning tightly packed planar 8-bit YUV420. The final
submission is registered as a frame consumer completion. Repeated observation
therefore reads the named historical frame rather than whichever decoder slot
was written most recently. The same-family path is qualified locally; the
explicit split-family release/acquire path remains unqualified on available
hardware.

This native path proves retention, exact observation, and safe recycling, not
public asynchronous decode or direct Render/ML consumption. Visible crop and
plane-stride metadata, packed RGB conversion, multi-slice AVC/HEVC, complete
AVC long-term MMCO handling, P010/P012 decode admission, and broader AV1/VP9
syntax remain open parity work. The admitted AV1 path fails closed on frame
IDs, decoder-model removal timing, short reference signaling, non-uniform tile
spacing, per-picture size override, super-resolution, distinct render size,
non-identity global motion, and applied film grain.

## Current player session

`oa::VideoPlayer` and `oa::video::VideoPlayer` are identity aliases for the
same composed local playback session. `VideoPlayer::open` owns one demuxer and
one decoder, decodes through the first display-order frame before returning,
and exposes that retained frame immediately. `advance` presents exactly one
frame independent of play/pause state; `tick` uses checked `Duration` pacing
and discards paused elapsed time. The session supports play, pause, toggle,
loop-policy changes, non-looping EOS, timestamp seek through the preceding
keyframe, reset, explicit flush, and idempotent close. `step_backward` and
signed `step_frames` operate in display order. The player first presents a
retained frame without decoding; an evicted-history miss flushes codec state,
seeks through a bounded preceding presentation window, and deterministically
replays through the requested display index. Its counters distinguish submitted
packets, presented frames, seek resets, loop restarts, cache hits/misses, and
replayed frames.

`seek_frame` selects an absolute zero-based display index. The player builds a
bounded timestamp index from the validated container sample table and resolves
each decoded frame back to that index by exact presentation time. Decode order
therefore cannot silently relabel B-frames as display order; a timestamp absent
from the container index fails as data loss. Nonresident seeks retain only the
bounded presentation window preceding the target.

The H.264 donor lifecycle test proves immediate first-frame identity, monotonic
display PTS across all 60 frames, non-looping EOS, reset, paced advance, and
close. A separate exact-frame test bounds retained history by both frame count
and planar byte budget, proves backward and subsequent forward cache hits do
not submit another packet, and hashes replayed evicted history against the
original frame. The lifecycle oracle also seeks directly to frame 30 after EOS
and matches its earlier linear-playback hash, exercising reordered H.264
presentation. `VideoFrame` clones share immutable planar backing, so cache
retention does not copy pixel storage. Explicit timestamp seek, flush, and loop
restart invalidate history. This first player does not yet provide audio, a
shared A/V clock, RGBA conversion, network sources, or native consumer leases.
Those belong to later Media and native-frame checkpoints rather than being
hidden fallbacks.

`video::query_decode_capabilities` performs the second, exact query for one
complete `VideoDecodeProfile`. H.264, H.265, AV1, and VP9 profiles carry their
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
decode or encode plus codec-specific capability structures under
`VkVideoCapabilitiesKHR`. Extension advertisement is checked before dispatch,
and Vulkan's profile-operation, profile-format, and profile-codec rejection
results become `MissingCapability` instead of being confused with runtime
failure. No `ash::vk` type crosses the public API. Released Ash 0.38 / Vulkan
Headers 1.3.281 remains the sole loader and resource owner required by
`vk-mem`. An exact official Ash revision generated from Vulkan Headers 1.4.350
supplies only the missing standard VP9 C-ABI structs. Those values are confined
to raw `pNext` payloads; no handle, loader, dispatch call, or allocator type
crosses between the two Ash versions. This bridge can be deleted once the main
released Ash dependency contains `VK_KHR_video_decode_vp9` without breaking
`vk-mem` compatibility.

## Current MP4 demux session

`VideoDemuxer` is the first admitted stateful Video session. It opens one local,
unfragmented ISO-BMFF/MP4 source, chooses the first supported video track, and
builds a bounded validated sample index. The public values are
`VideoContainerInfo`, `VideoTimeBase`, `VideoPacket`, `VideoCodec`, and
`VideoContainerKind`.

For AVC, the demuxer derives profile, chroma precision, and progressive versus
interlaced picture layout from the first bounded SPS in `avcC`. For HEVC it
reads the profile, chroma, and component depths from `hvcC`; for AV1 it reads
the profile and format fields from `av1C`; for VP9 it reads profile, depth, and
chroma from the `vpcC` FullBox without prepending that metadata to coded frame
bytes. `VideoContainerInfo::decode_profile`
therefore feeds the exact capability and format queries directly when the
stream profile is representable. It returns `None` for standard-video profiles
outside the typed binding rather than inventing a compatible profile.

Media payload remains in the file until `read_next_packet`; the demuxer bounds
movie metadata, sample-table expansion, and individual packet allocation.
H.264 and H.265 length-prefixed samples are converted to canonical four-byte
Annex-B start codes. AVC/HEVC parameter sets, or AV1 configuration OBUs, are
prepended to the first keyframe after open or seek. `seek` selects the closest
preceding random-access sample by presentation timestamp. EOS is explicit,
`close` is idempotent, and reads after close fail with `FailedPrecondition`.
Drop performs no seek, read, flush, or manufactured finalization.

The current session supports classic sample-table MP4 with H.264, H.265, AV1,
or VP9. Fragmented MP4, WebM/Matroska, MPEG-TS, network sources, and audio-track
demux remain Planned rather than silently routed through an external process or
hidden fallback.

## Current MP4 mux session

`VideoMuxer` is the matching stateful packet sink for H.264/H.265 MP4. It writes
media bytes incrementally after a fixed `ftyp` plus extended-size `mdat`
header, converts each Annex-B access unit to four-byte length-prefixed samples,
and retains only the metadata needed to build the final movie box. Callers
install raw SPS/PPS or VPS/SPS/PPS records explicitly. The final AVC record
derives profile, compatibility, and level bytes from SPS; the current HEVC
record is the donor's Main, 8-bit, 4:2:0 contract.

`EncodedVideoPacket` owns a non-empty access unit, microsecond presentation and
decode timestamps, and its random-access flag. The simple constructor makes
both timestamps equal; `with_timestamps` represents reordered streams. Packets
enter in strictly increasing decode order while presentation timestamps may
reorder. The muxer emits signed version-one `ctts` runs whenever presentation
differs from decode time. Sample tables preserve variable decode-timestamp
deltas, select `stco` or `co64` from actual offsets, and reject size, duration,
composition-offset, and version-zero timestamp overflow rather than truncating
it.

An optional second track accepts only `EncodedAudioPacket` PCM-S16 frames with
checked sample-rate/channel/duration agreement. Encoder priming is represented
by an edit list. Other audio codecs and arbitrary codec configuration blobs are
not exposed because no corresponding MP4 sample-entry implementation ships.

`finalize` requires at least one video packet and complete codec configuration,
patches the `mdat` size, appends `moov`, flushes, and closes. `close` abandons
the unfinished stream and is idempotent. Drop only releases the file handle: it
does not finalize, flush a codec, submit work, wait, or turn an interrupted
recording into an apparently complete file.

## Native backing contract

Native decoder output uses one retained private backing variant rather than raw
public Vulkan handles. The implemented variant owns or retains:

- the qualified NV12 or planar 8-bit 4:2:0 plane format;
- coded extent;
- image layout and queue-family state inside the runtime boundary;
- exact producer completion and the decoder pool slot lifetime;
- a consumer-completion edge before a recyclable slot can be reused.

It preserves the `VideoFrame` value contract without forcing native images into
Matrix, pretending a Texture is an Image, or exposing `ash::vk` handles through
`oa::video`. Visible crop, plane strides, 10/12-bit formats, and typed internal
views for Render/ML consumers remain required extensions.

## Session order

The remaining video checkpoints are dependency ordered:

1. asynchronous native-frame decode and display-order release;
2. typed Render/ML native-plane consumption and YUV-to-packed-Image conversion
   checked against an independent oracle;
3. visible crop, plane-stride, and qualified higher-bit-depth metadata;
4. broader AV1/VP9 syntax, then broader AVC/HEVC bitstream shapes;
5. encoding, capture, recording, and richer playback one qualified capability
   row at a time;
6. fragmented/container breadth.

`VideoEncoder`, capture devices, and recorders remain absent from source until
their state machine and one working path ship.
Drop will release safe ownership only; it will not drain, flush, submit, wait,
finalize a container, or conceal close failure.

## Verification

The current tests cover root/module type identity, timing and color metadata,
known-versus-unknown duration, packed Image retention and exact readback,
Texture retention without Image semantic erasure, single-frame batch
validation, zero-extent rejection, mixed Annex-B start-code
parsing, canonical emission, borrowed payload identity, H.264/H.265 parameter
extraction, malformed length-prefixed NAL rejection, hardware/device-session
capability separation, exact H.264/H.265/AV1/VP9 profile limits on advertised
implementations, bound session memory for each advertised typed codec profile,
profile-qualified output/DPB image allocation, live H.264/H.265/AV1 session-parameter
creation, single-VCL Vulkan bitstream normalization, aligned upload, result
status, cross-family image readback, exact FFmpeg pixel-oracle agreement,
full-fixture AVC/HEVC POC/DPB planning, 60-picture Vulkan submission with bounded
reference-slot resolution and per-picture `COMPLETE` result status, exact
decode/compute queue-family round trips, and a byte-exact full-stream FFmpeg
YUV420 differential,
open/read/seek/close over H.264, H.265, AV1, and VP9 donor MP4 fixtures,
synthetic multi-sample H.264 mux/demux, PCM-S16 second-track metadata, explicit
abandon behavior, signed composition-offset round-trip, and independent FFmpeg
decode of eight-packet remuxed donor AVC/HEVC streams including reorder timing.
Hardware-backed packed-frame tests use the same Matrix storage path as Image;
planar frames share immutable host bytes. The public AVC/HEVC/AV1/VP9 decoder has
reusable picture submission and complete-stream differential coverage. Separate
60-picture H.264, H.265, AV1, and VP9 native-output tests prove backend-neutral format and
extent reporting, absence of implicit host/Image/Texture materialization, exact
producer completion, consumer-event registration, decoder-identity rejection,
slot recycling, and native storage survival after decoder and original Engine
handles are dropped. Explicit retained-slot readback matches every display-order
frame byte-for-byte against independent FFmpeg planar YUV420 output for all
four codecs. AV1 parser tests cover bounded OBU structure, IVF framing, malformed
sizes, reserved/truncated sequence headers, the specification-correct reduced
still-picture branch, and exact profile/depth/chroma/extent metadata across the
complete donor MP4 packet inventory. The same inventory parses every
uncompressed header with evolving reference-map state and verifies every tile
range remains nonempty and bounded by its source access unit; synthetic cases
lock show-existing reference stability, one-tile offsets, and incompatible OBU
rejection. Public AV1 tests additionally cover complete packet assembly,
hidden/show-existing presentation, result-status completion, host caching by
physical reference slot, multi-lease native retention, and byte-exact retained
slot readback across the complete donor stream.
VP9 tests cover raw/IVF framing, complete and malformed superframe indices,
transactional parser/DPB failure, pointer-backed standard-video lowering,
complete donor packet inventory, exact typed profile/format queries, hidden and
show-existing handling, retained-slot recycling, and byte-exact host/native
FFmpeg differentials across all 60 display frames.
