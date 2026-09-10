# OA Rust media boundary

**Status:** Planned coordination architecture; independent Audio sessions and
packed VideoFrame are Experimental

**Updated:** 2026-09-10

**Authority:** [OA Rust Architecture](../architecture/oaArchitecture.md)

**Donor references:** OA C++ `docs/internal/audio/oaAudio.md`,
`docs/internal/vision/oaVideo.md`, and the Audio/Video architecture roadmaps

Media is the coordination layer for timed heterogeneous tracks. It does not
own the Audio, Image, VideoFrame, Texture, or Scene value definitions and is
not a parent namespace for every operation on them.

## Boundary

```text
audio                       video
  Audio                       VideoFrame
  decode/encode               decode/encode
  capture/output              demux/mux/capture
       \                       /
        \                     /
         media source + track coordination
           MediaClock
           Transport
           MediaPlayer
                    |
          AudioOutput + render Presenter
```

The shared user vocabulary—play, pause, stop, seek, rate, loop, position, and
end-of-stream—belongs to transport state. Payload processing remains typed.
Audio output owns device callback/ring behavior. Video decoding owns codec and
frame-pool state. Presentation consumes an Image, VideoFrame, Texture, or
rendered Scene through Render. Scene animation uses its own evaluation
contract and may share time primitives without pretending to be a compressed
media track.

## Contracts

| Kind | Examples | Responsibility |
|---|---|---|
| Value | `MediaTimestamp`, `TrackId`, source/track descriptions | Immutable backend-neutral metadata |
| Operation | bounded container inspection or timestamp conversion | Stateless and no retained protocol state |
| Session | `Demuxer`, `MediaPlayer` | Source state, track coordination, seek/flush, clock, terminal close |
| Sink session | `AudioOutput`, `Presenter`, recorder | External device or file lifecycle and backpressure |

`MediaPlayer` is admitted only for synchronized multi-track playback. It
composes Audio and Video sessions rather than replacing them. A universal
`Player<T>` is rejected until genuinely shared state and failure behavior can
be expressed without erasing payload-specific ownership.

## Module relationship

`audio`, `video`, and `image` remain direct public modules for ergonomic and
semantic stability. `media` imports their public values or private adapters as
needed. Neither of these shapes is permitted:

```text
media::audio::... as the only route       rejected deep ownership hierarchy
vision::video::...                        rejected historical C++ grouping
```

The intended public shape is:

```rust,ignore
let clip = oa::audio::decode_file(&engine, path)?;
let frame = video_decoder.decode(packet)?;
let player = oa::media::MediaPlayer::open(&engine, source)?;
let same_player_type = oa::MediaPlayer::open(&engine, source)?;
```

The names illustrate the target only; unimplemented declarations remain in
documentation rather than source stubs. `oa::MediaPlayer` is an explicit root
identity re-export of the media-owned type, not a second player abstraction.

## Image, VideoFrame, and Texture

- Image owns pixel meaning and remains reusable by Vision, Render, UI, Plot,
  codecs, and host sinks.
- VideoFrame owns plane, timing, producer-readiness, and retained-lifetime
  semantics.
- Texture owns sampling and render-usage semantics over admitted image or
  buffer backing.
- Zero-copy adapters preserve all source metadata and exact completion. They do
  not expose raw Vulkan handles or silently change format/layout.

## Session lifecycle

Every media session specifies:

- configuration and capability negotiation;
- `Created`, `Open`, active/paused, draining, failed, and closed transitions as
  applicable;
- queue/ring bounds, backpressure, underrun/overrun accounting, and thread
  behavior;
- exact producer/consumer readiness and timestamp discontinuities;
- seek and flush behavior for codec reference state and queued output;
- explicit `close`, `flush`, `drain`, or `abort` failure boundaries;
- non-blocking `Drop` that only releases safe host ownership.

## Port order

1. Complete Image value and one upload/transform/readback oracle.
2. Extend the Experimental packed-Image VideoFrame with native planes while
   preserving color/timing metadata, readiness, and retained lifetime.
3. Port one bounded demux/decode session and compare decoded output to an
   independent oracle.
4. Qualify the independent AudioPlayer/AudioCapture device sessions and add a
   Render Presenter as a separately owned sink.
5. Introduce MediaClock and synchronized A/V playback only after independent
   Video session behavior and cross-track clock policy are proven.
6. Add recording, external memory, and broader codecs one capability row at a
   time.

Vulkan Video, platform codecs, sound libraries, and WSI remain private
backends. Device or vendor support is reported from queried capabilities and
named validation evidence, never from module presence.
