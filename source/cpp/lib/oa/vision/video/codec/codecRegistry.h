// OA Vision — Video codec registry
// Singleton registry for codec parsers (H.264, H.265, AV1, VP9)

#pragma once

#include "videoCodecParser.h"
#include <oa/core/std/hashMap.h>
#include <oa/core/std/uniquePtr.h>
#include <oa/vision/type.h>

namespace oa {

// codec registry - manages codec parser instances
class VideoCodecRegistry {
public:
	// get the global registry instance
	static VideoCodecRegistry& getInstance();

	// get parser for a specific codec (returns nullptr if not registered)
	VideoCodecParser* getParser(oa::VideoCodec inCodec);

	// Create independent parser state for one decoder/stream. The registered
	// parsers remain available as stateless test/discovery entry points, but
	// their mutable parameter-set caches must never be shared by live decoders.
	oa::UniquePtr<VideoCodecParser> createParser(oa::VideoCodec inCodec) const;

	// register a codec parser (called during static initialization)
	void registerParser(oa::VideoCodec inCodec, oa::UniquePtr<VideoCodecParser> inParser);

	// register all built-in codec parsers (call this during decoder initialization)
	void registerAllParsers();

private:
	VideoCodecRegistry() = default;
	~VideoCodecRegistry() = default;

	// Non-copyable, non-movable
	VideoCodecRegistry(const VideoCodecRegistry&) = delete;
	VideoCodecRegistry& operator=(const VideoCodecRegistry&) = delete;

	oa::HashMap<oa::VideoCodec, oa::UniquePtr<VideoCodecParser>> parsers_;
};

} // namespace oa
