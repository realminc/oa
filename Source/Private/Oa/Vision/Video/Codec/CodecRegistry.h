// OA Vision — Video Codec Registry
// Singleton registry for codec parsers (H.264, H.265, AV1, VP9)

#pragma once

#include "VideoCodecParser.h"
#include <Oa/Core/Std/HashMap.h>
#include <Oa/Core/Std/UniquePtr.h>

// Forward declaration
enum class OaVideoCodec : OaU32;
class OaVideoCodecParser;

// Codec registry - manages codec parser instances
class OaVideoCodecRegistry {
public:
	// Get the global registry instance
	static OaVideoCodecRegistry& GetInstance();

	// Get parser for a specific codec (returns nullptr if not registered)
	OaVideoCodecParser* GetParser(OaVideoCodec InCodec);

	// Create independent parser state for one decoder/stream. The registered
	// parsers remain available as stateless test/discovery entry points, but
	// their mutable parameter-set caches must never be shared by live decoders.
	OaStdUniquePtr<OaVideoCodecParser> CreateParser(OaVideoCodec InCodec) const;

	// Register a codec parser (called during static initialization)
	void RegisterParser(OaVideoCodec InCodec, OaStdUniquePtr<OaVideoCodecParser> InParser);

	// Register all built-in codec parsers (call this during decoder initialization)
	void RegisterAllParsers();

private:
	OaVideoCodecRegistry() = default;
	~OaVideoCodecRegistry() = default;

	// Non-copyable, non-movable
	OaVideoCodecRegistry(const OaVideoCodecRegistry&) = delete;
	OaVideoCodecRegistry& operator=(const OaVideoCodecRegistry&) = delete;

	OaStdHashMap<OaVideoCodec, OaStdUniquePtr<OaVideoCodecParser>> Parsers_;
};
