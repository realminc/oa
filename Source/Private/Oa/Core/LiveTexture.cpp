// OaLiveTexture implementation

#include <Oa/Core/LiveTexture.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Core/Log.h>

// ─── Create ─────────────────────────────────────────────────────────────────────

OaResult<OaLiveTexture> OaLiveTexture::Create(
	OaEngine& InEngine,
	OaI32              InWidth,
	OaI32              InHeight)
{
	if (InWidth <= 0 or InHeight <= 0) {
		return OaStatus::Error("OaLiveTexture::Create: invalid dimensions");
	}

	OaLiveTexture live(InWidth, InHeight);
	live.Engine_ = &InEngine;

	// Allocate double-buffered textures
	// Both are buffer-backed RGBA8 for simplicity
	constexpr OaU32 bytesPerPixel = 4;  // RGBA8
	const OaU64 bufferSize = static_cast<OaU64>(InWidth) * static_cast<OaU64>(InHeight) * bytesPerPixel;

	auto buf1Result = InEngine.AllocBufferDevice(bufferSize);
	if (not buf1Result.IsOk()) {
		return OaStatus::Error("OaLiveTexture::Create: failed to allocate producer buffer");
	}
	live.ProducerTex_.DeviceBuf = std::move(buf1Result).GetValue();
	live.ProducerTex_.Width = InWidth;
	live.ProducerTex_.Height = InHeight;
	InEngine.RegisterBuffer(live.ProducerTex_.DeviceBuf);

	auto buf2Result = InEngine.AllocBufferDevice(bufferSize);
	if (not buf2Result.IsOk()) {
		InEngine.FreeBuffer(live.ProducerTex_.DeviceBuf);
		return OaStatus::Error("OaLiveTexture::Create: failed to allocate consumer buffer");
	}
	live.ConsumerTex_.DeviceBuf = std::move(buf2Result).GetValue();
	live.ConsumerTex_.Width = InWidth;
	live.ConsumerTex_.Height = InHeight;
	InEngine.RegisterBuffer(live.ConsumerTex_.DeviceBuf);

	return live;
}

// ─── Constructor / Destructor ───────────────────────────────────────────────────

OaLiveTexture::OaLiveTexture(OaI32 InWidth, OaI32 InHeight)
	: Width_(InWidth)
	, Height_(InHeight)
{
}

OaLiveTexture::~OaLiveTexture() {
	// Should be destroyed via Destroy() before engine shutdown
}

OaLiveTexture::OaLiveTexture(OaLiveTexture&& InOther) noexcept {
	MoveFrom(std::move(InOther));
}

OaLiveTexture& OaLiveTexture::operator=(OaLiveTexture&& InOther) noexcept {
	if (this != &InOther) {
		Destroy(*Engine_);
		MoveFrom(std::move(InOther));
	}
	return *this;
}

void OaLiveTexture::MoveFrom(OaLiveTexture&& InOther) noexcept {
	Width_  = InOther.Width_;
	Height_ = InOther.Height_;
	ProducerTex_ = std::move(InOther.ProducerTex_);
	ConsumerTex_ = std::move(InOther.ConsumerTex_);
	Sequence_.store(InOther.Sequence_.load(std::memory_order_relaxed), std::memory_order_relaxed);
	Engine_ = InOther.Engine_;

	InOther.Width_  = 0;
	InOther.Height_ = 0;
	InOther.Engine_ = nullptr;
}

// ─── Publish ───────────────────────────────────────────────────────────────────

void OaLiveTexture::Publish(const OaTexture& InTexture) {
	if (not IsValid() or not Engine_) {
		return;
	}

	if (not InTexture.IsValid()) {
		return;
	}

	// Copy the input texture into the producer buffer
	// For now, we do a synchronous copy. In a real implementation,
	// this would be an async GPU copy via OaContext.
	//
	// TODO: Integrate with OaContext for async copy
	// For now, we do a simple buffer copy (same extent required)
	if (InTexture.Width == Width_ and InTexture.Height == Height_) {
		// Record a copy operation into the default context
		auto& ctx = OaContext::GetDefault();

		OaVkBuffer bufs[2] = { InTexture.DeviceBuf, ProducerTex_.DeviceBuf };
		OaBufferAccess acc[2] = { OaBufferAccess::Read, OaBufferAccess::Write };
		const OaU32 pixelCount = static_cast<OaU32>(Width_) * static_cast<OaU32>(Height_);
		struct { OaU32 Count; } push{ pixelCount };

		constexpr OaU32 kGroupSize = 256;
		const OaU32 groupsX = (pixelCount + kGroupSize - 1) / kGroupSize;

		ctx.Add("Copy", OaSpan<OaVkBuffer>(bufs, 2), OaSpan<OaBufferAccess>(acc, 2),
			&push, sizeof(push), groupsX);

		// Swap buffers atomically
		std::swap(ProducerTex_, ConsumerTex_);

		// Increment sequence number
		Sequence_.fetch_add(1, std::memory_order_release);
	}
}

// ─── Destroy ───────────────────────────────────────────────────────────────────

void OaLiveTexture::Destroy(OaEngine& InEngine) {
	if (Engine_) {
		InEngine.DeregisterBuffer(ProducerTex_.DeviceBuf);
		InEngine.FreeBuffer(ProducerTex_.DeviceBuf);
		InEngine.DeregisterBuffer(ConsumerTex_.DeviceBuf);
		InEngine.FreeBuffer(ConsumerTex_.DeviceBuf);
	}

	Width_  = 0;
	Height_ = 0;
	Engine_ = nullptr;
	Sequence_.store(0, std::memory_order_relaxed);
}
