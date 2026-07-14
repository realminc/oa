// OaTimeCode — Exact frame/time representation with rational frame rates.
//
// Supports deterministic stepping, negative time, and lossless conversion.
// Uses rational frame rates (e.g., 30000/1001 for 29.97 FPS) for exact
// representation of video and film timecode.
//
// Design principles (CoreAnimationRenderArchitecture.md §3.2):
// - Use seconds only at API edges
// - Internally use exact tick-based time for editing and video
// - Support rational frame rates (not float) to prevent drift
// - Deterministic stepping for reliable playback and rendering
//
// Usage:
//   OaFrameRate rate{24, 1};           // 24 FPS
//   OaTimeCode tc{1000, 48000};        // 1000 ticks at 48k ticks/sec
//   OaF32 seconds = tc.ToSeconds();    // Convert to seconds for display
//   OaTimeCode frameTC = tc.ToFrame(rate);  // Snap to frame boundary
//
// Timecode arithmetic:
//   OaTimeCode a{48000, 48000};  // 1 second
//   OaTimeCode b{24000, 48000};  // 0.5 seconds
//   OaTimeCode c = a + b;        // 1.5 seconds
//
#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Math.h>

// ═════════════════════════════════════════════════════════════════════════════
// OaFrameRate — Rational frame rate (numerator/denominator)
// ═════════════════════════════════════════════════════════════════════════════

struct OaFrameRate {
	OaI64 Numerator = 24;     // e.g., 30000
	OaI64 Denominator = 1;   // e.g., 1001 for 29.97 FPS

	// Common frame rates
	[[nodiscard]] static constexpr OaFrameRate Fps24() noexcept { return {24, 1}; }
	[[nodiscard]] static constexpr OaFrameRate Fps25() noexcept { return {25, 1}; }
	[[nodiscard]] static constexpr OaFrameRate Fps30() noexcept { return {30, 1}; }
	[[nodiscard]] static constexpr OaFrameRate Fps29_97() noexcept { return {30000, 1001}; }
	[[nodiscard]] static constexpr OaFrameRate Fps59_94() noexcept { return {60000, 1001}; }
	[[nodiscard]] static constexpr OaFrameRate Fps60() noexcept { return {60, 1}; }
	[[nodiscard]] static constexpr OaFrameRate Fps120() noexcept { return {120, 1}; }

	// Convert to approximate float FPS
	[[nodiscard]] OaF32 ToFloat() const noexcept {
		return static_cast<OaF32>(Numerator) / static_cast<OaF32>(Denominator);
	}

	// Frame duration in seconds (as float)
	[[nodiscard]] OaF32 FrameDurationSeconds() const noexcept {
		return static_cast<OaF32>(Denominator) / static_cast<OaF32>(Numerator);
	}

	[[nodiscard]] bool operator==(const OaFrameRate& InOther) const noexcept {
		return Numerator * InOther.Denominator == Denominator * InOther.Numerator;
	}
	[[nodiscard]] bool operator!=(const OaFrameRate& InOther) const noexcept {
		return !(*this == InOther);
	}
};

// ═════════════════════════════════════════════════════════════════════════════
// OaTimeCode — Exact time representation with rational ticks
// ═════════════════════════════════════════════════════════════════════════════

struct OaTimeCode {
	OaI64 Ticks = 0;           // Time value in ticks
	OaI64 TicksPerSecond = 48000;  // Tick scale (default 48kHz for sub-frame precision)

	// Constructors
	constexpr OaTimeCode() noexcept = default;
	constexpr OaTimeCode(OaI64 InTicks, OaI64 InTicksPerSecond) noexcept
		: Ticks(InTicks), TicksPerSecond(InTicksPerSecond) {}

	// From seconds (API edge conversion)
	[[nodiscard]] static OaTimeCode FromSeconds(OaF32 InSeconds, OaI64 InTicksPerSecond = 48000) noexcept {
		return OaTimeCode{
			static_cast<OaI64>(InSeconds * static_cast<OaF32>(InTicksPerSecond)),
			InTicksPerSecond
		};
	}

	// From frame number at a specific frame rate
	[[nodiscard]] static OaTimeCode FromFrame(
		OaI64 InFrameNumber,
		const OaFrameRate& InRate,
		OaI64 InTicksPerSecond = 48000
	) noexcept {
		// frame / fps = seconds; seconds * ticks_per_second = ticks
		// frame * denominator / numerator = seconds
		OaI64 ticks = (InFrameNumber * InRate.Denominator * InTicksPerSecond) / InRate.Numerator;
		return OaTimeCode{ticks, InTicksPerSecond};
	}

	// Convert to seconds (API edge conversion)
	[[nodiscard]] OaF32 ToSeconds() const noexcept {
		return static_cast<OaF32>(Ticks) / static_cast<OaF32>(TicksPerSecond);
	}

	// Convert to frame number at a specific frame rate (truncates toward zero)
	[[nodiscard]] OaI64 ToFrame(const OaFrameRate& InRate) const noexcept {
		// ticks / tps = seconds; seconds * fps = frame
		// ticks * numerator / (tps * denominator) = frame
		return (Ticks * InRate.Numerator) / (TicksPerSecond * InRate.Denominator);
	}

	// Snap to nearest frame boundary at given rate
	[[nodiscard]] OaTimeCode SnapToFrame(const OaFrameRate& InRate) const noexcept {
		return FromFrame(ToFrame(InRate), InRate, TicksPerSecond);
	}

	// Frame delta to time offset
	[[nodiscard]] OaTimeCode FrameDelta(const OaFrameRate& InRate, OaI64 InFrameDelta) const noexcept {
		OaI64 tickDelta = (InFrameDelta * InRate.Denominator * TicksPerSecond) / InRate.Numerator;
		return OaTimeCode{Ticks + tickDelta, TicksPerSecond};
	}

	// Arithmetic
	[[nodiscard]] OaTimeCode operator+(const OaTimeCode& InOther) const noexcept {
		OaI64 thisTicks = Ticks;
		OaI64 otherTicks = InOther.ToTicks(TicksPerSecond);
		return OaTimeCode{thisTicks + otherTicks, TicksPerSecond};
	}

	[[nodiscard]] OaTimeCode operator-(const OaTimeCode& InOther) const noexcept {
		OaI64 thisTicks = Ticks;
		OaI64 otherTicks = InOther.ToTicks(TicksPerSecond);
		return OaTimeCode{thisTicks - otherTicks, TicksPerSecond};
	}

	[[nodiscard]] OaTimeCode operator*(OaF32 InScalar) const noexcept {
		return OaTimeCode{static_cast<OaI64>(static_cast<OaF32>(Ticks) * InScalar), TicksPerSecond};
	}

	// Convert to different tick scale
	[[nodiscard]] OaI64 ToTicks(OaI64 InTargetTicksPerSecond) const noexcept {
		if (TicksPerSecond == InTargetTicksPerSecond) return Ticks;
		return (Ticks * InTargetTicksPerSecond) / TicksPerSecond;
	}

	[[nodiscard]] OaTimeCode ToScale(OaI64 InTargetTicksPerSecond) const noexcept {
		if (TicksPerSecond == InTargetTicksPerSecond) return *this;
		return OaTimeCode{ToTicks(InTargetTicksPerSecond), InTargetTicksPerSecond};
	}

	// Comparison
	[[nodiscard]] int Compare(const OaTimeCode& InOther) const noexcept {
		OaI64 thisTicks = Ticks;
		OaI64 otherTicks = InOther.ToTicks(TicksPerSecond);
		if (thisTicks < otherTicks) return -1;
		if (thisTicks > otherTicks) return 1;
		return 0;
	}

	[[nodiscard]] bool operator<(const OaTimeCode& InOther) const noexcept { return Compare(InOther) < 0; }
	[[nodiscard]] bool operator>(const OaTimeCode& InOther) const noexcept { return Compare(InOther) > 0; }
	[[nodiscard]] bool operator<=(const OaTimeCode& InOther) const noexcept { return Compare(InOther) <= 0; }
	[[nodiscard]] bool operator>=(const OaTimeCode& InOther) const noexcept { return Compare(InOther) >= 0; }
	[[nodiscard]] bool operator==(const OaTimeCode& InOther) const noexcept { return Compare(InOther) == 0; }
	[[nodiscard]] bool operator!=(const OaTimeCode& InOther) const noexcept { return Compare(InOther) != 0; }

	// Absolute value (for duration calculations)
	[[nodiscard]] OaTimeCode Abs() const noexcept {
		return OaTimeCode{Ticks < 0 ? -Ticks : Ticks, TicksPerSecond};
	}

	// Zero check
	[[nodiscard]] bool IsZero() const noexcept { return Ticks == 0; }
	[[nodiscard]] bool IsPositive() const noexcept { return Ticks > 0; }
	[[nodiscard]] bool IsNegative() const noexcept { return Ticks < 0; }
};

// ═════════════════════════════════════════════════════════════════════════════
// OaTimeRange — Time range with loop behavior (exact timecode-based)
// ═════════════════════════════════════════════════════════════════════════════

struct OaTimeRange {
	OaTimeCode Start;
	OaTimeCode End;
	bool Loop = false;

	[[nodiscard]] bool IsEmpty() const noexcept {
		return Start == End;
	}

	[[nodiscard]] OaTimeCode Duration() const noexcept {
		return End - Start;
	}

	[[nodiscard]] bool Contains(const OaTimeCode& InTime) const noexcept {
		return InTime >= Start && InTime <= End;
	}

	// Clamp time to range bounds
	[[nodiscard]] OaTimeCode Clamp(const OaTimeCode& InTime) const noexcept {
		if (InTime < Start) return Start;
		if (InTime > End) return End;
		return InTime;
	}

	// Apply loop to time (wraps around range)
	[[nodiscard]] OaTimeCode ApplyLoop(const OaTimeCode& InTime) const noexcept {
		if (!Loop || IsEmpty()) return Clamp(InTime);

		OaTimeCode duration = Duration();
		if (duration.IsZero()) return Start;

		OaTimeCode local = InTime - Start;
		OaI64 durationTicks = duration.Ticks;
		OaI64 localTicks = local.ToTicks(Start.TicksPerSecond);

		// Handle negative time (loop backward)
		if (localTicks < 0) {
			OaI64 loops = ((-localTicks) / durationTicks) + 1;
			localTicks += loops * durationTicks;
		}

		OaI64 wrapped = localTicks % durationTicks;
		return Start + OaTimeCode{wrapped, Start.TicksPerSecond};
	}
};
