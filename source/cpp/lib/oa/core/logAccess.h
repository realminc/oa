#pragma once

#include <oa/core/log.h>

namespace oa {

class LogImpl;

// Thread selection is a weak reference to logger state, never process-global
// ownership. Destroying an engine on another thread therefore cannot leave a
// dangling logger pointer in the creating thread.
class LogSelection {
public:
	WeakPtr<LogImpl> state;
};

class LogAccess {
public:
	[[nodiscard]] static LogSelection select(Log* inLog) noexcept;
	static void restore(const LogSelection& inSelection) noexcept;
	static void restoreIfCurrent(
		Log* inExpected,
		const LogSelection& inSelection) noexcept;
	[[nodiscard]] static SharedPtr<LogImpl> current() noexcept;
	[[nodiscard]] static LogSelection currentSelection() noexcept;

	class Scope {
	public:
		explicit Scope(Log* inLog) noexcept
			: previous_(select(inLog)) {}
		explicit Scope(const LogSelection& inSelection) noexcept
			: previous_(currentSelection()) { restore(inSelection); }
		~Scope() { restore(previous_); }

		Scope(const Scope&) = delete;
		Scope& operator=(const Scope&) = delete;
		Scope(Scope&&) noexcept = delete;
		Scope& operator=(Scope&&) noexcept = delete;

	private:
		LogSelection previous_;
	};
};

} // namespace oa
