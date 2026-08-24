// oa::Iterator — base iterator interface
//
// Iteration sessions use `It<Thing>` names (for example `oa::ItTraining`). The contract
// the classic stateful cursor — construct, check isDone, advance with Next:
//
//   for (oa::ItTraining iter(...); !iter.isDone(); iter.next()) {
//     // do work for the current iteration
//   }
//
// Subclasses MUST override isDone() and next(). Default index() / reset()
// track an internal counter; override either to provide custom semantics.

#pragma once

#include <oa/core/types.h>

namespace oa {

class Iterator {
public:
	Iterator() = default;
	virtual ~Iterator() = default;

	Iterator(const Iterator&) = delete;
	Iterator& operator=(const Iterator&) = delete;
	Iterator(Iterator&&) noexcept = default;
	Iterator& operator=(Iterator&&) noexcept = default;

	// Required overrides
	[[nodiscard]] virtual bool isDone() const = 0;
	virtual void next() = 0;

	// Optional overrides — defaults manage the internal index_ counter
	virtual void reset() { index_ = 0; }
	[[nodiscard]] virtual oa::I64 index() const { return index_; }

protected:
	oa::I64 index_ = 0;
};

} // namespace oa
