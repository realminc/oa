// OaIterator — base iterator interface
//
// Subclasses live under `OaIt<Thing>` (e.g. OaItTraining). The contract
// the classic stateful cursor — construct, check IsDone, advance with Next:
//
//   for (OaItTraining iter(...); !iter.IsDone(); iter.Next()) {
//     // do work for the current iteration
//   }
//
// Subclasses MUST override IsDone() and Next(). Default Index() / Reset()
// track an internal counter; override either to provide custom semantics.

#pragma once

#include <Oa/Core/Types.h>

class OaIterator {
public:
	OaIterator() = default;
	virtual ~OaIterator() = default;

	OaIterator(const OaIterator&)            = delete;
	OaIterator& operator=(const OaIterator&) = delete;
	OaIterator(OaIterator&&) noexcept        = default;
	OaIterator& operator=(OaIterator&&) noexcept = default;

	// Required overrides
	[[nodiscard]] virtual bool IsDone() const = 0;
	virtual void Next() = 0;

	// Optional overrides — defaults manage the internal Index_ counter
	virtual void Reset() { Index_ = 0; }
	[[nodiscard]] virtual OaI64 Index() const { return Index_; }

protected:
	OaI64 Index_ = 0;
};
