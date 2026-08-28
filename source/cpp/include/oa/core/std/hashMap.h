#pragma once

// OA-owned open-addressed hash containers. Entries and probe metadata live in
// one allocation so lookup does not chase per-bucket allocations. Empty and
// erased slots remain distinct to preserve linear-probe search chains.

#include <oa/core/assert.h>
#include <oa/core/std/iter.h>
#include <oa/core/std/keyHash.h>
#include <oa/core/std/optional.h>
#include <oa/core/std/pair.h>
#include <oa/core/std/utility.h>
#include <oa/core/std/vector.h>

namespace oa {

template<typename K, typename V, typename Hash = oa::KeyHash<K>, typename KeyEq = oa::KeyEqual<K>>
class HashMap {
public:
	using KeyType = K;
	using MappedType = V;
	using ValueType = oa::Pair<const K, V>;
	using SizeType = oa::Usize;
	using HasherType = Hash;
	using KeyEqualType = KeyEq;
	using SlotType = ValueType;

	using key_type = KeyType;
	using mapped_type = MappedType;
	using value_type = ValueType;
	using size_type = SizeType;
	using hasher = HasherType;
	using key_equal = KeyEqualType;
	using slot_type = SlotType;

private:
	struct Slot {
		oa::Optional<SlotType> value{};
		bool erased{false};
	};

public:
	HashMap() = default;
	HashMap(const HashMap&) = default;
	HashMap& operator=(const HashMap&) = default;

	HashMap(HashMap&& inOther) noexcept(
		oa::IsNothrowMoveConstructibleV<Hash>
		&& oa::IsNothrowMoveConstructibleV<KeyEq>)
		requires(oa::IsNothrowMoveConstructibleV<Hash>
			&& oa::IsNothrowMoveConstructibleV<KeyEq>)
		: hasher_(oa::move(inOther.hasher_)), equal_(oa::move(inOther.equal_)) {
		// Move the potentially-throwing policy objects before stealing storage.
		// If policy construction fails, the source still owns every entry.
		slots_.swap(inOther.slots_);
		size_ = inOther.size_;
		erased_ = inOther.erased_;
		inOther.size_ = 0;
		inOther.erased_ = 0;
	}

	HashMap(HashMap&&)
		requires(!(oa::IsNothrowMoveConstructibleV<Hash>
			&& oa::IsNothrowMoveConstructibleV<KeyEq>)) = delete;

	HashMap& operator=(HashMap&& inOther) noexcept
		requires(oa::IsNothrowMoveConstructibleV<Hash>
			&& oa::IsNothrowMoveConstructibleV<KeyEq>
			&& oa::IsNothrowSwappableV<Hash>
			&& oa::IsNothrowSwappableV<KeyEq>) {
		if (this != &inOther) {
			HashMap replacement(oa::move(inOther));
			swap(replacement);
		}
		return *this;
	}

	HashMap& operator=(HashMap&&)
		requires(!(oa::IsNothrowMoveConstructibleV<Hash>
			&& oa::IsNothrowMoveConstructibleV<KeyEq>
			&& oa::IsNothrowSwappableV<Hash>
			&& oa::IsNothrowSwappableV<KeyEq>)) = delete;

	class const_iterator;

	class iterator {
		friend class HashMap;
		friend class const_iterator;

	public:
		using iterator_category = oa::ForwardIteratorTag;
		using difference_type = oa::Isize;
		using value_type = ValueType;
		using reference = ValueType&;
		using pointer = ValueType*;

		iterator() noexcept = default;

		reference operator*() const { return *value_; }

		pointer operator->() const { return value_; }

		iterator& operator++() {
			++index_;
			skipEmpty_();
			return *this;
		}

		iterator operator++(int) {
			iterator previous = *this;
			++*this;
			return previous;
		}

		bool operator==(const iterator& inOther) const noexcept {
			return slots_ == inOther.slots_ and index_ == inOther.index_;
		}

		bool operator!=(const iterator& inOther) const noexcept { return not (*this == inOther); }

	private:
		oa::Vector<Slot>* slots_{nullptr};
		SizeType index_{0};
		pointer value_{nullptr};

		iterator(
			oa::Vector<Slot>* inSlots,
			SizeType inIndex,
			pointer inValue,
			bool inScan
		) noexcept
			: slots_(inSlots), index_(inIndex), value_(inValue) {
			if (inScan) skipEmpty_();
		}

		void skipEmpty_() noexcept {
			if (slots_ == nullptr) return;
			while (index_ < slots_->size()) {
				value_ = (*slots_)[index_].value.get();
				if (value_ != nullptr) return;
				++index_;
			}
			value_ = nullptr;
		}
	};

	class const_iterator {
		friend class HashMap;

	public:
		using iterator_category = oa::ForwardIteratorTag;
		using difference_type = oa::Isize;
		using value_type = ValueType;
		using reference = const ValueType&;
		using pointer = const ValueType*;

		const_iterator() noexcept = default;

		const_iterator(iterator inIterator) noexcept
			: slots_(inIterator.slots_), index_(inIterator.index_), value_(inIterator.value_) {}

		reference operator*() const { return *value_; }

		pointer operator->() const { return value_; }

		const_iterator& operator++() {
			++index_;
			skipEmpty_();
			return *this;
		}

		const_iterator operator++(int) {
			const_iterator previous = *this;
			++*this;
			return previous;
		}

		bool operator==(const const_iterator& inOther) const noexcept {
			return slots_ == inOther.slots_ and index_ == inOther.index_;
		}

		bool operator!=(const const_iterator& inOther) const noexcept {
			return not (*this == inOther);
		}

	private:
		const oa::Vector<Slot>* slots_{nullptr};
		SizeType index_{0};
		pointer value_{nullptr};

		const_iterator(
			const oa::Vector<Slot>* inSlots,
			SizeType inIndex,
			pointer inValue,
			bool inScan
		) noexcept
			: slots_(inSlots), index_(inIndex), value_(inValue) {
			if (inScan) skipEmpty_();
		}

		void skipEmpty_() noexcept {
			if (slots_ == nullptr) return;
			while (index_ < slots_->size()) {
				value_ = (*slots_)[index_].value.get();
				if (value_ != nullptr) return;
				++index_;
			}
			value_ = nullptr;
		}
	};

	[[nodiscard]] SizeType size() const noexcept { return size_; }

	[[nodiscard]] bool empty() const noexcept { return size_ == 0; }

	void clear() noexcept {
		for (SizeType index = 0; index < slots_.size(); ++index) {
			slots_[index].value.reset();
			slots_[index].erased = false;
		}
		size_ = 0;
		erased_ = 0;
	}

	void reserve(SizeType inCount) {
		const SizeType capacity = capacityForCount_(inCount);
		if (capacity > slots_.size()) rehash_(capacity);
	}

	template<typename... Args>
	oa::Pair<iterator, bool> emplace(Args&&... inArgs) {
		SlotType value(oa::forward<Args>(inArgs)...);
		return insertStorage_(oa::move(value));
	}

	oa::Pair<iterator, bool> insert(const value_type& inValue) {
		SlotType value(inValue.first, inValue.second);
		return insertStorage_(oa::move(value));
	}

	oa::Pair<iterator, bool> insert(value_type&& inValue) {
		SlotType value(inValue.first, oa::move(inValue.second));
		return insertStorage_(oa::move(value));
	}

	[[nodiscard]] V& at(const K& inKey) {
		iterator it = find(inKey);
		OA_REQUIRE(it != end());
		return it->second;
	}

	[[nodiscard]] const V& at(const K& inKey) const {
		const_iterator it = find(inKey);
		OA_REQUIRE(it != end());
		return it->second;
	}

	[[nodiscard]] iterator find(const K& inKey) {
		const SizeType index = findIndex_(inKey);
		SlotType* value = index == slots_.size() ? nullptr : slots_[index].value.get();
		return iterator(&slots_, index, value, false);
	}

	[[nodiscard]] const_iterator find(const K& inKey) const {
		const SizeType index = findIndex_(inKey);
		const SlotType* value = index == slots_.size() ? nullptr : slots_[index].value.get();
		return const_iterator(&slots_, index, value, false);
	}

	[[nodiscard]] bool contains(const K& inKey) const {
		return findIndex_(inKey) != slots_.size();
	}

	SizeType erase(const K& inKey) {
		const SizeType index = findIndex_(inKey);
		if (index == slots_.size()) return 0;
		slots_[index].erased = true;
		--size_;
		++erased_;
		slots_[index].value.reset();
		return 1;
	}

	void swap(HashMap& inOther) noexcept
		requires(oa::IsNothrowSwappableV<Hash> && oa::IsNothrowSwappableV<KeyEq>) {
		if (this == &inOther) return;
		oa::swapValues(hasher_, inOther.hasher_);
		oa::swapValues(equal_, inOther.equal_);
		slots_.swap(inOther.slots_);
		oa::swapValues(size_, inOther.size_);
		oa::swapValues(erased_, inOther.erased_);
	}

	[[nodiscard]] iterator begin() noexcept { return iterator(&slots_, 0, nullptr, true); }

	[[nodiscard]] const_iterator begin() const noexcept {
		return const_iterator(&slots_, 0, nullptr, true);
	}

	[[nodiscard]] iterator end() noexcept {
		return iterator(&slots_, slots_.size(), nullptr, false);
	}

	[[nodiscard]] const_iterator end() const noexcept {
		return const_iterator(&slots_, slots_.size(), nullptr, false);
	}

private:
	static constexpr SizeType MinCapacity = 8;
	static constexpr SizeType LoadNumerator = 3;
	static constexpr SizeType LoadDenominator = 4;

	oa::Vector<Slot> slots_{};
	Hash hasher_{};
	KeyEq equal_{};
	SizeType size_{0};
	SizeType erased_{0};

	[[nodiscard]] static SizeType capacityForCount_(SizeType inCount) {
		if (inCount == 0) return 0;
		if (inCount > (static_cast<SizeType>(-1) - LoadNumerator + 1)
			/ LoadDenominator) {
			oa::allocationFailed(oa::AllocationError::SizeOverflow, inCount, alignof(Slot));
		}
		const SizeType required = (inCount * LoadDenominator + LoadNumerator - 1)
			/ LoadNumerator;
		SizeType capacity = MinCapacity;
		while (capacity < required) {
			if (capacity > static_cast<SizeType>(-1) / 2) {
				oa::allocationFailed(
					oa::AllocationError::SizeOverflow,
					required,
					alignof(Slot)
				);
			}
			capacity *= 2;
		}
		return capacity;
	}

	[[nodiscard]] SizeType findIndex_(const K& inKey) const {
		if (slots_.empty()) return 0;
		const SizeType mask = slots_.size() - 1;
		SizeType index = static_cast<SizeType>(hasher_(inKey)) & mask;
		for (SizeType probes = 0; probes < slots_.size(); ++probes) {
			const Slot& slot = slots_[index];
			if (slot.value.hasValue()) {
				if (equal_(slot.value->first, inKey)) return index;
			} else if (not slot.erased) {
				return slots_.size();
			}
			index = (index + 1) & mask;
		}
		return slots_.size();
	}

	void rehash_(SizeType inCapacity) {
		oa::Vector<Slot> replacement;
		replacement.resize(inCapacity);
		SizeType replacementSize = 0;

		if constexpr (oa::IsCopyConstructibleV<SlotType>) {
			// Copying keeps the original table byte-for-byte intact until every
			// hash and value construction in the replacement has succeeded.
			for (SizeType index = 0; index < slots_.size(); ++index) {
				if (slots_[index].value.hasValue()) {
					const SlotType& value = slots_[index].value.value();
					insertNoGrowHashed_(replacement, replacementSize,
						static_cast<SizeType>(hasher_(value.first)), value);
				}
			}
		} else {
			static_assert(oa::IsNothrowMoveConstructibleV<SlotType>,
				"HashMap rehash requires a copyable value or nothrow movable storage");
			// A throwing stateful hasher must run before the first source value is
			// moved. Once hashes are known, the admitted move-only path is no-throw.
			oa::Vector<SizeType> hashes(slots_.size());
			for (SizeType index = 0; index < slots_.size(); ++index) {
				if (slots_[index].value.hasValue()) {
					hashes[index] = static_cast<SizeType>(
						hasher_(slots_[index].value->first));
				}
			}
			for (SizeType index = 0; index < slots_.size(); ++index) {
				if (slots_[index].value.hasValue()) {
					insertNoGrowHashed_(replacement, replacementSize, hashes[index],
						oa::move(slots_[index].value.value()));
				}
			}
		}

		slots_.swap(replacement);
		size_ = replacementSize;
		erased_ = 0;
	}

	template<typename Value>
	void insertNoGrowHashed_(
		oa::Vector<Slot>& inSlots,
		SizeType& inSize,
		SizeType inHash,
		Value&& inValue
	) {
		const SizeType mask = inSlots.size() - 1;
		SizeType index = inHash & mask;
		while (inSlots[index].value.hasValue()) index = (index + 1) & mask;
		inSlots[index].value.emplace(oa::forward<Value>(inValue));
		++inSize;
	}

	oa::Pair<iterator, bool> insertStorage_(SlotType&& inValue) {
		if (slots_.empty()) {
			rehash_(MinCapacity);
		} else if (size_ + erased_ + 1
			> slots_.size() - slots_.size() / LoadDenominator) {
			const SizeType liveCapacity = capacityForCount_(size_ + 1);
			const SizeType nextCapacity = liveCapacity > slots_.size()
				? liveCapacity
				: slots_.size();
			rehash_(nextCapacity);
		}

		const SizeType mask = slots_.size() - 1;
		SizeType index = static_cast<SizeType>(hasher_(inValue.first)) & mask;
		SizeType insertion = slots_.size();
		for (;;) {
			Slot& slot = slots_[index];
			if (slot.value.hasValue()) {
				if (equal_(slot.value->first, inValue.first)) {
					return {iterator(&slots_, index, slot.value.get(), false), false};
				}
			} else if (slot.erased) {
				if (insertion == slots_.size()) insertion = index;
			} else {
				if (insertion == slots_.size()) insertion = index;
				break;
			}
			index = (index + 1) & mask;
		}

		Slot& destination = slots_[insertion];
		const bool reusedErased = destination.erased;
		destination.value.emplace(oa::move(inValue));
		if (reusedErased) {
			destination.erased = false;
			--erased_;
		}
		++size_;
		return {
			iterator(&slots_, insertion, destination.value.get(), false),
			true
		};
	}
};

template<typename K, typename Hash = oa::KeyHash<K>, typename KeyEq = oa::KeyEqual<K>>
class HashSet {
private:
	struct EmptyValue {};
	using Map = oa::HashMap<K, EmptyValue, Hash, KeyEq>;

public:
	using KeyType = K;
	using SizeType = oa::Usize;
	using HasherType = Hash;
	using KeyEqualType = KeyEq;

	using key_type = KeyType;
	using size_type = SizeType;
	using hasher = HasherType;
	using key_equal = KeyEqualType;

	HashSet() = default;
	HashSet(const HashSet&) = default;
	HashSet& operator=(const HashSet&) = default;
	HashSet(HashSet&&) = default;
	HashSet& operator=(HashSet&&) = default;

	class const_iterator;

	class iterator {
		friend class HashSet;
		friend class const_iterator;

	public:
		using iterator_category = oa::ForwardIteratorTag;
		using difference_type = oa::Isize;
		using value_type = KeyType;
		using reference = const KeyType&;
		using pointer = const KeyType*;

		iterator() noexcept = default;

		reference operator*() const { return inner_->first; }

		pointer operator->() const { return &inner_->first; }

		iterator& operator++() { ++inner_; return *this; }

		iterator operator++(int) {
			iterator previous = *this;
			++inner_;
			return previous;
		}

		bool operator==(const iterator& inOther) const noexcept {
			return inner_ == inOther.inner_;
		}

		bool operator!=(const iterator& inOther) const noexcept { return not (*this == inOther); }

	private:
		typename Map::iterator inner_{};

		explicit iterator(typename Map::iterator inInner) noexcept : inner_(inInner) {}
	};

	class const_iterator {
		friend class HashSet;

	public:
		using iterator_category = oa::ForwardIteratorTag;
		using difference_type = oa::Isize;
		using value_type = KeyType;
		using reference = const KeyType&;
		using pointer = const KeyType*;

		const_iterator() noexcept = default;

		const_iterator(iterator inIterator) noexcept : inner_(inIterator.inner_) {}

		reference operator*() const { return inner_->first; }

		pointer operator->() const { return &inner_->first; }

		const_iterator& operator++() { ++inner_; return *this; }

		const_iterator operator++(int) {
			const_iterator previous = *this;
			++inner_;
			return previous;
		}

		bool operator==(const const_iterator& inOther) const noexcept {
			return inner_ == inOther.inner_;
		}

		bool operator!=(const const_iterator& inOther) const noexcept {
			return not (*this == inOther);
		}

	private:
		typename Map::const_iterator inner_{};

		explicit const_iterator(typename Map::const_iterator inInner) noexcept : inner_(inInner) {}
	};

	[[nodiscard]] SizeType size() const noexcept { return map_.size(); }

	[[nodiscard]] bool empty() const noexcept { return map_.empty(); }

	void clear() noexcept { map_.clear(); }

	void reserve(SizeType inCount) { map_.reserve(inCount); }

	oa::Pair<iterator, bool> insert(const K& inKey) {
		auto inserted = map_.emplace(inKey, EmptyValue{});
		return {iterator(inserted.first), inserted.second};
	}

	oa::Pair<iterator, bool> insert(K&& inKey) {
		auto inserted = map_.emplace(oa::move(inKey), EmptyValue{});
		return {iterator(inserted.first), inserted.second};
	}

	[[nodiscard]] iterator find(const K& inKey) { return iterator(map_.find(inKey)); }

	[[nodiscard]] const_iterator find(const K& inKey) const {
		return const_iterator(map_.find(inKey));
	}

	[[nodiscard]] bool contains(const K& inKey) const { return map_.contains(inKey); }

	SizeType erase(const K& inKey) { return map_.erase(inKey); }

	[[nodiscard]] iterator begin() noexcept { return iterator(map_.begin()); }

	[[nodiscard]] const_iterator begin() const noexcept { return const_iterator(map_.begin()); }

	[[nodiscard]] iterator end() noexcept { return iterator(map_.end()); }

	[[nodiscard]] const_iterator end() const noexcept { return const_iterator(map_.end()); }

private:
	Map map_{};
};

template<typename K, typename V, typename Hash, typename KeyEq>
inline typename HashMap<K, V, Hash, KeyEq>::iterator begin(
	HashMap<K, V, Hash, KeyEq>& inMap
) noexcept {
	return inMap.begin();
}

template<typename K, typename V, typename Hash, typename KeyEq>
inline typename HashMap<K, V, Hash, KeyEq>::const_iterator begin(
	const HashMap<K, V, Hash, KeyEq>& inMap
) noexcept {
	return inMap.begin();
}

template<typename K, typename V, typename Hash, typename KeyEq>
inline typename HashMap<K, V, Hash, KeyEq>::iterator end(
	HashMap<K, V, Hash, KeyEq>& inMap
) noexcept {
	return inMap.end();
}

template<typename K, typename V, typename Hash, typename KeyEq>
inline typename HashMap<K, V, Hash, KeyEq>::const_iterator end(
	const HashMap<K, V, Hash, KeyEq>& inMap
) noexcept {
	return inMap.end();
}

template<typename K, typename Hash, typename KeyEq>
inline typename HashSet<K, Hash, KeyEq>::iterator begin(
	HashSet<K, Hash, KeyEq>& inSet
) noexcept {
	return inSet.begin();
}

template<typename K, typename Hash, typename KeyEq>
inline typename HashSet<K, Hash, KeyEq>::const_iterator begin(
	const HashSet<K, Hash, KeyEq>& inSet
) noexcept {
	return inSet.begin();
}

template<typename K, typename Hash, typename KeyEq>
inline typename HashSet<K, Hash, KeyEq>::iterator end(
	HashSet<K, Hash, KeyEq>& inSet
) noexcept {
	return inSet.end();
}

template<typename K, typename Hash, typename KeyEq>
inline typename HashSet<K, Hash, KeyEq>::const_iterator end(
	const HashSet<K, Hash, KeyEq>& inSet
) noexcept {
	return inSet.end();
}

} // namespace oa
