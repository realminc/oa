#pragma once

// phase 2b OA standard library — bucket chaining via oa::Vec; `stdMap`/`stdSet` copy to std for boundaries.
// Iterators: forward category, prefix and postfix ++. insert(pair&&) moves the mapped value into storage.

#include <oa/core/std/vec.h>

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace oa {

template<typename K, typename V, typename Hash = std::hash<K>, typename KeyEq = std::equal_to<K>>
class HashMap {
public:
	using KeyType = K;
	using MappedType = V;
	using ValueType = std::pair<const K, V>;
	using SizeType = std::size_t;
	using HasherType = Hash;
	using KeyEqualType = KeyEq;
	using SlotType = std::pair<K, V>;

	using key_type = KeyType;
	using mapped_type = MappedType;
	using value_type = ValueType;
	using size_type = SizeType;
	using hasher = HasherType;
	using key_equal = KeyEqualType;
	using slot_type = SlotType;

	class iterator {
		friend class HashMap;
		friend class const_iterator;

	public:
		using iterator_category = std::forward_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using value_type = ValueType;
		using reference = SlotType&;
		using pointer = SlotType*;

		iterator() noexcept = default;

		reference operator*() const { return map_->buckets_[bucket_][slot_]; }

		pointer operator->() const { return &map_->buckets_[bucket_][slot_]; }

		iterator& operator++() {
			++slot_;
			skip_();
			return *this;
		}

		iterator operator++(int) {
			iterator tmp = *this;
			++*this;
			return tmp;
		}

		bool operator==(const iterator& inO) const noexcept {
			return map_ == inO.map_ && bucket_ == inO.bucket_ && slot_ == inO.slot_;
		}

		bool operator!=(const iterator& inO) const noexcept { return !(*this == inO); }

	private:
		HashMap* map_{nullptr};
		SizeType bucket_{0};
		SizeType slot_{0};

		iterator(HashMap* inMap, SizeType inB, SizeType inS) noexcept
			: map_(inMap), bucket_(inB), slot_(inS) {
			skip_();
		}

		void skip_() noexcept {
			if (!map_) {
				return;
			}
			for (;;) {
				if (bucket_ >= map_->buckets_.size()) {
					return;
				}
				if (slot_ < map_->buckets_[bucket_].size()) {
					return;
				}
				++bucket_;
				slot_ = 0;
			}
		}
	};

	class const_iterator {
		friend class HashMap;

	public:
		using iterator_category = std::forward_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using value_type = ValueType;
		using reference = const SlotType&;
		using pointer = const SlotType*;

		const_iterator() noexcept = default;

		const_iterator(iterator inIt) noexcept
			: map_(inIt.map_), bucket_(inIt.bucket_), slot_(inIt.slot_) {}

		reference operator*() const { return map_->buckets_[bucket_][slot_]; }

		pointer operator->() const { return &map_->buckets_[bucket_][slot_]; }

		const_iterator& operator++() {
			++slot_;
			skip_();
			return *this;
		}

		const_iterator operator++(int) {
			const_iterator tmp = *this;
			++*this;
			return tmp;
		}

		bool operator==(const const_iterator& inO) const noexcept {
			return map_ == inO.map_ && bucket_ == inO.bucket_ && slot_ == inO.slot_;
		}

		bool operator!=(const const_iterator& inO) const noexcept { return !(*this == inO); }

	private:
		const HashMap* map_{nullptr};
		SizeType bucket_{0};
		SizeType slot_{0};

		const_iterator(const HashMap* inMap, SizeType inB, SizeType inS) noexcept
			: map_(inMap), bucket_(inB), slot_(inS) {
			skip_();
		}

		void skip_() noexcept {
			if (!map_) {
				return;
			}
			for (;;) {
				if (bucket_ >= map_->buckets_.size()) {
					return;
				}
				if (slot_ < map_->buckets_[bucket_].size()) {
					return;
				}
				++bucket_;
				slot_ = 0;
			}
		}
	};

	friend class iterator;
	friend class const_iterator;

	[[nodiscard]] SizeType size() const noexcept { return size_; }

	[[nodiscard]] bool empty() const noexcept { return size_ == 0; }

	void clear() noexcept {
		for (SizeType i = 0; i < buckets_.size(); ++i) {
			buckets_[i].clear();
		}
		size_ = 0;
	}

	void reserve(SizeType inN) {
		if (inN <= buckets_.size()) {
			return;
		}
		rehash_(inN);
	}

	template<typename... Args>
	std::pair<iterator, bool> emplace(Args&&... inArgs) {
		SlotType slot(std::forward<Args>(inArgs)...);
		return insertStorage_(std::move(slot));
	}

	std::pair<iterator, bool> insert(const value_type& inVal) {
		SlotType slot(inVal.first, inVal.second);
		return insertStorage_(std::move(slot));
	}

	std::pair<iterator, bool> insert(value_type&& inVal) {
		SlotType slot(inVal.first, std::move(inVal.second));
		return insertStorage_(std::move(slot));
	}

	[[nodiscard]] V& at(const K& inKey) {
		iterator it = find(inKey);
		if (it == end()) {
			throw std::out_of_range("HashMap::at");
		}
		return it->second;
	}

	[[nodiscard]] const V& at(const K& inKey) const {
		const_iterator it = find(inKey);
		if (it == end()) {
			throw std::out_of_range("HashMap::at");
		}
		return it->second;
	}

	[[nodiscard]] iterator find(const K& inKey) {
		ensureBuckets_();
		const SizeType b = bucketIndex_(inKey);
		oa::Vec<SlotType>& ch = buckets_[b];
		for (SizeType i = 0; i < ch.size(); ++i) {
			if (equal_(ch[i].first, inKey)) {
				return iterator(this, b, i);
			}
		}
		return end();
	}

	[[nodiscard]] const_iterator find(const K& inKey) const {
		if (buckets_.empty()) {
			return end();
		}
		const SizeType b = bucketIndex_(inKey);
		const oa::Vec<SlotType>& ch = buckets_[b];
		for (SizeType i = 0; i < ch.size(); ++i) {
			if (equal_(ch[i].first, inKey)) {
				return const_iterator(this, b, i);
			}
		}
		return end();
	}

	[[nodiscard]] bool contains(const K& inKey) const { return find(inKey) != end(); }

	SizeType erase(const K& inKey) {
		ensureBuckets_();
		const SizeType b = bucketIndex_(inKey);
		oa::Vec<SlotType>& ch = buckets_[b];
		for (SizeType i = 0; i < ch.size(); ++i) {
			if (equal_(ch[i].first, inKey)) {
				ch[i] = std::move(ch.back());
				ch.popBack();
				--size_;
				return 1;
			}
		}
		return 0;
	}

	[[nodiscard]] iterator begin() noexcept {
		ensureBuckets_();
		return iterator(this, 0, 0);
	}

	[[nodiscard]] const_iterator begin() const noexcept {
		if (buckets_.empty()) {
			return end();
		}
		return const_iterator(this, 0, 0);
	}

	[[nodiscard]] iterator end() noexcept {
		ensureBuckets_();
		return iterator(this, buckets_.size(), 0);
	}

	[[nodiscard]] const_iterator end() const noexcept {
		if (buckets_.empty()) {
			return const_iterator(nullptr, 0, 0);
		}
		return const_iterator(this, buckets_.size(), 0);
	}


	[[nodiscard]] std::unordered_map<K, V, Hash, KeyEq> stdMap() const {
		std::unordered_map<K, V, Hash, KeyEq> out;
		out.reserve(static_cast<std::size_t>(size_));
		for (const_iterator it = begin(), e = end(); it != e; ++it) {
			out.emplace(it->first, it->second);
		}
		return out;
	}

private:
	oa::Vec<oa::Vec<SlotType>> buckets_{};
	Hash hasher_{};
	KeyEq equal_{};
	SizeType size_{0};
	float maxLoad_{0.75F};

	void ensureBuckets_() {
		if (buckets_.empty()) {
			buckets_.resize(8);
		}
	}

	[[nodiscard]] SizeType bucketIndex_(const K& inKey) const noexcept {
		return static_cast<SizeType>(hasher_(inKey)) % buckets_.size();
	}

	void rehash_(SizeType inNewCap) {
		oa::Vec<oa::Vec<SlotType>> old = std::move(buckets_);
		buckets_.resize(inNewCap);
		size_ = 0;
		for (SizeType bi = 0; bi < old.size(); ++bi) {
			oa::Vec<SlotType>& chain = old[bi];
			const SizeType cnt = chain.size();
			for (SizeType si = 0; si < cnt; ++si) {
				insertNoGrow_(std::move(chain[si]));
			}
		}
	}

	void insertNoGrow_(SlotType&& inSlot) {
		const SizeType b = static_cast<SizeType>(hasher_(inSlot.first)) % buckets_.size();
		buckets_[b].emplaceBack(std::move(inSlot));
		++size_;
	}

	std::pair<iterator, bool> insertStorage_(SlotType&& inSlot) {
		ensureBuckets_();
		const K& key = inSlot.first;
		if (buckets_.size() > 0 &&
			static_cast<float>(size_ + 1) > maxLoad_ * static_cast<float>(buckets_.size())) {
			rehash_(buckets_.size() * 2);
		}
		SizeType b = static_cast<SizeType>(hasher_(key)) % buckets_.size();
		oa::Vec<SlotType>& ch = buckets_[b];
		for (SizeType i = 0; i < ch.size(); ++i) {
			if (equal_(ch[i].first, key)) {
				return {iterator(this, b, i), false};
			}
		}
		const SizeType idx = ch.size();
		ch.emplaceBack(std::move(inSlot));
		++size_;
		return {iterator(this, b, idx), true};
	}
};

template<typename K, typename Hash = std::hash<K>, typename KeyEq = std::equal_to<K>>
class HashSet {
public:
	using KeyType = K;
	using SizeType = std::size_t;
	using HasherType = Hash;
	using KeyEqualType = KeyEq;

	using key_type = KeyType;
	using size_type = SizeType;
	using hasher = HasherType;
	using key_equal = KeyEqualType;

	class iterator {
		friend class HashSet;
		friend class const_iterator;

	public:
		using iterator_category = std::forward_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using value_type = KeyType;
		using reference = KeyType&;
		using pointer = KeyType*;

		iterator() noexcept = default;

		reference operator*() const { return map_->buckets_[bucket_][slot_]; }

		pointer operator->() const { return &map_->buckets_[bucket_][slot_]; }

		iterator& operator++() {
			++slot_;
			skip_();
			return *this;
		}

		iterator operator++(int) {
			iterator tmp = *this;
			++*this;
			return tmp;
		}

		bool operator==(const iterator& inO) const noexcept {
			return map_ == inO.map_ && bucket_ == inO.bucket_ && slot_ == inO.slot_;
		}

		bool operator!=(const iterator& inO) const noexcept { return !(*this == inO); }

	private:
		HashSet* map_{nullptr};
		SizeType bucket_{0};
		SizeType slot_{0};

		iterator(HashSet* inMap, SizeType inB, SizeType inS) noexcept
			: map_(inMap), bucket_(inB), slot_(inS) {
			skip_();
		}

		void skip_() noexcept {
			if (!map_) {
				return;
			}
			for (;;) {
				if (bucket_ >= map_->buckets_.size()) {
					return;
				}
				if (slot_ < map_->buckets_[bucket_].size()) {
					return;
				}
				++bucket_;
				slot_ = 0;
			}
		}
	};

	class const_iterator {
		friend class HashSet;

	public:
		using iterator_category = std::forward_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using value_type = KeyType;
		using reference = const KeyType&;
		using pointer = const KeyType*;

		const_iterator() noexcept = default;

		const_iterator(iterator inIt) noexcept
			: map_(inIt.map_), bucket_(inIt.bucket_), slot_(inIt.slot_) {}

		reference operator*() const { return map_->buckets_[bucket_][slot_]; }

		pointer operator->() const { return &map_->buckets_[bucket_][slot_]; }

		const_iterator& operator++() {
			++slot_;
			skip_();
			return *this;
		}

		const_iterator operator++(int) {
			const_iterator tmp = *this;
			++*this;
			return tmp;
		}

		bool operator==(const const_iterator& inO) const noexcept {
			return map_ == inO.map_ && bucket_ == inO.bucket_ && slot_ == inO.slot_;
		}

		bool operator!=(const const_iterator& inO) const noexcept { return !(*this == inO); }

	private:
		const HashSet* map_{nullptr};
		SizeType bucket_{0};
		SizeType slot_{0};

		const_iterator(const HashSet* inMap, SizeType inB, SizeType inS) noexcept
			: map_(inMap), bucket_(inB), slot_(inS) {
			skip_();
		}

		void skip_() noexcept {
			if (!map_) {
				return;
			}
			for (;;) {
				if (bucket_ >= map_->buckets_.size()) {
					return;
				}
				if (slot_ < map_->buckets_[bucket_].size()) {
					return;
				}
				++bucket_;
				slot_ = 0;
			}
		}
	};

	friend class iterator;
	friend class const_iterator;

	[[nodiscard]] SizeType size() const noexcept { return size_; }

	[[nodiscard]] bool empty() const noexcept { return size_ == 0; }

	void clear() noexcept {
		for (SizeType i = 0; i < buckets_.size(); ++i) {
			buckets_[i].clear();
		}
		size_ = 0;
	}

	void reserve(SizeType inN) {
		if (inN <= buckets_.size()) {
			return;
		}
		rehash_(inN);
	}

	std::pair<iterator, bool> insert(const K& inKey) {
		return insertKey_(K(inKey));
	}

	std::pair<iterator, bool> insert(K&& inKey) { return insertKey_(std::move(inKey)); }

	[[nodiscard]] iterator find(const K& inKey) {
		ensureBuckets_();
		const SizeType b = bucketIndex_(inKey);
		oa::Vec<K>& ch = buckets_[b];
		for (SizeType i = 0; i < ch.size(); ++i) {
			if (equal_(ch[i], inKey)) {
				return iterator(this, b, i);
			}
		}
		return end();
	}

	[[nodiscard]] const_iterator find(const K& inKey) const {
		if (buckets_.empty()) {
			return end();
		}
		const SizeType b = bucketIndex_(inKey);
		const oa::Vec<K>& ch = buckets_[b];
		for (SizeType i = 0; i < ch.size(); ++i) {
			if (equal_(ch[i], inKey)) {
				return const_iterator(this, b, i);
			}
		}
		return end();
	}

	[[nodiscard]] bool contains(const K& inKey) const { return find(inKey) != end(); }

	SizeType erase(const K& inKey) {
		ensureBuckets_();
		const SizeType b = bucketIndex_(inKey);
		oa::Vec<K>& ch = buckets_[b];
		for (SizeType i = 0; i < ch.size(); ++i) {
			if (equal_(ch[i], inKey)) {
				ch[i] = std::move(ch.back());
				ch.popBack();
				--size_;
				return 1;
			}
		}
		return 0;
	}

	[[nodiscard]] iterator begin() noexcept {
		ensureBuckets_();
		return iterator(this, 0, 0);
	}

	[[nodiscard]] const_iterator begin() const noexcept {
		if (buckets_.empty()) {
			return end();
		}
		return const_iterator(this, 0, 0);
	}

	[[nodiscard]] iterator end() noexcept {
		ensureBuckets_();
		return iterator(this, buckets_.size(), 0);
	}

	[[nodiscard]] const_iterator end() const noexcept {
		if (buckets_.empty()) {
			return const_iterator(nullptr, 0, 0);
		}
		return const_iterator(this, buckets_.size(), 0);
	}


	[[nodiscard]] std::unordered_set<K, Hash, KeyEq> stdSet() const {
		std::unordered_set<K, Hash, KeyEq> out;
		out.reserve(static_cast<std::size_t>(size_));
		for (const_iterator it = begin(), e = end(); it != e; ++it) {
			out.insert(*it);
		}
		return out;
	}

private:
	oa::Vec<oa::Vec<K>> buckets_{};
	Hash hasher_{};
	KeyEq equal_{};
	SizeType size_{0};
	float maxLoad_{0.75F};

	void ensureBuckets_() {
		if (buckets_.empty()) {
			buckets_.resize(8);
		}
	}

	[[nodiscard]] SizeType bucketIndex_(const K& inKey) const noexcept {
		return static_cast<SizeType>(hasher_(inKey)) % buckets_.size();
	}

	void rehash_(SizeType inNewCap) {
		oa::Vec<oa::Vec<K>> old = std::move(buckets_);
		buckets_.resize(inNewCap);
		size_ = 0;
		for (SizeType bi = 0; bi < old.size(); ++bi) {
			oa::Vec<K>& chain = old[bi];
			const SizeType cnt = chain.size();
			for (SizeType si = 0; si < cnt; ++si) {
				insertNoGrow_(std::move(chain[si]));
			}
		}
	}

	void insertNoGrow_(K&& inKey) {
		const SizeType b = static_cast<SizeType>(hasher_(inKey)) % buckets_.size();
		buckets_[b].emplaceBack(std::move(inKey));
		++size_;
	}

	std::pair<iterator, bool> insertKey_(K&& inKey) {
		ensureBuckets_();
		if (buckets_.size() > 0 &&
			static_cast<float>(size_ + 1) > maxLoad_ * static_cast<float>(buckets_.size())) {
			rehash_(buckets_.size() * 2);
		}
		SizeType b = static_cast<SizeType>(hasher_(inKey)) % buckets_.size();
		oa::Vec<K>& ch = buckets_[b];
		for (SizeType i = 0; i < ch.size(); ++i) {
			if (equal_(ch[i], inKey)) {
				return {iterator(this, b, i), false};
			}
		}
		const SizeType idx = ch.size();
		ch.emplaceBack(std::move(inKey));
		++size_;
		return {iterator(this, b, idx), true};
	}
};

template<typename K, typename V, typename Hash, typename KeyEq>
inline typename HashMap<K, V, Hash, KeyEq>::iterator begin(HashMap<K, V, Hash, KeyEq>& inM) noexcept {
	return inM.begin();
}
template<typename K, typename V, typename Hash, typename KeyEq>
inline typename HashMap<K, V, Hash, KeyEq>::const_iterator begin(
	const HashMap<K, V, Hash, KeyEq>& inM) noexcept {
	return inM.begin();
}
template<typename K, typename V, typename Hash, typename KeyEq>
inline typename HashMap<K, V, Hash, KeyEq>::iterator end(HashMap<K, V, Hash, KeyEq>& inM) noexcept {
	return inM.end();
}
template<typename K, typename V, typename Hash, typename KeyEq>
inline typename HashMap<K, V, Hash, KeyEq>::const_iterator end(
	const HashMap<K, V, Hash, KeyEq>& inM) noexcept {
	return inM.end();
}

template<typename K, typename Hash, typename KeyEq>
inline typename HashSet<K, Hash, KeyEq>::iterator begin(HashSet<K, Hash, KeyEq>& inS) noexcept {
	return inS.begin();
}
template<typename K, typename Hash, typename KeyEq>
inline typename HashSet<K, Hash, KeyEq>::const_iterator begin(
	const HashSet<K, Hash, KeyEq>& inS) noexcept {
	return inS.begin();
}
template<typename K, typename Hash, typename KeyEq>
inline typename HashSet<K, Hash, KeyEq>::iterator end(HashSet<K, Hash, KeyEq>& inS) noexcept {
	return inS.end();
}
template<typename K, typename Hash, typename KeyEq>
inline typename HashSet<K, Hash, KeyEq>::const_iterator end(
	const HashSet<K, Hash, KeyEq>& inS) noexcept {
	return inS.end();
}

} // namespace oa
