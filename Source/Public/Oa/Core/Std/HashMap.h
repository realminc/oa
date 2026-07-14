#pragma once

// Phase 2b OaStd — bucket chaining via OaVec; `StdMap`/`StdSet` copy to std for boundaries.
// Iterators: forward category, prefix and postfix ++. Insert(pair&&) moves the mapped value into storage.

#include <Oa/Core/Std/Vec.h>

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

template<typename K, typename V, typename Hash = std::hash<K>, typename KeyEq = std::equal_to<K>>
class OaStdHashMap {
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
		friend class OaStdHashMap;
		friend class const_iterator;

	public:
		using iterator_category = std::forward_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using value_type = ValueType;
		using reference = SlotType&;
		using pointer = SlotType*;

		iterator() noexcept = default;

		reference operator*() const { return Map_->Buckets_[Bucket_][Slot_]; }

		pointer operator->() const { return &Map_->Buckets_[Bucket_][Slot_]; }

		iterator& operator++() {
			++Slot_;
			Skip();
			return *this;
		}

		iterator operator++(int) {
			iterator tmp = *this;
			++*this;
			return tmp;
		}

		bool operator==(const iterator& InO) const noexcept {
			return Map_ == InO.Map_ && Bucket_ == InO.Bucket_ && Slot_ == InO.Slot_;
		}

		bool operator!=(const iterator& InO) const noexcept { return !(*this == InO); }

	private:
		OaStdHashMap* Map_{nullptr};
		SizeType Bucket_{0};
		SizeType Slot_{0};

		iterator(OaStdHashMap* InMap, SizeType InB, SizeType InS) noexcept
			: Map_(InMap), Bucket_(InB), Slot_(InS) {
			Skip();
		}

		void Skip() noexcept {
			if (!Map_) {
				return;
			}
			for (;;) {
				if (Bucket_ >= Map_->Buckets_.Size()) {
					return;
				}
				if (Slot_ < Map_->Buckets_[Bucket_].Size()) {
					return;
				}
				++Bucket_;
				Slot_ = 0;
			}
		}
	};

	class const_iterator {
		friend class OaStdHashMap;

	public:
		using iterator_category = std::forward_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using value_type = ValueType;
		using reference = const SlotType&;
		using pointer = const SlotType*;

		const_iterator() noexcept = default;

		const_iterator(iterator InIt) noexcept
			: Map_(InIt.Map_), Bucket_(InIt.Bucket_), Slot_(InIt.Slot_) {}

		reference operator*() const { return Map_->Buckets_[Bucket_][Slot_]; }

		pointer operator->() const { return &Map_->Buckets_[Bucket_][Slot_]; }

		const_iterator& operator++() {
			++Slot_;
			Skip();
			return *this;
		}

		const_iterator operator++(int) {
			const_iterator tmp = *this;
			++*this;
			return tmp;
		}

		bool operator==(const const_iterator& InO) const noexcept {
			return Map_ == InO.Map_ && Bucket_ == InO.Bucket_ && Slot_ == InO.Slot_;
		}

		bool operator!=(const const_iterator& InO) const noexcept { return !(*this == InO); }

	private:
		const OaStdHashMap* Map_{nullptr};
		SizeType Bucket_{0};
		SizeType Slot_{0};

		const_iterator(const OaStdHashMap* InMap, SizeType InB, SizeType InS) noexcept
			: Map_(InMap), Bucket_(InB), Slot_(InS) {
			Skip();
		}

		void Skip() noexcept {
			if (!Map_) {
				return;
			}
			for (;;) {
				if (Bucket_ >= Map_->Buckets_.Size()) {
					return;
				}
				if (Slot_ < Map_->Buckets_[Bucket_].Size()) {
					return;
				}
				++Bucket_;
				Slot_ = 0;
			}
		}
	};

	friend class iterator;
	friend class const_iterator;

	[[nodiscard]] SizeType Size() const noexcept { return Size_; }

	[[nodiscard]] bool Empty() const noexcept { return Size_ == 0; }

	void Clear() noexcept {
		for (SizeType i = 0; i < Buckets_.Size(); ++i) {
			Buckets_[i].Clear();
		}
		Size_ = 0;
	}

	void Reserve(SizeType InN) {
		if (InN <= Buckets_.Size()) {
			return;
		}
		Rehash(InN);
	}

	template<typename... Args>
	std::pair<iterator, bool> Emplace(Args&&... InArgs) {
		SlotType slot(std::forward<Args>(InArgs)...);
		return InsertStorage(std::move(slot));
	}

	std::pair<iterator, bool> Insert(const value_type& InVal) {
		SlotType slot(InVal.first, InVal.second);
		return InsertStorage(std::move(slot));
	}

	std::pair<iterator, bool> Insert(value_type&& InVal) {
		SlotType slot(InVal.first, std::move(InVal.second));
		return InsertStorage(std::move(slot));
	}

	[[nodiscard]] V& At(const K& InKey) {
		iterator it = Find(InKey);
		if (it == End()) {
			throw std::out_of_range("OaStdHashMap::At");
		}
		return it->second;
	}

	[[nodiscard]] const V& At(const K& InKey) const {
		const_iterator it = Find(InKey);
		if (it == End()) {
			throw std::out_of_range("OaStdHashMap::At");
		}
		return it->second;
	}

	[[nodiscard]] iterator Find(const K& InKey) {
		EnsureBuckets();
		const SizeType b = BucketIndex(InKey);
		OaVec<SlotType>& ch = Buckets_[b];
		for (SizeType i = 0; i < ch.Size(); ++i) {
			if (Eq_(ch[i].first, InKey)) {
				return iterator(this, b, i);
			}
		}
		return End();
	}

	[[nodiscard]] const_iterator Find(const K& InKey) const {
		if (Buckets_.Empty()) {
			return End();
		}
		const SizeType b = BucketIndex(InKey);
		const OaVec<SlotType>& ch = Buckets_[b];
		for (SizeType i = 0; i < ch.Size(); ++i) {
			if (Eq_(ch[i].first, InKey)) {
				return const_iterator(this, b, i);
			}
		}
		return End();
	}

	[[nodiscard]] bool Contains(const K& InKey) const { return Find(InKey) != End(); }

	SizeType Erase(const K& InKey) {
		EnsureBuckets();
		const SizeType b = BucketIndex(InKey);
		OaVec<SlotType>& ch = Buckets_[b];
		for (SizeType i = 0; i < ch.Size(); ++i) {
			if (Eq_(ch[i].first, InKey)) {
				ch[i] = std::move(ch.Back());
				ch.PopBack();
				--Size_;
				return 1;
			}
		}
		return 0;
	}

	[[nodiscard]] iterator Begin() noexcept {
		EnsureBuckets();
		return iterator(this, 0, 0);
	}

	[[nodiscard]] const_iterator Begin() const noexcept {
		if (Buckets_.Empty()) {
			return End();
		}
		return const_iterator(this, 0, 0);
	}

	[[nodiscard]] iterator End() noexcept {
		EnsureBuckets();
		return iterator(this, Buckets_.Size(), 0);
	}

	[[nodiscard]] const_iterator End() const noexcept {
		if (Buckets_.Empty()) {
			return const_iterator(nullptr, 0, 0);
		}
		return const_iterator(this, Buckets_.Size(), 0);
	}

	[[nodiscard]] iterator begin() noexcept { return Begin(); }
	[[nodiscard]] const_iterator begin() const noexcept { return Begin(); }
	[[nodiscard]] iterator end() noexcept { return End(); }
	[[nodiscard]] const_iterator end() const noexcept { return End(); }

	[[nodiscard]] std::unordered_map<K, V, Hash, KeyEq> StdMap() const {
		std::unordered_map<K, V, Hash, KeyEq> out;
		out.reserve(static_cast<std::size_t>(Size_));
		for (const_iterator it = Begin(), e = End(); it != e; ++it) {
			out.emplace(it->first, it->second);
		}
		return out;
	}

private:
	OaVec<OaVec<SlotType>> Buckets_{};
	Hash Hasher_{};
	KeyEq Eq_{};
	SizeType Size_{0};
	float LoadMax_{0.75F};

	void EnsureBuckets() {
		if (Buckets_.Empty()) {
			Buckets_.Resize(8);
		}
	}

	[[nodiscard]] SizeType BucketIndex(const K& InKey) const noexcept {
		return static_cast<SizeType>(Hasher_(InKey)) % Buckets_.Size();
	}

	void Rehash(SizeType InNewCap) {
		OaVec<OaVec<SlotType>> old = std::move(Buckets_);
		Buckets_.Resize(InNewCap);
		Size_ = 0;
		for (SizeType bi = 0; bi < old.Size(); ++bi) {
			OaVec<SlotType>& chain = old[bi];
			const SizeType cnt = chain.Size();
			for (SizeType si = 0; si < cnt; ++si) {
				InsertNoGrow(std::move(chain[si]));
			}
		}
	}

	void InsertNoGrow(SlotType&& InSlot) {
		const SizeType b = static_cast<SizeType>(Hasher_(InSlot.first)) % Buckets_.Size();
		Buckets_[b].EmplaceBack(std::move(InSlot));
		++Size_;
	}

	std::pair<iterator, bool> InsertStorage(SlotType&& InSlot) {
		EnsureBuckets();
		const K& key = InSlot.first;
		if (Buckets_.Size() > 0 &&
			static_cast<float>(Size_ + 1) > LoadMax_ * static_cast<float>(Buckets_.Size())) {
			Rehash(Buckets_.Size() * 2);
		}
		SizeType b = static_cast<SizeType>(Hasher_(key)) % Buckets_.Size();
		OaVec<SlotType>& ch = Buckets_[b];
		for (SizeType i = 0; i < ch.Size(); ++i) {
			if (Eq_(ch[i].first, key)) {
				return {iterator(this, b, i), false};
			}
		}
		const SizeType idx = ch.Size();
		ch.EmplaceBack(std::move(InSlot));
		++Size_;
		return {iterator(this, b, idx), true};
	}
};

template<typename K, typename Hash = std::hash<K>, typename KeyEq = std::equal_to<K>>
class OaStdHashSet {
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
		friend class OaStdHashSet;
		friend class const_iterator;

	public:
		using iterator_category = std::forward_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using value_type = KeyType;
		using reference = KeyType&;
		using pointer = KeyType*;

		iterator() noexcept = default;

		reference operator*() const { return Map_->Buckets_[Bucket_][Slot_]; }

		pointer operator->() const { return &Map_->Buckets_[Bucket_][Slot_]; }

		iterator& operator++() {
			++Slot_;
			Skip();
			return *this;
		}

		iterator operator++(int) {
			iterator tmp = *this;
			++*this;
			return tmp;
		}

		bool operator==(const iterator& InO) const noexcept {
			return Map_ == InO.Map_ && Bucket_ == InO.Bucket_ && Slot_ == InO.Slot_;
		}

		bool operator!=(const iterator& InO) const noexcept { return !(*this == InO); }

	private:
		OaStdHashSet* Map_{nullptr};
		SizeType Bucket_{0};
		SizeType Slot_{0};

		iterator(OaStdHashSet* InMap, SizeType InB, SizeType InS) noexcept
			: Map_(InMap), Bucket_(InB), Slot_(InS) {
			Skip();
		}

		void Skip() noexcept {
			if (!Map_) {
				return;
			}
			for (;;) {
				if (Bucket_ >= Map_->Buckets_.Size()) {
					return;
				}
				if (Slot_ < Map_->Buckets_[Bucket_].Size()) {
					return;
				}
				++Bucket_;
				Slot_ = 0;
			}
		}
	};

	class const_iterator {
		friend class OaStdHashSet;

	public:
		using iterator_category = std::forward_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using value_type = KeyType;
		using reference = const KeyType&;
		using pointer = const KeyType*;

		const_iterator() noexcept = default;

		const_iterator(iterator InIt) noexcept
			: Map_(InIt.Map_), Bucket_(InIt.Bucket_), Slot_(InIt.Slot_) {}

		reference operator*() const { return Map_->Buckets_[Bucket_][Slot_]; }

		pointer operator->() const { return &Map_->Buckets_[Bucket_][Slot_]; }

		const_iterator& operator++() {
			++Slot_;
			Skip();
			return *this;
		}

		const_iterator operator++(int) {
			const_iterator tmp = *this;
			++*this;
			return tmp;
		}

		bool operator==(const const_iterator& InO) const noexcept {
			return Map_ == InO.Map_ && Bucket_ == InO.Bucket_ && Slot_ == InO.Slot_;
		}

		bool operator!=(const const_iterator& InO) const noexcept { return !(*this == InO); }

	private:
		const OaStdHashSet* Map_{nullptr};
		SizeType Bucket_{0};
		SizeType Slot_{0};

		const_iterator(const OaStdHashSet* InMap, SizeType InB, SizeType InS) noexcept
			: Map_(InMap), Bucket_(InB), Slot_(InS) {
			Skip();
		}

		void Skip() noexcept {
			if (!Map_) {
				return;
			}
			for (;;) {
				if (Bucket_ >= Map_->Buckets_.Size()) {
					return;
				}
				if (Slot_ < Map_->Buckets_[Bucket_].Size()) {
					return;
				}
				++Bucket_;
				Slot_ = 0;
			}
		}
	};

	friend class iterator;
	friend class const_iterator;

	[[nodiscard]] SizeType Size() const noexcept { return Size_; }

	[[nodiscard]] bool Empty() const noexcept { return Size_ == 0; }

	void Clear() noexcept {
		for (SizeType i = 0; i < Buckets_.Size(); ++i) {
			Buckets_[i].Clear();
		}
		Size_ = 0;
	}

	void Reserve(SizeType InN) {
		if (InN <= Buckets_.Size()) {
			return;
		}
		Rehash(InN);
	}

	std::pair<iterator, bool> Insert(const K& InKey) {
		return InsertKey(K(InKey));
	}

	std::pair<iterator, bool> Insert(K&& InKey) { return InsertKey(std::move(InKey)); }

	[[nodiscard]] iterator Find(const K& InKey) {
		EnsureBuckets();
		const SizeType b = BucketIndex(InKey);
		OaVec<K>& ch = Buckets_[b];
		for (SizeType i = 0; i < ch.Size(); ++i) {
			if (Eq_(ch[i], InKey)) {
				return iterator(this, b, i);
			}
		}
		return End();
	}

	[[nodiscard]] const_iterator Find(const K& InKey) const {
		if (Buckets_.Empty()) {
			return End();
		}
		const SizeType b = BucketIndex(InKey);
		const OaVec<K>& ch = Buckets_[b];
		for (SizeType i = 0; i < ch.Size(); ++i) {
			if (Eq_(ch[i], InKey)) {
				return const_iterator(this, b, i);
			}
		}
		return End();
	}

	[[nodiscard]] bool Contains(const K& InKey) const { return Find(InKey) != End(); }

	SizeType Erase(const K& InKey) {
		EnsureBuckets();
		const SizeType b = BucketIndex(InKey);
		OaVec<K>& ch = Buckets_[b];
		for (SizeType i = 0; i < ch.Size(); ++i) {
			if (Eq_(ch[i], InKey)) {
				ch[i] = std::move(ch.Back());
				ch.PopBack();
				--Size_;
				return 1;
			}
		}
		return 0;
	}

	[[nodiscard]] iterator Begin() noexcept {
		EnsureBuckets();
		return iterator(this, 0, 0);
	}

	[[nodiscard]] const_iterator Begin() const noexcept {
		if (Buckets_.Empty()) {
			return End();
		}
		return const_iterator(this, 0, 0);
	}

	[[nodiscard]] iterator End() noexcept {
		EnsureBuckets();
		return iterator(this, Buckets_.Size(), 0);
	}

	[[nodiscard]] const_iterator End() const noexcept {
		if (Buckets_.Empty()) {
			return const_iterator(nullptr, 0, 0);
		}
		return const_iterator(this, Buckets_.Size(), 0);
	}

	[[nodiscard]] iterator begin() noexcept { return Begin(); }
	[[nodiscard]] const_iterator begin() const noexcept { return Begin(); }
	[[nodiscard]] iterator end() noexcept { return End(); }
	[[nodiscard]] const_iterator end() const noexcept { return End(); }

	[[nodiscard]] std::unordered_set<K, Hash, KeyEq> StdSet() const {
		std::unordered_set<K, Hash, KeyEq> out;
		out.reserve(static_cast<std::size_t>(Size_));
		for (const_iterator it = Begin(), e = End(); it != e; ++it) {
			out.insert(*it);
		}
		return out;
	}

private:
	OaVec<OaVec<K>> Buckets_{};
	Hash Hasher_{};
	KeyEq Eq_{};
	SizeType Size_{0};
	float LoadMax_{0.75F};

	void EnsureBuckets() {
		if (Buckets_.Empty()) {
			Buckets_.Resize(8);
		}
	}

	[[nodiscard]] SizeType BucketIndex(const K& InKey) const noexcept {
		return static_cast<SizeType>(Hasher_(InKey)) % Buckets_.Size();
	}

	void Rehash(SizeType InNewCap) {
		OaVec<OaVec<K>> old = std::move(Buckets_);
		Buckets_.Resize(InNewCap);
		Size_ = 0;
		for (SizeType bi = 0; bi < old.Size(); ++bi) {
			OaVec<K>& chain = old[bi];
			const SizeType cnt = chain.Size();
			for (SizeType si = 0; si < cnt; ++si) {
				InsertNoGrow(std::move(chain[si]));
			}
		}
	}

	void InsertNoGrow(K&& InKey) {
		const SizeType b = static_cast<SizeType>(Hasher_(InKey)) % Buckets_.Size();
		Buckets_[b].EmplaceBack(std::move(InKey));
		++Size_;
	}

	std::pair<iterator, bool> InsertKey(K&& InKey) {
		EnsureBuckets();
		if (Buckets_.Size() > 0 &&
			static_cast<float>(Size_ + 1) > LoadMax_ * static_cast<float>(Buckets_.Size())) {
			Rehash(Buckets_.Size() * 2);
		}
		SizeType b = static_cast<SizeType>(Hasher_(InKey)) % Buckets_.Size();
		OaVec<K>& ch = Buckets_[b];
		for (SizeType i = 0; i < ch.Size(); ++i) {
			if (Eq_(ch[i], InKey)) {
				return {iterator(this, b, i), false};
			}
		}
		const SizeType idx = ch.Size();
		ch.EmplaceBack(std::move(InKey));
		++Size_;
		return {iterator(this, b, idx), true};
	}
};

template<typename K, typename V, typename Hash, typename KeyEq>
inline typename OaStdHashMap<K, V, Hash, KeyEq>::iterator begin(OaStdHashMap<K, V, Hash, KeyEq>& InM) noexcept {
	return InM.Begin();
}
template<typename K, typename V, typename Hash, typename KeyEq>
inline typename OaStdHashMap<K, V, Hash, KeyEq>::const_iterator begin(
	const OaStdHashMap<K, V, Hash, KeyEq>& InM) noexcept {
	return InM.Begin();
}
template<typename K, typename V, typename Hash, typename KeyEq>
inline typename OaStdHashMap<K, V, Hash, KeyEq>::iterator end(OaStdHashMap<K, V, Hash, KeyEq>& InM) noexcept {
	return InM.End();
}
template<typename K, typename V, typename Hash, typename KeyEq>
inline typename OaStdHashMap<K, V, Hash, KeyEq>::const_iterator end(
	const OaStdHashMap<K, V, Hash, KeyEq>& InM) noexcept {
	return InM.End();
}

template<typename K, typename Hash, typename KeyEq>
inline typename OaStdHashSet<K, Hash, KeyEq>::iterator begin(OaStdHashSet<K, Hash, KeyEq>& InS) noexcept {
	return InS.Begin();
}
template<typename K, typename Hash, typename KeyEq>
inline typename OaStdHashSet<K, Hash, KeyEq>::const_iterator begin(
	const OaStdHashSet<K, Hash, KeyEq>& InS) noexcept {
	return InS.Begin();
}
template<typename K, typename Hash, typename KeyEq>
inline typename OaStdHashSet<K, Hash, KeyEq>::iterator end(OaStdHashSet<K, Hash, KeyEq>& InS) noexcept {
	return InS.End();
}
template<typename K, typename Hash, typename KeyEq>
inline typename OaStdHashSet<K, Hash, KeyEq>::const_iterator end(
	const OaStdHashSet<K, Hash, KeyEq>& InS) noexcept {
	return InS.End();
}
