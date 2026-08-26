// oa::SecureBuffer — cryptographic key buffer
//
// Non-owning view that securely erases its backing range on destruction.
// Linux mlock() is best-effort and observable through isLocked(). The caller
// retains allocation ownership and must keep the range alive for this view.

#pragma once

#include <oa/core/types.h>
#include <oa/core/memory.h>

#ifdef OA_PLATFORM_LINUX
#include <sys/mman.h>
#endif

namespace oa {

class SecureBuffer {
public:
	SecureBuffer() = default;

	SecureBuffer(void* inPtr, oa::U64 inSize) : ptr_(inPtr), size_(inSize) {
		if (ptr_ && size_ > 0) {
#ifdef OA_PLATFORM_LINUX
			locked_ = mlock(ptr_, size_) == 0;
#endif
		}
	}

	~SecureBuffer() {
		reset();
	}

	SecureBuffer(const SecureBuffer&) = delete;
	SecureBuffer& operator=(const SecureBuffer&) = delete;

	SecureBuffer(SecureBuffer&& inOther) noexcept
		: ptr_(inOther.ptr_), size_(inOther.size_), locked_(inOther.locked_) {
		inOther.ptr_ = nullptr;
		inOther.size_ = 0;
		inOther.locked_ = false;
	}

	SecureBuffer& operator=(SecureBuffer&& inOther) noexcept {
		if (this != &inOther) {
			reset();
			ptr_ = inOther.ptr_;
			size_ = inOther.size_;
			locked_ = inOther.locked_;
			inOther.ptr_ = nullptr;
			inOther.size_ = 0;
			inOther.locked_ = false;
		}
		return *this;
	}

	void secureZero() noexcept {
		if (!ptr_ || size_ == 0) return;
		// Volatile stores are deliberately simple and portable; erasing a few KiB
		// of key material is not a throughput path.
		volatile oa::U8* vp = static_cast<volatile oa::U8*>(ptr_);
		for (oa::U64 i = 0; i < size_; ++i) {
			vp[i] = 0;
		}
	}

	void reset() noexcept {
		secureZero();
#ifdef OA_PLATFORM_LINUX
		if (locked_) {
			munlock(ptr_, size_);
		}
#endif
		ptr_ = nullptr;
		size_ = 0;
		locked_ = false;
	}

	[[nodiscard]] oa::U8* data() { return static_cast<oa::U8*>(ptr_); }
	[[nodiscard]] const oa::U8* data() const { return static_cast<const oa::U8*>(ptr_); }
	[[nodiscard]] oa::U64 sizeBytes() const { return size_; }
	[[nodiscard]] bool isValid() const { return ptr_ != nullptr && size_ > 0; }
	[[nodiscard]] bool isLocked() const { return locked_; }

private:
	void* ptr_ = nullptr;
	oa::U64 size_ = 0;
	bool locked_ = false;
};

} // namespace oa
