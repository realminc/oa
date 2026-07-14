// OaSecureBuffer — Cryptographic key buffer
//
// Non-owning view that securely erases its backing range on destruction.
// Linux mlock() is best-effort and observable through IsLocked(). The caller
// retains allocation ownership and must keep the range alive for this view.

#pragma once

#include <cstring>

#include <Oa/Core/Types.h>
#include <Oa/Core/Memory.h>

#ifdef OA_PLATFORM_LINUX
#include <sys/mman.h>
#endif

class OaSecureBuffer {
public:
	OaSecureBuffer() = default;

	OaSecureBuffer(void* InPtr, OaU64 InSize) : Ptr_(InPtr), Size_(InSize) {
		if (Ptr_ && Size_ > 0) {
#ifdef OA_PLATFORM_LINUX
			Locked_ = mlock(Ptr_, Size_) == 0;
#endif
		}
	}

	~OaSecureBuffer() {
		Reset();
	}

	OaSecureBuffer(const OaSecureBuffer&) = delete;
	OaSecureBuffer& operator=(const OaSecureBuffer&) = delete;

	OaSecureBuffer(OaSecureBuffer&& InOther) noexcept
		: Ptr_(InOther.Ptr_), Size_(InOther.Size_), Locked_(InOther.Locked_) {
		InOther.Ptr_ = nullptr;
		InOther.Size_ = 0;
		InOther.Locked_ = false;
	}

	OaSecureBuffer& operator=(OaSecureBuffer&& InOther) noexcept {
		if (this != &InOther) {
			Reset();
			Ptr_ = InOther.Ptr_;
			Size_ = InOther.Size_;
			Locked_ = InOther.Locked_;
			InOther.Ptr_ = nullptr;
			InOther.Size_ = 0;
			InOther.Locked_ = false;
		}
		return *this;
	}

	void SecureZero() noexcept {
		if (!Ptr_ || Size_ == 0) return;
		// Volatile stores are deliberately simple and portable; erasing a few KiB
		// of key material is not a throughput path.
		volatile OaU8* vp = static_cast<volatile OaU8*>(Ptr_);
		for (OaU64 i = 0; i < Size_; ++i) {
			vp[i] = 0;
		}
	}

	void Reset() noexcept {
		SecureZero();
#ifdef OA_PLATFORM_LINUX
		if (Locked_) {
			munlock(Ptr_, Size_);
		}
#endif
		Ptr_ = nullptr;
		Size_ = 0;
		Locked_ = false;
	}

	[[nodiscard]] OaU8* Data() { return static_cast<OaU8*>(Ptr_); }
	[[nodiscard]] const OaU8* Data() const { return static_cast<const OaU8*>(Ptr_); }
	[[nodiscard]] OaU64 SizeBytes() const { return Size_; }
	[[nodiscard]] bool IsValid() const { return Ptr_ != nullptr && Size_ > 0; }
	[[nodiscard]] bool IsLocked() const { return Locked_; }

private:
	void* Ptr_ = nullptr;
	OaU64 Size_ = 0;
	bool Locked_ = false;
};
