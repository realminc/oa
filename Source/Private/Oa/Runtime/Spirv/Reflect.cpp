// SPIR-V push-constant block reflection.
//
// Parses just enough of a compute module to compute the exact byte size of its
// PushConstant block. This backs the bindless buffer-binding contract assert in
// OaContext::Record: the GPU push the runtime assembles is
//   [numBuffers * 4 bytes of auto-prepended buffer indices] ++ [host push tail]
// and that must exactly fill the shader's declared PushConstants struct. A wrong
// buffer count (the recurring bug class: SwigluBwd 6→5, MaxPool2dBwd 5→3,
// RmsNormBwd 7→5, GRU fused, LeakyReluBwd) silently shifts every binding; this
// reflection lets us catch it deterministically at record time.
//
// Conservative by design: any member type we cannot size exactly makes the whole
// reflection return 0 (= "unknown"), so a non-zero result is always trustworthy
// and the caller simply skips the assert when 0.

#include <Oa/Runtime/Spirv.h>

#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

// SPIR-V constants we care about.
constexpr OaU32 kSpirvMagic = 0x07230203u;
constexpr OaU32 kOpTypeInt = 21;
constexpr OaU32 kOpTypeFloat = 22;
constexpr OaU32 kOpTypeVector = 23;
constexpr OaU32 kOpTypeStruct = 30;
constexpr OaU32 kOpTypePointer = 32;
constexpr OaU32 kOpMemberDecorate = 72;
constexpr OaU32 kStorageClassPushConstant = 9;
constexpr OaU32 kDecorationOffset = 35;

struct ReflectState {
	// Indexed by SPIR-V result id (< bound).
	std::vector<OaU8>  Kind;        // 0=unknown, 1=scalar, 2=vector, 3=struct
	std::vector<OaU32> ScalarSize;  // bytes, for Kind==1
	std::vector<OaU32> VecComp;     // component type id, for Kind==2
	std::vector<OaU32> VecCount;    // component count, for Kind==2
	std::vector<std::vector<OaU32>> StructMembers;  // member type ids
	std::vector<std::vector<OaU32>> MemberOffset;   // byte offset per member
	OaU32 PushStructId = 0;

	OaU32 SizeOfType(OaU32 InId) const {
		if (InId == 0 || InId >= Kind.size()) return 0;
		if (Kind[InId] == 1) return ScalarSize[InId];
		if (Kind[InId] == 2) {
			const OaU32 cs = SizeOfType(VecComp[InId]);
			if (cs == 0) return 0;
			return cs * VecCount[InId];
		}
		return 0;
	}
};

}  // namespace

OaU32 OaSpvPushConstantBlockSize(const OaU8* InSpirv, OaU32 InSizeBytes) {
	if (!InSpirv || InSizeBytes < 20 || (InSizeBytes % 4) != 0) return 0;
	const OaU32* words = reinterpret_cast<const OaU32*>(InSpirv);
	const OaU32 wordCount = InSizeBytes / 4;
	if (words[0] != kSpirvMagic) return 0;
	const OaU32 bound = words[3];
	if (bound == 0 || bound > (1u << 22)) return 0;  // sanity cap

	ReflectState s;
	s.Kind.assign(bound, 0);
	s.ScalarSize.assign(bound, 0);
	s.VecComp.assign(bound, 0);
	s.VecCount.assign(bound, 0);
	s.StructMembers.assign(bound, {});
	s.MemberOffset.assign(bound, {});

	auto safeId = [&](OaU32 id) -> bool { return id != 0 && id < bound; };

	// Instruction stream starts after the 5-word header.
	OaU32 i = 5;
	while (i < wordCount) {
		const OaU32 first = words[i];
		const OaU32 count = first >> 16;
		const OaU32 op = first & 0xFFFFu;
		if (count == 0 || i + count > wordCount) break;  // malformed → stop
		const OaU32* ops = &words[i + 1];
		const OaU32 nOps = count - 1;

		switch (op) {
			case kOpTypeInt:    // [result, width, signedness]
			case kOpTypeFloat:  // [result, width]
				if (nOps >= 2 && safeId(ops[0])) {
					s.Kind[ops[0]] = 1;
					s.ScalarSize[ops[0]] = ops[1] / 8;
				}
				break;
			case kOpTypeVector:  // [result, component_type, count]
				if (nOps >= 3 && safeId(ops[0])) {
					s.Kind[ops[0]] = 2;
					s.VecComp[ops[0]] = ops[1];
					s.VecCount[ops[0]] = ops[2];
				}
				break;
			case kOpTypeStruct:  // [result, member0, member1, ...]
				if (nOps >= 1 && safeId(ops[0])) {
					s.Kind[ops[0]] = 3;
					s.StructMembers[ops[0]].assign(ops + 1, ops + nOps);
				}
				break;
			case kOpTypePointer:  // [result, storage_class, type]
				// Slang emits multiple PushConstant pointers: one to the block
				// struct and bare ones to individual members (for OpAccessChain).
				// Only the struct-pointee is the block we want to size.
				if (nOps >= 3 && ops[1] == kStorageClassPushConstant &&
					safeId(ops[2]) && s.Kind[ops[2]] == 3) {
					s.PushStructId = ops[2];
				}
				break;
			case kOpMemberDecorate:  // [struct, member, decoration, literals...]
				if (nOps >= 4 && ops[2] == kDecorationOffset && safeId(ops[0])) {
					auto& offs = s.MemberOffset[ops[0]];
					const OaU32 member = ops[1];
					if (member >= offs.size()) offs.resize(member + 1, 0);
					offs[member] = ops[3];
				}
				break;
			default:
				break;
		}
		i += count;
	}

	// A PushConstant pointer commonly targets the struct directly; some toolchains
	// point it at a wrapper. We only handle the direct-struct case (which is what
	// slang emits for these kernels) — anything else stays "unknown" (0).
	const OaU32 sid = s.PushStructId;
	if (!safeId(sid) || s.StructMembers[sid].empty()) return 0;
	const auto& members = s.StructMembers[sid];
	const auto& offsets = s.MemberOffset[sid];
	if (offsets.size() != members.size()) return 0;

	OaU32 total = 0;
	for (size_t m = 0; m < members.size(); ++m) {
		const OaU32 msz = s.SizeOfType(members[m]);
		if (msz == 0) return 0;  // unsizeable member → bail conservatively
		const OaU32 end = offsets[m] + msz;
		if (end > total) total = end;
	}
	return total;
}

OaU32 OaSpvPushConstantBlockSizeByName(const char* InName) {
	if (!InName) return 0;
	static std::mutex mtx;
	static std::unordered_map<std::string, OaU32> cache;
	{
		std::lock_guard<std::mutex> lock(mtx);
		auto it = cache.find(InName);
		if (it != cache.end()) return it->second;
	}
	const OaSpvEntry* entry = OaSpvFindAny(InName);
	const OaU32 size = entry ? OaSpvPushConstantBlockSize(entry->Data, entry->Size) : 0;
	{
		std::lock_guard<std::mutex> lock(mtx);
		cache[InName] = size;
	}
	return size;
}

OaU64 OaSpvContentHash(const OaU8* InSpirv, OaU32 InSizeBytes) {
	if (InSpirv == nullptr or InSizeBytes == 0U) return 0U;
	OaU64 hash = 0xcbf29ce484222325ULL;
	for (OaU32 i = 0; i < InSizeBytes; ++i) {
		hash ^= InSpirv[i];
		hash *= 0x100000001b3ULL;
	}
	return hash;
}
