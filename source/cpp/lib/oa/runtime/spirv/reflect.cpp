// SPIR-V push-constant block reflection.
//
// Parses just enough of a compute module to compute the exact byte size of its
// PushConstant block. This backs the bindless buffer-binding contract assert in
// private context dispatch recording: the GPU push the runtime assembles is
//   [numBuffers * 4 bytes of auto-prepended buffer indices] ++ [host push tail]
// and that must exactly fill the shader's declared PushConstants struct. A wrong
// buffer count (the recurring bug class: SwigluBwd 6→5, MaxPool2dBwd 5→3,
// RmsNormBwd 7→5, GRU fused, LeakyReluBwd) silently shifts every binding; this
// reflection lets us catch it deterministically at record time.
//
// Conservative by design: any member type we cannot size exactly makes the whole
// reflection return 0 (= "unknown"), so a non-zero result is always trustworthy
// and the caller simply skips the assert when 0.

#include <oa/runtime/spirv.h>
#include <oa/core/std/hashMap.h>
#include <oa/core/std/sync.h>
#include <oa/core/std/vec.h>

namespace {

// SPIR-V constants we care about.
constexpr oa::U32 kSpirvMagic = 0x07230203u;
constexpr oa::U32 kOpTypeInt = 21;
constexpr oa::U32 kOpTypeFloat = 22;
constexpr oa::U32 kOpTypeVector = 23;
constexpr oa::U32 kOpTypeStruct = 30;
constexpr oa::U32 kOpTypePointer = 32;
constexpr oa::U32 kOpMemberDecorate = 72;
constexpr oa::U32 kStorageClassPushConstant = 9;
constexpr oa::U32 kDecorationOffset = 35;

struct ReflectState {
	// Indexed by SPIR-V result id (< bound).
	oa::Vec<oa::U8> kind;        // 0=unknown, 1=scalar, 2=vector, 3=struct
	oa::Vec<oa::U32> scalarSize; // bytes, for kind==1
	oa::Vec<oa::U32> vecComp;    // component type id, for kind==2
	oa::Vec<oa::U32> vecCount;   // component count, for kind==2
	oa::Vec<oa::Vec<oa::U32>> structMembers; // member type ids
	oa::Vec<oa::Vec<oa::U32>> memberOffset;  // byte offset per member
	oa::U32 pushStructId = 0;

	oa::U32 sizeOfType(oa::U32 inId) const {
		if (inId == 0 || inId >= kind.size()) return 0;
		if (kind[inId] == 1) return scalarSize[inId];
		if (kind[inId] == 2) {
			const oa::U32 cs = sizeOfType(vecComp[inId]);
			if (cs == 0) return 0;
			return cs * vecCount[inId];
		}
		return 0;
	}
};

}  // namespace

oa::U32 oavk::spirvPushConstantBlockSize(const oa::U8* inSpirv, oa::U32 inSizeBytes) {
	if (!inSpirv || inSizeBytes < 20 || (inSizeBytes % 4) != 0) return 0;
	const oa::U32* words = reinterpret_cast<const oa::U32*>(inSpirv);
	const oa::U32 wordCount = inSizeBytes / 4;
	if (words[0] != kSpirvMagic) return 0;
	const oa::U32 bound = words[3];
	if (bound == 0 || bound > (1u << 22)) return 0;  // sanity cap

	ReflectState s;
	s.kind.assign(bound, 0);
	s.scalarSize.assign(bound, 0);
	s.vecComp.assign(bound, 0);
	s.vecCount.assign(bound, 0);
	s.structMembers.assign(bound, oa::Vec<oa::U32>{});
	s.memberOffset.assign(bound, oa::Vec<oa::U32>{});

	auto safeId = [&](oa::U32 id) -> bool { return id != 0 && id < bound; };

	// Instruction stream starts after the 5-word header.
	oa::U32 i = 5;
	while (i < wordCount) {
		const oa::U32 first = words[i];
		const oa::U32 count = first >> 16;
		const oa::U32 op = first & 0xFFFFu;
		if (count == 0 || i + count > wordCount) break;  // malformed → stop
		const oa::U32* ops = &words[i + 1];
		const oa::U32 nOps = count - 1;

		switch (op) {
			case kOpTypeInt:    // [result, width, signedness]
			case kOpTypeFloat:  // [result, width]
				if (nOps >= 2 && safeId(ops[0])) {
					s.kind[ops[0]] = 1;
					s.scalarSize[ops[0]] = ops[1] / 8;
				}
				break;
			case kOpTypeVector:  // [result, component_type, count]
				if (nOps >= 3 && safeId(ops[0])) {
					s.kind[ops[0]] = 2;
					s.vecComp[ops[0]] = ops[1];
					s.vecCount[ops[0]] = ops[2];
				}
				break;
			case kOpTypeStruct:  // [result, member0, member1, ...]
				if (nOps >= 1 && safeId(ops[0])) {
					s.kind[ops[0]] = 3;
					s.structMembers[ops[0]].assign(ops + 1, ops + nOps);
				}
				break;
			case kOpTypePointer:  // [result, storage_class, type]
				// slang emits multiple PushConstant pointers: one to the block
				// struct and bare ones to individual members (for OpAccessChain).
				// Only the struct-pointee is the block we want to size.
				if (nOps >= 3 && ops[1] == kStorageClassPushConstant &&
					safeId(ops[2]) && s.kind[ops[2]] == 3) {
					s.pushStructId = ops[2];
				}
				break;
			case kOpMemberDecorate:  // [struct, member, decoration, literals...]
				if (nOps >= 4 && ops[2] == kDecorationOffset && safeId(ops[0])) {
					auto& offs = s.memberOffset[ops[0]];
					const oa::U32 member = ops[1];
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
	const oa::U32 sid = s.pushStructId;
	if (!safeId(sid) || s.structMembers[sid].empty()) return 0;
	const auto& members = s.structMembers[sid];
	const auto& offsets = s.memberOffset[sid];
	if (offsets.size() != members.size()) return 0;

	oa::U32 total = 0;
	for (oa::Usize m = 0; m < members.size(); ++m) {
		const oa::U32 msz = s.sizeOfType(members[m]);
		if (msz == 0) return 0;  // unsizeable member → bail conservatively
		const oa::U32 end = offsets[m] + msz;
		if (end > total) total = end;
	}
	return total;
}

oa::U32 oavk::spirvPushConstantBlockSizeByName(const char* inName) {
	if (!inName) return 0;
	static oa::Mutex mutex;
	static oa::HashMap<oa::String, oa::U32> cache;
	const oa::String name(inName);
	{
		oa::ScopedLock lock(mutex);
		auto it = cache.find(name);
		if (it != cache.end()) return it->second;
	}
	const oavk::SpirvEntry* entry = oavk::findSpirv(inName);
	const oa::U32 size = entry ? oavk::spirvPushConstantBlockSize(entry->data, entry->size) : 0;
	{
		oa::ScopedLock lock(mutex);
		cache.emplace(name, size);
	}
	return size;
}
