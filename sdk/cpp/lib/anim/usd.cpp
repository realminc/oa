#include <anim/usd.h>

#include <core/streamText.h>

#include <cctype>
#include <cstdlib>
#include <set>
#include <sstream>
#include <string>
#include <utility>

bool oa::UsdSkelClip::isValid() const noexcept {
	const oa::I32 n = jointCount();
	if (n == 0 || frameCount == 0 || fps <= 0.0f) {
		return false;
	}
	const oa::Usize expect = static_cast<oa::Usize>(frameCount) * static_cast<oa::Usize>(n);
	if (translations.size() != expect || rotations.size() != expect) {
		return false;
	}
	// Bind/rest are optional, but if present must be one per joint.
	if (bindTransforms.size() != 0 && bindTransforms.size() != static_cast<oa::Usize>(n)) {
		return false;
	}
	if (restTransforms.size() != 0 && restTransforms.size() != static_cast<oa::Usize>(n)) {
		return false;
	}
	return true;
}

// ── Writer ───────────────────────────────────────────────────────────────────

namespace {

oa::String mat4Usd(const oa::vlm::Mat4& m) {
	std::ostringstream s;
	s.precision(9);
	s << "( ";
	for (int r = 0; r < 4; ++r) {
		s << "(" << m.m[r][0] << ", " << m.m[r][1] << ", " << m.m[r][2] << ", " << m.m[r][3] << ")";
		s << (r < 3 ? ", " : " ");
	}
	s << ")";
	return oa::sdk::fromStdString(s.str());
}

// Comma-separated quoted joint token list shared by Skeleton + SkelAnimation.
// USD convention: the root joint should be named "root", not the empty string.
oa::String jointsToken(const oa::UsdSkelClip& inClip) {
	const oa::I32 n = inClip.jointCount();
	std::ostringstream joints;
	for (oa::I32 j = 0; j < n; ++j) {
		const oa::String& path = inClip.jointPaths[static_cast<oa::Usize>(j)];
		const char* jointName = (j == 0 && path.empty()) ? "root" : path.cStr();
		joints << (j ? ", " : "") << "\"" << jointName << "\"";
	}
	return oa::sdk::fromStdString(joints.str());
}

// USD prim names must be valid identifiers: [A-Za-z_][A-Za-z0-9_]*.
oa::String sanitizePrimName(const oa::String& inName) {
	std::string s = oa::sdk::toStdString(inName);
	if (s.empty()) { return oa::String("Anim"); }
	std::string out;
	out.reserve(s.size());
	for (char c : s) {
		out.push_back((std::isalnum(static_cast<unsigned char>(c)) || c == '_') ? c : '_');
	}
	if (std::isdigit(static_cast<unsigned char>(out[0]))) { out.insert(out.begin(), '_'); }
	return oa::sdk::fromStdString(out);
}

// Emit one `def SkelAnimation "<animName>" { ... }` prim at indent `ind`. When
// `split`/`source` are non-null an `customData.oa` dictionary is written so the
// dataset partition + source name round-trip.
void emitSkelAnimation(std::ostream& o, const char* ind, const oa::String& inAnimName,
                       const oa::UsdSkelClip& clip, const oa::String& jointsTok,
                       const oa::String* split, const oa::String* source) {
	const oa::I32 n = clip.jointCount();

	o << "\n" << ind << "def SkelAnimation \"" << inAnimName << "\"";
	if (split != nullptr || source != nullptr) {
		o << " (\n";
		o << ind << "    customData = {\n";
		o << ind << "        dictionary oa = {\n";
		if (split  != nullptr) { o << ind << "            string split = \""  << split->cStr()  << "\"\n"; }
		if (source != nullptr) { o << ind << "            string source = \"" << source->cStr() << "\"\n"; }
		o << ind << "        }\n";
		o << ind << "    }\n";
		o << ind << ")";
	}
	o << "\n" << ind << "{\n";
	o << ind << "    uniform token[] joints = [" << jointsTok << "]\n";

	// Static rest rotations (frame 0 convention) + per-frame timeSamples.
	o << ind << "    quatf[] rotations = [";
	for (oa::I32 j = 0; j < n; ++j) {
		const oa::vlm::Quat& q = clip.rotations[static_cast<oa::Usize>(j)];
		o << (j ? ", " : "") << "(" << q.w << ", " << q.x << ", " << q.y << ", " << q.z << ")";
	}
	o << "]\n";
	o << ind << "    quatf[] rotations.timeSamples = {\n";
	for (oa::U32 f = 0; f < clip.frameCount; ++f) {
		o << ind << "        " << f << ": [";
		for (oa::I32 j = 0; j < n; ++j) {
			const oa::vlm::Quat& q = clip.rotations[static_cast<oa::Usize>(f) * n + j];
			o << (j ? ", " : "") << "(" << q.w << ", " << q.x << ", " << q.y << ", " << q.z << ")";
		}
		o << "],\n";
	}
	o << ind << "    }\n";

	o << ind << "    float3[] translations = [";
	for (oa::I32 j = 0; j < n; ++j) {
		const oa::vlm::Vec3& t = clip.translations[static_cast<oa::Usize>(j)];
		o << (j ? ", " : "") << "(" << t.x << ", " << t.y << ", " << t.z << ")";
	}
	o << "]\n";
	o << ind << "    float3[] translations.timeSamples = {\n";
	for (oa::U32 f = 0; f < clip.frameCount; ++f) {
		o << ind << "        " << f << ": [";
		for (oa::I32 j = 0; j < n; ++j) {
			const oa::vlm::Vec3& t = clip.translations[static_cast<oa::Usize>(f) * n + j];
			o << (j ? ", " : "") << "(" << t.x << ", " << t.y << ", " << t.z << ")";
		}
		o << "],\n";
	}
	o << ind << "    }\n";

	o << ind << "    half3[] scales = [";
	for (oa::I32 j = 0; j < n; ++j) {
		o << (j ? ", " : "") << "(1, 1, 1)";
	}
	o << "]\n";
	o << ind << "}\n";
}

// Shared stage emitter for both single- and multi-clip writers. Skeleton joints +
// bind/rest come from the first clip; one SkelAnimation prim is emitted per clip.
oa::Status writeStage(const oa::Path& inPath, oa::Span<const oa::UsdNamedClip> inClips,
                    oa::StringView inDefaultPrim, bool inEmitCustomData) {
	if (inClips.empty()) {
		return oa::Status::invalidArgument("oa::Usd::WriteStage: no clips");
	}
	for (oa::Usize i = 0; i < inClips.size(); ++i) {
		if (!inClips[i].clip.isValid()) {
			return oa::Status::invalidArgument("oa::Usd::WriteStage: invalid clip");
		}
	}

	const oa::UsdSkelClip& c0 = inClips[0].clip;
	const oa::I32 n = c0.jointCount();
	const oa::String prim(inDefaultPrim);
	const char* skelPrim = "root";

	// Resolve unique, valid prim names up-front (the Skeleton's animationSource rel
	// must reference the first one).
	oa::Vec<oa::String> animNames;
	animNames.reserve(inClips.size());
	{
		std::set<std::string> used;
		for (oa::Usize i = 0; i < inClips.size(); ++i) {
			oa::String base = inClips[i].name.empty() ? oa::String("Animation")
			                                        : sanitizePrimName(inClips[i].name);
			std::string cand = oa::sdk::toStdString(base);
			oa::I32 suffix = 1;
			while (used.count(cand) != 0) {
				cand = oa::sdk::toStdString(base) + "_" + std::to_string(suffix++);
			}
			used.insert(cand);
			animNames.pushBack(oa::sdk::fromStdString(cand));
		}
	}

	oa::U32 maxFrames = 0;
	for (oa::Usize i = 0; i < inClips.size(); ++i) {
		if (inClips[i].clip.frameCount > maxFrames) { maxFrames = inClips[i].clip.frameCount; }
	}

	std::ostringstream o;
	o.precision(9);

	// stage metadata.
	o << "#usda 1.0\n(\n";
	o << "    defaultPrim = \"" << prim << "\"\n";
	o << "    endTimeCode = " << (maxFrames > 0 ? maxFrames - 1 : 0) << "\n";
	o << "    framesPerSecond = " << c0.fps << "\n";
	o << "    metersPerUnit = 0.01\n";
	o << "    startTimeCode = 0\n";
	o << "    timeCodesPerSecond = " << c0.fps << "\n";
	o << "    upAxis = \"" << (c0.upAxis == 1 ? 'Y' : 'Z') << "\"\n";
	o << "    doc = \"OA Gen3dAnim squashed-USD clip\"\n";
	o << ")\n\n";

	const oa::String jointsTok = jointsToken(c0);

	o << "def SkelRoot \"" << prim << "\" (\n";
	o << "    kind = \"assembly\"\n";
	o << ")\n{\n";

	// Skeleton nested under rig/root, matching Maya/UE squashed-USD exports.
	o << "    def Skeleton \"" << skelPrim << "\" (\n";
	o << "        prepend apiSchemas = [\"SkelBindingAPI\"]\n";
	o << "        customData = {\n";
	o << "            dictionary Maya = {\n";
	o << "                bool generated = 1\n";
	o << "            }\n";
	o << "        }\n";
	o << "    )\n    {\n";
	o << "        uniform token[] joints = [" << jointsTok << "]\n";
	if (c0.bindTransforms.size() == static_cast<oa::Usize>(n)) {
		o << "        uniform matrix4d[] bindTransforms = [";
		for (oa::I32 j = 0; j < n; ++j) {
			o << (j ? ", " : "") << mat4Usd(c0.bindTransforms[static_cast<oa::Usize>(j)]);
		}
		o << "]\n";
	}
	if (c0.restTransforms.size() == static_cast<oa::Usize>(n)) {
		o << "        uniform matrix4d[] restTransforms = [";
		for (oa::I32 j = 0; j < n; ++j) {
			o << (j ? ", " : "") << mat4Usd(c0.restTransforms[static_cast<oa::Usize>(j)]);
		}
		o << "]\n";
	}
	o << "        rel skel:animationSource = </" << prim << "/" << skelPrim << "/"
	  << animNames[0] << ">\n";

	// One SkelAnimation prim per clip, nested inside the skeleton (UE layout).
	for (oa::Usize i = 0; i < inClips.size(); ++i) {
		const oa::UsdNamedClip& nc = inClips[i];
		const oa::String* split  = inEmitCustomData ? &nc.split : nullptr;
		const oa::String* source = inEmitCustomData ? &nc.name  : nullptr;
		emitSkelAnimation(o, "        ", animNames[i], nc.clip, jointsToken(nc.clip), split, source);
	}

	o << "    }\n";  // close Skeleton
	o << "}\n";      // close SkelRoot

	return oa::Filesystem::writeText(inPath, oa::sdk::fromStdString(o.str()));
}

} // namespace

oa::Status oa::Usd::writeUsda(const oa::Path& inPath, const oa::UsdSkelClip& inClip, oa::StringView inDefaultPrim) {
	if (!inClip.isValid()) {
		return oa::Status::invalidArgument("oa::Usd::WriteUsda: invalid clip");
	}
	oa::UsdNamedClip nc;
	nc.name = oa::String("Animation");
	nc.clip = inClip;
	const oa::UsdNamedClip clips[1] = { nc };
	// Single-clip: no customData (keeps byte-for-byte compat with prior output).
	return writeStage(inPath, oa::Span<const oa::UsdNamedClip>(clips, 1), inDefaultPrim, /*customData=*/false);
}

oa::Status oa::Usd::writeUsdaMulti(const oa::Path& inPath, oa::Span<const oa::UsdNamedClip> inClips,
                               oa::StringView inDefaultPrim) {
	return writeStage(inPath, inClips, inDefaultPrim, /*customData=*/true);
}

// ── Reader ───────────────────────────────────────────────────────────────────

namespace {

// find the substring inside the first open/close pair at or after `from`,
// honouring quoted strings (so brackets inside quotes are ignored). Returns the
// [innerBegin, innerEnd) bounds into `t`; false if no balanced pair is found.
bool balanced(const std::string& t, oa::Usize from, char open, char close,
              oa::Usize& outBegin, oa::Usize& outEnd) {
	const oa::Usize npos = std::string::npos;
	oa::Usize openPos = t.find(open, from);
	if (openPos == npos) {
		return false;
	}
	int depth = 0;
	bool inStr = false;
	for (oa::Usize i = openPos; i < t.size(); ++i) {
		const char c = t[i];
		if (c == '"') {
			inStr = !inStr;
			continue;
		}
		if (inStr) {
			continue;
		}
		if (c == open) {
			++depth;
		} else if (c == close) {
			if (--depth == 0) {
				outBegin = openPos + 1;
				outEnd = i;
				return true;
			}
		}
	}
	return false;
}

// All numeric literals in [p, end), in order.
oa::Vec<oa::F32> parseFloats(const char* p, const char* end) {
	oa::Vec<oa::F32> out;
	while (p < end) {
		const char c = *p;
		const bool numStart = (c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.';
		if (!numStart) {
			++p;
			continue;
		}
		char* fin = nullptr;
		const double v = std::strtod(p, &fin);
		if (fin == p) {
			++p;
			continue;
		}
		out.pushBack(static_cast<oa::F32>(v));
		p = fin;
	}
	return out;
}

// Locate the attribute then the next bracket array and parse all its floats.
bool floatArrayAfter(const std::string& t, const char* attr, oa::Vec<oa::F32>& out) {
	const oa::Usize npos = std::string::npos;
	const oa::Usize key = t.find(attr);
	if (key == npos) {
		return false;
	}
	oa::Usize b = 0, e = 0;
	if (!balanced(t, key, '[', ']', b, e)) {
		return false;
	}
	out = parseFloats(t.data() + b, t.data() + e);
	return true;
}

// first quoted string after `key`, e.g. customData `string split = "train"`.
bool stringValueAfter(const std::string& t, const char* key, oa::String& out) {
	const oa::Usize npos = std::string::npos;
	const oa::Usize k = t.find(key);
	if (k == npos) { return false; }
	const oa::Usize q1 = t.find('"', k);
	if (q1 == npos) { return false; }
	const oa::Usize q2 = t.find('"', q1 + 1);
	if (q2 == npos) { return false; }
	out = oa::sdk::fromStdString(t.substr(q1 + 1, q2 - q1 - 1));
	return true;
}

// All `{...}` numeric literals after `attr` (a timeSamples block).
bool rawTimeSamples(const std::string& t, const char* attr, oa::Vec<oa::F32>& out) {
	const oa::Usize npos = std::string::npos;
	const oa::Usize key = t.find(attr);
	if (key == npos) { return false; }
	oa::Usize b = 0, e = 0;
	if (!balanced(t, key, '{', '}', b, e)) { return false; }
	out = parseFloats(t.data() + b, t.data() + e);
	return true;
}

// Each frame block is: frameKey then n component-tuples (translations: 1 + 3n
// floats per frame; rotations: 1 + 4n). Returns each frame's integer key and the
// key-stripped values, so rotation/translation tracks can be aligned by frame
// index — UE exports occasionally drop or add a sample on one track, leaving the
// two unequal/sparse (see the keyed alignment in ParseAnimClipFromText).
bool deinterleaveKeyed(const oa::Vec<oa::F32>& raw, int comps, oa::I32 n,
                       oa::Vec<oa::I32>& keys, oa::Vec<oa::F32>& out) {
	const oa::Usize perFrame = 1 + static_cast<oa::Usize>(comps) * static_cast<oa::Usize>(n);
	if (raw.size() == 0 || raw.size() % perFrame != 0) {
		return false;
	}
	const oa::U32 frames = static_cast<oa::U32>(raw.size() / perFrame);
	keys.reserve(frames);
	out.reserve(static_cast<oa::Usize>(frames) * comps * n);
	for (oa::U32 f = 0; f < frames; ++f) {
		const oa::Usize key = static_cast<oa::Usize>(f) * perFrame;
		keys.pushBack(static_cast<oa::I32>(raw[key]));
		for (oa::I32 c = 0; c < comps * n; ++c) {
			out.pushBack(raw[key + 1 + static_cast<oa::Usize>(c)]);
		}
	}
	return true;
}

// flat row-major 16-per-joint floats → per-joint matrices.
void matricesFrom(const oa::Vec<oa::F32>& flat, oa::I32 n, oa::Vec<oa::vlm::Mat4>& out) {
	if (flat.size() != static_cast<oa::Usize>(n) * 16) {
		return;
	}
	out.resize(static_cast<oa::Usize>(n));
	for (oa::I32 j = 0; j < n; ++j) {
		oa::vlm::Mat4 m{};
		for (int r = 0; r < 4; ++r) {
			for (int c = 0; c < 4; ++c) {
				m.m[r][c] = flat[static_cast<oa::Usize>(j) * 16 + r * 4 + c];
			}
		}
		out[static_cast<oa::Usize>(j)] = m;
	}
}

// parse the joints + per-frame translations/rotations from a SkelAnimation text
// region `t` (the whole stage for a single-anim file, or one prim body for a
// multi-anim stage). Does NOT fill bind/rest — the caller applies the shared
// Skeleton transforms.
oa::Status parseAnimClipFromText(const std::string& t, oa::F32 inFps, oa::I32 inUpAxis, oa::UsdSkelClip& clip) {
	const oa::Usize npos = std::string::npos;
	clip.fps = inFps;
	clip.upAxis = inUpAxis;

	// joints: first "joints =" array of quoted paths. Anchored on the '=' so the
	// data '[' is matched, not the '[' inside the "token[]" type tag.
	{
		const oa::Usize key = t.find("joints =");
		if (key == npos) {
			return oa::Status::invalidArgument("oa::Usd: no joints array");
		}
		oa::Usize b = 0, e = 0;
		if (!balanced(t, key, '[', ']', b, e)) {
			return oa::Status::invalidArgument("oa::Usd: malformed joints array");
		}
		oa::Usize i = b;
		while (i < e) {
			if (t[i] == '"') {
				const oa::Usize q = t.find('"', i + 1);
				if (q == npos || q > e) {
					break;
				}
				clip.jointPaths.pushBack(
					oa::sdk::fromStdString(t.substr(i + 1, q - i - 1)));
				i = q + 1;
			} else {
				++i;
			}
		}
	}
	const oa::I32 n = clip.jointCount();
	if (n == 0) {
		return oa::Status::invalidArgument("oa::Usd: empty joints array");
	}

	// rotation + translation tracks, each either time-sampled (animation) or a
	// single static array (a UE aim-offset / pose asset with no timeSamples → one
	// frame). UE exports occasionally leave the two tracks UNEQUAL or SPARSE (a
	// sample dropped/added on one), so we don't assume equal dense frames: each
	// track keeps its own frame KEYS and the clip timeline is the union of both.
	// A missing sample holds the previous value (step) — exact when a track is
	// merely shorter, harmless for a one-off dropped key.
	oa::Vec<oa::I32> rotKeys, transKeys;
	oa::Vec<oa::F32> rotVals, transVals;   // rotVals: frames*(4n); transVals: frames*(3n)
	bool rotStatic = false, transStatic = false;

	if (oa::Vec<oa::F32> raw; rawTimeSamples(t, "rotations.timeSamples", raw)) {
		if (!deinterleaveKeyed(raw, 4, n, rotKeys, rotVals)) {
			return oa::Status::invalidArgument("oa::Usd: rotations count mismatch");
		}
	} else if (floatArrayAfter(t, "rotations =", rotVals)
	           && rotVals.size() == static_cast<oa::Usize>(n) * 4) {
		rotKeys.pushBack(0); rotStatic = true;   // single static pose
	} else {
		return oa::Status::invalidArgument("oa::Usd: no rotations (timeSamples or static)");
	}

	if (oa::Vec<oa::F32> raw; rawTimeSamples(t, "translations.timeSamples", raw)) {
		if (!deinterleaveKeyed(raw, 3, n, transKeys, transVals)) {
			return oa::Status::invalidArgument("oa::Usd: translations count mismatch");
		}
	} else if (floatArrayAfter(t, "translations =", transVals)
	           && transVals.size() == static_cast<oa::Usize>(n) * 3) {
		transKeys.pushBack(0); transStatic = true;   // single static pose → broadcast
	} else {
		return oa::Status::invalidArgument("oa::Usd: no translations (timeSamples or static)");
	}

	// clip timeline = union of the animated tracks' keys (a static track broadcasts
	// and contributes no keys of its own). Two static tracks → a single-frame pose.
	oa::Vec<oa::I32> timeline;
	{
		std::set<oa::I32> keySet;
		if (!rotStatic)   { for (oa::I32 k : rotKeys)   { keySet.insert(k); } }
		if (!transStatic) { for (oa::I32 k : transKeys) { keySet.insert(k); } }
		if (keySet.empty()) { keySet.insert(0); }
		for (oa::I32 k : keySet) { timeline.pushBack(k); }   // std::set iterates ascending
	}

	// index of the last key <= q (hold-previous); keys are ascending, clamp to 0.
	auto holdIdx = [](const oa::Vec<oa::I32>& keys, oa::I32 q) -> oa::Usize {
		oa::Usize idx = 0;
		for (oa::Usize i = 0; i < keys.size(); ++i) {
			if (keys[i] <= q) { idx = i; } else { break; }
		}
		return idx;
	};

	const oa::U32 frames = static_cast<oa::U32>(timeline.size());
	clip.frameCount = frames;
	clip.rotations.resize(static_cast<oa::Usize>(frames) * n);
	clip.translations.resize(static_cast<oa::Usize>(frames) * n);
	for (oa::U32 f = 0; f < frames; ++f) {
		const oa::I32 q = timeline[f];
		const oa::Usize rB = (rotStatic   ? 0 : holdIdx(rotKeys, q))   * static_cast<oa::Usize>(n) * 4;
		const oa::Usize tB = (transStatic ? 0 : holdIdx(transKeys, q)) * static_cast<oa::Usize>(n) * 3;
		for (oa::I32 j = 0; j < n; ++j) {
			const oa::Usize o  = static_cast<oa::Usize>(f) * n + j;
			const oa::Usize r  = rB + static_cast<oa::Usize>(j) * 4;
			const oa::Usize tr = tB + static_cast<oa::Usize>(j) * 3;
			// USD quatf order (w, x, y, z) -> oa::vlm::Quat (x, y, z, w).
			clip.rotations[o]    = { rotVals[r + 1], rotVals[r + 2], rotVals[r + 3], rotVals[r + 0] };
			clip.translations[o] = { transVals[tr + 0], transVals[tr + 1], transVals[tr + 2] };
		}
	}
	return oa::Status::ok();
}

// stage-level fps + upAxis (defaults applied).
void parseStageMeta(const std::string& text, oa::F32& outFps, oa::I32& outUpAxis) {
	const oa::Usize npos = std::string::npos;
	outFps = 30.0f;
	outUpAxis = 2;
	if (oa::Usize k = text.find("timeCodesPerSecond"); k != npos) {
		if (oa::Usize eq = text.find('=', k); eq != npos) {
			outFps = static_cast<oa::F32>(std::strtod(text.data() + eq + 1, nullptr));
		}
	}
	if (outFps <= 0.0f) { outFps = 30.0f; }
	if (oa::Usize k = text.find("upAxis"); k != npos) {
		if (oa::Usize q = text.find('"', k); q != npos && q < k + 40) {
			outUpAxis = (text[q + 1] == 'Y' || text[q + 1] == 'y') ? 1 : 2;
		}
	}
}

} // namespace

oa::Result<oa::UsdSkelClip> oa::Usd::readUsda(const oa::Path& inPath) {
	auto textResult = oa::Filesystem::readText(inPath);
	if (!textResult.isOk()) {
		return textResult.getStatus();
	}
	const std::string text = oa::sdk::toStdString(*textResult);

	oa::F32 fps = 30.0f;
	oa::I32 upAxis = 2;
	parseStageMeta(text, fps, upAxis);

	oa::UsdSkelClip clip;
	if (auto st = parseAnimClipFromText(text, fps, upAxis, clip); st.isError()) {
		return st;
	}
	const oa::I32 n = clip.jointCount();

	// Optional bind/rest transforms (Skeleton-level): 16 floats per joint.
	if (oa::Vec<oa::F32> f; floatArrayAfter(text, "bindTransforms", f)) {
		matricesFrom(f, n, clip.bindTransforms);
	}
	if (oa::Vec<oa::F32> f; floatArrayAfter(text, "restTransforms", f)) {
		matricesFrom(f, n, clip.restTransforms);
	}

	if (!clip.isValid()) {
		return oa::Status::invalidArgument("oa::Usd::ReadUsda: assembled clip invalid");
	}
	return clip;
}

oa::Result<oa::Vec<oa::UsdNamedClip>> oa::Usd::readUsdaMulti(const oa::Path& inPath) {
	auto textResult = oa::Filesystem::readText(inPath);
	if (!textResult.isOk()) {
		return textResult.getStatus();
	}
	const std::string text = oa::sdk::toStdString(*textResult);
	const oa::Usize npos = std::string::npos;

	oa::F32 fps = 30.0f;
	oa::I32 upAxis = 2;
	parseStageMeta(text, fps, upAxis);

	// Shared Skeleton bind/rest (first occurrence — applied to every clip).
	oa::Vec<oa::F32> bindFlat, restFlat;
	const bool hasBind = floatArrayAfter(text, "bindTransforms", bindFlat);
	const bool hasRest = floatArrayAfter(text, "restTransforms", restFlat);

	oa::Vec<oa::UsdNamedClip> out;

	oa::Usize pos = 0;
	while ((pos = text.find("def SkelAnimation", pos)) != npos) {
		// Prim name (first quoted token after the keyword).
		const oa::Usize q1 = text.find('"', pos);
		const oa::Usize q2 = (q1 == npos) ? npos : text.find('"', q1 + 1);
		oa::String primName = (q1 != npos && q2 != npos)
			? oa::sdk::fromStdString(text.substr(q1 + 1, q2 - q1 - 1))
			: oa::String("Animation");

		// step past an optional metadata paren block `( ... )` (which itself contains
		// `{ }` for customData) so the prim BODY brace is matched, not customData.
		oa::Usize headerEnd = (q2 == npos) ? pos : q2 + 1;
		while (headerEnd < text.size() && std::isspace(static_cast<unsigned char>(text[headerEnd]))) {
			++headerEnd;
		}
		oa::String split  = oa::String("train");
		oa::String source = primName;
		if (headerEnd < text.size() && text[headerEnd] == '(') {
			oa::Usize pb = 0, pe = 0;
			if (balanced(text, headerEnd, '(', ')', pb, pe)) {
				const std::string header = text.substr(pb, pe - pb);
				stringValueAfter(header, "split", split);
				stringValueAfter(header, "source", source);
				headerEnd = pe + 1;
			}
		}

		// Prim body.
		oa::Usize bb = 0, be = 0;
		if (!balanced(text, headerEnd, '{', '}', bb, be)) {
			pos = (q2 == npos) ? pos + 17 : q2 + 1;
			continue;
		}
		const std::string block = text.substr(bb, be - bb);

		oa::UsdNamedClip nc;
		nc.name = source;
		nc.split = split;
		if (parseAnimClipFromText(block, fps, upAxis, nc.clip).isOk()) {
			const oa::I32 n = nc.clip.jointCount();
			if (hasBind) { matricesFrom(bindFlat, n, nc.clip.bindTransforms); }
			if (hasRest) { matricesFrom(restFlat, n, nc.clip.restTransforms); }
			if (nc.clip.isValid()) {
				out.pushBack(std::move(nc));
			}
		}
		pos = be;
	}

	if (out.empty()) {
		return oa::Status::invalidArgument("oa::Usd::ReadUsdaMulti: no valid SkelAnimation prims");
	}
	return out;
}
