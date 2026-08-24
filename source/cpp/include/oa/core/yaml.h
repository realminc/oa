#pragma once

#include <oa/core/types.h>

// Stub implementation when yaml-cpp is not available
// To enable: install yaml-cpp via vcpkg and define OA_HAS_YAML_CPP
#ifdef OA_HAS_YAML_CPP
#include <yaml-cpp/yaml.h>

namespace YAML {

template<>
struct convert<oa::String> {
	static Node encode(const oa::String& inRhs) { return Node(inRhs.stdStr()); }

	static bool decode(const Node& inNode, oa::String& outRhs) {
		if (!inNode.IsScalar()) {
			return false;
		}
		outRhs = oa::String(inNode.as<std::string>());
		return true;
	}
};

} // namespace YAML

namespace oa {

class Yaml {
public:
	using Node = YAML::Node;
	using Exception = YAML::Exception;

	[[nodiscard]] static Node loadFile(const oa::String& inPath) {
		return YAML::LoadFile(inPath.stdStr());
	}

	template<typename T>
	[[nodiscard]] static T get(const Node& inNode, const oa::String& inKey, const T& inDefault) {
		if (inNode && inNode[inKey.stdStr()]) {
			try {
				return inNode[inKey.stdStr()].as<T>();
			} catch (...) {
				return inDefault;
			}
		}
		return inDefault;
	}

	[[nodiscard]] static oa::Vec<oa::String> getList(const Node& inNode, const oa::String& inKey) {
		oa::Vec<oa::String> result;
		if (inNode && inNode[inKey.stdStr()] && inNode[inKey.stdStr()].IsSequence()) {
			for (const auto& item : inNode[inKey.stdStr()]) {
				result.pushBack(item.as<oa::String>());
			}
		}
		return result;
	}
};

} // namespace oa

#else
// Stub implementation when yaml-cpp is not available
namespace oa {

class Yaml {
public:
	struct Node {
		operator bool() const { return false; }
		Node operator[](const char*) const { return Node(); }
		Node operator[](const std::string&) const { return Node(); }
		template<typename T> T as() const { return T(); }
		bool isSequence() const { return false; }
		Node begin() const { return Node(); }
		Node end() const { return Node(); }
	};
	
	struct Exception : public std::exception {
		const char* what() const noexcept override { return "yaml-cpp not available"; }
	};

	[[nodiscard]] static Node loadFile(const oa::String&) { return Node(); }

	template<typename T>
	[[nodiscard]] static T get(const Node&, const oa::String&, const T& inDefault) {
		return inDefault;
	}

	[[nodiscard]] static oa::Vec<oa::String> getList(const Node&, const oa::String&) {
		return oa::Vec<oa::String>();
	}
};

} // namespace oa
#endif
