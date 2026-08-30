#pragma once

#include <oa/core/types.h>

// Stub implementation when yaml-cpp is not available
// To enable: install yaml-cpp via vcpkg and define OA_HAS_YAML_CPP
#ifdef OA_HAS_YAML_CPP
#include <yaml-cpp/yaml.h>

namespace YAML {

template<>
struct convert<oa::String> {
	static Node encode(const oa::String& inRhs) {
		return Node(std::string(inRhs.data(), inRhs.size()));
	}

	static bool decode(const Node& inNode, oa::String& outRhs) {
		if (!inNode.IsScalar()) {
			return false;
		}
		const std::string value = inNode.as<std::string>();
		outRhs = oa::String(value.data(), value.size());
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
		return YAML::LoadFile(std::string(inPath.data(), inPath.size()));
	}

	template<typename T>
	[[nodiscard]] static T get(const Node& inNode, const oa::String& inKey, const T& inDefault) {
		const std::string key(inKey.data(), inKey.size());
		if (inNode && inNode[key]) {
			try {
				return inNode[key].as<T>();
			} catch (...) {
				return inDefault;
			}
		}
		return inDefault;
	}

	[[nodiscard]] static oa::Vector<oa::String> getList(const Node& inNode, const oa::String& inKey) {
		oa::Vector<oa::String> result;
		const std::string key(inKey.data(), inKey.size());
		if (inNode && inNode[key] && inNode[key].IsSequence()) {
			for (const auto& item : inNode[key]) {
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
		template<typename T> Node operator[](const T&) const { return Node(); }
		template<typename T> T as() const { return T(); }
		bool isSequence() const { return false; }
		Node begin() const { return Node(); }
		Node end() const { return Node(); }
	};

	struct Exception {
		const char* what() const noexcept { return "yaml-cpp not available"; }
	};

	[[nodiscard]] static Node loadFile(const oa::String&) { return Node(); }

	template<typename T>
	[[nodiscard]] static T get(const Node&, const oa::String&, const T& inDefault) {
		return inDefault;
	}

	[[nodiscard]] static oa::Vector<oa::String> getList(const Node&, const oa::String&) {
		return oa::Vector<oa::String>();
	}
};

} // namespace oa
#endif
