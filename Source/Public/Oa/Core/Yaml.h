#pragma once

#include <Oa/Core/Types.h>

// Stub implementation when yaml-cpp is not available
// To enable: install yaml-cpp via vcpkg and define OA_HAS_YAML_CPP
#ifdef OA_HAS_YAML_CPP
#include <yaml-cpp/yaml.h>

namespace YAML {

template<>
struct convert<OaStdString> {
	static Node encode(const OaStdString& InRhs) { return Node(InRhs.StdStr()); }

	static bool decode(const Node& InNode, OaStdString& OutRhs) {
		if (!InNode.IsScalar()) {
			return false;
		}
		OutRhs = OaStdString(InNode.as<std::string>());
		return true;
	}
};

} // namespace YAML

class OaYaml {
public:
	using Node = YAML::Node;
	using Exception = YAML::Exception;

	[[nodiscard]] static Node LoadFile(const OaString& InPath) {
		return YAML::LoadFile(InPath.StdStr());
	}

	template<typename T>
	[[nodiscard]] static T Get(const Node& InNode, const OaString& InKey, const T& InDefault) {
		if (InNode && InNode[InKey.StdStr()]) {
			try {
				return InNode[InKey.StdStr()].as<T>();
			} catch (...) {
				return InDefault;
			}
		}
		return InDefault;
	}

	[[nodiscard]] static OaVec<OaString> GetList(const Node& InNode, const OaString& InKey) {
		OaVec<OaString> result;
		if (InNode && InNode[InKey.StdStr()] && InNode[InKey.StdStr()].IsSequence()) {
			for (const auto& item : InNode[InKey.StdStr()]) {
				result.PushBack(item.as<OaString>());
			}
		}
		return result;
	}
};

#else
// Stub implementation when yaml-cpp is not available
class OaYaml {
public:
	struct Node {
		operator bool() const { return false; }
		Node operator[](const char*) const { return Node(); }
		Node operator[](const std::string&) const { return Node(); }
		template<typename T> T as() const { return T(); }
		bool IsSequence() const { return false; }
		Node begin() const { return Node(); }
		Node end() const { return Node(); }
	};
	
	struct Exception : public std::exception {
		const char* what() const noexcept override { return "yaml-cpp not available"; }
	};

	[[nodiscard]] static Node LoadFile(const OaString&) { return Node(); }

	template<typename T>
	[[nodiscard]] static T Get(const Node&, const OaString&, const T& InDefault) {
		return InDefault;
	}

	[[nodiscard]] static OaVec<OaString> GetList(const Node&, const OaString&) {
		return OaVec<OaString>();
	}
};
#endif
