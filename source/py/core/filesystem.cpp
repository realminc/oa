// OA Python bindings — oa::Path, oa::Filesystem, and named OA locations.
#include "../binding.h"

#include <oa/core/filesystem.h>
#include <oa/core/hostText.h>
#include <oa/core/paths.h>

namespace {

template <typename T>
T unwrap(oa::Result<T>&& result) {
    throwIfError(result.getStatus());
    return std::move(result).getValue();
}

std::vector<oa::Path> pathsFrom(oa::Vector<oa::Path>&& paths) {
    std::vector<oa::Path> result;
    result.reserve(paths.size());
    for (auto& path : paths) {
        result.push_back(std::move(path));
    }
    return result;
}

std::vector<std::string> stringsFrom(oa::Vector<oa::String>&& strings) {
    std::vector<std::string> result;
    result.reserve(strings.size());
    for (const auto& value : strings) {
		result.push_back(oa::hostText::copy(value));
    }
    return result;
}

} // namespace

void bindCoreFilesystem(nb::module_& m) {
    nb::class_<oa::Path>(m, "Path")
        .def(nb::init<>())
        .def("__init__", [](oa::Path* self, nb::handle path) {
            new (self) oa::Path(pathFromPython(path));
        }, nb::arg("path"))
		.def("__str__", [](const oa::Path& self) { return oa::hostText::copy(self.string()); })
		.def("__repr__", [](const oa::Path& self) {
			const std::string value = oa::hostText::copy(self.string());
            return std::string("oa.Path(") + nb::cast<std::string>(
                nb::repr(nb::str(value.c_str()))) + ")";
        })
		.def("__fspath__", [](const oa::Path& self) { return oa::hostText::copy(self.string()); })
        .def("__truediv__", [](const oa::Path& self, nb::handle child) {
            return self / pathFromPython(child);
        }, nb::arg("child"))
        .def("__eq__", [](const oa::Path& self, const oa::Path& other) {
            return self == other;
        }, nb::arg("other"))
		.def("string", [](const oa::Path& self) { return oa::hostText::copy(self.string()); })
		.def("genericString", [](const oa::Path& self) {
			return oa::hostText::copy(self.genericString());
        })
        .def("parentPath", &oa::Path::parentPath)
        .def("filename", &oa::Path::filename)
        .def("stem", &oa::Path::stem)
        .def("extension", &oa::Path::extension)
        .def("isAbsolute", &oa::Path::isAbsolute)
        .def("isRelative", &oa::Path::isRelative)
        .def("lexicallyNormal", &oa::Path::lexicallyNormal)
        .def("empty", &oa::Path::empty);

    nb::class_<oa::Paths>(m, "Paths")
        .def_static("asset", [] { return oa::Paths::asset(); })
        .def_static("asset", [](nb::handle relative) {
            const auto path = pathFromPython(relative);
            return oa::Paths::asset(path.genericString());
        }, nb::arg("relative"))
        .def_static("var", [] { return oa::Paths::var(); })
        .def_static("var", [](nb::handle relative) {
            const auto path = pathFromPython(relative);
            return oa::Paths::var(path.genericString());
        }, nb::arg("relative"))
		.def_static("data", [] { return oa::Paths::data(); })
		.def_static("data", [](nb::handle relative) {
			const auto path = pathFromPython(relative);
			return oa::Paths::data(path.genericString());
		}, nb::arg("relative"))
        .def_static("current", &oa::Paths::current)
        .def_static("home", &oa::Paths::home)
        .def_static("temp", &oa::Paths::temp);

    nb::class_<oa::Filesystem>(m, "Filesystem")
        .def_static("exists", [](nb::handle path) {
            return oa::Filesystem::exists(pathFromPython(path));
        }, nb::arg("path"))
        .def_static("isFile", [](nb::handle path) {
            return oa::Filesystem::isFile(pathFromPython(path));
        }, nb::arg("path"))
        .def_static("isDirectory", [](nb::handle path) {
            return oa::Filesystem::isDirectory(pathFromPython(path));
        }, nb::arg("path"))
        .def_static("getFileSize", [](nb::handle path) {
            return unwrap(oa::Filesystem::getFileSize(pathFromPython(path)));
        }, nb::arg("path"))
        .def_static("getLastModified", [](nb::handle path) {
            return unwrap(oa::Filesystem::getLastModified(pathFromPython(path)));
        }, nb::arg("path"))
        .def_static("createDirectory", [](nb::handle path) {
            throwIfError(oa::Filesystem::createDirectory(pathFromPython(path)));
        }, nb::arg("path"))
        .def_static("createDirectories", [](nb::handle path) {
            throwIfError(oa::Filesystem::createDirectories(pathFromPython(path)));
        }, nb::arg("path"))
        .def_static("removeFile", [](nb::handle path) {
            throwIfError(oa::Filesystem::removeFile(pathFromPython(path)));
        }, nb::arg("path"))
        .def_static("removeDirectory", [](nb::handle path, bool recursive) {
            throwIfError(oa::Filesystem::removeDirectory(
                pathFromPython(path), recursive));
        }, nb::arg("path"), nb::arg("recursive") = false)
        .def_static("copy", [](nb::handle from, nb::handle to) {
            throwIfError(oa::Filesystem::copy(
                pathFromPython(from), pathFromPython(to)));
        }, nb::arg("fromPath"), nb::arg("toPath"))
        .def_static("move", [](nb::handle from, nb::handle to) {
            throwIfError(oa::Filesystem::move(
                pathFromPython(from), pathFromPython(to)));
        }, nb::arg("fromPath"), nb::arg("toPath"))
        .def_static("listFiles", [](nb::handle directory,
                                    const std::string& extension) {
            return pathsFrom(unwrap(oa::Filesystem::listFiles(
                pathFromPython(directory),
                oa::StringView(extension.data(), extension.size()))));
        }, nb::arg("directory"), nb::arg("extension") = "")
        .def_static("listDirectories", [](nb::handle directory) {
            return pathsFrom(unwrap(oa::Filesystem::listDirectories(
                pathFromPython(directory))));
        }, nb::arg("directory"))
        .def_static("listAll", [](nb::handle directory, bool recursive) {
            return pathsFrom(unwrap(oa::Filesystem::listAll(
                pathFromPython(directory), recursive)));
        }, nb::arg("directory"), nb::arg("recursive") = false)
        .def_static("readText", [](nb::handle path) {
			return oa::hostText::copy(unwrap(oa::Filesystem::readText(pathFromPython(path))));
        }, nb::arg("path"))
        .def_static("writeText", [](nb::handle path, const std::string& content) {
            throwIfError(oa::Filesystem::writeText(
                pathFromPython(path),
                oa::StringView(content.data(), content.size())));
        }, nb::arg("path"), nb::arg("content"))
        .def_static("appendText", [](nb::handle path, const std::string& content) {
            throwIfError(oa::Filesystem::appendText(
                pathFromPython(path),
                oa::StringView(content.data(), content.size())));
        }, nb::arg("path"), nb::arg("content"))
        .def_static("readLines", [](nb::handle path) {
            return stringsFrom(unwrap(oa::Filesystem::readLines(
                pathFromPython(path))));
        }, nb::arg("path"))
        .def_static("readBinary", [](nb::handle path) {
            auto data = unwrap(oa::Filesystem::readBinary(pathFromPython(path)));
            return nb::bytes(
                reinterpret_cast<const char*>(data.data()), data.size());
        }, nb::arg("path"))
        .def_static("writeBinary", [](nb::handle path, nb::bytes data) {
            throwIfError(oa::Filesystem::writeBinary(
                pathFromPython(path),
                oa::Span<const oa::U8>(
                    reinterpret_cast<const oa::U8*>(data.data()), data.size())));
        }, nb::arg("path"), nb::arg("data"))
        .def_static("absolute", [](nb::handle path) {
            return unwrap(oa::Filesystem::absolute(pathFromPython(path)));
        }, nb::arg("path"))
        .def_static("glob", [](nb::handle directory, const std::string& pattern) {
            return pathsFrom(unwrap(oa::Filesystem::glob(
                pathFromPython(directory),
                oa::StringView(pattern.data(), pattern.size()))));
        }, nb::arg("directory"), nb::arg("pattern"));
}
