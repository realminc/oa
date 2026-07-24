// OA Python bindings — OaPath, OaFilesystem, and named OA locations.
#include "../Binding.h"

#include <Oa/Core/Filesystem.h>
#include <Oa/Core/Paths.h>

namespace {

template <typename T>
T unwrap(OaResult<T>&& result) {
    throw_if_error(result.GetStatus());
    return std::move(result).GetValue();
}

std::vector<OaPath> paths_from(OaVec<OaPath>&& paths) {
    std::vector<OaPath> result;
    result.reserve(paths.Size());
    for (auto& path : paths) {
        result.push_back(std::move(path));
    }
    return result;
}

std::vector<std::string> strings_from(OaVec<OaString>&& strings) {
    std::vector<std::string> result;
    result.reserve(strings.Size());
    for (const auto& value : strings) {
        result.push_back(value.StdStr());
    }
    return result;
}

} // namespace

void BindCoreFilesystem(nb::module_& m) {
    nb::class_<OaPath>(m, "OaPath")
        .def(nb::init<>())
        .def("__init__", [](OaPath* self, nb::handle path) {
            new (self) OaPath(path_from_python(path));
        }, nb::arg("Path"))
        .def("__str__", [](const OaPath& self) { return self.String().StdStr(); })
        .def("__repr__", [](const OaPath& self) {
            const std::string value = self.String().StdStr();
            return std::string("OaPath(") + nb::cast<std::string>(
                nb::repr(nb::str(value.c_str()))) + ")";
        })
        .def("__fspath__", [](const OaPath& self) { return self.String().StdStr(); })
        .def("__truediv__", [](const OaPath& self, nb::handle child) {
            return self / path_from_python(child);
        }, nb::arg("Child"))
        .def("__eq__", [](const OaPath& self, const OaPath& other) {
            return self == other;
        }, nb::arg("Other"))
        .def("String", [](const OaPath& self) { return self.String().StdStr(); })
        .def("GenericString", [](const OaPath& self) {
            return self.GenericString().StdStr();
        })
        .def("ParentPath", &OaPath::ParentPath)
        .def("Filename", &OaPath::Filename)
        .def("Stem", &OaPath::Stem)
        .def("Extension", &OaPath::Extension)
        .def("IsAbsolute", &OaPath::IsAbsolute)
        .def("IsRelative", &OaPath::IsRelative)
        .def("LexicallyNormal", &OaPath::LexicallyNormal)
        .def("Empty", &OaPath::Empty);

    nb::class_<OaPaths>(m, "OaPaths")
        .def_static("Asset", [] { return OaPaths::Asset(); })
        .def_static("Asset", [](nb::handle relative) {
            const auto path = path_from_python(relative);
            return OaPaths::Asset(path.GenericString());
        }, nb::arg("Relative"))
        .def_static("Var", [] { return OaPaths::Var(); })
        .def_static("Var", [](nb::handle relative) {
            const auto path = path_from_python(relative);
            return OaPaths::Var(path.GenericString());
        }, nb::arg("Relative"))
        .def_static("Current", &OaPaths::Current)
        .def_static("Home", &OaPaths::Home)
        .def_static("Temp", &OaPaths::Temp);

    nb::class_<OaFilesystem>(m, "OaFilesystem")
        .def_static("Exists", [](nb::handle path) {
            return OaFilesystem::Exists(path_from_python(path));
        }, nb::arg("Path"))
        .def_static("IsFile", [](nb::handle path) {
            return OaFilesystem::IsFile(path_from_python(path));
        }, nb::arg("Path"))
        .def_static("IsDirectory", [](nb::handle path) {
            return OaFilesystem::IsDirectory(path_from_python(path));
        }, nb::arg("Path"))
        .def_static("GetFileSize", [](nb::handle path) {
            return unwrap(OaFilesystem::GetFileSize(path_from_python(path)));
        }, nb::arg("Path"))
        .def_static("GetLastModified", [](nb::handle path) {
            return unwrap(OaFilesystem::GetLastModified(path_from_python(path)));
        }, nb::arg("Path"))
        .def_static("CreateDirectory", [](nb::handle path) {
            throw_if_error(OaFilesystem::CreateDirectory(path_from_python(path)));
        }, nb::arg("Path"))
        .def_static("CreateDirectories", [](nb::handle path) {
            throw_if_error(OaFilesystem::CreateDirectories(path_from_python(path)));
        }, nb::arg("Path"))
        .def_static("RemoveFile", [](nb::handle path) {
            throw_if_error(OaFilesystem::RemoveFile(path_from_python(path)));
        }, nb::arg("Path"))
        .def_static("RemoveDirectory", [](nb::handle path, bool recursive) {
            throw_if_error(OaFilesystem::RemoveDirectory(
                path_from_python(path), recursive));
        }, nb::arg("Path"), nb::arg("Recursive") = false)
        .def_static("Copy", [](nb::handle from, nb::handle to) {
            throw_if_error(OaFilesystem::Copy(
                path_from_python(from), path_from_python(to)));
        }, nb::arg("FromPath"), nb::arg("ToPath"))
        .def_static("Move", [](nb::handle from, nb::handle to) {
            throw_if_error(OaFilesystem::Move(
                path_from_python(from), path_from_python(to)));
        }, nb::arg("FromPath"), nb::arg("ToPath"))
        .def_static("ListFiles", [](nb::handle directory,
                                    const std::string& extension) {
            return paths_from(unwrap(OaFilesystem::ListFiles(
                path_from_python(directory),
                OaStringView(extension.data(), extension.size()))));
        }, nb::arg("Directory"), nb::arg("Extension") = "")
        .def_static("ListDirectories", [](nb::handle directory) {
            return paths_from(unwrap(OaFilesystem::ListDirectories(
                path_from_python(directory))));
        }, nb::arg("Directory"))
        .def_static("ListAll", [](nb::handle directory, bool recursive) {
            return paths_from(unwrap(OaFilesystem::ListAll(
                path_from_python(directory), recursive)));
        }, nb::arg("Directory"), nb::arg("Recursive") = false)
        .def_static("ReadText", [](nb::handle path) {
            return unwrap(OaFilesystem::ReadText(path_from_python(path))).StdStr();
        }, nb::arg("Path"))
        .def_static("WriteText", [](nb::handle path, const std::string& content) {
            throw_if_error(OaFilesystem::WriteText(
                path_from_python(path),
                OaStringView(content.data(), content.size())));
        }, nb::arg("Path"), nb::arg("Content"))
        .def_static("AppendText", [](nb::handle path, const std::string& content) {
            throw_if_error(OaFilesystem::AppendText(
                path_from_python(path),
                OaStringView(content.data(), content.size())));
        }, nb::arg("Path"), nb::arg("Content"))
        .def_static("ReadLines", [](nb::handle path) {
            return strings_from(unwrap(OaFilesystem::ReadLines(
                path_from_python(path))));
        }, nb::arg("Path"))
        .def_static("ReadBinary", [](nb::handle path) {
            auto data = unwrap(OaFilesystem::ReadBinary(path_from_python(path)));
            return nb::bytes(
                reinterpret_cast<const char*>(data.Data()), data.Size());
        }, nb::arg("Path"))
        .def_static("WriteBinary", [](nb::handle path, nb::bytes data) {
            throw_if_error(OaFilesystem::WriteBinary(
                path_from_python(path),
                OaSpan<const OaU8>(
                    reinterpret_cast<const OaU8*>(data.data()), data.size())));
        }, nb::arg("Path"), nb::arg("Data"))
        .def_static("Absolute", [](nb::handle path) {
            return unwrap(OaFilesystem::Absolute(path_from_python(path)));
        }, nb::arg("Path"))
        .def_static("Glob", [](nb::handle directory, const std::string& pattern) {
            return paths_from(unwrap(OaFilesystem::Glob(
                path_from_python(directory),
                OaStringView(pattern.data(), pattern.size()))));
        }, nb::arg("Directory"), nb::arg("Pattern"));
}
