// OA Python binding — blocking OaViewer convenience session.
#include "../Binding.h"

#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Ui/Viewer.h>

void BindViewer(nb::module_& m) {
    nb::class_<OaViewer>(m, "OaViewer")
        .def_static("Show", [](const OaMatrix& image, const std::string& title) {
            OaViewerConfig config;
            config.Title = title.c_str();
            OaStatus status;
            {
                nb::gil_scoped_release release;
                status = PythonViewerShow(image, config);
            }
            throw_if_error(status);
        }, nb::arg("Image"), nb::arg("Title") = "OaViewer",
           "Display a matrix in a blocking OaViewer session.")
        .def_static("Show", [](const OaImage& image, const std::string& title) {
            OaViewerConfig config;
            config.Title = title.c_str();
            OaStatus status;
            {
                nb::gil_scoped_release release;
                status = PythonViewerShow(image, config);
            }
            throw_if_error(status);
        }, nb::arg("Image"), nb::arg("Title") = "OaViewer",
           "Display a semantic image in a blocking OaViewer session.");
}
