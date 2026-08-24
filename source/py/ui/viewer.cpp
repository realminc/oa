// OA Python binding — blocking oa::Viewer convenience session.
#include "../binding.h"

#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
#include <oa/ui/viewer.h>

void bindViewer(nb::module_& m) {
    nb::class_<oa::Viewer>(m, "Viewer")
        .def_static("show", [](const oa::Matrix& image, const std::string& title) {
            oa::ViewerConfig config;
            config.title = title.c_str();
            oa::Status status;
            {
                nb::gil_scoped_release release;
                status = pythonViewerShow(image, config);
            }
            throwIfError(status);
        }, nb::arg("image"), nb::arg("title") = "OA Viewer",
           "Display a matrix in a blocking oa::Viewer session.")
        .def_static("show", [](const oa::Image& image, const std::string& title) {
            oa::ViewerConfig config;
            config.title = title.c_str();
            oa::Status status;
            {
                nb::gil_scoped_release release;
                status = pythonViewerShow(image, config);
            }
            throwIfError(status);
        }, nb::arg("image"), nb::arg("title") = "OA Viewer",
           "Display a semantic image in a blocking oa::Viewer session.");
}
