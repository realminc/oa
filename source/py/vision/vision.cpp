// OA Python bindings — Vision registration in dependency order.
#include "../binding.h"

void bindVision(
	nb::module_& m,
	nb::module_& inFnImage,
	nb::module_& inFnDetection) {
    bindVisionType(m);
	bindVisionImage(m, inFnImage);
	bindVisionDetection(m, inFnDetection);
	bindVisionCodec(m, inFnImage);
    bindVisionVideo(m);
}
