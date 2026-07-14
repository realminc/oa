// Shared image-coordinate and NCHW storage helpers.
// Border values intentionally match OaBorderMode in VisionTypes.toml.

import Storage;

export static const uint OA_BORDER_CONSTANT   = 0;
export static const uint OA_BORDER_REPLICATE  = 1;
export static const uint OA_BORDER_REFLECT    = 2;
export static const uint OA_BORDER_REFLECT101 = 3;
export static const uint OA_BORDER_WRAP       = 4;

int OaPositiveModulo(int value, int modulus) {
    int result = value % modulus;
    return result < 0 ? result + modulus : result;
}

export bool OaMapBorderCoordinate(int coordinate, uint size, uint border_mode,
                                  out uint mapped) {
    if (coordinate >= 0 && coordinate < int(size)) {
        mapped = uint(coordinate);
        return true;
    }
    if (size == 0 || border_mode == OA_BORDER_CONSTANT) {
        mapped = 0;
        return false;
    }
    if (size == 1) {
        mapped = 0;
        return true;
    }
    if (border_mode == OA_BORDER_REPLICATE) {
        mapped = uint(clamp(coordinate, 0, int(size) - 1));
        return true;
    }
    if (border_mode == OA_BORDER_WRAP) {
        mapped = uint(OaPositiveModulo(coordinate, int(size)));
        return true;
    }
    if (border_mode == OA_BORDER_REFLECT) {
        int period = int(size) * 2;
        int reflected = OaPositiveModulo(coordinate, period);
        if (reflected >= int(size)) reflected = period - reflected - 1;
        mapped = uint(reflected);
        return true;
    }
    if (border_mode == OA_BORDER_REFLECT101) {
        int period = (int(size) - 1) * 2;
        int reflected = OaPositiveModulo(coordinate, period);
        if (reflected >= int(size)) reflected = period - reflected;
        mapped = uint(reflected);
        return true;
    }

    mapped = 0;
    return false;
}

export float OaLoadNchwBorder(RWByteAddressBuffer input, uint b, uint c,
                              int y, int x, uint channels, uint height, uint width,
                              uint border_mode, float border_value) {
    uint mapped_y;
    uint mapped_x;
    if (!OaMapBorderCoordinate(y, height, border_mode, mapped_y) ||
        !OaMapBorderCoordinate(x, width, border_mode, mapped_x)) {
        return border_value;
    }
    uint index = b * (channels * height * width) +
                 c * (height * width) + mapped_y * width + mapped_x;
    return OaLoad(input, index);
}

export void OaStoreNchw(RWByteAddressBuffer output, uint b, uint c,
                        uint y, uint x, uint channels, uint height, uint width,
                        float value) {
    uint index = b * (channels * height * width) +
                 c * (height * width) + y * width + x;
    OaStore(output, index, value);
}
