// Shared RGB -> YUV conversion math for buffer- and image-backed encoder input.

export static const uint COLOR_SPACE_BT601  = 0;
export static const uint COLOR_SPACE_BT709  = 1;
export static const uint COLOR_SPACE_BT2020 = 2;

export float3 RgbToYuv_BT601_Limited(float3 rgb) {
    float Y  =  0.257f * rgb.r + 0.504f * rgb.g + 0.098f * rgb.b + 16.0f / 255.0f;
    float Cb = -0.148f * rgb.r - 0.291f * rgb.g + 0.439f * rgb.b + 128.0f / 255.0f;
    float Cr =  0.439f * rgb.r - 0.368f * rgb.g - 0.071f * rgb.b + 128.0f / 255.0f;
    return saturate(float3(Y, Cb, Cr));
}

export float3 RgbToYuv_BT601_Full(float3 rgb) {
    float Y  =  0.299f * rgb.r + 0.587f * rgb.g + 0.114f * rgb.b;
    float Cb = -0.169f * rgb.r - 0.331f * rgb.g + 0.500f * rgb.b + 0.5f;
    float Cr =  0.500f * rgb.r - 0.419f * rgb.g - 0.081f * rgb.b + 0.5f;
    return saturate(float3(Y, Cb, Cr));
}

export float3 RgbToYuv_BT709_Limited(float3 rgb) {
    float Y  =  0.183f * rgb.r + 0.614f * rgb.g + 0.062f * rgb.b + 16.0f / 255.0f;
    float Cb = -0.101f * rgb.r - 0.339f * rgb.g + 0.439f * rgb.b + 128.0f / 255.0f;
    float Cr =  0.439f * rgb.r - 0.399f * rgb.g - 0.040f * rgb.b + 128.0f / 255.0f;
    return saturate(float3(Y, Cb, Cr));
}

export float3 RgbToYuv_BT709_Full(float3 rgb) {
    float Y  =  0.2126f * rgb.r + 0.7152f * rgb.g + 0.0722f * rgb.b;
    float Cb = -0.1146f * rgb.r - 0.3854f * rgb.g + 0.5000f * rgb.b + 0.5f;
    float Cr =  0.5000f * rgb.r - 0.4542f * rgb.g - 0.0458f * rgb.b + 0.5f;
    return saturate(float3(Y, Cb, Cr));
}

export float3 RgbToYuv_BT2020_Limited(float3 rgb) {
    float Y  =  0.225f * rgb.r + 0.580f * rgb.g + 0.051f * rgb.b + 16.0f / 255.0f;
    float Cb = -0.122f * rgb.r - 0.316f * rgb.g + 0.439f * rgb.b + 128.0f / 255.0f;
    float Cr =  0.439f * rgb.r - 0.403f * rgb.g - 0.036f * rgb.b + 128.0f / 255.0f;
    return saturate(float3(Y, Cb, Cr));
}

export float3 RgbToYuv_BT2020_Full(float3 rgb) {
    float Y  =  0.2627f * rgb.r + 0.6780f * rgb.g + 0.0593f * rgb.b;
    float Cb = -0.1396f * rgb.r - 0.3604f * rgb.g + 0.5000f * rgb.b + 0.5f;
    float Cr =  0.5000f * rgb.r - 0.4598f * rgb.g - 0.0402f * rgb.b + 0.5f;
    return saturate(float3(Y, Cb, Cr));
}

export float3 RgbToYuv(float3 rgb, uint color_space, uint full_range) {
    if (color_space == COLOR_SPACE_BT709) {
        return full_range != 0 ? RgbToYuv_BT709_Full(rgb) : RgbToYuv_BT709_Limited(rgb);
    }
    if (color_space == COLOR_SPACE_BT2020) {
        return full_range != 0 ? RgbToYuv_BT2020_Full(rgb) : RgbToYuv_BT2020_Limited(rgb);
    }
    return full_range != 0 ? RgbToYuv_BT601_Full(rgb) : RgbToYuv_BT601_Limited(rgb);
}
