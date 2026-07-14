// OaAudio Mel Filterbank Kernel — Convert STFT magnitude to mel spectrogram
//
// Applies triangular mel-scale filters to power spectrum.
// Input:  [Channels, Frames, FreqBins] F32 magnitude spectrogram
// Output: [Channels, Frames, NumMels] F32 mel spectrogram
//
// Dispatch: (ceil(NumMels/32), Frames, Channels)
// Each thread computes one mel bin for one frame.

import Storage;

static const float OA_PI = 3.14159265358979323846f;

// ─── Push Constants ───────────────────────────────────────────────────────────

struct MelParams {
    uint InputIdx;       // STFT magnitude [Channels, Frames, FreqBins]
    uint OutputIdx;      // Mel spectrogram [Channels, Frames, NumMels]
    uint FilterbankIdx;  // Precomputed filterbank [NumMels, FreqBins]
    uint Channels;
    uint Frames;
    uint FreqBins;
    uint NumMels;
    uint LogScale;       // 0=linear, 1=log(mel + 1e-9)
};

[[vk::push_constant]] MelParams Params;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

// ─── Mel Filterbank Application ──────────────────────────────────────────────

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint melIdx = gid.x * 32 + tid.x;
    uint frameIdx = gid.y;
    uint channelIdx = gid.z;
    
    if (melIdx >= Params.NumMels || frameIdx >= Params.Frames || channelIdx >= Params.Channels) {
        return;
    }
    
    // Compute mel bin by applying triangular filter
    float melValue = 0.0f;
    
    uint inputBase = (channelIdx * Params.Frames + frameIdx) * Params.FreqBins;
    uint filterBase = melIdx * Params.FreqBins;
    
    for (uint k = 0; k < Params.FreqBins; k++) {
        float magnitude = OaLoad(heap[Params.InputIdx], inputBase + k);
        float filterWeight = OaLoad(heap[Params.FilterbankIdx], filterBase + k);
        melValue += magnitude * filterWeight;
    }
    
    // Apply log scale if requested
    if (Params.LogScale != 0) {
        melValue = log(melValue + 1e-9f);
    }
    
    // Store result
    uint outputIdx = (channelIdx * Params.Frames + frameIdx) * Params.NumMels + melIdx;
    OaStore(heap[Params.OutputIdx], outputIdx, melValue);
}
