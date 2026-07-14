// OaAudio STFT Kernel — GPU-accelerated Short-Time Fourier Transform
//
// Cooley-Tukey radix-2 FFT with Hann windowing, batched over channels and frames.
// Input:  [Channels, Samples] F32 planar audio buffer
// Output: [Channels, Frames, FreqBins] F32 magnitude spectrogram
//         where FreqBins = FftSize/2 + 1 (positive frequencies only)
//
// Dispatch: (Frames, 1, Channels) — one workgroup per frame per channel
// Shared memory: 2 * FftSize * sizeof(float) = 8KB for FftSize=1024

import Storage;

static const float OA_PI = 3.14159265358979323846f;

// ─── Push Constants ───────────────────────────────────────────────────────────

struct StftParams {
    uint InputIdx;       // Bindless index of input audio buffer [Channels, Samples]
    uint OutputIdx;      // Bindless index of output spectrogram [Channels, Frames, FreqBins]
    uint FftSize;        // FFT size (must be power of 2)
    uint HopSize;        // Hop length in samples
    uint WinSize;        // Active window length (<= FftSize)
    uint Window;         // 0=Hann, 1=Hamming, 2=Blackman, 3=rectangular
    uint Channels;       // Number of audio channels
    uint Samples;        // Total samples per channel
    uint Frames;         // Number of STFT frames
    uint FreqBins;       // FftSize/2 + 1
};

[[vk::push_constant]] StftParams Params;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

// Shared memory for FFT (real and imaginary parts)
groupshared float SharedRe[1024];
groupshared float SharedIm[1024];

// ─── Window Functions ─────────────────────────────────────────────────────────

// Periodic Hann (denominator N, not N-1) — matches librosa/torch stft.
float WindowValue(uint k, uint N, uint kind) {
    if (k >= N) return 0.0f;
    if (kind == 3) return 1.0f;
    float phase = 2.0f * OA_PI * float(k) / float(N);
    if (kind == 1) return 0.54f - 0.46f * cos(phase);
    if (kind == 2) return 0.42f - 0.5f * cos(phase) + 0.08f * cos(2.0f * phase);
    return 0.5f * (1.0f - cos(phase));
}

// ─── Radix-2 DIT FFT (in-place, power-of-two N ≤ 1024) ───────────────────────
// All loops stride by the workgroup size (256) so any N up to the shared-memory
// limit is fully covered — a plain `if (tid < …)` guard silently skips
// elements ≥ 256 and corrupts every FftSize above 256.

void FftInPlace(uint tid, uint N) {
    uint bits = 0;
    { uint t = N; while (t > 1) { bits++; t >>= 1; } }

    // Bit-reversal permutation (strided)
    for (uint i = tid; i < N; i += 256) {
        uint j = 0;
        uint v = i;
        for (uint b = 0; b < bits; b++) {
            j = (j << 1) | (v & 1);
            v >>= 1;
        }
        if (i < j) {
            float tmpRe = SharedRe[i];
            float tmpIm = SharedIm[i];
            SharedRe[i] = SharedRe[j];
            SharedIm[i] = SharedIm[j];
            SharedRe[j] = tmpRe;
            SharedIm[j] = tmpIm;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Butterfly passes: N/2 butterflies per pass, strided over threads.
    // Twiddle computed directly per butterfly (exact angle, no serial
    // accumulation error).
    for (uint len = 2; len <= N; len <<= 1) {
        uint halfLen = len >> 1;
        for (uint pair = tid; pair < (N >> 1); pair += 256) {
            uint g = pair / halfLen;
            uint k = pair % halfLen;
            uint i = g * len + k;
            uint j = i + halfLen;

            float ang = -2.0f * OA_PI * float(k) / float(len);
            float wRe = cos(ang);
            float wIm = sin(ang);

            float uRe = SharedRe[i];
            float uIm = SharedIm[i];
            float vRe = SharedRe[j] * wRe - SharedIm[j] * wIm;
            float vIm = SharedRe[j] * wIm + SharedIm[j] * wRe;

            SharedRe[i] = uRe + vRe;
            SharedIm[i] = uIm + vIm;
            SharedRe[j] = uRe - vRe;
            SharedIm[j] = uIm - vIm;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

// ─── Main Kernel ──────────────────────────────────────────────────────────────

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint frameIdx = gid.x;
    uint channelIdx = gid.z;
    uint localTid = tid.x;
    
    if (frameIdx >= Params.Frames || channelIdx >= Params.Channels) {
        return;
    }
    
    // Load windowed samples into shared memory
    uint frameOffset = frameIdx * Params.HopSize;
    uint channelOffset = channelIdx * Params.Samples;
    
    for (uint k = localTid; k < Params.FftSize; k += 256) {
        uint sampleIdx = frameOffset + k;
        float sample = 0.0f;
        
        if (sampleIdx < Params.Samples) {
            sample = OaLoad(heap[Params.InputIdx], channelOffset + sampleIdx);
        }
        
        float window = WindowValue(k, Params.WinSize, Params.Window);
        SharedRe[k] = sample * window;
        SharedIm[k] = 0.0f;
    }
    GroupMemoryBarrierWithGroupSync();
    
    // Perform FFT
    FftInPlace(localTid, Params.FftSize);
    
    // Compute magnitude and store positive frequencies only
    uint outputBase = (channelIdx * Params.Frames + frameIdx) * Params.FreqBins;
    
    for (uint k = localTid; k < Params.FreqBins; k += 256) {
        float re = SharedRe[k];
        float im = SharedIm[k];
        float magnitude = sqrt(re * re + im * im);
        OaStore(heap[Params.OutputIdx], outputBase + k, magnitude);
    }
}
