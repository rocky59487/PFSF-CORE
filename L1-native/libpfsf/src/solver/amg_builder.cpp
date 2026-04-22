#include "amg_builder.h"

namespace pfsf {

namespace {
constexpr uint8_t VTYPE_ANCHOR = 2;
}

AMGData buildAMG(const float* sigma, const uint8_t* vtype,
                 int32_t lx, int32_t ly, int32_t lz) {
    AMGData result{};
    const int64_t N = static_cast<int64_t>(lx) * ly * lz;
    if (N <= 0 || !sigma) return result;

    const int32_t Lxc = (lx + 1) / 2;
    const int32_t Lyc = (ly + 1) / 2;
    const int32_t Lzc = (lz + 1) / 2;
    const int32_t N_coarse = Lxc * Lyc * Lzc;

    result.n_coarse = N_coarse;
    result.aggregation.assign(N, -1);
    result.weights.assign(N, 0.0f);
    result.diag_c.assign(N_coarse, 0.0f);

    // diag6[i] = face conductivity sum — used as strength-of-connection weight
    std::vector<float> diag6(N, 0.0f);
    for (int64_t i = 0; i < N; ++i) {
        for (int d = 0; d < 6; ++d) {
            diag6[i] += sigma[d * N + i];
        }
    }

    // Pass 1: assign aggregates + accumulate agg_diag_sum[j]
    std::vector<float> agg_sum(N_coarse, 0.0f);
    for (int32_t gz = 0; gz < lz; ++gz) {
        for (int32_t gy = 0; gy < ly; ++gy) {
            for (int32_t gx = 0; gx < lx; ++gx) {
                const int64_t i = gx + lx * (gy + ly * gz);
                // Anchors pinned at phi=0: no correction needed
                if (vtype && vtype[i] == VTYPE_ANCHOR) continue;
                // Void voxels carry no conductivity: no meaningful z
                if (diag6[i] < 1e-20f) continue;
                const int32_t j = (gx / 2) + Lxc * ((gy / 2) + Lyc * (gz / 2));
                result.aggregation[i] = j;
                agg_sum[j] += diag6[i];
            }
        }
    }

    // Pass 2: compute prolongation weights P[i] and coarse diagonal D_c
    // P[i] = diag6[i] / agg_sum[agg[i]]  (conductivity-proportional, sums to 1 in agg)
    // D_c[j] = Σ_{i in agg j} P[i]² · diag6[i]  (Galerkin diagonal approximation)
    for (int64_t i = 0; i < N; ++i) {
        const int32_t j = result.aggregation[i];
        if (j < 0) continue;
        const float s = agg_sum[j];
        if (s < 1e-20f) continue;
        const float pi = diag6[i] / s;
        result.weights[i] = pi;
        result.diag_c[j] += pi * pi * diag6[i];
    }

    return result;
}

} // namespace pfsf
