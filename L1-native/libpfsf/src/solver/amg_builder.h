#pragma once

#include <cstdint>
#include <vector>

namespace pfsf {

struct AMGData {
    int32_t              n_coarse = 0;
    std::vector<int32_t> aggregation;  // int[N_fine]: fine→coarse index, -1 = excluded
    std::vector<float>   weights;      // float[N_fine]: prolongation weights P[i]
    std::vector<float>   diag_c;       // float[N_coarse]: Galerkin coarse diagonal D_c
};

/**
 * Build 2×2×2 block aggregation with conductivity-proportional prolongation.
 *
 * Aggregation mirrors GMG L1 coarsening (same dims as lx_l1/ly_l1/lz_l1).
 * Anchor voxels (vtype == 2) and void voxels (diag6 ≈ 0) are assigned
 * aggregation index -1 and receive zero weight.
 *
 * @param sigma    float[6 * N], SoA conductivity (dir * N + flat_index)
 * @param vtype    uint8_t[N], voxel type (2 = anchor, excluded from correction)
 * @param lx/ly/lz fine-grid dimensions
 */
AMGData buildAMG(const float* sigma, const uint8_t* vtype,
                 int32_t lx, int32_t ly, int32_t lz);

} // namespace pfsf
