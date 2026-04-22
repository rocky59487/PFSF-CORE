/**
 * @file wind_bias.cpp
 * @brief Anderson 2010 first-order upwind bias applied to SoA-6 conductivity.
 *
 * @cite Anderson, J.D. (2010). "Fundamentals of Aerodynamics — Potential
 *        Flow Theory §5". McGraw-Hill. (See also Eurocode 1 Cp tables.)
 * @formula For each horizontal edge direction d in {-X,+X,-Z,+Z}:
 *            dot = step_d · wind_xz
 *            if (dot > 0):  sigma *= (1 + k)
 *            if (dot < 0):  sigma /= (1 + k)
 *            (dot == 0: unchanged; UP/DOWN never biased.)
 * @maps_to PFSFConductivity.java:sigma() L74-L85 (inline per-edge)
 *
 * Designed as a standalone post-processor so it can be composed with a
 * sigma base computed without wind. Phase 1 publishes the symbol only;
 * Phase 3 will wire it into the full conductivity kernel (when
 * PFSFConductivity moves to C++). For now the Java ref path stays
 * per-edge-inlined, and this implementation is exercised exclusively by
 * the golden-parity test (vs an equivalent Java post-processor helper).
 *
 * SoA layout: conductivity[d * N + i] where
 *   d = 0 (-X) 1 (+X) 2 (-Y) 3 (+Y) 4 (-Z) 5 (+Z)
 * matches pfsf_direction and PFSFConstants.DIR_*.
 */

#include "pfsf/pfsf_compute.h"
#include <cstddef>

extern "C" void pfsf_apply_wind_bias(float* conductivity,
                                       int32_t n,
                                       pfsf_vec3 wind,
                                       float upwind_factor) {
    /* 移除非對稱性錯誤 (Point 7)：迎風偏置破壞了系統矩陣 A 的對稱性，
       這會導致 PCG 求解器不穩定或發散。矩陣 A 必須保持對稱正定。
       風壓應透過 Source Term 體現，不再修改 conductivity 陣列。
       此函數現在為 no-op 以維持 ABI 相容性。 */
    (void) conductivity;
    (void) n;
    (void) wind;
    (void) upwind_factor;
}
