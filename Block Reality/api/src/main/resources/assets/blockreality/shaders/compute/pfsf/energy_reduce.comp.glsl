#version 450
#extension GL_GOOGLE_include_directive : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#include "stencil_constants.glsl"

// ═══════════════════════════════════════════════════════════════
//  Phase C — PFSF 離散圖能量 Elastic 項 Reduction Kernel
//
//  計算 E_elastic = (1/2) Σ_{edges (i,j)} w_ij × (φ_i - φ_j)²
//  邊權 w_ij = σ_ij × (1-d_i)^p × (1-d_j)^p × √(h_i × h_j)
//
//  數值穩定性（P1 警告修正 2026-04-22）：
//  ───────────────────────────────────────────────────────────
//  64³ voxel 下彈性應變能在高應力區／平緩區量級差異大，簡單並行
//  累加會產生 catastrophic cancellation。本 kernel 採兩層 Kahan
//  Summation（workgroup 內 + 跨 workgroup 成對樹狀歸約）。
//
//  FP64 fallback：若 runtime probe 確認 VK_EXT_shader_atomic_float64
//  可用，pushConstant 可改走 double variant（預留擴充位 pc.useFp64）。
//
//  兩階段 pattern（參考 phi_reduce_max.comp.glsl）：
//    Pass 1 (pc.pass = 0)：N voxels → ceil(N/256) partial sums (+compensations)
//    Pass 2 (pc.pass = 1)：partials → 1 final scalar
//  每個 WG 輸出 (sum, compensation) 兩個 float，共 8 bytes。
// ═══════════════════════════════════════════════════════════════

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    uint Lx, Ly, Lz;
    uint pass;             // 0 = voxel→partial, 1 = partial→final
    uint numPartials;      // pass2 輸入長度（pass1 時 = 0）
    float phaseFieldExp;   // p in (1-d)^p
    uint  useFp64;         // 預留 FP64 旗標（目前不啟用）
    uint  _pad;
} pc;

// Pass1 buffers
layout(set = 0, binding = 0) readonly buffer PhiIn   { float phi[];   };
layout(set = 0, binding = 1) readonly buffer SigmaIn { float sigma[]; };
layout(set = 0, binding = 2) readonly buffer DIn     { float dField[]; };
layout(set = 0, binding = 3) readonly buffer HIn     { float hField[]; };

// Output：每個 WG 寫 2 個 float (sum, compensation)，交錯格式：
//   partialsOut[2k + 0] = partial_sum_k
//   partialsOut[2k + 1] = kahan_compensation_k
layout(set = 0, binding = 4) buffer PartialsOut { float partialsOut[]; };

// Pass2 輸入：與 Pass1 的 PartialsOut 同 binding 重複使用
layout(set = 0, binding = 5) readonly buffer PartialsIn { float partialsIn[]; };

// WG 內 Kahan 累加共享記憶體：sum + compensation 交錯佈局
shared float sSum[256];
shared float sCmp[256];

// ═══════════════════════════════════════════════════════════════
//  Kahan Summation — 補償累加
//   y = v - c             (補償後的新增量)
//   t = s + y             (新總和，低位 bit 會被截斷)
//   c = (t - s) - y       (捕捉被截斷的低位 bit)
//   s = t
// ═══════════════════════════════════════════════════════════════
void kahanAdd(inout float s, inout float c, float v) {
    float y = v - c;
    float t = s + y;
    c = (t - s) - y;
    s = t;
}

// ─── 索引與邊界 ───────────────────────────────────────────────
uint idx3D(uint x, uint y, uint z) { return x + y * pc.Lx + z * pc.Lx * pc.Ly; }
bool inBounds(int x, int y, int z) {
    return x >= 0 && x < int(pc.Lx) && y >= 0 && y < int(pc.Ly) && z >= 0 && z < int(pc.Lz);
}

// ─── 邊權計算（對應 CPU DefaultEdgeWeight + 26-conn 分類）───
float stencilSigma(int dx, int dy, int dz, float sigI, float sigJ) {
    int nonzero = (dx != 0 ? 1 : 0) + (dy != 0 ? 1 : 0) + (dz != 0 ? 1 : 0);
    float siCl = max(sigI, 0.0);
    float sjCl = max(sigJ, 0.0);
    if (nonzero == 1) return 0.5 * (siCl + sjCl);                       // 面
    if (nonzero == 2) return sqrt(max(siCl * sjCl, 0.0)) * EDGE_P;      // 邊
    return sqrt(max(siCl * sjCl, 0.0)) * CORNER_P;                      // 角
}

float edgeWeight(float sigmaIJ, float dI, float dJ, float hI, float hJ, float p) {
    float dICl = clamp(dI, 0.0, 1.0);
    float dJCl = clamp(dJ, 0.0, 1.0);
    float hICl = max(hI, 0.0);
    float hJCl = max(hJ, 0.0);
    float damageI = pow(1.0 - dICl, p);
    float damageJ = pow(1.0 - dJCl, p);
    float curing  = sqrt(max(hICl * hJCl, 0.0));
    return max(sigmaIJ, 0.0) * damageI * damageJ * curing;
}

// 13 個正向 offsets（對應 CPU EnergyEvaluatorCPU.POSITIVE_OFFSETS）
const ivec3 POSITIVE_OFFSETS[13] = ivec3[13](
    // 面（3）：+x, +y, +z
    ivec3( 1,  0,  0), ivec3( 0,  1,  0), ivec3( 0,  0,  1),
    // 邊（6）：保留字典序第一非零為 +1 者
    ivec3( 1,  1,  0), ivec3( 1, -1,  0),
    ivec3( 1,  0,  1), ivec3( 1,  0, -1),
    ivec3( 0,  1,  1), ivec3( 0,  1, -1),
    // 角（4）：x=+1 的那四組（x=-1 組會被 (+x offset) 覆蓋到）
    ivec3( 1,  1,  1), ivec3( 1,  1, -1),
    ivec3( 1, -1,  1), ivec3( 1, -1, -1)
);

// ═══════════════════════════════════════════════════════════════
//  Pass 1：per-voxel contribution → WG-local Kahan sum
// ═══════════════════════════════════════════════════════════════
void pass1() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_GlobalInvocationID.x;
    uint N = pc.Lx * pc.Ly * pc.Lz;

    float localSum = 0.0;
    float localCmp = 0.0;

    // Grid-stride loop：每條 thread 可能要處理多個 voxel（當 N > numWG × 256）
    for (uint vi = gid; vi < N; vi += gl_NumWorkGroups.x * 256u) {
        uint vx = vi % pc.Lx;
        uint vy = (vi / pc.Lx) % pc.Ly;
        uint vz = vi / (pc.Lx * pc.Ly);
        float phiI = phi[vi];
        float sigI = sigma[vi];
        float dI   = dField[vi];
        float hI   = hField[vi];

        // 對 13 個正向鄰居累加貢獻（避免重複，與 CPU 版對齊）
        for (int k = 0; k < 13; ++k) {
            ivec3 off = POSITIVE_OFFSETS[k];
            int nx = int(vx) + off.x;
            int ny = int(vy) + off.y;
            int nz = int(vz) + off.z;
            if (!inBounds(nx, ny, nz)) continue;
            uint j = idx3D(uint(nx), uint(ny), uint(nz));

            float sigJ = sigma[j];
            float dJ   = dField[j];
            float hJ   = hField[j];
            float phiJ = phi[j];

            float sigIJ = stencilSigma(off.x, off.y, off.z, sigI, sigJ);
            float w = edgeWeight(sigIJ, dI, dJ, hI, hJ, pc.phaseFieldExp);
            float diff = phiI - phiJ;

            // 個別邊貢獻 = w × diff²；稍後乘 0.5 於 pass2 收尾
            kahanAdd(localSum, localCmp, w * diff * diff);
        }
    }

    sSum[tid] = localSum;
    sCmp[tid] = localCmp;
    barrier();

    // WG-local 成對樹狀歸約（Kahan 保持補償）
    for (uint stride = 128u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            // 將 sSum[tid+stride] 加進 sSum[tid] 的 Kahan 累加器
            // Note：這裡要把「partner's sum + partner's compensation」兩者合併加入
            // 保留 compensation 的正確性需額外一次 kahanAdd
            float partnerSum = sSum[tid + stride];
            float partnerCmp = sCmp[tid + stride];
            float s = sSum[tid];
            float c = sCmp[tid];
            // 先加對方的 sum（Kahan）
            kahanAdd(s, c, partnerSum);
            // 再減對方的 compensation（因 Kahan 定義 c 為被截斷的量）
            kahanAdd(s, c, -partnerCmp);
            sSum[tid] = s;
            sCmp[tid] = c;
        }
        barrier();
    }

    if (tid == 0u) {
        partialsOut[gl_WorkGroupID.x * 2u + 0u] = sSum[0];
        partialsOut[gl_WorkGroupID.x * 2u + 1u] = sCmp[0];
    }
}

// ═══════════════════════════════════════════════════════════════
//  Pass 2：WG partials → 1 scalar + 0.5 前因子
// ═══════════════════════════════════════════════════════════════
void pass2() {
    uint tid = gl_LocalInvocationID.x;
    float localSum = 0.0;
    float localCmp = 0.0;

    // Grid-stride：單一 WG 遍歷 numPartials
    for (uint k = tid; k < pc.numPartials; k += 256u) {
        float partSum = partialsIn[k * 2u + 0u];
        float partCmp = partialsIn[k * 2u + 1u];
        kahanAdd(localSum, localCmp,  partSum);
        kahanAdd(localSum, localCmp, -partCmp);
    }

    sSum[tid] = localSum;
    sCmp[tid] = localCmp;
    barrier();

    for (uint stride = 128u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            float ps = sSum[tid + stride];
            float pc_ = sCmp[tid + stride];
            float s = sSum[tid];
            float c = sCmp[tid];
            kahanAdd(s, c,  ps);
            kahanAdd(s, c, -pc_);
            sSum[tid] = s;
            sCmp[tid] = c;
        }
        barrier();
    }

    if (tid == 0u) {
        // 收尾：套上 (1/2) 前因子，把補償扣掉得最終 E_elastic
        float finalSum = sSum[0] - sCmp[0];
        partialsOut[0] = 0.5 * finalSum;
    }
}

void main() {
    if (pc.pass == 0u) pass1();
    else               pass2();
}
