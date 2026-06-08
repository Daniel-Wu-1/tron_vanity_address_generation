typedef unsigned char      u8;
typedef unsigned int       u32;
typedef unsigned long long u64;

__constant__ u64 P_MOD[4] = {
    0xFFFFFFFEFFFFFC2FULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL
};

#define P_REDUCE_C 0x1000003D1ULL

__constant__ u64 GX[4] = {
    0x59F2815B16F81798ULL,
    0x029BFCDB2DCE28D9ULL,
    0x55A06295CE870B07ULL,
    0x79BE667EF9DCBBACULL
};
__constant__ u64 GY[4] = {
    0x9C47D08FFB10D4B8ULL,
    0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL,
    0x483ADA7726A3C465ULL
};

__device__ __forceinline__ u64 addc(u64 a, u64 b, u64 *carry) {
    u64 r = a + b;
    u64 c1 = (r < a) ? 1ULL : 0ULL;
    u64 r2 = r + *carry;
    u64 c2 = (r2 < r) ? 1ULL : 0ULL;
    *carry = c1 + c2;
    return r2;
}

__device__ __forceinline__ u64 subb(u64 a, u64 b, u64 *borrow) {
    u64 r = a - b;
    u64 b1 = (a < b) ? 1ULL : 0ULL;
    u64 r2 = r - *borrow;
    u64 b2 = (r < *borrow) ? 1ULL : 0ULL;
    *borrow = b1 + b2;
    return r2;
}

__device__ __forceinline__ u64 add4(u64 r[4], const u64 a[4], const u64 b[4]) {
    u64 c = 0;
    r[0] = addc(a[0], b[0], &c);
    r[1] = addc(a[1], b[1], &c);
    r[2] = addc(a[2], b[2], &c);
    r[3] = addc(a[3], b[3], &c);
    return c;
}

__device__ __forceinline__ u64 sub4(u64 r[4], const u64 a[4], const u64 b[4]) {
    u64 b_ = 0;
    r[0] = subb(a[0], b[0], &b_);
    r[1] = subb(a[1], b[1], &b_);
    r[2] = subb(a[2], b[2], &b_);
    r[3] = subb(a[3], b[3], &b_);
    return b_;
}

__device__ __forceinline__ bool ge_p(const u64 a[4]) {
    if (a[3] != P_MOD[3]) return a[3] > P_MOD[3];
    if (a[2] != P_MOD[2]) return a[2] > P_MOD[2];
    if (a[1] != P_MOD[1]) return a[1] > P_MOD[1];
    return a[0] >= P_MOD[0];
}

__device__ void f_sub(u64 r[4], const u64 a[4], const u64 b[4]) {
    u64 br = sub4(r, a, b);
    if (br) {
        u64 tmp[4];
        add4(tmp, r, P_MOD);
        r[0]=tmp[0]; r[1]=tmp[1]; r[2]=tmp[2]; r[3]=tmp[3];
    }
}

__device__ __forceinline__ void mul_full(u64 hi_lo[8], const u64 a[4], const u64 b[4]) {
    u64 t[8] = {0,0,0,0,0,0,0,0};
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        u64 carry = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            u64 lo = a[i] * b[j];
            u64 hi = __umul64hi(a[i], b[j]);

            u64 s = t[i+j] + lo;
            u64 c1 = (s < t[i+j]) ? 1ULL : 0ULL;
            u64 s2 = s + carry;
            u64 c2 = (s2 < s) ? 1ULL : 0ULL;
            t[i+j] = s2;
            carry = hi + c1 + c2;
        }
        t[i+4] = carry;
    }
    #pragma unroll
    for (int k = 0; k < 8; k++) hi_lo[k] = t[k];
}

__device__ void f_reduce(u64 r[4], const u64 c[8]) {

    const u64 K = P_REDUCE_C;

    u64 acc[5];
    acc[0]=c[0]; acc[1]=c[1]; acc[2]=c[2]; acc[3]=c[3]; acc[4]=0;

    u64 cy = 0;
    {
        u64 lo0 = c[4] * K, hi0 = __umul64hi(c[4], K);
        u64 lo1 = c[5] * K, hi1 = __umul64hi(c[5], K);
        u64 lo2 = c[6] * K, hi2 = __umul64hi(c[6], K);
        u64 lo3 = c[7] * K, hi3 = __umul64hi(c[7], K);

        u64 s, p;
        cy = 0;
        s = acc[0] + lo0; cy = (s < acc[0]) ? 1ULL : 0ULL; acc[0] = s;

        p = acc[1];
        s = p + lo1; u64 c1 = (s < p) ? 1ULL : 0ULL;
        p = s;
        s = p + hi0; u64 c2 = (s < p) ? 1ULL : 0ULL;
        p = s;
        s = p + cy;  u64 c3 = (s < p) ? 1ULL : 0ULL;
        acc[1] = s;
        cy = c1 + c2 + c3;

        p = acc[2];
        s = p + lo2; c1 = (s < p) ? 1ULL : 0ULL;
        p = s;
        s = p + hi1; c2 = (s < p) ? 1ULL : 0ULL;
        p = s;
        s = p + cy;  c3 = (s < p) ? 1ULL : 0ULL;
        acc[2] = s;
        cy = c1 + c2 + c3;

        p = acc[3];
        s = p + lo3; c1 = (s < p) ? 1ULL : 0ULL;
        p = s;
        s = p + hi2; c2 = (s < p) ? 1ULL : 0ULL;
        p = s;
        s = p + cy;  c3 = (s < p) ? 1ULL : 0ULL;
        acc[3] = s;
        cy = c1 + c2 + c3;

        acc[4] = hi3 + cy;
    }

    {
        u64 lo = acc[4] * K;
        u64 hi = __umul64hi(acc[4], K);
        u64 s, p, c1;
        u64 carry2 = 0;

        p = acc[0];
        s = p + lo; c1 = (s < p) ? 1ULL : 0ULL;
        acc[0] = s;
        carry2 = c1;

        p = acc[1];
        s = p + hi; c1 = (s < p) ? 1ULL : 0ULL;
        p = s;
        s = p + carry2; u64 c2 = (s < p) ? 1ULL : 0ULL;
        acc[1] = s;
        carry2 = c1 + c2;

        p = acc[2];
        s = p + carry2; c1 = (s < p) ? 1ULL : 0ULL;
        acc[2] = s;
        carry2 = c1;

        p = acc[3];
        s = p + carry2; c1 = (s < p) ? 1ULL : 0ULL;
        acc[3] = s;
        carry2 = c1;

        if (carry2) {
            u64 lo2 = carry2 * K;

            u64 c1b = 0;
            p = acc[0];
            s = p + lo2; c1b = (s < p) ? 1ULL : 0ULL;
            acc[0] = s;

            for (int k = 1; k < 4 && c1b; k++) {
                u64 pp = acc[k];
                u64 ss = pp + 1ULL;
                acc[k] = ss;
                c1b = (ss < pp) ? 1ULL : 0ULL;
            }
        }
    }

    r[0] = acc[0]; r[1] = acc[1]; r[2] = acc[2]; r[3] = acc[3];
    if (ge_p(r)) {
        u64 tmp[4];
        sub4(tmp, r, P_MOD);
        r[0]=tmp[0]; r[1]=tmp[1]; r[2]=tmp[2]; r[3]=tmp[3];
    }
}

__device__ void f_mul(u64 r[4], const u64 a[4], const u64 b[4]) {
    u64 c[8];
    mul_full(c, a, b);
    f_reduce(r, c);
}

__device__ void f_sqr(u64 r[4], const u64 a[4]) {
    u64 c[8];
    mul_full(c, a, a);
    f_reduce(r, c);
}

__device__ void f_inv(u64 r[4], const u64 a[4]) {

    u64 exp[4] = {
        0xFFFFFFFEFFFFFC2DULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL
    };

    u64 result[4] = {1ULL, 0ULL, 0ULL, 0ULL};
    u64 base[4]   = {a[0], a[1], a[2], a[3]};
    u64 tmp[4];

    #pragma unroll 1
    for (int i = 0; i < 256; i++) {
        int limb = i >> 6;
        int bit = i & 63;
        if ((exp[limb] >> bit) & 1ULL) {
            f_mul(tmp, result, base);
            result[0]=tmp[0]; result[1]=tmp[1]; result[2]=tmp[2]; result[3]=tmp[3];
        }
        f_sqr(tmp, base);
        base[0]=tmp[0]; base[1]=tmp[1]; base[2]=tmp[2]; base[3]=tmp[3];
    }
    r[0]=result[0]; r[1]=result[1]; r[2]=result[2]; r[3]=result[3];
}

__device__ void point_add(u64 rx[4], u64 ry[4],
                           const u64 px[4], const u64 py[4],
                           const u64 qx[4], const u64 qy[4]) {
    u64 dx[4], dy[4], dx_inv[4], s[4], s2[4], tmp[4];

    f_sub(dx, qx, px);
    f_sub(dy, qy, py);
    f_inv(dx_inv, dx);
    f_mul(s, dy, dx_inv);
    f_sqr(s2, s);

    f_sub(tmp, s2, px);
    f_sub(rx, tmp, qx);

    f_sub(tmp, px, rx);
    u64 prod[4];
    f_mul(prod, s, tmp);
    f_sub(ry, prod, py);
}

#ifndef POINTS_PER_THREAD
#define POINTS_PER_THREAD 8
#endif

__device__ void f_inv_batch(u64 a[POINTS_PER_THREAD][4]) {
    u64 prefix[POINTS_PER_THREAD][4];

    #pragma unroll
    for (int i = 0; i < 4; i++) prefix[0][i] = a[0][i];

    #pragma unroll
    for (int k = 1; k < POINTS_PER_THREAD; k++) {
        f_mul(prefix[k], prefix[k-1], a[k]);
    }

    u64 inv[4];
    f_inv(inv, prefix[POINTS_PER_THREAD - 1]);

    u64 tmp[4], ak_inv[4];
    #pragma unroll
    for (int k = POINTS_PER_THREAD - 1; k >= 1; k--) {

        f_mul(ak_inv, inv, prefix[k - 1]);

        f_mul(tmp, inv, a[k]);
        #pragma unroll
        for (int i = 0; i < 4; i++) inv[i] = tmp[i];

        #pragma unroll
        for (int i = 0; i < 4; i++) a[k][i] = ak_inv[i];
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) a[0][i] = inv[i];
}

__constant__ u64 KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
    0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ __forceinline__ u64 rotl64(u64 x, int n) {
    return (x << n) | (x >> (64 - n));
}

__device__ __forceinline__ void keccak_round(u64 st[25], u64 rc) {
    u64 bc0, bc1, bc2, bc3, bc4;

    bc0 = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
    bc1 = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
    bc2 = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
    bc3 = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
    bc4 = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];
    u64 t0 = bc4 ^ rotl64(bc1, 1);
    u64 t1 = bc0 ^ rotl64(bc2, 1);
    u64 t2 = bc1 ^ rotl64(bc3, 1);
    u64 t3 = bc2 ^ rotl64(bc4, 1);
    u64 t4 = bc3 ^ rotl64(bc0, 1);
    st[0] ^= t0; st[5] ^= t0; st[10] ^= t0; st[15] ^= t0; st[20] ^= t0;
    st[1] ^= t1; st[6] ^= t1; st[11] ^= t1; st[16] ^= t1; st[21] ^= t1;
    st[2] ^= t2; st[7] ^= t2; st[12] ^= t2; st[17] ^= t2; st[22] ^= t2;
    st[3] ^= t3; st[8] ^= t3; st[13] ^= t3; st[18] ^= t3; st[23] ^= t3;
    st[4] ^= t4; st[9] ^= t4; st[14] ^= t4; st[19] ^= t4; st[24] ^= t4;

    u64 tt = st[1], tmp;
    tmp = st[10]; st[10] = rotl64(tt, 1);  tt = tmp;
    tmp = st[7];  st[7]  = rotl64(tt, 3);  tt = tmp;
    tmp = st[11]; st[11] = rotl64(tt, 6);  tt = tmp;
    tmp = st[17]; st[17] = rotl64(tt, 10); tt = tmp;
    tmp = st[18]; st[18] = rotl64(tt, 15); tt = tmp;
    tmp = st[3];  st[3]  = rotl64(tt, 21); tt = tmp;
    tmp = st[5];  st[5]  = rotl64(tt, 28); tt = tmp;
    tmp = st[16]; st[16] = rotl64(tt, 36); tt = tmp;
    tmp = st[8];  st[8]  = rotl64(tt, 45); tt = tmp;
    tmp = st[21]; st[21] = rotl64(tt, 55); tt = tmp;
    tmp = st[24]; st[24] = rotl64(tt, 2);  tt = tmp;
    tmp = st[4];  st[4]  = rotl64(tt, 14); tt = tmp;
    tmp = st[15]; st[15] = rotl64(tt, 27); tt = tmp;
    tmp = st[23]; st[23] = rotl64(tt, 41); tt = tmp;
    tmp = st[19]; st[19] = rotl64(tt, 56); tt = tmp;
    tmp = st[13]; st[13] = rotl64(tt, 8);  tt = tmp;
    tmp = st[12]; st[12] = rotl64(tt, 25); tt = tmp;
    tmp = st[2];  st[2]  = rotl64(tt, 43); tt = tmp;
    tmp = st[20]; st[20] = rotl64(tt, 62); tt = tmp;
    tmp = st[14]; st[14] = rotl64(tt, 18); tt = tmp;
    tmp = st[22]; st[22] = rotl64(tt, 39); tt = tmp;
    tmp = st[9];  st[9]  = rotl64(tt, 61); tt = tmp;
    tmp = st[6];  st[6]  = rotl64(tt, 20); tt = tmp;
                  st[1]  = rotl64(tt, 44);

    #pragma unroll
    for (int j = 0; j < 25; j += 5) {
        u64 a0 = st[j+0], a1 = st[j+1], a2 = st[j+2], a3 = st[j+3], a4 = st[j+4];
        st[j+0] = a0 ^ ((~a1) & a2);
        st[j+1] = a1 ^ ((~a2) & a3);
        st[j+2] = a2 ^ ((~a3) & a4);
        st[j+3] = a3 ^ ((~a4) & a0);
        st[j+4] = a4 ^ ((~a0) & a1);
    }

    st[0] ^= rc;
}

__device__ void keccak_f(u64 st[25]) {

    #pragma unroll 1
    for (int round = 0; round < 24; round++) {
        keccak_round(st, KECCAK_RC[round]);
    }
}

__device__ void keccak256_64(u8 out[32], const u8 in[64]) {
    u64 st[25];
    #pragma unroll
    for (int i = 0; i < 25; i++) st[i] = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        u64 v = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            v |= ((u64)in[i*8 + j]) << (j*8);
        }
        st[i] = v;
    }

    st[8] ^= 0x0000000000000001ULL;
    st[16] ^= 0x8000000000000000ULL;

    keccak_f(st);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        u64 v = st[i];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            out[i*8 + j] = (u8)(v >> (j*8));
        }
    }
}

__constant__ u32 SHA_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__device__ __forceinline__ u32 rotr32(u32 x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ void sha256_21(u8 out[32], const u8 in[21]) {

    u8 buf[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) buf[i] = 0;
    #pragma unroll
    for (int i = 0; i < 21; i++) buf[i] = in[i];
    buf[21] = 0x80;

    buf[63] = 168;

    u32 H[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
    };
    u32 W[64];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = ((u32)buf[i*4] << 24) | ((u32)buf[i*4+1] << 16) |
               ((u32)buf[i*4+2] << 8) | (u32)buf[i*4+3];
    }
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        u32 s0 = rotr32(W[i-15],7) ^ rotr32(W[i-15],18) ^ (W[i-15]>>3);
        u32 s1 = rotr32(W[i-2],17) ^ rotr32(W[i-2],19) ^ (W[i-2]>>10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }
    u32 a=H[0],b=H[1],c=H[2],d=H[3],e=H[4],f=H[5],g=H[6],h=H[7];
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        u32 S1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
        u32 ch = (e & f) ^ ((~e) & g);
        u32 t1 = h + S1 + ch + SHA_K[i] + W[i];
        u32 S0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
        u32 mj = (a & b) ^ (a & c) ^ (b & c);
        u32 t2 = S0 + mj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    H[0]+=a; H[1]+=b; H[2]+=c; H[3]+=d; H[4]+=e; H[5]+=f; H[6]+=g; H[7]+=h;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i*4]   = (u8)(H[i] >> 24);
        out[i*4+1] = (u8)(H[i] >> 16);
        out[i*4+2] = (u8)(H[i] >> 8);
        out[i*4+3] = (u8)(H[i]);
    }
}

__device__ void sha256_32(u8 out[32], const u8 in[32]) {
    u8 buf[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) buf[i] = 0;
    #pragma unroll
    for (int i = 0; i < 32; i++) buf[i] = in[i];
    buf[32] = 0x80;
    buf[62] = 1;
    buf[63] = 0;

    u32 H[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
    };
    u32 W[64];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = ((u32)buf[i*4] << 24) | ((u32)buf[i*4+1] << 16) |
               ((u32)buf[i*4+2] << 8) | (u32)buf[i*4+3];
    }
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        u32 s0 = rotr32(W[i-15],7) ^ rotr32(W[i-15],18) ^ (W[i-15]>>3);
        u32 s1 = rotr32(W[i-2],17) ^ rotr32(W[i-2],19) ^ (W[i-2]>>10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }
    u32 a=H[0],b=H[1],c=H[2],d=H[3],e=H[4],f=H[5],g=H[6],h=H[7];
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        u32 S1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
        u32 ch = (e & f) ^ ((~e) & g);
        u32 t1 = h + S1 + ch + SHA_K[i] + W[i];
        u32 S0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
        u32 mj = (a & b) ^ (a & c) ^ (b & c);
        u32 t2 = S0 + mj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    H[0]+=a; H[1]+=b; H[2]+=c; H[3]+=d; H[4]+=e; H[5]+=f; H[6]+=g; H[7]+=h;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i*4]   = (u8)(H[i] >> 24);
        out[i*4+1] = (u8)(H[i] >> 16);
        out[i*4+2] = (u8)(H[i] >> 8);
        out[i*4+3] = (u8)(H[i]);
    }
}

__constant__ char B58_ALPHABET[59] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

__device__ void base58_tail(char out[], const u8 in[25], int k) {
    u8 buf[25];
    #pragma unroll
    for (int i = 0; i < 25; i++) buf[i] = in[i];

    int produced = 0;
    int top = 0;
    while (top < 25 && buf[top] == 0) top++;

    while (produced < k && top < 25) {
        u32 rem = 0;
        for (int i = top; i < 25; i++) {
            u32 v = (rem << 8) | buf[i];
            buf[i] = (u8)(v / 58);
            rem = v % 58;
        }
        out[produced++] = B58_ALPHABET[rem];
        while (top < 25 && buf[top] == 0) top++;
    }

    while (produced < k) {
        out[produced++] = B58_ALPHABET[0];
    }
}

__device__ void base58_encode_25(char out[34], const u8 in[25]) {

    u8 buf[25];
    #pragma unroll
    for (int i = 0; i < 25; i++) buf[i] = in[i];

    char tmp[40];
    int idx = 0;

    int start = 0;
    while (start < 25 && buf[start] == 0) {
        tmp[idx++] = B58_ALPHABET[0];
        start++;
    }

    int top = start;
    while (top < 25) {
        u32 rem = 0;
        for (int i = top; i < 25; i++) {
            u32 v = (rem << 8) | buf[i];
            buf[i] = (u8)(v / 58);
            rem = v % 58;
        }
        tmp[idx++] = B58_ALPHABET[rem];
        while (top < 25 && buf[top] == 0) top++;
    }

    for (int i = 0; i < 34; i++) {
        if (i < idx) out[i] = tmp[idx - 1 - i];
        else out[i] = B58_ALPHABET[0];
    }
}

struct PatternParams {
    int prefix_len;
    int suffix_len;
    int repeat_n;
    char prefix[40];
    char suffix[40];
};

struct MatchRecord {
    u32  thread_id;
    u32  _pad;
    u64  step;
    char address[34];
    char _pad2[6];
};

extern "C" __global__ void vanity_kernel(
    u64       *cur_x,
    u64       *cur_y,
    int        steps,
    u64        step_offset,
    const PatternParams *params_in,
    MatchRecord *out,
    u32        *out_count,
    u32         max_matches
) {
    PatternParams params = *params_in;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    u64 px[POINTS_PER_THREAD][4], py[POINTS_PER_THREAD][4];
    #pragma unroll
    for (int p = 0; p < POINTS_PER_THREAD; p++) {
        int base = (tid * POINTS_PER_THREAD + p) * 4;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            px[p][i] = cur_x[base + i];
            py[p][i] = cur_y[base + i];
        }
    }

    u64 gx[4], gy[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) { gx[i] = GX[i]; gy[i] = GY[i]; }

    for (int step = 0; step < steps; step++) {

        #pragma unroll 1
        for (int p = 0; p < POINTS_PER_THREAD; p++) {

            u8 xy[64];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                u64 v = px[p][3 - i];
                #pragma unroll
                for (int j = 0; j < 8; j++) xy[i*8 + j] = (u8)(v >> (56 - j*8));
            }
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                u64 v = py[p][3 - i];
                #pragma unroll
                for (int j = 0; j < 8; j++) xy[32 + i*8 + j] = (u8)(v >> (56 - j*8));
            }

            u8 hash[32];
            keccak256_64(hash, xy);

            u8 payload[21];
            payload[0] = 0x41;
            #pragma unroll
            for (int i = 0; i < 20; i++) payload[1 + i] = hash[12 + i];

            u8 h1[32], h2[32];
            sha256_21(h1, payload);
            sha256_32(h2, h1);

            u8 raw[25];
            #pragma unroll
            for (int i = 0; i < 21; i++) raw[i] = payload[i];
            raw[21] = h2[0]; raw[22] = h2[1]; raw[23] = h2[2]; raw[24] = h2[3];

            int sfx_len = params.suffix_len;
            int pfx_len = params.prefix_len;
            int rep_n = params.repeat_n;
            if (sfx_len < 0 || sfx_len > 34) sfx_len = 0;
            if (pfx_len < 0 || pfx_len > 34) pfx_len = 0;
            if (rep_n < 0 || rep_n > 34) rep_n = 0;

            int tail_k = sfx_len > rep_n ? sfx_len : rep_n;
            bool ok = true;

            if (tail_k > 0) {

                char tail_chars[34];
                base58_tail(tail_chars, raw, tail_k);

                if (sfx_len > 0) {
                    for (int i = 0; i < sfx_len; i++) {

                        if (tail_chars[sfx_len - 1 - i] != params.suffix[i]) {
                            ok = false; break;
                        }
                    }
                }

                if (ok && rep_n > 0) {
                    char c0 = tail_chars[0];
                    for (int i = 1; i < rep_n; i++) {
                        if (tail_chars[i] != c0) { ok = false; break; }
                    }
                }
            }

            if (!ok) continue;

            char addr[34];
            base58_encode_25(addr, raw);

            if (pfx_len > 0) {
                for (int i = 0; i < pfx_len; i++) {
                    if (addr[i] != params.prefix[i]) { ok = false; break; }
                }
            }

            if (ok) {
                u32 idx = atomicAdd(out_count, 1u);
                if (idx < max_matches) {

                    out[idx].thread_id = ((u32)tid << 4) | ((u32)p & 0xF);
                    out[idx].step = step_offset + (u64)step;
                    #pragma unroll
                    for (int i = 0; i < 34; i++) out[idx].address[i] = addr[i];
                }
            }
        }

        u64 dx[POINTS_PER_THREAD][4], dy[POINTS_PER_THREAD][4];
        #pragma unroll
        for (int p = 0; p < POINTS_PER_THREAD; p++) {
            f_sub(dx[p], gx, px[p]);
            f_sub(dy[p], gy, py[p]);
        }

        f_inv_batch(dx);

        #pragma unroll
        for (int p = 0; p < POINTS_PER_THREAD; p++) {
            u64 s[4], s2[4], tmp[4], nx[4], ny[4];
            f_mul(s, dy[p], dx[p]);
            f_sqr(s2, s);
            f_sub(tmp, s2, px[p]);
            f_sub(nx, tmp, gx);
            f_sub(tmp, px[p], nx);
            u64 prod[4];
            f_mul(prod, s, tmp);
            f_sub(ny, prod, py[p]);
            #pragma unroll
            for (int i = 0; i < 4; i++) { px[p][i] = nx[i]; py[p][i] = ny[i]; }
        }
    }

    #pragma unroll
    for (int p = 0; p < POINTS_PER_THREAD; p++) {
        int base = (tid * POINTS_PER_THREAD + p) * 4;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            cur_x[base + i] = px[p][i];
            cur_y[base + i] = py[p][i];
        }
    }
}
