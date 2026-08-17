// Hand-written small GEMM: cache blocking + AVX2/FMA 4x8 micro-kernel.
// Companion to compare_small_gemm.py
//
//   g++ -O3 -mavx2 -mfma -fopenmp -shared -fPIC -o libgemm_avx2.so gemm_avx2_blocked.cpp

#include <immintrin.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int MR = 4;
constexpr int NR = 8;
constexpr int BLOCK = 64;

// C[rows x cols] += Ap[k x 4] * Bp[k x 8]  (packed panels)
inline void micro_kernel_avx2(const float* Ap, const float* Bp, float* C, int ldc, int k,
                              int rows, int cols) {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();

  for (int p = 0; p < k; ++p) {
    const __m256 b = _mm256_loadu_ps(Bp + p * NR);
    acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + p * MR + 0), b, acc0);
    acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + p * MR + 1), b, acc1);
    acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + p * MR + 2), b, acc2);
    acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(Ap + p * MR + 3), b, acc3);
  }

  alignas(32) float tmp[MR * NR];
  _mm256_store_ps(tmp + 0 * NR, acc0);
  _mm256_store_ps(tmp + 1 * NR, acc1);
  _mm256_store_ps(tmp + 2 * NR, acc2);
  _mm256_store_ps(tmp + 3 * NR, acc3);

  for (int r = 0; r < rows; ++r) {
    float* crow = C + r * ldc;
    const float* trow = tmp + r * NR;
    for (int c = 0; c < cols; ++c) {
      crow[c] += trow[c];
    }
  }
}

}  // namespace

extern "C" void gemm_avx2_blocked(const float* A, const float* B, float* C, int M, int N,
                                  int K) {
  std::memset(C, 0, sizeof(float) * static_cast<size_t>(M) * N);

#pragma omp parallel
  {
    float* Ap = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * BLOCK * MR));
    float* Bp = static_cast<float*>(std::aligned_alloc(32, sizeof(float) * BLOCK * NR));

#pragma omp for schedule(static)
    for (int i0 = 0; i0 < M; i0 += BLOCK) {
      const int ib = std::min(BLOCK, M - i0);
      for (int p0 = 0; p0 < K; p0 += BLOCK) {
        const int pb = std::min(BLOCK, K - p0);
        for (int j0 = 0; j0 < N; j0 += BLOCK) {
          const int jb = std::min(BLOCK, N - j0);

          for (int ii = 0; ii < ib; ii += MR) {
            const int rows = std::min(MR, ib - ii);
            // Pack A micro-panel: Ap[p*4 + r] = A[i0+ii+r, p0+p]
            for (int p = 0; p < pb; ++p) {
              for (int r = 0; r < MR; ++r) {
                const int i = i0 + ii + r;
                Ap[p * MR + r] = (r < rows) ? A[i * K + (p0 + p)] : 0.f;
              }
            }

            for (int jj = 0; jj < jb; jj += NR) {
              const int cols = std::min(NR, jb - jj);
              // Pack B micro-panel: Bp[p*8 + c] = B[p0+p, j0+jj+c]
              for (int p = 0; p < pb; ++p) {
                for (int c = 0; c < NR; ++c) {
                  const int j = j0 + jj + c;
                  Bp[p * NR + c] = (c < cols) ? B[(p0 + p) * N + j] : 0.f;
                }
              }
              micro_kernel_avx2(Ap, Bp, C + (i0 + ii) * N + (j0 + jj), N, pb, rows, cols);
            }
          }
        }
      }
    }

    std::free(Ap);
    std::free(Bp);
  }
}

extern "C" void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
  std::memset(C, 0, sizeof(float) * static_cast<size_t>(M) * N);
  for (int i = 0; i < M; ++i) {
    for (int p = 0; p < K; ++p) {
      const float a = A[i * K + p];
      const float* brow = B + p * N;
      float* crow = C + i * N;
      for (int j = 0; j < N; ++j) {
        crow[j] += a * brow[j];
      }
    }
  }
}
