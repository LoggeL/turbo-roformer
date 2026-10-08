// Single-core throughput of vdpbf16ps (zmm/ymm) vs fp32 FMA, to put the kernel numbers into perspective.
//   g++ -O2 -march=native -o dpbf16_peak benchmarks/dpbf16_peak.cpp && ./dpbf16_peak
// 12 independent accumulator chains per loop hide the instruction latency; GFLOP/s counts 2 flops per MAC.
#include <immintrin.h>
#include <chrono>
#include <cstdio>
template<int K> __attribute__((noinline)) float run_dp(long n){
  __m512 a[K]; for(int i=0;i<K;i++) a[i]=_mm512_set1_ps(i*1e-3f);
  __m512bh x=(__m512bh)_mm512_set1_epi32(0x3f803f80), y=(__m512bh)_mm512_set1_epi32(0x3f003f00);
  for(long i=0;i<n;i++){
#pragma GCC unroll 32
    for(int k=0;k<K;k++) a[k]=_mm512_dpbf16_ps(a[k],x,y);
    __asm__ volatile("" : "+v"(x));
  }
  float s=0; for(int i=0;i<K;i++) s+=_mm512_reduce_add_ps(a[i]); return s;}
template<int K> __attribute__((noinline)) float run_fma(long n){
  __m512 a[K]; for(int i=0;i<K;i++) a[i]=_mm512_set1_ps(i*1e-3f);
  __m512 x=_mm512_set1_ps(1.0001f), y=_mm512_set1_ps(0.9999f);
  for(long i=0;i<n;i++){
#pragma GCC unroll 32
    for(int k=0;k<K;k++) a[k]=_mm512_fmadd_ps(a[k],x,y);
    __asm__ volatile("" : "+v"(x));
  }
  float s=0; for(int i=0;i<K;i++) s+=_mm512_reduce_add_ps(a[i]); return s;}
template<int K> __attribute__((noinline)) float run_dp256(long n){
  __m256 a[K]; for(int i=0;i<K;i++) a[i]=_mm256_set1_ps(i*1e-3f);
  __m256bh x=(__m256bh)_mm256_set1_epi32(0x3f803f80), y=(__m256bh)_mm256_set1_epi32(0x3f003f00);
  for(long i=0;i<n;i++){
#pragma GCC unroll 32
    for(int k=0;k<K;k++) a[k]=_mm256_dpbf16_ps(a[k],x,y);
    __asm__ volatile("" : "+v"(x));
  }
  float s=0; for(int i=0;i<K;i++) s+=a[i][0]; return s;}
int main(){
  long n=50000000;
  auto t=std::chrono::steady_clock::now(); float s=run_dp<12>(n); double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
  printf("dpbf16 zmm: %.2f Ginstr/s (%.0f GFLOP/s) %f\n", n*12/dt/1e9, n*12*64/dt/1e9, s);
  t=std::chrono::steady_clock::now(); s=run_fma<12>(n); dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
  printf("fma zmm: %.2f Ginstr/s (%.0f GFLOP/s) %f\n", n*12/dt/1e9, n*12*32/dt/1e9, s);
  t=std::chrono::steady_clock::now(); s=run_dp256<12>(n); dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
  printf("dpbf16 ymm: %.2f Ginstr/s (%.0f GFLOP/s) %f\n", n*12/dt/1e9, n*12*32/dt/1e9, s);
}
