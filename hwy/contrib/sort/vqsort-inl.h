// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Per-target
#if defined(HIGHWAY_HWY_CONTRIB_SORT_VQSORT_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_SORT_VQSORT_INL_H_
#undef HIGHWAY_HWY_CONTRIB_SORT_VQSORT_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_SORT_VQSORT_INL_H_
#endif

// Makes it harder for adversaries to predict our sampling locations, at the
// cost of 1-2% increased runtime.
#ifndef VQSORT_SECURE_RNG
#define VQSORT_SECURE_RNG 0
#endif

// Seed source for SFC generator: 1=getrandom, 2=CryptGenRandom
// (not all Android support the getrandom wrapper)
#ifndef VQSORT_SECURE_SEED
#if (defined(linux) || defined(__linux__)) && \
    !(defined(ANDROID) || defined(__ANDROID__))
#define VQSORT_SECURE_SEED 1
#elif defined(_WIN32) || defined(_WIN64)
#define VQSORT_SECURE_SEED 2
#else
#define VQSORT_SECURE_SEED 0
#endif

#if VQSORT_SECURE_RNG
#include "third_party/absl/random/random.h"
#else
#include <time.h>
#if VQSORT_SECURE_SEED == 1
#include <sys/random.h>
#elif VQSORT_SECURE_SEED == 2
#include <windows.h>
// Must come after windows.h.
#include <wincrypt.h>
#endif  // VQSORT_SECURE_SEED
#endif  // VQSORT_SECURE_RNG

#endif
#include "hwy/aligned_allocator.h"
#include "hwy/cache_control.h"
#include "hwy/contrib/sort/shared-inl.h"
#include "hwy/contrib/sort/sorting_networks-inl.h"
#include "hwy/highway.h"

#if HWY_IS_MSAN
#include <sanitizer/msan_interface.h>
#endif

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace detail {

#if HWY_TARGET == HWY_SCALAR

template <typename T>
void Swap(T* a, T* b) {
  T t = *a;
  *a = *b;
  *b = t;
}

// Scalar version of HeapSort (see below)
template <class Traits, typename T>
void HeapSort(Traits st, T* HWY_RESTRICT keys, const size_t num) {
  if (num < 2) return;

  // Build heap.
  for (size_t i = 1; i < num; i += 1) {
    size_t j = i;
    while (j != 0) {
      const size_t idx_parent = ((j - 1) / 1 / 2);
      if (!st.Compare1(keys + idx_parent, keys + j)) {
        break;
      }
      Swap(keys + j, keys + idx_parent);
      j = idx_parent;
    }
  }

  for (size_t i = num - 1; i != 0; i -= 1) {
    // Swap root with last
    Swap(keys + 0, keys + i);

    // Sift down the new root.
    size_t j = 0;
    while (j < i) {
      const size_t left = 2 * j + 1;
      const size_t right = 2 * j + 2;
      if (left >= i) break;
      size_t idx_larger = j;
      if (st.Compare1(keys + j, keys + left)) {
        idx_larger = left;
      }
      if (right < i && st.Compare1(keys + idx_larger, keys + right)) {
        idx_larger = right;
      }
      if (idx_larger == j) break;
      Swap(keys + j, keys + idx_larger);
      j = idx_larger;
    }
  }
}

#else

using Constants = hwy::SortConstants;

// ------------------------------ HeapSort

// Heapsort: O(1) space, O(N*logN) worst-case comparisons.
// Based on LLVM sanitizer_common.h, licensed under Apache-2.0.
template <class Traits, typename T>
void HeapSort(Traits st, T* HWY_RESTRICT keys, const size_t num) {
  constexpr size_t N1 = st.LanesPerKey();
  const FixedTag<T, N1> d;

  if (num < 2 * N1) return;

  // Build heap.
  for (size_t i = N1; i < num; i += N1) {
    size_t j = i;
    while (j != 0) {
      const size_t idx_parent = ((j - N1) / N1 / 2) * N1;
      if (AllFalse(d, st.Compare(d, st.SetKey(d, keys + idx_parent),
                                 st.SetKey(d, keys + j)))) {
        break;
      }
      st.Swap(keys + j, keys + idx_parent);
      j = idx_parent;
    }
  }

  for (size_t i = num - N1; i != 0; i -= N1) {
    // Swap root with last
    st.Swap(keys + 0, keys + i);

    // Sift down the new root.
    size_t j = 0;
    while (j < i) {
      const size_t left = 2 * j + N1;
      const size_t right = 2 * j + 2 * N1;
      if (left >= i) break;
      size_t idx_larger = j;
      const auto key_j = st.SetKey(d, keys + j);
      if (AllTrue(d, st.Compare(d, key_j, st.SetKey(d, keys + left)))) {
        idx_larger = left;
      }
      if (right < i && AllTrue(d, st.Compare(d, st.SetKey(d, keys + idx_larger),
                                             st.SetKey(d, keys + right)))) {
        idx_larger = right;
      }
      if (idx_larger == j) break;
      st.Swap(keys + j, keys + idx_larger);
      j = idx_larger;
    }
  }
}

// ------------------------------ BaseCase

// Sorts `keys` within the range [0, num) via sorting network.
template <class D, class Traits, typename T = TFromD<D> >
HWY_NOINLINE void BaseCase(D d, Traits st, T* HWY_RESTRICT keys, size_t num,
                           T* HWY_RESTRICT buf) {
  const size_t N = Lanes(d);
  using V = decltype(Zero(d));

  // _Nonzero32 requires num - 1 != 0.
  if (HWY_UNLIKELY(num <= 1)) return;

  // Reshape into a matrix with kMaxRows rows, and columns limited by the
  // 1D `num`, which is upper-bounded by the vector width (see BaseCaseNum).
  const size_t num_pow2 = 1u << (32 - Num0BitsAboveMS1Bit_Nonzero32(
                                          static_cast<uint32_t>(num - 1)));
  HWY_DASSERT(num <= num_pow2 && num_pow2 <= Constants::BaseCaseNum(N));
  const size_t cols =
      HWY_MAX(st.LanesPerKey(), num_pow2 >> Constants::kMaxRowsLog2);
  HWY_DASSERT(cols <= N);

  // Copy `keys` to `buf`.
  size_t i;
  for (i = 0; i + N <= num; i += N) {
    Store(LoadU(d, keys + i), d, buf + i);
  }
  for (; i < num; ++i) {
    buf[i] = keys[i];
  }

  // Fill with padding - last in sort order, not copied to keys.
  const V kPadding = st.LastValue(d);
  // Initialize an extra vector because SortingNetwork loads full vectors,
  // which may exceed cols*kMaxRows.
  for (; i < (cols * Constants::kMaxRows + N); i += N) {
    StoreU(kPadding, d, buf + i);
  }

  SortingNetwork(st, buf, cols);

  for (i = 0; i + N <= num; i += N) {
    StoreU(Load(d, buf + i), d, keys + i);
  }
  for (; i < num; ++i) {
    keys[i] = buf[i];
  }
}

// ------------------------------ Partition

// Consumes from `left` until a multiple of kUnroll*N remains.
// Temporarily stores the right side into `buf`, then moves behind `right`.
template <class D, class Traits, class T = TFromD<D>, class V = Vec<D> >
HWY_NOINLINE void PartitionToMultipleOfUnroll(D d, Traits st,
                                              T* HWY_RESTRICT keys,
                                              size_t& left, size_t& right,
                                              const V pivot,
                                              T* HWY_RESTRICT buf) {
  constexpr size_t kUnroll = Constants::kPartitionUnroll;
  const size_t N = Lanes(d);
  size_t readL = left;
  size_t bufR = 0;
  const size_t num = right - left;
  // Partition requires both a multiple of kUnroll*N and at least
  // 2*kUnroll*N for the initial loads. If less, consume all here.
  const size_t num_rem =
      (num < 2 * kUnroll * N) ? num : (num & (kUnroll * N - 1));
  size_t i = 0;
  for (; i + N <= num_rem; i += N) {
    const V vL = LoadU(d, keys + readL);
    readL += N;

    const auto comp = st.Compare(d, pivot, vL);
    left += CompressBlendedStore(vL, Not(comp), d, keys + left);
    bufR += CompressStore(vL, comp, d, buf + bufR);
  }
  // Last iteration: only use valid lanes.
  if (HWY_LIKELY(i != num_rem)) {
    const auto mask = FirstN(d, num_rem - i);
    const V vL = LoadU(d, keys + readL);

    const auto comp = st.Compare(d, pivot, vL);
    left += CompressBlendedStore(vL, AndNot(comp, mask), d, keys + left);
    bufR += CompressStore(vL, And(comp, mask), d, buf + bufR);
  }

  // MSAN seems not to understand CompressStore. buf[0, bufR) are valid.
#if HWY_IS_MSAN
  __msan_unpoison(buf, bufR * sizeof(T));
#endif

  // Everything we loaded was put into buf, or behind the new `left`, after
  // which there is space for bufR items. First move items from `right` to
  // `left` to free up space, then copy `buf` into the vacated `right`.
  // A loop with masked loads from `buf` is insufficient - we would also need to
  // mask from `right`. Combining a loop with memcpy for the remainders is
  // slower than just memcpy, so we use that for simplicity.
  right -= bufR;
  memcpy(keys + left, keys + right, bufR * sizeof(T));
  memcpy(keys + right, buf, bufR * sizeof(T));
}

template <class D, class Traits, class V = Vec<D> >
HWY_INLINE void StoreLeftRight(D d, Traits st, const V v, const V pivot,
                               TFromD<D>* HWY_RESTRICT keys, size_t& writeL,
                               size_t& writeR) {
  const size_t N = Lanes(d);

  const auto comp = st.Compare(d, pivot, v);
  const size_t num_left = CompressBlendedStore(v, Not(comp), d, keys + writeL);
  writeL += num_left;

  writeR -= (N - num_left);
  (void)CompressBlendedStore(v, comp, d, keys + writeR);
}

template <class D, class Traits, class V = Vec<D> >
HWY_INLINE void StoreLeftRight4(D d, Traits st, const V v0, const V v1, const V v2,
                                const V v3, const V pivot,
                                TFromD<D>* HWY_RESTRICT keys, size_t& writeL,
                                size_t& writeR) {
  StoreLeftRight(d, st, v0, pivot, keys, writeL, writeR);
  StoreLeftRight(d, st, v1, pivot, keys, writeL, writeR);
  StoreLeftRight(d, st, v2, pivot, keys, writeL, writeR);
  StoreLeftRight(d, st, v3, pivot, keys, writeL, writeR);
}

// Moves "<= pivot" keys to the front, and others to the back. pivot is
// broadcasted. Time-critical!
//
// Aligned loads do not seem to be worthwhile (not bottlenecked by load ports).
template <class D, class Traits, typename T = TFromD<D> >
HWY_NOINLINE size_t Partition(D d, Traits st, T* HWY_RESTRICT keys, size_t left,
                              size_t right, const Vec<D> pivot,
                              T* HWY_RESTRICT buf) {
  using V = decltype(Zero(d));
  const size_t N = Lanes(d);

  // StoreLeftRight will CompressBlendedStore ending at `writeR`. Unless all
  // lanes happen to be in the right-side partition, this will overrun `keys`,
  // which triggers asan errors. Avoid by special-casing the last vector.
  HWY_DASSERT(right - left > 2 * N);  // ensured by HandleSpecialCases
  right -= N;
  const size_t last = right;
  const V vlast = LoadU(d, keys + last);

  PartitionToMultipleOfUnroll(d, st, keys, left, right, pivot, buf);
  constexpr size_t kUnroll = Constants::kPartitionUnroll;

  // Invariant: [left, writeL) and [writeR, right) are already partitioned.
  size_t writeL = left;
  size_t writeR = right;

  const size_t num = right - left;
  // Cannot load if there were fewer than 2 * kUnroll * N.
  if (HWY_LIKELY(num != 0)) {
    HWY_DASSERT(num >= 2 * kUnroll * N);
    HWY_DASSERT((num & (kUnroll * N - 1)) == 0);

    // Make space for writing in-place by reading from left and right.
    const V vL0 = LoadU(d, keys + left + 0 * N);
    const V vL1 = LoadU(d, keys + left + 1 * N);
    const V vL2 = LoadU(d, keys + left + 2 * N);
    const V vL3 = LoadU(d, keys + left + 3 * N);
    left += kUnroll * N;
    right -= kUnroll * N;
    const V vR0 = LoadU(d, keys + right + 0 * N);
    const V vR1 = LoadU(d, keys + right + 1 * N);
    const V vR2 = LoadU(d, keys + right + 2 * N);
    const V vR3 = LoadU(d, keys + right + 3 * N);

    // The left/right updates may consume all inputs, so check before the loop.
    while (left != right) {
      V v0, v1, v2, v3;

      // Free up capacity for writing by loading from the side that has less.
      // Data-dependent but branching is faster than forcing branch-free.
      const size_t capacityL = left - writeL;
      const size_t capacityR = writeR - right;
      HWY_DASSERT(capacityL <= num && capacityR <= num);  // >= 0
      if (capacityR < capacityL) {
        right -= kUnroll * N;
        v0 = LoadU(d, keys + right + 0 * N);
        v1 = LoadU(d, keys + right + 1 * N);
        v2 = LoadU(d, keys + right + 2 * N);
        v3 = LoadU(d, keys + right + 3 * N);
        hwy::Prefetch(keys + right - 3 * kUnroll * N);
      } else {
        v0 = LoadU(d, keys + left + 0 * N);
        v1 = LoadU(d, keys + left + 1 * N);
        v2 = LoadU(d, keys + left + 2 * N);
        v3 = LoadU(d, keys + left + 3 * N);
        left += kUnroll * N;
        hwy::Prefetch(keys + left + 3 * kUnroll * N);
      }

      StoreLeftRight4(d, st, v0, v1, v2, v3, pivot, keys, writeL, writeR);
    }

    // Now finish writing the initial left/right to the middle.
    StoreLeftRight4(d, st, vL0, vL1, vL2, vL3, pivot, keys, writeL, writeR);
    StoreLeftRight4(d, st, vR0, vR1, vR2, vR3, pivot, keys, writeL, writeR);
  }

  // We have partitioned [left, right) such that writeL is the boundary.
  HWY_DASSERT(writeL == writeR);
  // Make space for inserting vlast: move up to N of the first right-side keys
  // into the unused space starting at last. If we have fewer, ensure they are
  // the last items in that vector by subtracting from the *load* address,
  // which is safe because we have at least two vectors (checked above).
  const size_t totalR = last - writeL;
  const size_t startR = totalR < N ? writeL + totalR - N : writeL;
  StoreU(LoadU(d, keys + startR), d, keys + last);

  // Partition vlast: write L, then R, into the single-vector gap at writeL.
  const auto comp = st.Compare(d, pivot, vlast);
  writeL += CompressBlendedStore(vlast, Not(comp), d, keys + writeL);
  (void)CompressBlendedStore(vlast, comp, d, keys + writeL);

  return writeL;
}

// ------------------------------ Pivot

template <class Traits, class V>
HWY_INLINE V MedianOf3(Traits st, V v0, V v1, V v2) {
  const DFromV<V> d;
  // Slightly faster for 128-bit, apparently because not serially dependent.
  if (st.Is128()) {
    // Median = XOR-sum 'minus' the first and last. Calling First twice is
    // slightly faster than Compare + 2 IfThenElse or even IfThenElse + XOR.
    const auto sum = Xor(Xor(v0, v1), v2);
    const auto first = st.First(d, st.First(d, v0, v1), v2);
    const auto last = st.Last(d, st.Last(d, v0, v1), v2);
    return Xor(Xor(sum, first), last);
  }
  st.Sort2(d, v0, v2);
  v1 = st.Last(d, v0, v1);
  v1 = st.First(d, v1, v2);
  return v1;
}

// Replaces triplets with their median and recurses until less than 3 keys
// remain. Ignores leftover values (non-whole triplets)!
template <class D, class Traits, typename T = TFromD<D> >
Vec<D> RecursiveMedianOf3(D d, Traits st, T* HWY_RESTRICT keys, size_t num,
                          T* HWY_RESTRICT buf) {
  const size_t N = Lanes(d);
  constexpr size_t N1 = st.LanesPerKey();

  if (num < 3 * N1) return st.SetKey(d, keys);

  size_t read = 0;
  size_t written = 0;

  // Triplets of vectors
  for (; read + 3 * N <= num; read += 3 * N) {
    const auto v0 = Load(d, keys + read + 0 * N);
    const auto v1 = Load(d, keys + read + 1 * N);
    const auto v2 = Load(d, keys + read + 2 * N);
    Store(MedianOf3(st, v0, v1, v2), d, buf + written);
    written += N;
  }

  // Triplets of keys
  for (; read + 3 * N1 <= num; read += 3 * N1) {
    const auto v0 = st.SetKey(d, keys + read + 0 * N1);
    const auto v1 = st.SetKey(d, keys + read + 1 * N1);
    const auto v2 = st.SetKey(d, keys + read + 2 * N1);
    StoreU(MedianOf3(st, v0, v1, v2), d, buf + written);
    written += N1;
  }

  // Tail recursion; swap buffers
  return RecursiveMedianOf3(d, st, buf, written, keys);
}

#if VQSORT_SECURE_RNG
using Generator = absl::BitGen;
#else
static inline void Fill24Bytes(const void* seed_heap, size_t seed_num,
                               void* bytes) {
#if VQSORT_SECURE_SEED == 1
  // May block if urandom is not yet initialized.
  const ssize_t ret = getrandom(bytes, 24, /*flags=*/0);
  if (ret == 24) return;
#elif VQSORT_SECURE_SEED == 2
  HCRYPTPROV hProvider{};
  if (CryptAcquireContextA(&hProvider, nullptr, nullptr, PROV_RSA_FULL,
                           CRYPT_VERIFYCONTEXT)) {
    const BOOL ok =
        CryptGenRandom(hProvider, 24, reinterpret_cast<BYTE*>(bytes));
    CryptReleaseContext(hProvider, 0);
    if (ok) return;
  }
#endif

  // VQSORT_SECURE_SEED == 0, or one of the above failed. Get some entropy from
  // stack/heap/code addresses and the clock() timer.
  uint64_t* words = reinterpret_cast<uint64_t*>(bytes);
  uint64_t** seed_stack = &words;
  void (*seed_code)(const void*, size_t, void*) = &Fill24Bytes;
  const uintptr_t bits_stack = reinterpret_cast<uintptr_t>(seed_stack);
  const uintptr_t bits_heap = reinterpret_cast<uintptr_t>(seed_heap);
  const uintptr_t bits_code = reinterpret_cast<uintptr_t>(seed_code);
  const uint64_t bits_time = clock();
  words[0] = bits_stack ^ bits_time;
  words[1] = bits_heap ^ bits_time;
  words[2] = bits_code ^ bits_time;
}

// Based on https://github.com/numpy/numpy/issues/16313#issuecomment-641897028
#pragma pack(push, 1)
class Generator {
 public:
  Generator(const void* heap, size_t num) {
    Fill24Bytes(heap, num, &a_);
    k_ = 1;  // stream index: must be odd
  }

  uint64_t operator()() {
    const uint64_t b = b_;
    w_ += k_;
    const uint64_t next = a_ ^ w_;
    a_ = (b + (b << 3)) ^ (b >> 11);
    const uint64_t rot = (b << 24) | (b >> 40);
    b_ = rot + next;
    return next;
  }

 private:
  uint64_t a_;
  uint64_t b_;
  uint64_t w_;
  uint64_t k_;  // increment
};
#pragma pack(pop)

#endif  // !VQSORT_SECURE_RNG

// Returns slightly biased random index of a chunk in [0, num_chunks).
// See https://www.pcg-random.org/posts/bounded-rands.html.
HWY_INLINE size_t RandomChunkIndex(const uint32_t num_chunks, uint32_t bits) {
  const uint64_t chunk_index = (static_cast<uint64_t>(bits) * num_chunks) >> 32;
  HWY_DASSERT(chunk_index < num_chunks);
  return chunk_index;
}

template <class D, class Traits, typename T = TFromD<D> >
HWY_NOINLINE Vec<D> ChoosePivot(D d, Traits st, T* HWY_RESTRICT keys,
                                const size_t begin, const size_t end,
                                T* HWY_RESTRICT buf, Generator& rng) {
  using V = decltype(Zero(d));
  const size_t N = Lanes(d);

  // Power of two
  const size_t lanes_per_chunk = Constants::LanesPerChunk(sizeof(T), N);

  keys += begin;
  size_t num = end - begin;

  // Align start of keys to chunks. We always have at least 2 chunks because the
  // base case would have handled anything up to 16 vectors, i.e. >= 4 chunks.
  HWY_DASSERT(num >= 2 * lanes_per_chunk);
  const size_t misalign =
      (reinterpret_cast<uintptr_t>(keys) / sizeof(T)) & (lanes_per_chunk - 1);
  if (misalign != 0) {
    const size_t consume = lanes_per_chunk - misalign;
    keys += consume;
    num -= consume;
  }

  // Generate enough random bits for 9 uint32
  uint64_t* bits64 = reinterpret_cast<uint64_t*>(buf);
  for (size_t i = 0; i < 5; ++i) {
    bits64[i] = rng();
  }
  const uint32_t* bits = reinterpret_cast<const uint32_t*>(buf);

  const uint32_t lpc32 = static_cast<uint32_t>(lanes_per_chunk);
  // Avoid division
  const size_t log2_lpc = Num0BitsBelowLS1Bit_Nonzero32(lpc32);
  const size_t num_chunks64 = num >> log2_lpc;
  // Clamp to uint32 for RandomChunkIndex
  const uint32_t num_chunks =
      static_cast<uint32_t>(HWY_MIN(num_chunks64, 0xFFFFFFFFull));

  const size_t offset0 = RandomChunkIndex(num_chunks, bits[0]) << log2_lpc;
  const size_t offset1 = RandomChunkIndex(num_chunks, bits[1]) << log2_lpc;
  const size_t offset2 = RandomChunkIndex(num_chunks, bits[2]) << log2_lpc;
  const size_t offset3 = RandomChunkIndex(num_chunks, bits[3]) << log2_lpc;
  const size_t offset4 = RandomChunkIndex(num_chunks, bits[4]) << log2_lpc;
  const size_t offset5 = RandomChunkIndex(num_chunks, bits[5]) << log2_lpc;
  const size_t offset6 = RandomChunkIndex(num_chunks, bits[6]) << log2_lpc;
  const size_t offset7 = RandomChunkIndex(num_chunks, bits[7]) << log2_lpc;
  const size_t offset8 = RandomChunkIndex(num_chunks, bits[8]) << log2_lpc;
  for (size_t i = 0; i < lanes_per_chunk; i += N) {
    const V v0 = Load(d, keys + offset0 + i);
    const V v1 = Load(d, keys + offset1 + i);
    const V v2 = Load(d, keys + offset2 + i);
    const V medians0 = MedianOf3(st, v0, v1, v2);
    Store(medians0, d, buf + i);

    const V v3 = Load(d, keys + offset3 + i);
    const V v4 = Load(d, keys + offset4 + i);
    const V v5 = Load(d, keys + offset5 + i);
    const V medians1 = MedianOf3(st, v3, v4, v5);
    Store(medians1, d, buf + i + lanes_per_chunk);

    const V v6 = Load(d, keys + offset6 + i);
    const V v7 = Load(d, keys + offset7 + i);
    const V v8 = Load(d, keys + offset8 + i);
    const V medians2 = MedianOf3(st, v6, v7, v8);
    Store(medians2, d, buf + i + lanes_per_chunk * 2);
  }

  return RecursiveMedianOf3(d, st, buf, 3 * lanes_per_chunk,
                            buf + 3 * lanes_per_chunk);
}

// Compute exact min/max to detect all-equal partitions. Only called after a
// degenerate Partition (none in the right partition).
template <class D, class Traits, typename T = TFromD<D>, class V = Vec<D> >
HWY_NOINLINE void ScanMinMax(D d, Traits st, const T* HWY_RESTRICT keys,
                             size_t num, T* HWY_RESTRICT buf, V& first,
                             V& last) {
  const size_t N = Lanes(d);

  first = st.LastValue(d);
  last = st.FirstValue(d);

  size_t i = 0;
  for (; i + N <= num; i += N) {
    const V v = LoadU(d, keys + i);
    first = st.First(d, v, first);
    last = st.Last(d, v, last);
  }
  if (HWY_LIKELY(i != num)) {
    HWY_DASSERT(num >= N);  // See HandleSpecialCases
    const V v = LoadU(d, keys + num - N);
    first = st.First(d, v, first);
    last = st.Last(d, v, last);
  }

  first = st.FirstOfLanes(d, first, buf);
  last = st.LastOfLanes(d, last, buf);
}

template <class D, class Traits, typename T = TFromD<D> >
void Recurse(D d, Traits st, T* HWY_RESTRICT keys, const size_t begin,
             const size_t end, const Vec<D> pivot, T* HWY_RESTRICT buf,
             Generator& rng, size_t remaining_levels) {
  HWY_DASSERT(begin + 1 < end);
  const size_t num = end - begin;  // >= 2

  // Too many degenerate partitions. This is extremely unlikely to happen
  // because we select pivots from large (though still O(1)) samples.
  if (HWY_UNLIKELY(remaining_levels == 0)) {
    HeapSort(st, keys + begin, num);  // Slow but N*logN.
    return;
  }

  const size_t base_case_num = Constants::BaseCaseNum(Lanes(d));
  const size_t bound = Partition(d, st, keys, begin, end, pivot, buf);

  const ptrdiff_t num_left =
      static_cast<ptrdiff_t>(bound) - static_cast<ptrdiff_t>(begin);
  const ptrdiff_t num_right =
      static_cast<ptrdiff_t>(end) - static_cast<ptrdiff_t>(bound);

  // Check for degenerate partitions (i.e. Partition did not move any keys):
  if (HWY_UNLIKELY(num_right == 0)) {
    // Because the pivot is one of the keys, it must have been equal to the
    // first or last key in sort order. Scan for the actual min/max:
    // passing the current pivot as the new bound is insufficient because one of
    // the partitions might not actually include that key.
    Vec<D> first, last;
    ScanMinMax(d, st, keys + begin, num, buf, first, last);
    if (AllTrue(d, Eq(first, last))) return;

    // Separate recursion to make sure that we don't pick `last` as the
    // pivot - that would again lead to a degenerate partition.
    Recurse(d, st, keys, begin, end, first, buf, rng, remaining_levels - 1);
    return;
  }

  if (HWY_UNLIKELY(num_left <= base_case_num)) {
    BaseCase(d, st, keys + begin, num_left, buf);
  } else {
    const Vec<D> next_pivot = ChoosePivot(d, st, keys, begin, bound, buf, rng);
    Recurse(d, st, keys, begin, bound, next_pivot, buf, rng,
            remaining_levels - 1);
  }
  if (HWY_UNLIKELY(num_right <= base_case_num)) {
    BaseCase(d, st, keys + bound, num_right, buf);
  } else {
    const Vec<D> next_pivot = ChoosePivot(d, st, keys, bound, end, buf, rng);
    Recurse(d, st, keys, bound, end, next_pivot, buf, rng,
            remaining_levels - 1);
  }
}

// Returns true if sorting is finished.
template <class D, class Traits, typename T = TFromD<D> >
bool HandleSpecialCases(D d, Traits st, T* HWY_RESTRICT keys, size_t num,
                        T* HWY_RESTRICT buf) {
  const size_t N = Lanes(d);
  const size_t base_case_num = Constants::BaseCaseNum(N);

  // 128-bit keys require vectors with at least two u64 lanes, which is always
  // the case unless `d` requests partial vectors (e.g. fraction = 1/2) AND the
  // hardware vector width is less than 128bit / fraction.
  const bool partial_128 = N < 2 && st.Is128();
  // Partition assumes its input is at least two vectors. If vectors are huge,
  // base_case_num may actually be smaller. If so, which is only possible on
  // RVV, pass a capped or partial d (LMUL < 1).
  constexpr bool kPotentiallyHuge =
      HWY_MAX_BYTES / sizeof(T) > Constants::kMaxRows * Constants::kMaxCols;
  const bool huge_vec = kPotentiallyHuge && (2 * N > base_case_num);
  if (partial_128 || huge_vec) {
    // PERFORMANCE WARNING: falling back to HeapSort.
    HeapSort(st, keys, num);
    return true;
  }

  // Small arrays: use sorting network, no need for other checks.
  if (HWY_UNLIKELY(num <= base_case_num)) {
    BaseCase(d, st, keys, num, buf);
    return true;
  }

  // We could also check for already sorted/reverse/equal, but that's probably
  // counterproductive if vqsort is used as a base case.

  return false;  // not finished sorting
}

#endif  // HWY_TARGET != HWY_SCALAR
}  // namespace detail

// Sorts `keys[0..num-1]` according to the order defined by `st.Compare`.
// In-place i.e. O(1) additional storage. Worst-case N*logN comparisons.
// Non-stable (order of equal keys may change), except for the common case where
// the upper bits of T are the key, and the lower bits are a sequential or at
// least unique ID.
// There is no upper limit on `num`, but note that pivots may be chosen by
// sampling only from the first 256 GiB.
//
// `d` is typically SortTag<T> (chooses between full and partial vectors).
// `st` is SharedTraits<{Lane|Block}Traits<Order*>>. This abstraction layer
//   bridges differences in sort order and single-lane vs 128-bit keys.
template <class D, class Traits, typename T = TFromD<D> >
void Sort(D d, Traits st, T* HWY_RESTRICT keys, size_t num,
          T* HWY_RESTRICT buf) {
#if HWY_TARGET == HWY_SCALAR
  // PERFORMANCE WARNING: vqsort is not enabled for the non-SIMD target
  return detail::HeapSort(st, keys, num);
#else
  if (detail::HandleSpecialCases(d, st, keys, num, buf)) return;

  // Pulled out of the recursion so we can special-case degenerate partitions.
  detail::Generator rng(keys, num);
  const Vec<D> pivot = detail::ChoosePivot(d, st, keys, 0, num, buf, rng);

  // Introspection: switch to worst-case N*logN heapsort after this many.
  const size_t max_levels = 2 * hwy::CeilLog2(num) + 4;

  detail::Recurse(d, st, keys, 0, num, pivot, buf, rng, max_levels);
#endif
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_SORT_VQSORT_INL_H_
