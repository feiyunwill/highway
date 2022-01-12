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

#include "hwy/aligned_allocator.h"
#include "hwy/contrib/sort/vqsort.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy/contrib/sort/vqsort.cc"
#include "hwy/foreach_target.h"

// After foreach_target
#include "hwy/tests/include_farm_sve.h"
// ^ must come before highway.h.

#include "hwy/contrib/sort/shared-inl.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

size_t VectorSize() { return Lanes(ScalableTag<uint8_t, 3>()); }

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_EXPORT(VectorSize);

HWY_INLINE size_t PivotBufNum(size_t sizeof_t, size_t N) {
  // 3 chunks of medians, 1 chunk of median medians plus two padding vectors.
  const size_t lpc = SortConstants::LanesPerChunk(sizeof_t, N);
  return (3 + 1) * lpc + 2 * N;
}

}  // namespace

Sorter::Sorter() {
  // Determine the largest buffer size required for any type by trying them all.
  // (The capping of N in BaseCaseNum means that smaller N but larger sizeof_t
  // may require a larger buffer.)
  const size_t vector_size = HWY_DYNAMIC_DISPATCH(VectorSize)();
  size_t max_bytes = 0;
  for (size_t sizeof_t :
       {sizeof(uint16_t), sizeof(uint32_t), sizeof(uint64_t)}) {
    const size_t N = vector_size / sizeof_t;
    // One extra for padding plus another for full-vector loads.
    const size_t base_case = SortConstants::BaseCaseNum(N) + 2 * N;
    const size_t partition_num = SortConstants::PartitionBufNum(N);
    const size_t buf_lanes =
        HWY_MAX(base_case, HWY_MAX(partition_num, PivotBufNum(sizeof_t, N)));
    max_bytes = HWY_MAX(max_bytes, buf_lanes * sizeof_t);
  }

  ptr_ = hwy::AllocateAlignedBytes(max_bytes, nullptr, nullptr);

  // Prevent msan errors by initializing.
  memset(ptr_, 0, max_bytes);
}

void Sorter::Delete() {
  FreeAlignedBytes(ptr_, nullptr, nullptr);
  ptr_ = nullptr;
}

}  // namespace hwy
#endif  // HWY_ONCE
