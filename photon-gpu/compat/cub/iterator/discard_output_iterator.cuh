// CUDA 13 / CCCL 3 compatibility: cub::DiscardOutputIterator became
// thrust::discard_iterator (see transform_input_iterator.cuh).
#pragma once
#include <thrust/iterator/discard_iterator.h>
namespace cub {
template <typename OffsetT = std::ptrdiff_t>
using DiscardOutputIterator = thrust::discard_iterator<>;
}
