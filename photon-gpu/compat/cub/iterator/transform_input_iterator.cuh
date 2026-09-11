// CUDA 13 / CCCL 3 compatibility for Team 971's CUDA AprilTag detector.
//
// CCCL 3.0 removed cub::TransformInputIterator in favour of
// thrust::transform_iterator; the constructor order (iterator, op) is the
// same, only the template parameter order differs. This directory is put on
// the include path ahead of CCCL's so the 971 sources build unchanged.
#pragma once
#include <thrust/iterator/transform_iterator.h>
namespace cub {
template <typename ValueType, typename ConversionOp, typename InputIteratorT>
using TransformInputIterator =
    thrust::transform_iterator<ConversionOp, InputIteratorT, thrust::use_default, ValueType>;
}
