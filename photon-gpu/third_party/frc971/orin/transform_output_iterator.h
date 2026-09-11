#pragma once

namespace frc971::apriltag {

// template class that allows conversions at the output of a cub algorithm
template <typename InputType, typename OutputType, typename ConversionOp,
          typename OffsetT = ptrdiff_t>
class TransformOutputIterator {
 private:
  // proxy object to be able to convert when assigning value
  struct Reference {
    OutputType *ptr;
    ConversionOp convert_op;
    __host__ __device__ Reference(OutputType *ptr, ConversionOp convert_op)
        : ptr(ptr), convert_op(convert_op) {}
    __host__ __device__ Reference operator=(InputType val) {
      *ptr = convert_op(val);
      return *this;
    }
  };

 public:
  // typedefs may not be neeeded for iterator to work but is here to maintain
  // similarity to cub's CacheModifiedOutputIterator
  typedef TransformOutputIterator self_type;
  typedef OffsetT difference_type;
  typedef void value_type;
  typedef void *pointer;
  typedef Reference reference;

  TransformOutputIterator(OutputType *ptr, const ConversionOp convert_op)
      : convert_op(convert_op), ptr(ptr) {}

  // postfix addition
  __host__ __device__ __forceinline__ self_type operator++(int) {
    self_type retval = *this;
    ptr++;
    return retval;
  }

  // prefix addition
  __host__ __device__ __forceinline__ self_type operator++() {
    ptr++;
    return *this;
  }

  // postfix subtraction
  __host__ __device__ __forceinline__ self_type operator--(int) {
    self_type retval = *this;
    ptr--;
    return retval;
  }

  // prefix subtraction
  __host__ __device__ __forceinline__ self_type operator--() {
    ptr--;
    return *this;
  }

  // Random access, needed by CCCL 3 (CUDA 13): DeviceSelect now indexes its
  // output iterator with 64-bit offsets (it + n) instead of it[n].
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator+(Distance n) const {
    self_type retval = *this;
    retval.ptr += n;
    return retval;
  }
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type &operator+=(Distance n) {
    ptr += n;
    return *this;
  }
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator-(Distance n) const {
    self_type retval = *this;
    retval.ptr -= n;
    return retval;
  }

  // indirection
  __host__ __device__ __forceinline__ reference operator*() const {
    return Reference(ptr, convert_op);
  }

  // array index
  __host__ __device__ __forceinline__ reference operator[](int num) const {
    return Reference(ptr + num, convert_op);
  }

  // equal to
  __host__ __device__ __forceinline__ bool operator==(
      const TransformOutputIterator &rhs) const {
    return ptr == rhs.ptr;
  }

  // not equal to
  __host__ __device__ __forceinline__ bool operator!=(
      const TransformOutputIterator &rhs) const {
    return ptr != rhs.ptr;
  }

 private:
  const ConversionOp convert_op;
  OutputType *ptr;
};

}  // namespace frc971::apriltag


