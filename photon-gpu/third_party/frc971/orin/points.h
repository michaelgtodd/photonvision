#pragma once

#include <stdint.h>

#include <cuda/std/tuple>
#include <iomanip>
#include <ostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace frc971::apriltag {

// Class to hold the 2 adjacent blob IDs, a point in decimated image space, the
// half pixel offset, and the gradient.
//
// rep0 and rep1 are the two blob ids, and are each allocated 19 bits.  They
// are the dense numbers GpuDetector gives the blobs of 25 pixels or more (the
// only ones that make boundary points), not the union-find pixel indices,
// which need 22 bits at full resolution.
// point is the base point and is allocated 11 bits for x and 11 bits for y,
// enough for a 2048x2048 image undecimated.
// dx and dy are allocated 2 bits, and can only take on set values.
// black_to_white captures the direction of the gradient in 1 bit.
//
// This adds up to 63 bits so we can load this with one big load.
struct QuadBoundaryPoint {
  static constexpr size_t kRepEndBit = 25;
  static constexpr size_t kBitsInKey = 64;

  __forceinline__ __host__ __device__ QuadBoundaryPoint() : key(0) {}

  // Sets rep0, the 0th blob id.  This only respects the bottom 19 bits.
  __forceinline__ __host__ __device__ void set_rep0(uint32_t rep0) {
    key = (key & 0xfffff00001ffffffull) |
          (static_cast<uint64_t>(rep0 & 0x7ffff) << 25);
  }
  // Returns rep0.
  __forceinline__ __host__ __device__ uint32_t rep0() const {
    return ((key >> 25) & 0x7ffff);
  }

  // Sets rep1, the 1st blob id.  This only respects the bottom 19 bits.
  __forceinline__ __host__ __device__ void set_rep1(uint32_t rep1) {
    key = (key & 0x80000fffffffffffull) |
          (static_cast<uint64_t>(rep1 & 0x7ffff) << 44);
  }
  // Returns rep1.
  __forceinline__ __host__ __device__ uint32_t rep1() const {
    return ((key >> 44) & 0x7ffff);
  }

  // Returns both rep0 and rep1 concatenated into a single 38 bit number.
  __forceinline__ __host__ __device__ uint64_t rep01() const {
    return ((key >> 25) & 0x3fffffffffull);
  }

  // Returns all the bits used to hold position and gradient information.
  __forceinline__ __host__ __device__ uint32_t point_bits() const {
    return key & 0x1ffffff;
  }

  // Sets the 11 bit x and y.
  __forceinline__ __host__ __device__ void set_base_xy(uint32_t x, uint32_t y) {
    key = (key & 0xfffffffffe000007ull) |
          (static_cast<uint64_t>(x & 0x7ff) << 14) |
          (static_cast<uint64_t>(y & 0x7ff) << 3);
  }

  // Returns the base 11 bit x and y.
  __forceinline__ __host__ __device__ uint32_t base_x() const {
    return ((key >> 14) & 0x7ff);
  }
  __forceinline__ __host__ __device__ uint32_t base_y() const {
    return ((key >> 3) & 0x7ff);
  }

  // Sets dxy, the integer representing which of the 4 search directions we
  // went.
  __forceinline__ __host__ __device__ void set_dxy(uint64_t dxy) {
    key = (key & 0xfffffffffffffffcull) | (static_cast<uint64_t>(dxy & 0x3));
  }

  // Returns the change in x derived from the search direction.
  __forceinline__ __host__ __device__ int32_t dx() const {
    switch (key & 0x3) {
      case 0:
        return 1;
      case 1:
        return 1;
      case 2:
        return 0;
      case 3:
        return -1;
    }
    return 0;
  }

  // Returns the change in y derived from the search direction.
  __forceinline__ __host__ __device__ int32_t dy() const {
    switch (key & 0x3) {
      case 0:
        return 0;
      case 1:
      case 2:
      case 3:
        return 1;
    }
    return 0;
  }

  // Returns the un-decimated x and y positions.
  __forceinline__ __host__ __device__ uint32_t x() const {
    return static_cast<int32_t>(base_x() * 2) + dx();
  }
  __forceinline__ __host__ __device__ uint32_t y() const {
    return static_cast<int32_t>(base_y() * 2) + dy();
  }

  // Returns the gradient that this point represents, taking into account which
  // direction the color transitioned.
  __forceinline__ __host__ __device__ int8_t gx() const {
    return black_to_white() ? dx() : -dx();
  }
  __forceinline__ __host__ __device__ int8_t gy() const {
    return black_to_white() ? dy() : -dy();
  }

  // Returns the black to white or white to black bit.
  __forceinline__ __host__ __device__ void set_black_to_white(
      bool black_to_white) {
    key = (key & 0xfffffffffffffffbull) |
          (static_cast<uint64_t>(black_to_white) << 2);
  }
  __forceinline__ __host__ __device__ bool black_to_white() const {
    return (key & 0x4) != 0;
  }

  // Various operators to make it easy to compare points.
  __forceinline__ __host__ __device__ bool operator!=(
      const QuadBoundaryPoint other) const {
    return other.key != key;
  }
  __forceinline__ __host__ __device__ bool operator==(
      const QuadBoundaryPoint other) const {
    return other.key == key;
  }
  __forceinline__ __host__ __device__ bool operator<(
      const QuadBoundaryPoint other) const {
    return key < other.key;
  }

  // Returns true if this point has been set.  Zero is reserved for "invalid"
  __forceinline__ __host__ __device__ bool nonzero() const {
    return key != 0ull;
  }

  // Returns true if this point is about the other point.
  bool near(QuadBoundaryPoint other) const { return other == *this; }

  // The key.  This shouldn't be parsed directly.
  uint64_t key;
};

std::ostream &operator<<(std::ostream &os, const QuadBoundaryPoint &point);

// Holds a compacted blob index, the angle to the X axis from the center of the
// blob, and the coordinate of the point.
//
// The blob index is 16 bits, the angle is 23 bits, and the point is 25 bits.
struct IndexPoint {
  // Max number of blob IDs we can hold.  The blob index is 16 bits and
  // Peak::kNoPeak() (0xffff) is reserved.
  static constexpr size_t kMaxBlobs = 0xffff;

  static constexpr size_t kRepEndBit = 25;
  static constexpr size_t kBitsInKey = 64;

  __forceinline__ __host__ __device__ IndexPoint() : key(0) {}

  // Constructor to build a point with just the blob index, and point bits.  The
  // point bits should be grabbed from a QuadBoundaryPoint rather than built up
  // by hand.
  __forceinline__ __host__ __device__ IndexPoint(uint32_t blob_index,
                                                 uint32_t point_bits)
      : key((static_cast<uint64_t>(blob_index & 0xffff) << 48) |
            (static_cast<uint64_t>(point_bits & 0x1ffffff))) {}

  // Sets and gets the 16 bit blob index.
  __forceinline__ __host__ __device__ void set_blob_index(uint32_t blob_index) {
    key = (key & 0x0000ffffffffffffull) |
          (static_cast<uint64_t>(blob_index & 0xffff) << 48);
  }
  __forceinline__ __host__ __device__ uint32_t blob_index() const {
    return ((key >> 48) & 0xffff);
  }

  // Sets and gets the 23 bit angle.
  __forceinline__ __host__ __device__ void set_theta(uint32_t theta) {
    key = (key & 0xffff000001ffffffull) |
          (static_cast<uint64_t>(theta & 0x7fffff) << 25);
  }
  __forceinline__ __host__ __device__ uint32_t theta() const {
    return ((key >> 25) & 0x7fffff);
  }

  // See QuadBoundaryPoint for a description of the rest of these.
  // Sets the 11 bit x and y.
  __forceinline__ __host__ __device__ void set_base_xy(uint32_t x, uint32_t y) {
    key = (key & 0xfffffffffe000007ull) |
          (static_cast<uint64_t>(x & 0x7ff) << 14) |
          (static_cast<uint64_t>(y & 0x7ff) << 3);
  }

  __forceinline__ __host__ __device__ uint32_t base_x() const {
    return ((key >> 14) & 0x7ff);
  }

  __forceinline__ __host__ __device__ uint32_t base_y() const {
    return ((key >> 3) & 0x7ff);
  }

  __forceinline__ __host__ __device__ void set_dxy(uint64_t dxy) {
    key = (key & 0xfffffffffffffffcull) | (static_cast<uint64_t>(dxy & 0x3));
  }

  __forceinline__ __host__ __device__ int32_t dx() const {
    switch (key & 0x3) {
      case 0:
        return 1;
      case 1:
        return 1;
      case 2:
        return 0;
      case 3:
        return -1;
    }
    return 0;
  }

  __forceinline__ __host__ __device__ int32_t dy() const {
    switch (key & 0x3) {
      case 0:
        return 0;
      case 1:
      case 2:
      case 3:
        return 1;
    }
    return 0;
  }

  __forceinline__ __host__ __device__ uint32_t x() const {
    return static_cast<int32_t>(base_x() * 2) + dx();
  }
  __forceinline__ __host__ __device__ uint32_t y() const {
    return static_cast<int32_t>(base_y() * 2) + dy();
  }

  __forceinline__ __host__ __device__ int8_t gx() const {
    return black_to_white() ? dx() : -dx();
  }
  __forceinline__ __host__ __device__ int8_t gy() const {
    return black_to_white() ? dy() : -dy();
  }

  __forceinline__ __host__ __device__ uint32_t point_bits() const {
    return key & 0x1ffffff;
  }

  __forceinline__ __host__ __device__ void set_black_to_white(
      bool black_to_white) {
    key = (key & 0xfffffffffffffffbull) |
          (static_cast<uint64_t>(black_to_white) << 2);
  }
  __forceinline__ __host__ __device__ bool black_to_white() const {
    return (key & 0x4) != 0;
  }

  // The key.  This shouldn't be parsed directly.
  uint64_t key;
};

std::ostream &operator<<(std::ostream &os, const IndexPoint &point);

// Decomposer for sorting which just returns the key.
struct QuadBoundaryPointDecomposer {
  __host__ __device__ ::cuda::std::tuple<uint64_t &> operator()(
      QuadBoundaryPoint &key) const {
    return {key.key};
  }
};

// Decomposer for sorting which just returns the key.
struct QuadIndexPointDecomposer {
  __host__ __device__ ::cuda::std::tuple<uint64_t &> operator()(
      IndexPoint &key) const {
    return {key.key};
  }
};

}  // namespace frc971::apriltag


