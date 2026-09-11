#include "frc971/orin/cuda.h"

#include <cstdlib>

namespace frc971::apriltag {

size_t overall_memory = 0;

void CheckAndSynchronize(std::string_view message) {
  CHECK_CUDA(cudaDeviceSynchronize()) << message;
  CHECK_CUDA(cudaGetLastError()) << message;
}

// photon-gpu: the upstream --sync flag became the PHOTONGPU_SYNC environment
// variable (any value): synchronise and check after every stage, to find
// which one fails. Slow; for debugging only.
static const bool kSyncEveryStage = getenv("PHOTONGPU_SYNC") != nullptr;

void MaybeCheckAndSynchronize() {
  if (kSyncEveryStage) CheckAndSynchronize();
}

void MaybeCheckAndSynchronize(std::string_view message) {
  if (kSyncEveryStage) CheckAndSynchronize(message);
}

}  // namespace frc971::apriltag

// photon-gpu: see cuda.h
namespace frc971::apriltag {
std::atomic<long> cuda_check_failures{0};
void ReportCudaCheckFailure(const char *condition, cudaError_t err, const std::string &message) {
  long n = ++cuda_check_failures;
  if (n <= 5 || kSyncEveryStage) {
    std::cerr << "photon-gpu: CUDA check failed: " << condition << " (" << cudaGetErrorString(err) << ") "
              << message << (n == 5 && !kSyncEveryStage ? " -- further failures counted silently" : "")
              << std::endl;
  }
}
}  // namespace frc971::apriltag
