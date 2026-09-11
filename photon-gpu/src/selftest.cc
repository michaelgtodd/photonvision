// photongpu-selftest [width height [frame.raw]]: runs the library PhotonVision
// loads through one detector and reports CUDA check failures. Exit 0 = healthy.
// frame.raw is an 8-bit grey image of width x height (default: a synthetic
// square, which contains no tag).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
extern "C" long photongpu_selftest(int width, int height, int decimate, const uint8_t *frame, int *ndet);
int main(int argc, char **argv) {
  int w = argc > 1 ? atoi(argv[1]) : 1920, h = argc > 2 ? atoi(argv[2]) : 1200;
  std::vector<uint8_t> frame;
  if (argc > 3) {
    frame.resize(static_cast<size_t>(w) * h);
    FILE *f = fopen(argv[3], "rb");
    if (!f || fread(frame.data(), 1, frame.size(), f) != frame.size()) { perror(argv[3]); return 2; }
    fclose(f);
  }
  int ndet = 0;
  long failures = photongpu_selftest(w, h, 2, frame.empty() ? nullptr : frame.data(), &ndet);
  printf("photongpu selftest %dx%d decimate 2: %d detections, %ld CUDA check failures\n", w, h, ndet, failures);
  return failures == 0 ? 0 : 1;
}
