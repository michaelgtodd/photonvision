// photongpu-selftest [width height [frame.raw [frames]]]: runs the library
// PhotonVision loads through one detector and reports CUDA check failures
// and per-frame heap growth. Exit 0 = healthy. frame.raw is an 8-bit grey
// image of width x height (default, or "-": a synthetic square, which
// contains no tag); frames defaults to 300.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
extern "C" long photongpu_selftest(int width, int height, int decimate, const uint8_t *frame, int frames,
                                   int *ndet, long *heap_growth);
int main(int argc, char **argv) {
  int w = argc > 1 ? atoi(argv[1]) : 1920, h = argc > 2 ? atoi(argv[2]) : 1200;
  int frames = argc > 4 ? atoi(argv[4]) : 300;
  std::vector<uint8_t> frame;
  if (argc > 3 && strcmp(argv[3], "-") != 0) {
    frame.resize(static_cast<size_t>(w) * h);
    FILE *f = fopen(argv[3], "rb");
    if (!f || fread(frame.data(), 1, frame.size(), f) != frame.size()) { perror(argv[3]); return 2; }
    fclose(f);
  }
  int ndet = 0;
  long growth = 0;
  long failures = photongpu_selftest(w, h, 2, frame.empty() ? nullptr : frame.data(), frames, &ndet, &growth);
  // a few KB of one-off allocations is normal; a leak grows with every frame
  bool leak = growth > 64 * 1024 + 16L * frames;
  printf("photongpu selftest %dx%d decimate 2, %d frames: %d detections, %ld CUDA check failures, heap %+ld KB%s\n",
         w, h, frames, ndet, failures, growth / 1024, leak ? " (LEAK)" : "");
  return (failures == 0 && !leak) ? 0 : 1;
}
