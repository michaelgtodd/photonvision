/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/*
 * Newest-frame capture for 16-bit grey (Y16) V4L2 cameras.
 *
 * cscore's USB camera copies every frame it dequeues into its own image
 * (SourceImpl::PutFrame), whether or not a sink will ever use it. For a
 * 1920x1200 Y16 head that is 4.6 MB per frame; at 117 fps on four heads it
 * cost about 2.2 cores on a Jetson Orin Nano, most of it on frames the
 * pipeline never processed.
 *
 * cscore only dequeues while one of its sinks is enabled, and keeps the
 * device open with its buffers allocated either way (controls keep working
 * through it). So with the camera's sink disabled, this streams on the very
 * file descriptor cscore holds -- no second buffer queue, no second owner --
 * and keeps only the newest filled buffer, requeueing every older one at
 * once without touching its pixels. grab() converts just that newest buffer
 * to 8-bit grey. If cscore reopens the device (a video mode change), the old
 * descriptor fails, and this detaches and re-attaches to the new one.
 */

#include <dirent.h>
#include <fcntl.h>
#include <jni.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <wpi/timestamp.h>

namespace {

int Xioctl(int fd, unsigned long req, void* arg) {
  int r;
  do {
    r = ioctl(fd, req, arg);
  } while (r == -1 && errno == EINTR);
  return r;
}

struct Buffer {
  void* data = nullptr;
  size_t length = 0;
};

struct Grabber {
  std::string device;  // resolved /dev/videoN
  int width = 0;
  int height = 0;
  std::thread thread;
  std::atomic<bool> running{true};

  // Everything below is guarded by mutex.
  std::mutex mutex;
  std::condition_variable cv;
  int fd = -1;
  bool streaming = false;
  std::vector<Buffer> buffers;
  int latest = -1;  // newest filled buffer, held (not queued)
  int64_t latestTimeUs = 0;
  uint64_t latestSeq = 0;
  int inUse = -1;  // buffer grab() is converting
  uint64_t returnedSeq = 0;
  std::string error;
};

int64_t MonotonicUs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000 + ts.tv_nsec / 1000;
}

// cscore's descriptor for the device: one of ours that points at it and has
// buffers allocated.
int FindDeviceFd(const std::string& device) {
  DIR* dir = opendir("/proc/self/fd");
  if (!dir) {
    return -1;
  }
  int found = -1;
  while (struct dirent* e = readdir(dir)) {
    char* end;
    long n = std::strtol(e->d_name, &end, 10);
    if (*end != '\0' || n <= 2) {
      continue;
    }
    char link[64];
    char target[PATH_MAX];
    std::snprintf(link, sizeof(link), "/proc/self/fd/%ld", n);
    ssize_t len = readlink(link, target, sizeof(target) - 1);
    if (len <= 0) {
      continue;
    }
    target[len] = '\0';
    if (device != target) {
      continue;
    }
    struct v4l2_buffer b;
    std::memset(&b, 0, sizeof(b));
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    b.memory = V4L2_MEMORY_MMAP;
    b.index = 0;
    if (Xioctl(static_cast<int>(n), VIDIOC_QUERYBUF, &b) == 0) {
      found = static_cast<int>(n);
      break;
    }
  }
  closedir(dir);
  return found;
}

void Unmap(std::vector<Buffer>& buffers) {
  for (auto& b : buffers) {
    if (b.data) {
      munmap(b.data, b.length);
    }
  }
  buffers.clear();
}

void Detach(Grabber& g, const char* why) {
  std::unique_lock lock{g.mutex};
  // Never unmap a buffer grab() is still reading.
  g.cv.wait(lock, [&] { return g.inUse < 0; });
  if (g.streaming && g.fd >= 0) {
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    Xioctl(g.fd, VIDIOC_STREAMOFF, &type);
  }
  Unmap(g.buffers);
  g.fd = -1;
  g.streaming = false;
  g.latest = -1;
  if (why) {
    g.error = why;
  }
}

bool Attach(Grabber& g) {
  int fd = FindDeviceFd(g.device);
  if (fd < 0) {
    std::scoped_lock lock{g.mutex};
    g.error = "no open descriptor with buffers for " + g.device;
    return false;
  }
  std::vector<Buffer> buffers;
  for (unsigned i = 0;; ++i) {
    struct v4l2_buffer b;
    std::memset(&b, 0, sizeof(b));
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    b.memory = V4L2_MEMORY_MMAP;
    b.index = i;
    if (Xioctl(fd, VIDIOC_QUERYBUF, &b) != 0) {
      break;
    }
    void* p = mmap(nullptr, b.length, PROT_READ, MAP_SHARED, fd, b.m.offset);
    if (p == MAP_FAILED) {
      Unmap(buffers);
      std::scoped_lock lock{g.mutex};
      g.error = std::string("mmap: ") + std::strerror(errno);
      return false;
    }
    buffers.push_back({p, b.length});
  }
  // Three are the minimum for one held newest, one being converted and one
  // the VI can fill.
  if (buffers.size() < 3) {
    Unmap(buffers);
    std::scoped_lock lock{g.mutex};
    g.error = "fewer than 3 buffers";
    return false;
  }
  // cscore is not streaming (its sink is disabled), so every buffer is idle.
  // If it still is, the QBUF fails and this retries.
  for (unsigned i = 0; i < buffers.size(); ++i) {
    struct v4l2_buffer b;
    std::memset(&b, 0, sizeof(b));
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    b.memory = V4L2_MEMORY_MMAP;
    b.index = i;
    if (Xioctl(fd, VIDIOC_QBUF, &b) != 0) {
      int err = errno;
      Unmap(buffers);
      std::scoped_lock lock{g.mutex};
      g.error = std::string("QBUF (is cscore still streaming?): ") + std::strerror(err);
      return false;
    }
  }
  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (Xioctl(fd, VIDIOC_STREAMON, &type) != 0) {
    int err = errno;
    Unmap(buffers);
    std::scoped_lock lock{g.mutex};
    g.error = std::string("STREAMON: ") + std::strerror(err);
    return false;
  }
  std::scoped_lock lock{g.mutex};
  g.fd = fd;
  g.buffers = std::move(buffers);
  g.streaming = true;
  g.latest = -1;
  g.error.clear();
  return true;
}

void Run(Grabber* g) {
  while (g->running) {
    int fd;
    size_t nbuf;
    {
      std::scoped_lock lock{g->mutex};
      fd = g->streaming ? g->fd : -1;
      nbuf = g->buffers.size();
    }
    if (fd < 0) {
      if (!Attach(*g)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
      continue;
    }

    struct pollfd p = {fd, POLLIN, 0};
    int r = poll(&p, 1, 200);
    if (r < 0 && errno == EINTR) {
      continue;
    }
    if (r == 0) {
      continue;  // no frame yet; also lets running be checked
    }
    if (r < 0 || (p.revents & (POLLERR | POLLNVAL | POLLHUP))) {
      Detach(*g, "device descriptor went away (cscore reopened it?)");
      continue;
    }

    struct v4l2_buffer b;
    std::memset(&b, 0, sizeof(b));
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    b.memory = V4L2_MEMORY_MMAP;
    if (Xioctl(fd, VIDIOC_DQBUF, &b) != 0) {
      if (errno == EAGAIN) {
        continue;
      }
      Detach(*g, "DQBUF failed");
      continue;
    }
    if ((b.flags & V4L2_BUF_FLAG_ERROR) || b.index >= nbuf) {
      Xioctl(fd, VIDIOC_QBUF, &b);
      continue;
    }

    // The buffer's CLOCK_MONOTONIC capture time, in wpi::Now()'s timebase
    // (the translation cscore makes for the same buffers).
    int64_t bufUs = static_cast<int64_t>(b.timestamp.tv_sec) * 1000000 +
                    b.timestamp.tv_usec;
    int64_t timeUs = static_cast<int64_t>(wpi::Now()) - (MonotonicUs() - bufUs);

    int requeue = -1;
    {
      std::scoped_lock lock{g->mutex};
      if (g->latest >= 0 && g->latest != g->inUse) {
        requeue = g->latest;
      }
      g->latest = static_cast<int>(b.index);
      g->latestTimeUs = timeUs;
      g->latestSeq++;
    }
    g->cv.notify_all();
    if (requeue >= 0) {
      struct v4l2_buffer q;
      std::memset(&q, 0, sizeof(q));
      q.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      q.memory = V4L2_MEMORY_MMAP;
      q.index = static_cast<unsigned>(requeue);
      if (Xioctl(fd, VIDIOC_QBUF, &q) != 0) {
        Detach(*g, "QBUF failed");
      }
    }
  }
  Detach(*g, nullptr);
}

}  // namespace

extern "C" {

/*
 * Class:     org_photonvision_jni_V4l2LatestFrame
 * Method:    create
 * Signature: (Ljava/lang/String;II)J
 */
JNIEXPORT jlong JNICALL
Java_org_photonvision_jni_V4l2LatestFrame_create
  (JNIEnv* env, jclass, jstring path, jint width, jint height)
{
  const char* p = env->GetStringUTFChars(path, nullptr);
  char resolved[PATH_MAX];
  std::string device = realpath(p, resolved) ? resolved : p;
  env->ReleaseStringUTFChars(path, p);

  auto* g = new Grabber;
  g->device = device;
  g->width = width;
  g->height = height;
  g->thread = std::thread(Run, g);
  return reinterpret_cast<jlong>(g);
}

/*
 * Class:     org_photonvision_jni_V4l2LatestFrame
 * Method:    grab
 * Signature: (JJD)J
 */
JNIEXPORT jlong JNICALL
Java_org_photonvision_jni_V4l2LatestFrame_grab
  (JNIEnv*, jclass, jlong handle, jlong matPtr, jdouble timeout)
{
  auto* g = reinterpret_cast<Grabber*>(handle);
  auto* out = reinterpret_cast<cv::Mat*>(matPtr);
  std::unique_lock lock{g->mutex};
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                      std::chrono::duration<double>(timeout));
  // A frame newer than the last one handed out.
  if (!g->cv.wait_until(lock, deadline, [&] {
        return g->latest >= 0 && g->latestSeq > g->returnedSeq;
      })) {
    return 0;
  }
  int idx = g->latest;
  int fd = g->fd;
  g->inUse = idx;
  g->returnedSeq = g->latestSeq;
  int64_t timeUs = g->latestTimeUs;
  Buffer buf = g->buffers[idx];
  lock.unlock();

  // 10 bits MSB-aligned in 16: keep the top 8.
  bool ok = buf.length >= static_cast<size_t>(g->width) * g->height * 2;
  if (ok) {
    cv::Mat in(g->height, g->width, CV_16UC1, buf.data);
    in.convertTo(*out, CV_8U, 1.0 / 256.0);
  }

  lock.lock();
  g->inUse = -1;
  bool requeue = idx != g->latest && g->streaming && g->fd == fd;
  lock.unlock();
  g->cv.notify_all();
  if (requeue) {
    struct v4l2_buffer q;
    std::memset(&q, 0, sizeof(q));
    q.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    q.memory = V4L2_MEMORY_MMAP;
    q.index = static_cast<unsigned>(idx);
    Xioctl(fd, VIDIOC_QBUF, &q);  // a failure surfaces in the capture thread
  }
  return ok ? timeUs : 0;
}

/*
 * Class:     org_photonvision_jni_V4l2LatestFrame
 * Method:    lastError
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_org_photonvision_jni_V4l2LatestFrame_lastError
  (JNIEnv* env, jclass, jlong handle)
{
  auto* g = reinterpret_cast<Grabber*>(handle);
  std::scoped_lock lock{g->mutex};
  return env->NewStringUTF(g->error.c_str());
}

/*
 * Class:     org_photonvision_jni_V4l2LatestFrame
 * Method:    destroy
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_org_photonvision_jni_V4l2LatestFrame_destroy
  (JNIEnv*, jclass, jlong handle)
{
  auto* g = reinterpret_cast<Grabber*>(handle);
  g->running = false;
  if (g->thread.joinable()) {
    g->thread.join();
  }
  delete g;
}

}  // extern "C"
