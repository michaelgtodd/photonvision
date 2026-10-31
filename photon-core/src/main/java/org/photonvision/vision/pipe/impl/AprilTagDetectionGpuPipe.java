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

package org.photonvision.vision.pipe.impl;

import edu.wpi.first.apriltag.AprilTagDetection;
import java.util.HashMap;
import java.util.List;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.jni.GpuAprilTagJNI;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.opencv.Releasable;
import org.photonvision.vision.pipe.CVPipe;

/**
 * AprilTag detection on the GPU (Team 971's CUDA detector via photon-gpu). A drop-in for {@link
 * AprilTagDetectionPipe}: same input (a grey CVMat), same output (WPILib {@link AprilTagDetection}s
 * with corners refined on the full-resolution image). The native detector is sized to the frame and
 * recreated when the frame size or the parameters change.
 *
 * <p>Quad finding runs at decimate 2 above 1024 px unless full resolution is asked for
 * explicitly ({@code gpuFullResolution}): the pipeline decimate setting of 1 that existing
 * pipelines carry would otherwise silently switch them to the far more expensive full-resolution
 * search. Decode and edge refinement use the full-resolution image in either case.
 */
public class AprilTagDetectionGpuPipe
        extends CVPipe<
                CVMat, List<AprilTagDetection>, AprilTagDetectionGpuPipe.AprilTagDetectionGpuPipeParams>
        implements Releasable {
    private final Logger logger = new Logger(AprilTagDetectionGpuPipe.class, LogGroup.VisionModule);

    private long handle = 0;
    private int handleWidth = 0, handleHeight = 0;
    private AprilTagDetectionGpuPipeParams handleParams = null;

    public static record AprilTagDetectionGpuPipeParams(
            AprilTagFamily family,
            int decimate,
            int cpuThreads,
            int minWhiteBlackDiff,
            int minBlobSize,
            boolean refineEdges,
            float decodeSharpening,
            boolean fullResolution) {}

    /** The decimate the detector will actually use for a frame of this size. */
    public static int effectiveDecimate(int requested, int width, int height, boolean fullResolution) {
        if (fullResolution) return 1;
        if (width > 1024 || height > 1024) return 2;
        return requested >= 2 ? 2 : 1;
    }

    // Full-resolution detectors are shared by every pipe that asks for one with the same frame
    // size and parameters. Each holds 1.5-2 GB of device buffers; on the 8 GB Orin Nano, a second
    // one beside four half-resolution detectors ran the GPU out of memory (NvMap ENOMEM), after
    // which every CUDA call in the process failed. Full-resolution feeds run at a few fps each, so
    // one detector serves them all: GpuAprilTagJNI.detect serialises calls on a handle.
    private record SharedKey(int width, int height, AprilTagDetectionGpuPipeParams params) {}

    private static final class SharedDetector {
        final long handle;
        int users = 0;

        SharedDetector(long handle) {
            this.handle = handle;
        }
    }

    private static final HashMap<SharedKey, SharedDetector> sharedDetectors = new HashMap<>();
    private SharedKey sharedKey = null;

    private long createDetector(int width, int height, int decimate) {
        try {
            return GpuAprilTagJNI.create(
                    width,
                    height,
                    decimate,
                    params.cpuThreads(),
                    params.minWhiteBlackDiff(),
                    params.minBlobSize(),
                    params.refineEdges(),
                    params.decodeSharpening(),
                    params.family().getNativeName());
        } catch (IllegalArgumentException e) {
            logger.error("GPU AprilTag detector rejected the frame/parameters", e);
            return 0;
        }
    }

    private boolean ensureDetector(int width, int height) {
        boolean same =
                handle != 0
                        && handleWidth == width
                        && handleHeight == height
                        && handleParams != null
                        && handleParams.equals(params);
        if (same) return true;

        releaseHandle();
        if (!GpuAprilTagJNI.isAvailable()) return false;
        int decimate =
                effectiveDecimate(params.decimate(), width, height, params.fullResolution());
        boolean reused = false;
        if (params.fullResolution()) {
            var key = new SharedKey(width, height, params);
            synchronized (sharedDetectors) {
                var shared = sharedDetectors.get(key);
                if (shared == null) {
                    long created = createDetector(width, height, decimate);
                    if (created == 0) return false;
                    shared = new SharedDetector(created);
                    sharedDetectors.put(key, shared);
                } else {
                    reused = true;
                }
                shared.users++;
                handle = shared.handle;
                sharedKey = key;
            }
        } else {
            handle = createDetector(width, height, decimate);
        }
        if (handle == 0) return false;
        handleWidth = width;
        handleHeight = height;
        handleParams = params;
        logger.info(
                (reused ? "GPU AprilTag detector shared for " : "GPU AprilTag detector created for ")
                        + width
                        + "x"
                        + height
                        + " ("
                        + params.family().getNativeName()
                        + ", decimate "
                        + decimate
                        + ", "
                        + params.cpuThreads()
                        + " decode threads)");
        return true;
    }

    private void releaseHandle() {
        if (handle == 0) return;
        if (sharedKey != null) {
            synchronized (sharedDetectors) {
                var shared = sharedDetectors.get(sharedKey);
                if (shared != null && --shared.users == 0) {
                    GpuAprilTagJNI.destroy(shared.handle);
                    sharedDetectors.remove(sharedKey);
                }
            }
            sharedKey = null;
        } else {
            GpuAprilTagJNI.destroy(handle);
        }
        handle = 0;
    }

    @Override
    protected List<AprilTagDetection> process(CVMat in) {
        Mat mat = in.getMat();
        if (mat.empty()) return List.of();
        if (mat.type() != CvType.CV_8UC1) {
            logger.error("GPU AprilTag detector needs an 8-bit grey frame, got type " + mat.type());
            return List.of();
        }
        if (!ensureDetector(mat.cols(), mat.rows())) return List.of();

        AprilTagDetection[] ret =
                GpuAprilTagJNI.detect(handle, mat.dataAddr(), mat.cols(), mat.rows(), mat.step1());
        if (ret == null) {
            reportCudaFailure();
            return List.of();
        }
        return List.of(ret);
    }

    private long reportedFailures = 0;
    private long droppedFrames = 0;

    /** A frame was dropped because a CUDA call failed; log it without flooding. */
    private void reportCudaFailure() {
        droppedFrames++;
        long failures = GpuAprilTagJNI.cudaFailures();
        // the first, then every 1000 dropped frames
        if (droppedFrames == 1 || droppedFrames % 1000 == 0 || failures - reportedFailures > 1000) {
            logger.error(
                    "GPU AprilTag detector: CUDA call failed, frame dropped ("
                            + droppedFrames
                            + " frames dropped by this pipe, "
                            + failures
                            + " CUDA failures in this process; see stderr)");
            reportedFailures = failures;
        }
    }

    @Override
    public void release() {
        releaseHandle();
    }
}
