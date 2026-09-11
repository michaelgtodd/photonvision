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
 * <p>Quad finding runs at decimate 2 regardless of the pipeline's decimate setting at resolutions
 * above 1024 px (the detector's packed coordinates are sized for the half-resolution image); decode
 * and edge refinement use the full-resolution image in either case.
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
            boolean refineEdges,
            float decodeSharpening) {}

    /** The decimate the detector will actually use for a frame of this size. */
    public static int effectiveDecimate(int requested, int width, int height) {
        if (width > 1024 || height > 1024) return 2;
        return requested >= 2 ? 2 : 1;
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
        int decimate = effectiveDecimate(params.decimate(), width, height);
        try {
            handle =
                    GpuAprilTagJNI.create(
                            width,
                            height,
                            decimate,
                            params.cpuThreads(),
                            params.minWhiteBlackDiff(),
                            params.refineEdges(),
                            params.decodeSharpening(),
                            params.family().getNativeName());
        } catch (IllegalArgumentException e) {
            logger.error("GPU AprilTag detector rejected the frame/parameters", e);
            handle = 0;
        }
        if (handle == 0) return false;
        handleWidth = width;
        handleHeight = height;
        handleParams = params;
        logger.info(
                "GPU AprilTag detector created for "
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
        if (handle != 0) {
            GpuAprilTagJNI.destroy(handle);
            handle = 0;
        }
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
        return ret == null ? List.of() : List.of(ret);
    }

    @Override
    public void release() {
        releaseHandle();
    }
}
