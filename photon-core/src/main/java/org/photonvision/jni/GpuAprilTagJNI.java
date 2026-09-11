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

package org.photonvision.jni;

import edu.wpi.first.apriltag.AprilTagDetection;
import java.io.File;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;

/**
 * Team 971's CUDA AprilTag detector, from the photon-gpu library built on the Jetson
 * (photon-gpu/build.sh). Loaded on demand from {@code /opt/photonvision/lib/libphotongpu.so}
 * (system property {@code photon.gpu.lib} overrides); absent or unloadable means {@link
 * #isAvailable()} is false and pipelines fall back to the CPU detector.
 *
 * <p>Handles are per camera: each carries its own CUDA stream and decode worker pool.
 */
public class GpuAprilTagJNI {
    private static final Logger logger = new Logger(GpuAprilTagJNI.class, LogGroup.VisionModule);
    private static final String DEFAULT_LIB = "/opt/photonvision/lib/libphotongpu.so";

    private static Boolean loaded = null;

    /** Try once to load the native library; safe to call repeatedly. */
    public static synchronized boolean isAvailable() {
        if (loaded != null) return loaded;
        String path = System.getProperty("photon.gpu.lib", DEFAULT_LIB);
        try {
            if (!new File(path).isFile()) {
                logger.info("GPU AprilTag detector not installed (" + path + " missing)");
                loaded = false;
            } else {
                System.load(path);
                logger.info("Loaded GPU AprilTag detector from " + path);
                loaded = true;
            }
        } catch (UnsatisfiedLinkError | SecurityException e) {
            logger.error("Failed to load GPU AprilTag detector from " + path, e);
            loaded = false;
        }
        return loaded;
    }

    /**
     * Create a detector for frames of a fixed size.
     *
     * @param width frame width (multiple of 8)
     * @param height frame height (multiple of 8)
     * @param decimate quad decimation, 2 at 1920x1200 (1 only for frames up to 1024x1024)
     * @param cpuThreads worker threads for the CPU decode stage
     * @param minWhiteBlackDiff quad threshold parameter, as in the CPU detector
     * @param refineEdges refine quad edges on the full-resolution image
     * @param decodeSharpening decode sharpening, as in the CPU detector
     * @param family "tag36h11" or "tag16h5"
     * @return native handle, 0 on failure
     */
    public static native long create(
            int width,
            int height,
            int decimate,
            int cpuThreads,
            int minWhiteBlackDiff,
            boolean refineEdges,
            float decodeSharpening,
            String family);

    /**
     * Detect tags in an 8-bit grey image described like a cv::Mat.
     *
     * @return detections in raw pixel coordinates; corners refined on the full-resolution image
     */
    public static native AprilTagDetection[] detect(
            long handle, long dataPtr, int width, int height, long step);

    public static native void destroy(long handle);
}
