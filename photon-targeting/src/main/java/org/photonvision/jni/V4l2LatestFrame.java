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

/**
 * Newest-frame capture for a 16-bit grey (Y16) V4L2 camera, on the device descriptor cscore holds
 * open for it (see V4l2LatestFrame.cpp). The camera's cscore sink must be disabled first, so that
 * cscore stops streaming and copying every frame; controls keep working through cscore.
 */
public class V4l2LatestFrame {
    /**
     * Start capturing from the device cscore has open.
     *
     * @param devicePath The camera's path as cscore opened it (symlinks are resolved).
     * @param width Frame width in pixels.
     * @param height Frame height in pixels.
     * @return A handle for {@link #grab}, {@link #lastError} and {@link #destroy}.
     */
    public static native long create(String devicePath, int width, int height);

    /**
     * Convert the newest frame not yet returned to 8-bit grey (top 8 of the 16 bits).
     *
     * @param handle From {@link #create}.
     * @param matPtr Native address of the destination cv::Mat (reallocated to CV_8UC1 if needed).
     * @param timeoutSeconds How long to wait for a new frame.
     * @return The frame's capture time in microseconds, in wpi::Now()'s timebase; 0 on timeout.
     */
    public static native long grab(long handle, long matPtr, double timeoutSeconds);

    /**
     * Convert on the GPU from now on: grab() then returns frames in pinned host memory, which
     * the GPU AprilTag detector reads in place.
     *
     * @param handle From {@link #create}.
     * @param captureOps GpuAprilTagJNI.captureOps() (photon-core).
     * @return Whether the ops were accepted.
     */
    public static native boolean setGpu(long handle, long captureOps);

    /**
     * Register another reader of this capture (a secondary feed of the same camera). Each reader
     * gets every newest frame once, in its own buffer.
     *
     * @param handle From {@link #create}.
     * @return The reader id for {@link #grabFor}; {@link #grab} is reader 0.
     */
    public static native int addConsumer(long handle);

    /**
     * {@link #grab} for reader {@code consumer}.
     *
     * @param handle From {@link #create}.
     * @param consumer From {@link #addConsumer}.
     * @param matPtr Native address of the destination cv::Mat.
     * @param timeoutSeconds How long to wait for a frame this reader has not had.
     * @return The frame's capture time in microseconds (wpi::Now()'s timebase); 0 on timeout.
     */
    public static native long grabFor(long handle, int consumer, long matPtr, double timeoutSeconds);

    /** The most recent attach/capture problem, for logging; empty if none. */
    public static native String lastError(long handle);

    /** Stop capturing and free the handle. */
    public static native void destroy(long handle);
}
