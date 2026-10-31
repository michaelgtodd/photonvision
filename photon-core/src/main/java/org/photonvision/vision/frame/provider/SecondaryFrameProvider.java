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

package org.photonvision.vision.frame.provider;

import org.opencv.core.Mat;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.jni.V4l2LatestFrame;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.processes.VisionSourceSettables;

/**
 * Frames for a secondary feed (PVSecondaryFeedInfo): the parent camera's newest frames, read from
 * its registered capture as another reader (V4l2LatestFrame.addConsumer), at most maxFps times a
 * second. Each frame is this feed's own buffer (pinned and converted on the GPU when the parent's
 * capture is set up for it), so this pipeline and the parent's never share pixels.
 */
public class SecondaryFrameProvider extends CpuImageProcessor {
    private static final Logger logger = new Logger(SecondaryFrameProvider.class, LogGroup.Camera);

    private final String parentPath;
    private final double maxFps;
    private final VisionSourceSettables settables;
    private long readerHandle = 0;
    private int reader = -1;
    private long nextDueNanos = 0;
    private boolean everConnected = false;

    public SecondaryFrameProvider(String parentPath, double maxFps, VisionSourceSettables settables) {
        this.parentPath = parentPath;
        this.maxFps = maxFps;
        this.settables = settables;
    }

    @Override
    public CapturedFrame getInputMat() {
        // The newest frame, maxFps times a second on average. Time lost waiting (for the shared
        // full-resolution detector, say) is made up by up to one period; a pipeline slower than
        // maxFps simply takes the next frame at once, and no backlog builds up.
        if (maxFps > 0) {
            long period = (long) (1e9 / maxFps);
            long now = System.nanoTime();
            if (nextDueNanos > now && !sleepNanos(nextDueNanos - now)) return empty();
            nextDueNanos = Math.max(nextDueNanos + period, System.nanoTime() - period);
        }
        var capture = SharedCaptures.get(parentPath);
        if (capture == null) {
            sleepNanos(100_000_000L);
            return empty();
        }
        capture.lock.readLock().lock();
        try {
            if (capture.closed) return empty();
            if (readerHandle != capture.handle) {
                reader = V4l2LatestFrame.addConsumer(capture.handle);
                readerHandle = capture.handle;
                logger.info("Reading " + parentPath + "'s capture as reader " + reader);
            }
            var gray = new Mat();
            long timeUs =
                    V4l2LatestFrame.grabFor(capture.handle, reader, gray.getNativeObjAddr(), 1.0);
            if (timeUs == 0) {
                gray.release();
                return empty();
            }
            everConnected = true;
            // monochrome: the "colour" image is a second header on the grey pixels
            var color = gray.submat(0, gray.rows(), 0, gray.cols());
            return new CapturedFrame(
                    new CVMat(color), new CVMat(gray), settables.getFrameStaticProperties(), timeUs * 1000);
        } finally {
            capture.lock.readLock().unlock();
        }
    }

    private CapturedFrame empty() {
        return new CapturedFrame(new CVMat(), settables.getFrameStaticProperties(), 0);
    }

    private static boolean sleepNanos(long nanos) {
        try {
            Thread.sleep(nanos / 1_000_000L, (int) (nanos % 1_000_000L));
            return true;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    @Override
    public String getName() {
        return "SecondaryFrameProvider - " + parentPath;
    }

    @Override
    public void release() {
        // The reader slot belongs to the parent's capture and goes with it.
    }

    @Override
    public boolean checkCameraConnected() {
        return SharedCaptures.get(parentPath) != null;
    }

    @Override
    public boolean isConnected() {
        return SharedCaptures.get(parentPath) != null;
    }

    @Override
    public boolean hasConnected() {
        return everConnected || isConnected();
    }
}
