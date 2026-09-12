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

package org.photonvision.vision.apriltag;

import edu.wpi.first.apriltag.AprilTagDetection;
import edu.wpi.first.apriltag.AprilTagDetector;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;

/**
 * Tier 2 of the tiered AprilTag pipeline: a full-resolution search of the far-field band, run on
 * its own thread at a limited rate, so the per-frame path never waits for it.
 *
 * <p>The band is copied out of the frame on the caller's thread (sub-millisecond) and searched in
 * the background at full resolution, optionally upsampled. Results are seeds: tags with positions
 * from an older frame, which the ROI tracker (tier 3) re-detects in the current frame. They are not
 * reported directly.
 */
public class FarFieldSearch implements AutoCloseable {
    private final Logger logger;
    private final RegionDetector detector = new RegionDetector();
    private final ExecutorService executor;
    private final AtomicBoolean busy = new AtomicBoolean(false);
    private final Object lock = new Object();

    private long lastSubmitNanos = 0;
    private List<AprilTagDetection> latest = new ArrayList<>();
    private long latestSeq = -1;
    private double lastSearchMs = 0;

    public FarFieldSearch(String name) {
        logger = new Logger(FarFieldSearch.class, name, LogGroup.VisionModule);
        executor =
                Executors.newSingleThreadExecutor(
                        r -> {
                            var t = new Thread(r, "FarFieldSearch - " + name);
                            t.setDaemon(true);
                            return t;
                        });
    }

    private record Config(
            String family, int threads, AprilTagDetector.QuadThresholdParameters quadParams) {}

    private volatile Config wanted = null;

    /**
     * Record the detector configuration; it is applied on the search thread before the next search.
     * Called every frame by the pipeline, so it must not wait for a running search.
     */
    public void configure(
            String family, int threads, AprilTagDetector.QuadThresholdParameters quadParams) {
        var c = new Config(family, threads, quadParams);
        if (!c.equals(wanted)) wanted = c;
    }

    /**
     * Offer a frame. If the search is idle and at least {@code minIntervalSec} has passed since the
     * last one, the band is copied and searched in the background.
     *
     * @return true if a search was started on this frame
     */
    public boolean offer(Mat grey, Rect band, double upsample, long seq, double minIntervalSec) {
        long now = System.nanoTime();
        if (now - lastSubmitNanos < minIntervalSec * 1e9) return false;
        if (!busy.compareAndSet(false, true)) return false;
        lastSubmitNanos = now;

        Rect r = RegionDetector.clip(band, grey.cols(), grey.rows());
        if (r.width < 8 || r.height < 8) {
            busy.set(false);
            return false;
        }
        Mat view = grey.submat(r);
        Mat copy = view.clone();
        view.release();
        executor.submit(
                () -> {
                    long t0 = System.nanoTime();
                    try {
                        var c = wanted;
                        if (c != null) detector.configure(c.family(), c.threads(), c.quadParams());
                        List<AprilTagDetection> found = detector.detectRegionImage(copy, r.x, r.y, upsample);
                        synchronized (lock) {
                            latest = found;
                            latestSeq = seq;
                            lastSearchMs = (System.nanoTime() - t0) / 1e6;
                        }
                    } catch (Exception e) {
                        logger.error("Far-field search failed", e);
                    } finally {
                        copy.release();
                        busy.set(false);
                    }
                });
        return true;
    }

    /**
     * Newest results since the last call (empty if none); the frame they came from via {@link
     * #lastSeq()}.
     */
    public List<AprilTagDetection> takeResults() {
        synchronized (lock) {
            var out = latest;
            latest = new ArrayList<>();
            return out;
        }
    }

    public long lastSeq() {
        synchronized (lock) {
            return latestSeq;
        }
    }

    /** Duration of the most recent search, for diagnostics. */
    public double lastSearchMillis() {
        synchronized (lock) {
            return lastSearchMs;
        }
    }

    @Override
    public void close() {
        executor.shutdownNow();
        try {
            executor.awaitTermination(2, java.util.concurrent.TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        detector.close();
    }
}
