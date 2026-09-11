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
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.opencv.core.Mat;
import org.opencv.core.Rect;

/**
 * Tier 3 of the tiered AprilTag pipeline: full-resolution re-detection, every frame, of tags the
 * GPU tier did not see this frame but that are known to be there -- from the far-field search (tier
 * 2) or from earlier frames. A 256x256 full-resolution crop costs ~2.6 ms, so a handful per frame
 * is affordable, and a far tag, once acquired, is tracked at frame rate with corners refined on the
 * full-resolution image of the current frame.
 */
public class RoiTracker implements AutoCloseable {
    private static class Track {
        AprilTagDetection last;
        long lastSeenSeq;
        int misses;
    }

    private final RegionDetector detector = new RegionDetector();
    private final Map<Integer, Track> tracks = new HashMap<>();

    public record Params(
            double margin, // ROI side = margin x tag size (2 = one tag width of slack each side)
            int padPx, // plus this many pixels each side
            int maxRois, // per frame, most recently seen first
            int maxMisses, // drop a track after this many consecutive frames without it
            double upsample) {}

    public void configure(
            String family, int threads, AprilTagDetector.QuadThresholdParameters quadParams) {
        detector.configure(family, threads, quadParams);
    }

    /**
     * Update with this frame.
     *
     * @param grey the frame (8-bit grey, full resolution)
     * @param seq the frame's sequence number
     * @param seen detections already made on this frame (the GPU tier)
     * @param seeds detections from an older frame (the far-field search), if any
     * @return re-detections made on this frame for ids not in {@code seen}, full-frame coordinates
     */
    public List<AprilTagDetection> update(
            Mat grey, long seq, List<AprilTagDetection> seen, List<AprilTagDetection> seeds, Params p) {
        // Tags the GPU tier saw need no help; remember where they are.
        var seenIds = new HashMap<Integer, AprilTagDetection>();
        for (var d : seen) {
            seenIds.put(d.getId(), d);
            remember(d, seq);
        }
        // Seeds from the far-field search: known positions from an older frame.
        for (var d : seeds) {
            if (!seenIds.containsKey(d.getId())) rememberIfNewer(d, seq - 1);
        }

        // Candidates: tracked ids not seen this frame, most recently seen first.
        var candidates = new ArrayList<Map.Entry<Integer, Track>>();
        for (var e : tracks.entrySet()) {
            if (!seenIds.containsKey(e.getKey())) candidates.add(e);
        }
        candidates.sort(
                Comparator.comparingLong((Map.Entry<Integer, Track> e) -> -e.getValue().lastSeenSeq));

        var out = new LinkedHashMap<Integer, AprilTagDetection>();
        int rois = 0;
        for (var e : candidates) {
            Track t = e.getValue();
            if (rois >= p.maxRois()) {
                t.misses++;
                continue;
            }
            rois++;
            Rect roi = roiAround(t.last, p);
            var found = detector.detect(grey, roi, p.upsample());
            AprilTagDetection hit = null;
            for (var d : found) {
                if (d.getId() == e.getKey()) {
                    hit = d;
                    break;
                }
            }
            if (hit != null) {
                t.last = hit;
                t.lastSeenSeq = seq;
                t.misses = 0;
                out.put(hit.getId(), hit);
            } else {
                t.misses++;
            }
        }
        tracks.entrySet().removeIf(e -> e.getValue().misses > p.maxMisses());
        return new ArrayList<>(out.values());
    }

    public int trackCount() {
        return tracks.size();
    }

    private void remember(AprilTagDetection d, long seq) {
        Track t = tracks.computeIfAbsent(d.getId(), k -> new Track());
        t.last = d;
        t.lastSeenSeq = seq;
        t.misses = 0;
    }

    private void rememberIfNewer(AprilTagDetection d, long seq) {
        Track t = tracks.get(d.getId());
        if (t == null || t.lastSeenSeq <= seq) remember(d, seq);
    }

    private static Rect roiAround(AprilTagDetection d, Params p) {
        double[] c = d.getCorners();
        double minX = c[0], maxX = c[0], minY = c[1], maxY = c[1];
        for (int i = 1; i < 4; i++) {
            minX = Math.min(minX, c[2 * i]);
            maxX = Math.max(maxX, c[2 * i]);
            minY = Math.min(minY, c[2 * i + 1]);
            maxY = Math.max(maxY, c[2 * i + 1]);
        }
        double size = Math.max(maxX - minX, maxY - minY);
        double half = size * p.margin() / 2.0 + p.padPx();
        double cx = (minX + maxX) / 2.0, cy = (minY + maxY) / 2.0;
        return new Rect(
                (int) Math.floor(cx - half),
                (int) Math.floor(cy - half),
                (int) Math.ceil(2 * half),
                (int) Math.ceil(2 * half));
    }

    @Override
    public void close() {
        detector.close();
    }
}
