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
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Full-resolution AprilTag detection on a region of a frame, with the results mapped back into
 * full-frame coordinates. The CPU building block of the far-field search (tier 2) and the ROI
 * tracker (tier 3): a detector run on a crop costs in proportion to the crop, so a band or a few
 * small ROIs at full resolution are affordable every frame even where the whole frame is not.
 *
 * <p>The region may be upsampled before detection (bicubic). Measured on the bench, quad finding on
 * a 1.5x upsampled region finds blurred tags about one size step smaller (22 px vs 24 px) at 2.25x
 * the cost; corners and homography are scaled back so callers never see the upsampling.
 */
public class RegionDetector implements AutoCloseable {
    private AprilTagDetector detector = new AprilTagDetector();
    private AprilTagDetector.Config config = null;
    private AprilTagDetector.QuadThresholdParameters quadParams = null;
    private String family = null;

    // scratch, reused across calls
    private final Mat scaled = new Mat();

    /** Configure like the pipeline's detector, but always at full resolution (decimate 1). */
    public void configure(
            String family, int threads, AprilTagDetector.QuadThresholdParameters quadParams) {
        var cfg = new AprilTagDetector.Config();
        cfg.numThreads = Math.max(1, threads);
        cfg.quadDecimate = 1.0f;
        cfg.quadSigma = 0.0f;
        cfg.refineEdges = true;
        cfg.decodeSharpening = 0.25;
        boolean familyChanged = !family.equals(this.family);
        if (this.config == null || this.config.numThreads != cfg.numThreads) {
            detector.setConfig(cfg);
            this.config = cfg;
        }
        if (this.quadParams == null || !sameQuadParams(quadParams, this.quadParams)) {
            detector.setQuadThresholdParameters(quadParams);
            this.quadParams = quadParams;
        }
        if (familyChanged) {
            detector.clearFamilies();
            detector.addFamily(family);
            this.family = family;
        }
    }

    /**
     * Detect in {@code region} of {@code frame} (8-bit grey), optionally upsampled by {@code
     * upsample} (>= 1), returning detections in full-frame coordinates.
     */
    public List<AprilTagDetection> detect(Mat frame, Rect region, double upsample) {
        Rect r = clip(region, frame.cols(), frame.rows());
        if (r.width < 8 || r.height < 8) return List.of();
        Mat crop = frame.submat(r);
        try {
            return detectRegionImage(crop, r.x, r.y, upsample);
        } finally {
            crop.release();
        }
    }

    /**
     * Detect in an image that is itself a region of a frame whose top-left corner sits at (x0, y0) in
     * the frame; results are in full-frame coordinates.
     */
    public List<AprilTagDetection> detectRegionImage(Mat region, int x0, int y0, double upsample) {
        Mat input = region;
        double s = upsample > 1.0 ? upsample : 1.0;
        if (s > 1.0) {
            Imgproc.resize(
                    region,
                    scaled,
                    new Size(Math.round(region.cols() * s), Math.round(region.rows() * s)),
                    0,
                    0,
                    Imgproc.INTER_CUBIC);
            input = scaled;
        }

        AprilTagDetection[] found = detector.detect(input);
        if (found == null || found.length == 0) return List.of();

        List<AprilTagDetection> out = new ArrayList<>(found.length);
        for (var d : found) out.add(mapToFrame(d, x0, y0, s));
        return out;
    }

    /** Translate (and un-scale) a detection made on a region back into full-frame coordinates. */
    public static AprilTagDetection mapToFrame(AprilTagDetection d, double x0, double y0, double s) {
        double inv = 1.0 / s;
        double[] c = d.getCorners();
        double[] corners = new double[8];
        for (int i = 0; i < 4; i++) {
            corners[2 * i] = c[2 * i] * inv + x0;
            corners[2 * i + 1] = c[2 * i + 1] * inv + y0;
        }
        // H' = T(x0, y0) * S(1/s) * H  (row-major 3x3): rows 0 and 1 scaled by 1/s, then the
        // translation adds x0/y0 times the third row.
        double[] h = d.getHomography();
        double[] hp = new double[9];
        for (int j = 0; j < 3; j++) {
            hp[j] = h[j] * inv + x0 * h[6 + j];
            hp[3 + j] = h[3 + j] * inv + y0 * h[6 + j];
            hp[6 + j] = h[6 + j];
        }
        return new AprilTagDetection(
                d.getFamily(),
                d.getId(),
                d.getHamming(),
                d.getDecisionMargin(),
                hp,
                d.getCenterX() * inv + x0,
                d.getCenterY() * inv + y0,
                corners);
    }

    private static boolean sameQuadParams(
            AprilTagDetector.QuadThresholdParameters a, AprilTagDetector.QuadThresholdParameters b) {
        return a.minClusterPixels == b.minClusterPixels
                && a.maxNumMaxima == b.maxNumMaxima
                && a.criticalAngle == b.criticalAngle
                && a.maxLineFitMSE == b.maxLineFitMSE
                && a.minWhiteBlackDiff == b.minWhiteBlackDiff
                && a.deglitch == b.deglitch;
    }

    /** Clip a rectangle to the frame, keeping x/y even (the detector likes aligned crops). */
    public static Rect clip(Rect r, int width, int height) {
        int x = Math.max(0, r.x) & ~1;
        int y = Math.max(0, r.y) & ~1;
        int x2 = Math.min(width, r.x + r.width);
        int y2 = Math.min(height, r.y + r.height);
        return new Rect(x, y, Math.max(0, x2 - x), Math.max(0, y2 - y));
    }

    @Override
    public void close() {
        if (detector != null) {
            detector.close();
            detector = null;
        }
        scaled.release();
    }
}
