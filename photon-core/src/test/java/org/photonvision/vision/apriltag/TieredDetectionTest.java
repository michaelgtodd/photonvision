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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import edu.wpi.first.apriltag.AprilTagDetection;
import edu.wpi.first.apriltag.AprilTagDetector;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.Objdetect;
import org.photonvision.common.LoadJNI;

/**
 * Tiers 2 and 3 on synthetic frames: small, blurred tag36h11 markers on a noisy 1920x1200 grey
 * background, like the far-field frames used for the bench experiments.
 */
public class TieredDetectionTest {
    static final int W = 1920, H = 1200;

    @BeforeAll
    public static void setup() {
        LoadJNI.loadLibraries();
    }

    /** A frame with markers of side {@code sidePx} (black border included) at the given positions. */
    public static Mat frameWithTags(Map<Integer, int[]> tags, int sidePx, double blurSigma) {
        var rng = new Random(1);
        Mat img = new Mat(H, W, CvType.CV_8UC1, new Scalar(110));
        // some texture so the background is not flat
        Mat noise = new Mat(H, W, CvType.CV_8UC1);
        Core.randn(noise, 0, 12);
        Core.add(img, noise, img);
        var dict = Objdetect.getPredefinedDictionary(Objdetect.DICT_APRILTAG_36h11);
        for (var e : tags.entrySet()) {
            Mat big = new Mat();
            dict.generateImageMarker(e.getKey(), 8 * 32, big, 1);
            int quiet = 32;
            Mat canvas = new Mat(8 * 32 + 2 * quiet, 8 * 32 + 2 * quiet, CvType.CV_8UC1, new Scalar(255));
            big.copyTo(canvas.submat(new Rect(quiet, quiet, 8 * 32, 8 * 32)));
            int total = (int) Math.round(sidePx * 10.0 / 8.0);
            Mat small = new Mat();
            Imgproc.resize(canvas, small, new Size(total, total), 0, 0, Imgproc.INTER_AREA);
            // contrast like a printed tag under so-so exposure: white 200, black 25
            small.convertTo(small, CvType.CV_8UC1, (200 - 25) / 255.0, 25);
            int x = e.getValue()[0], y = e.getValue()[1];
            small.copyTo(img.submat(new Rect(x, y, total, total)));
        }
        if (blurSigma > 0) Imgproc.GaussianBlur(img, img, new Size(0, 0), blurSigma);
        return img;
    }

    static AprilTagDetector.QuadThresholdParameters quadParams() {
        var q = new AprilTagDetector.QuadThresholdParameters();
        q.minClusterPixels = 5;
        q.maxNumMaxima = 10;
        q.criticalAngle = 45 * Math.PI / 180.0;
        q.maxLineFitMSE = 10.0f;
        q.minWhiteBlackDiff = 5;
        q.deglitch = false;
        return q;
    }

    static AprilTagDetection byId(List<AprilTagDetection> l, int id) {
        for (var d : l) if (d.getId() == id) return d;
        return null;
    }

    @Test
    public void regionDetectorMapsBackToFrame() {
        // a 40 px tag at (1400, 700): detect it via a crop and via the whole frame; the corners
        // and homography must agree
        var img = frameWithTags(Map.of(7, new int[] {1400, 700}), 40, 0.6);
        try (var rd = new RegionDetector()) {
            rd.configure("tag36h11", 1, quadParams());
            var whole = rd.detect(img, new Rect(0, 0, W, H), 1.0);
            var crop = rd.detect(img, new Rect(1300, 600, 300, 300), 1.0);
            var up = rd.detect(img, new Rect(1300, 600, 300, 300), 1.5);
            assertEquals(1, whole.size());
            assertEquals(1, crop.size());
            assertEquals(1, up.size());
            var a = whole.get(0);
            for (var b : List.of(crop.get(0), up.get(0))) {
                assertEquals(a.getId(), b.getId());
                for (int i = 0; i < 8; i++) assertEquals(a.getCorners()[i], b.getCorners()[i], 0.5);
                assertEquals(a.getCenterX(), b.getCenterX(), 0.5);
                assertEquals(a.getCenterY(), b.getCenterY(), 0.5);
                // the homography maps the tag's centre (0,0,1) to the image centre of the tag
                double[] h = b.getHomography();
                assertEquals(b.getCenterX(), h[2] / h[8], 0.5);
                assertEquals(b.getCenterY(), h[5] / h[8], 0.5);
            }
        }
    }

    @Test
    public void farFieldSeedsRoiTrackingAcrossFrames() throws Exception {
        // a 24 px blurred tag in the far-field band: too small for the main detector at decimate
        // 2 (it is not even offered here), found by the band search, then tracked by ROI
        int side = 24;
        var frame0 = frameWithTags(Map.of(11, new int[] {900, 560}), side, 1.0);
        try (var far = new FarFieldSearch("test");
                var roi = new RoiTracker()) {
            far.configure("tag36h11", 2, quadParams());
            roi.configure("tag36h11", 1, quadParams());
            var band = FarFieldBand.fromFractions(W, H, 0.4, 0.6);
            assertTrue(far.offer(frame0, band, 1.5, 1, 0));
            List<AprilTagDetection> seeds = List.of();
            for (int i = 0; i < 200 && seeds.isEmpty(); i++) {
                Thread.sleep(20);
                seeds = far.takeResults();
            }
            assertEquals(1, seeds.size(), "far-field search finds the small tag");
            assertEquals(11, seeds.get(0).getId());

            var params = new RoiTracker.Params(2.0, 16, 8, 10, 1.5);
            // frame 1: the tag moved a little; the main detector saw nothing
            var frame1 = frameWithTags(Map.of(11, new int[] {912, 566}), side, 1.0);
            var t3 = roi.update(frame1, 2, List.of(), seeds, params);
            assertEquals(1, t3.size(), "ROI re-detection in the new frame");
            var d = byId(t3, 11);
            assertEquals(912 + side * 10.0 / 8.0 / 2.0, d.getCenterX(), 3.0);
            // frame 2: still tracked with no new seeds
            var frame2 = frameWithTags(Map.of(11, new int[] {920, 570}), side, 1.0);
            t3 = roi.update(frame2, 3, List.of(), List.of(), params);
            assertEquals(1, t3.size());
            assertEquals(1, roi.trackCount());
            // tag gone: dropped after maxMisses frames
            var empty = frameWithTags(Map.of(), side, 1.0);
            for (int i = 0; i < 11; i++) roi.update(empty, 4 + i, List.of(), List.of(), params);
            assertEquals(0, roi.trackCount());
        }
    }

    @Test
    public void bandFromMountGeometry() {
        // camera at 0.5 m, level, fy 806 px (100 deg lens at 1920), cy 600: tags 0.2..1.5 m at 5 m
        var band = FarFieldBand.fromMount(W, H, 806, 600, 0.5, 0, 5.0, 0.2, 1.5, 3.0, 32);
        // 1.5 m tag: atan(1.0/5)=11.3 deg +3 -> row 600-806*tan(14.3)=395, minus pad -> ~363
        assertTrue(band.y < 400 && band.y > 300, "top " + band.y);
        // 0.2 m tag: atan(-0.3/5)=-3.4 deg -3 -> row 600+806*tan(6.4)=690, plus pad -> ~723
        assertTrue(
                band.y + band.height > 700 && band.y + band.height < 760,
                "bottom " + (band.y + band.height));
        assertEquals(W, band.width);
    }
}
