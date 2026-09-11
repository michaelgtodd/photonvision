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

package org.photonvision.vision.pipeline;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.imgcodecs.Imgcodecs;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.apriltag.TieredDetectionTest;
import org.photonvision.vision.camera.QuirkyCamera;
import org.photonvision.vision.frame.provider.FileFrameProvider;
import org.photonvision.vision.pipeline.result.CVPipelineResult;

/**
 * The whole AprilTag pipeline with the far-field and ROI tiers on: a frame holding a 60 px tag (the
 * main detector's) and a 22 px blurred tag that only a full-resolution search finds. With the main
 * detector at decimate 2 (standing in for the GPU tier), the pipeline must report both.
 */
public class AprilTagTieredPipelineTest {
    @BeforeAll
    public static void setup() {
        LoadJNI.loadLibraries();
        ConfigManager.getInstance().load();
    }

    static Set<Integer> ids(CVPipelineResult r) {
        var s = new HashSet<Integer>();
        for (var t : r.targets) s.add(t.getFiducialId());
        return s;
    }

    @Test
    public void farTagReportedThroughTheTiers() throws Exception {
        var near = TieredDetectionTest.frameWithTags(Map.of(3, new int[] {300, 200}), 60, 0.6);
        // the far tag sits in the middle band; blur like a distant tag
        var far = TieredDetectionTest.frameWithTags(Map.of(21, new int[] {1200, 560}), 22, 1.0);
        // combine: paste the far frame's band into the near frame
        far.submat(400, 800, 0, 1920).copyTo(near.submat(400, 800, 0, 1920));
        var png = Files.createTempFile("tiered", ".png");
        Imgcodecs.imwrite(png.toString(), near);

        var pipeline = new AprilTagPipeline();
        var s = pipeline.getSettings();
        s.tagFamily = AprilTagFamily.kTag36h11;
        s.decimate = 2;
        s.threads = 2;
        s.solvePNPEnabled = false;
        s.farFieldEnabled = true;
        s.farFieldRateHz = 50;
        s.farFieldUpsample = 1.5;
        s.farFieldBandTop = 0.35;
        s.farFieldBandBottom = 0.65;
        s.roiTrackEnabled = true;
        s.roiUpsample = 1.5;

        var provider = new FileFrameProvider(png, 100, 30);
        provider.requestFrameThresholdType(pipeline.getThresholdType());

        // frame 1: main detector only (the far-field search has just been started)
        var first = pipeline.run(provider.get(), QuirkyCamera.DefaultCamera);
        assertTrue(ids(first).contains(3), "near tag from the main detector: " + ids(first));

        // subsequent frames: the search completes, seeds the tracker, which re-detects each frame
        Set<Integer> got = Set.of();
        for (int i = 0; i < 60 && !got.contains(21); i++) {
            Thread.sleep(30);
            got = ids(pipeline.run(provider.get(), QuirkyCamera.DefaultCamera));
        }
        assertEquals(Set.of(3, 21), got, "both tags once the tiers have run");

        // and it stays tracked without new seeds arriving every frame
        for (int i = 0; i < 3; i++) {
            got = ids(pipeline.run(provider.get(), QuirkyCamera.DefaultCamera));
            assertEquals(Set.of(3, 21), got);
        }
        pipeline.release();
        Files.deleteIfExists(png);
    }
}
