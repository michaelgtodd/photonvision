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

package org.photonvision.common.configuration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Path;
import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.util.TestUtils;
import org.photonvision.vision.camera.PVCameraInfo;
import org.photonvision.vision.pipeline.AprilTagPipelineSettings;

/**
 * A 2026.3.4-format database from the Jetson bench (one GMSL head and its full-resolution secondary
 * feed, both on the GPU AprilTag detector) loads after the 2027 migrations: the feed is still a
 * feed of the same camera, and the GPU settings survive.
 */
public class SecondaryFeedMigrationTest {
    static final String HEAD_PATH = "/dev/v4l/by-path/gmsl-linkA-ar0234-0013";

    @TempDir private Path tmpDir;

    @BeforeAll
    public static void init() {
        LoadJNI.loadLibraries();
    }

    private PhotonConfiguration load(Path folder) {
        var provider = new ConfigProvider(folder);
        provider.load();
        return provider.getConfig();
    }

    private void check(PhotonConfiguration config) {
        var cameras = config.getCameraConfigurations().values();
        assertEquals(2, cameras.size());
        var head =
                cameras.stream().filter(c -> c.nickname.equals("gmsl-linkA")).findFirst().orElseThrow();
        var feed =
                cameras.stream()
                        .filter(c -> c.nickname.equals("gmsl-linkA-full"))
                        .findFirst()
                        .orElseThrow();

        assertInstanceOf(PVCameraInfo.PVUsbCameraInfo.class, head.matchedCameraInfo);
        assertEquals(HEAD_PATH, head.matchedCameraInfo.uniquePath());

        var info = assertInstanceOf(PVCameraInfo.PVSecondaryFeedInfo.class, feed.matchedCameraInfo);
        assertEquals(HEAD_PATH, info.parentPath);
        assertEquals("gmsl-linkA-full", info.name);
        assertEquals(5.0, info.maxFps);
        assertEquals(HEAD_PATH + "#gmsl-linkA-full", info.uniquePath());

        var headPipe = assertInstanceOf(AprilTagPipelineSettings.class, head.pipelineSettings.get(0));
        assertTrue(headPipe.gpuDetector);
        assertEquals(false, headPipe.gpuFullResolution);
        assertEquals(2, headPipe.threads);

        var feedPipe = assertInstanceOf(AprilTagPipelineSettings.class, feed.pipelineSettings.get(0));
        assertTrue(feedPipe.gpuDetector);
        assertTrue(feedPipe.gpuFullResolution);
        assertEquals(100, feedPipe.gpuMinBlobSize);
        assertEquals(20, feedPipe.gpuMinWhiteBlackDiff);
        assertEquals(4, feedPipe.threads);
    }

    @Test
    public void migratesAndRoundTrips() throws IOException {
        var folder = tmpDir.resolve("2026.3.4-jetson-feeds");
        FileUtils.copyDirectory(
                TestUtils.getConfigDirectoriesPath(false).resolve("2026.3.4-jetson-feeds").toFile(),
                folder.toFile());

        check(load(folder));

        // saved in the 2027 format and read back
        var provider = new ConfigProvider(folder);
        provider.load();
        provider.saveToDisk();
        check(load(folder));
    }
}
