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

package org.photonvision.vision.camera;

import java.util.List;
import org.photonvision.common.configuration.CameraConfiguration;
import org.photonvision.vision.frame.FrameProvider;
import org.photonvision.vision.frame.FrameStaticProperties;
import org.photonvision.vision.frame.provider.SecondaryFrameProvider;
import org.photonvision.vision.frame.provider.SharedCaptures;
import org.photonvision.vision.processes.VisionSource;
import org.photonvision.vision.processes.VisionSourceSettables;
import org.wpilib.util.PixelFormat;
import org.wpilib.vision.camera.VideoMode;

/**
 * A second pipeline on another camera's frames (PVCameraInfo.PVSecondaryFeedInfo). It has its own
 * VisionModule -- pipelines, streams, NetworkTables name -- but no device: frames come from the
 * parent's capture, and resolution, FOV and calibration from the parent's settings. Exposure, gain
 * and the like belong to the parent; here they do nothing.
 */
public class SecondaryVisionSource extends VisionSource {
    private final SecondaryFrameProvider frameProvider;
    private final SecondarySettables settables;

    public SecondaryVisionSource(CameraConfiguration cameraConfiguration) {
        super(cameraConfiguration);
        if (!(cameraConfiguration.matchedCameraInfo instanceof PVCameraInfo.PVSecondaryFeedInfo info)) {
            throw new IllegalArgumentException(
                    "Not a secondary feed: " + cameraConfiguration.matchedCameraInfo);
        }
        if (cameraConfiguration.cameraQuirks == null)
            cameraConfiguration.cameraQuirks = QuirkyCamera.DefaultCamera;
        settables = new SecondarySettables(cameraConfiguration, info.parentPath, info.maxFps);
        frameProvider = new SecondaryFrameProvider(info.parentPath, info.maxFps, settables);
    }

    @Override
    public FrameProvider getFrameProvider() {
        return frameProvider;
    }

    @Override
    public VisionSourceSettables getSettables() {
        return settables;
    }

    @Override
    public boolean isVendorCamera() {
        return false;
    }

    @Override
    public void remakeSettables() {}

    @Override
    public boolean hasLEDs() {
        return false;
    }

    @Override
    public void release() {
        frameProvider.release();
    }

    public static class SecondarySettables extends VisionSourceSettables {
        // Until the parent's capture is up; the GMSL heads' mode.
        private static final int DEFAULT_WIDTH = 1920, DEFAULT_HEIGHT = 1200;

        private final String parentPath;
        private final int fps;

        SecondarySettables(CameraConfiguration cameraConfiguration, String parentPath, double maxFps) {
            super(cameraConfiguration);
            this.parentPath = parentPath;
            this.fps = (int) Math.max(1, Math.round(maxFps));
            videoModes.add(getCurrentVideoMode());
            calculateFrameStaticProps();
        }

        /** The parent's frame properties (its calibration included) once its capture is up. */
        @Override
        public FrameStaticProperties getFrameStaticProperties() {
            var capture = SharedCaptures.get(parentPath);
            if (capture != null) {
                var parent = capture.staticProperties();
                if (parent != null) return parent;
            }
            return super.getFrameStaticProperties();
        }

        @Override
        public VideoMode getCurrentVideoMode() {
            var capture = SharedCaptures.get(parentPath);
            var parent = capture != null ? capture.staticProperties() : null;
            if (parent != null) {
                return new VideoMode(PixelFormat.Y16, parent.imageWidth, parent.imageHeight, fps);
            }
            return new VideoMode(PixelFormat.Y16, DEFAULT_WIDTH, DEFAULT_HEIGHT, fps);
        }

        @Override
        protected void setVideoModeInternal(VideoMode videoMode) {}

        @Override
        public List<VideoMode> getAllVideoModes() {
            return videoModes;
        }

        @Override
        public void setExposureRaw(double exposureRaw) {}

        @Override
        public double getMinExposureRaw() {
            return 1;
        }

        @Override
        public double getMaxExposureRaw() {
            return 100;
        }

        @Override
        public void setAutoExposure(boolean cameraAutoExposure) {}

        @Override
        public void setWhiteBalanceTemp(double temp) {}

        @Override
        public void setAutoWhiteBalance(boolean autowb) {}

        @Override
        public void setBrightness(int brightness) {}

        @Override
        public void setGain(int gain) {}

        @Override
        public double getMinWhiteBalanceTemp() {
            return 1;
        }

        @Override
        public double getMaxWhiteBalanceTemp() {
            return 2;
        }
    }
}
