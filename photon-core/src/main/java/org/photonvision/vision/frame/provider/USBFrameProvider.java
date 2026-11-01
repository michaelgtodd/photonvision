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

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.jni.CscoreExtras;
import org.photonvision.jni.GpuAprilTagJNI;
import org.photonvision.jni.V4l2LatestFrame;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.processes.VisionSourceSettables;
import org.wpilib.util.PixelFormat;
import org.wpilib.util.RawFrame;
import org.wpilib.vision.camera.CvSink;
import org.wpilib.vision.camera.UsbCamera;
import org.wpilib.vision.camera.VideoMode;
import org.wpilib.vision.stream.CameraServer;

public class USBFrameProvider extends CpuImageProcessor {
    private final Logger logger;

    private UsbCamera camera = null;
    private CvSink cvSink = null;

    @SuppressWarnings("SpellCheckingInspection")
    private VisionSourceSettables settables;

    private Runnable connectedCallback;

    private long lastTime = 0;

    /**
     * Y16 sources are captured newest-frame-only on cscore's own descriptor (V4l2LatestFrame) unless
     * PHOTONVISION_Y16_LATEST_FRAME=0: cscore copies every frame it dequeues, and at sensor rates
     * well above what the pipeline processes that copy dominated the CPU.
     */
    private static final boolean Y16_LATEST_FRAME =
            !"0".equals(System.getenv("PHOTONVISION_Y16_LATEST_FRAME"));

    private long latestFrame = 0;
    // latestFrame as published to secondary feeds of this camera (SharedCaptures)
    private SharedCaptures.Capture sharedCapture = null;

    @SuppressWarnings("SpellCheckingInspection")
    public USBFrameProvider(
            UsbCamera camera, VisionSourceSettables visionSettables, Runnable connectedCallback) {
        this.camera = camera;
        this.cvSink = CameraServer.getVideo(this.camera);
        this.logger =
                new Logger(
                        USBFrameProvider.class, visionSettables.getConfiguration().nickname, LogGroup.Camera);
        this.cvSink.setEnabled(true);

        this.settables = visionSettables;

        this.connectedCallback = connectedCallback;
    }

    final double CSCORE_DEFAULT_FRAME_TIMEOUT = 1.0 / 4.0;

    @Override
    public CapturedFrame getInputMat() {
        if (!cameraPropertiesCached && camera.isConnected()) {
            onCameraConnected();
        }

        // 16-bit grey sources are converted here, never by cscore (see grabY16)
        var currentMode = settables.getCurrentVideoMode();
        if (currentMode != null && currentMode.pixelFormat == PixelFormat.Y16) {
            return grabY16(currentMode);
        }

        if (m_blockForFrames) {
            // We allocate memory so we don't fill a Mat in use by another thread (memory model is easier)
            var mat = new CVMat();
            // This is from wpi::nt::Now, or WPIUtilJNI.now(). The epoch from grabFrame is nS since
            // Hal::initialize was called
            // TODO - under the hood, this incurs an extra copy. We should avoid this, if we
            // can.
            long captureTimeNs = cvSink.grabFrame(mat.getMat(), CSCORE_DEFAULT_FRAME_TIMEOUT);

            if (captureTimeNs == 0) {
                var error = cvSink.getError();
                logger.error("Error grabbing image: " + error);
            }

            return new CapturedFrame(mat, settables.getFrameStaticProperties(), captureTimeNs);
        } else {
            // We allocate memory so we don't fill a Mat in use by another thread (memory model is easier)
            // TODO - consider a frame pool
            // TODO - getCurrentVideoMode is a JNI call for us, but profiling indicates it's fast
            var cameraMode = settables.getCurrentVideoMode();
            var frame = new RawFrame();
            frame.setInfo(
                    cameraMode.width,
                    cameraMode.height,
                    // hard-coded 3 channel
                    cameraMode.width * 3,
                    PixelFormat.BGR);

            // This is from wpi::nt::Now, or WPIUtilJNI.now(). The epoch from grabFrame is nS since
            // Hal::initialize was called
            long captureTimeNs =
                    CscoreExtras.grabRawSinkFrameTimeoutLastTime(
                            cvSink.getHandle(), frame.getNativeObj(), CSCORE_DEFAULT_FRAME_TIMEOUT, lastTime);
            lastTime = captureTimeNs;

            CVMat ret;

            if (captureTimeNs == 0) {
                var error = cvSink.getError();
                logger.error("Error grabbing image: " + error);

                frame.close();
                ret = new CVMat();
            } else {
                // No error! yay
                var mat = new Mat(CscoreExtras.wrapRawFrame(frame.getNativeObj()));

                ret = new CVMat(mat, frame);
            }

            return new CapturedFrame(ret, settables.getFrameStaticProperties(), captureTimeNs);
        }
    }

    /**
     * Grab a frame from a Y16 source and convert it to 8-bit BGR ourselves.
     *
     * <p>Monochrome sensors behind a raw V4L2 capture pipeline (for instance the AR0234 heads on an
     * NVIDIA Jetson VI node) advertise Y16: one little-endian 16-bit word per pixel, MSB aligned.
     * cscore must not be asked to convert it: its Y16-to-grey conversion is a per-frame min/max
     * stretch (cv::normalize), which would make exposure and gain meaningless for vision processing,
     * and in 2026.x it throws a cv::Exception from inside the frame pipeline and takes the JVM down
     * with it. So request the frame in Y16, which cscore hands over without conversion, keep the top
     * 8 bits, and replicate to BGR for the rest of the pipeline.
     */
    private CapturedFrame grabY16(VideoMode mode) {
        if (Y16_LATEST_FRAME) {
            return grabY16Latest(mode);
        }
        try (var frame = new RawFrame()) {
            frame.setInfo(mode.width, mode.height, mode.width * 2, PixelFormat.Y16);

            // This is from wpi::nt::Now, or WPIUtilJNI.now(). The epoch from grabFrame is nS since
            // Hal::initialize was called
            long captureTimeNs =
                    CscoreExtras.grabRawSinkFrameTimeoutLastTime(
                            cvSink.getHandle(), frame.getNativeObj(), CSCORE_DEFAULT_FRAME_TIMEOUT, lastTime);
            lastTime = captureTimeNs;

            if (captureTimeNs == 0) {
                logger.error("Error grabbing image: " + cvSink.getError());
                return new CapturedFrame(new CVMat(), settables.getFrameStaticProperties(), 0);
            }

            // View the frame's buffer as 16-bit grey (no copy). The Java-side RawFrame fields are
            // not filled in by our JNI grab, so the geometry/format check happens natively; 0 means
            // cscore converted or resized behind our back and the data would be misread.
            long rawPtr =
                    CscoreExtras.wrapRawFrameAs(
                            frame.getNativeObj(),
                            mode.width,
                            mode.height,
                            PixelFormat.Y16.getValue(),
                            CvType.CV_16UC1);
            if (rawPtr == 0) {
                logger.error(
                        "Expected a "
                                + mode.width
                                + "x"
                                + mode.height
                                + " Y16 frame from cscore, got something else");
                return new CapturedFrame(new CVMat(), settables.getFrameStaticProperties(), 0);
            }

            var raw = new Mat(rawPtr);
            var gray = new Mat();
            // Keep the top 8 bits (for a 10-bit sensor in [15:6], the 8 MSBs of the sample)
            raw.convertTo(gray, CvType.CV_8UC1, 1.0 / 256.0);
            raw.release();

            var bgr = new Mat();
            Imgproc.cvtColor(gray, bgr, Imgproc.COLOR_GRAY2BGR);

            // Both go out: BGR for drawing and streaming, grey for greyscale pipelines (which
            // would otherwise convert the BGR straight back)
            return new CapturedFrame(
                    new CVMat(bgr), new CVMat(gray), settables.getFrameStaticProperties(), captureTimeNs);
        }
    }

    /**
     * The newest frame from a Y16 source, as 8-bit grey, without cscore copying every frame.
     *
     * <p>Disabling the sink stops cscore streaming (it keeps the device open, its buffers allocated
     * and its controls working); V4l2LatestFrame then streams on that same descriptor, keeps only the
     * newest filled buffer and converts it here. Frames the pipeline does not get to are requeued
     * untouched.
     */
    private CapturedFrame grabY16Latest(VideoMode mode) {
        if (latestFrame == 0) {
            cvSink.setEnabled(false);
            latestFrame = V4l2LatestFrame.create(camera.getPath(), mode.width, mode.height);
            logger.info("Y16: newest-frame capture on cscore's descriptor for " + camera.getPath());
            // Convert on the GPU into pinned frames the GPU detector reads in place, unless
            // PHOTONVISION_Y16_GPU_CAPTURE=0 or there is no GPU library.
            if (!"0".equals(System.getenv("PHOTONVISION_Y16_GPU_CAPTURE"))
                    && GpuAprilTagJNI.isAvailable()) {
                boolean on = V4l2LatestFrame.setGpu(latestFrame, GpuAprilTagJNI.captureOps());
                logger.info(
                        "Y16: GPU conversion " + (on ? "on" : "unavailable") + " for " + camera.getPath());
            }
            // Secondary feeds of this camera read the same capture.
            sharedCapture =
                    SharedCaptures.register(
                            sharedPath(), latestFrame, () -> settables.getFrameStaticProperties());
        }
        var gray = new Mat();
        long captureTimeNs =
                V4l2LatestFrame.grab(latestFrame, gray.getNativeObjAddr(), CSCORE_DEFAULT_FRAME_TIMEOUT);
        if (captureTimeNs == 0) {
            gray.release();
            logger.error("No new Y16 frame: " + V4l2LatestFrame.lastError(latestFrame));
            return new CapturedFrame(new CVMat(), settables.getFrameStaticProperties(), 0);
        }
        // The heads are monochrome: the "colour" image is a second header on the grey pixels
        // (refcounted, no copy) rather than a 3-channel copy of every frame. Streams and
        // snapshots draw on their own resized or converted copies (OutputStreamPipeline).
        var color = gray.submat(0, gray.rows(), 0, gray.cols());
        return new CapturedFrame(
                new CVMat(color), new CVMat(gray), settables.getFrameStaticProperties(), captureTimeNs);
    }

    private String sharedPath() {
        return settables.getConfiguration().matchedCameraInfo.uniquePath();
    }

    @Override
    public String getName() {
        return "USBFrameProvider - " + cvSink.getName();
    }

    @Override
    public void release() {
        if (sharedCapture != null) {
            SharedCaptures.unregister(sharedPath(), sharedCapture); // waits out any reader
            sharedCapture = null;
        }
        if (latestFrame != 0) {
            V4l2LatestFrame.destroy(latestFrame);
            latestFrame = 0;
        }
        CameraServer.removeServer(cvSink.getName());
        cvSink.close();
        cvSink = null;
    }

    @Override
    public void onCameraConnected() {
        logger.info("Camera connected! running callback");

        super.onCameraConnected();

        this.connectedCallback.run();
    }

    @Override
    public boolean checkCameraConnected() {
        return camera.isConnected();
    }

    public void updateSettables(VisionSourceSettables settables) {
        this.settables = settables;
    }
}
