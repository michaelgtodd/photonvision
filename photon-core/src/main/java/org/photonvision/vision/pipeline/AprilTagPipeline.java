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

import edu.wpi.first.apriltag.AprilTagDetection;
import edu.wpi.first.apriltag.AprilTagDetector;
import edu.wpi.first.apriltag.AprilTagPoseEstimate;
import edu.wpi.first.apriltag.AprilTagPoseEstimator.Config;
import edu.wpi.first.math.geometry.CoordinateSystem;
import edu.wpi.first.math.geometry.Pose3d;
import edu.wpi.first.math.geometry.Rotation3d;
import edu.wpi.first.math.geometry.Transform3d;
import edu.wpi.first.math.util.Units;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.dataflow.structures.Packet;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.math.MathUtils;
import org.photonvision.estimation.TargetModel;
import org.photonvision.jni.GpuAprilTagJNI;
import org.photonvision.targeting.MultiTargetPNPResult;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.apriltag.FarFieldBand;
import org.photonvision.vision.apriltag.FarFieldSearch;
import org.photonvision.vision.apriltag.RoiTracker;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.pipe.CVPipe.CVPipeResult;
import org.photonvision.vision.pipe.impl.AprilTagDetectionGpuPipe;
import org.photonvision.vision.pipe.impl.AprilTagDetectionGpuPipe.AprilTagDetectionGpuPipeParams;
import org.photonvision.vision.pipe.impl.AprilTagDetectionPipe;
import org.photonvision.vision.pipe.impl.AprilTagDetectionPipe.AprilTagDetectionPipeParams;
import org.photonvision.vision.pipe.impl.AprilTagPoseEstimatorPipe;
import org.photonvision.vision.pipe.impl.AprilTagPoseEstimatorPipe.AprilTagPoseEstimatorPipeParams;
import org.photonvision.vision.pipe.impl.CalculateFPSPipe;
import org.photonvision.vision.pipe.impl.MultiTargetPNPPipe;
import org.photonvision.vision.pipe.impl.MultiTargetPNPPipe.MultiTargetPNPPipeParams;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.photonvision.vision.target.TrackedTarget;
import org.photonvision.vision.target.TrackedTarget.TargetCalculationParameters;

public class AprilTagPipeline extends CVPipeline<CVPipelineResult, AprilTagPipelineSettings> {
    private static final Logger logger = new Logger(AprilTagPipeline.class, LogGroup.VisionModule);

    private final AprilTagDetectionPipe aprilTagDetectionPipe = new AprilTagDetectionPipe();
    // The GPU detector (photon-gpu on a Jetson); created on first use, kept for the pipeline's life
    private AprilTagDetectionGpuPipe gpuDetectionPipe = null;
    private boolean useGpuDetector = false;

    // Tiers 2 and 3 (see docs/apriltag-rate-plan.md in jetson-gmsl-quad-ar0234): a background
    // full-resolution search of the far-field band seeds a per-frame full-resolution ROI
    // re-detection of tags the main detector did not see.
    private FarFieldSearch farFieldSearch = null;
    private RoiTracker roiTracker = null;
    private AprilTagDetector.QuadThresholdParameters lastQuadParams = null;
    private long tierSeq = 0;
    private long lastTierLogNanos = 0;
    private final AprilTagPoseEstimatorPipe singleTagPoseEstimatorPipe =
            new AprilTagPoseEstimatorPipe();
    private final MultiTargetPNPPipe multiTagPNPPipe = new MultiTargetPNPPipe();
    private final CalculateFPSPipe calculateFPSPipe = new CalculateFPSPipe();

    private static final FrameThresholdType PROCESSING_TYPE = FrameThresholdType.GREYSCALE;

    public AprilTagPipeline() {
        super(PROCESSING_TYPE);
        settings = new AprilTagPipelineSettings();
    }

    public AprilTagPipeline(AprilTagPipelineSettings settings) {
        super(PROCESSING_TYPE);
        this.settings = settings;
    }

    @Override
    protected void setPipeParamsImpl() {
        // Sanitize thread count - not supported to have fewer than 1 threads
        settings.threads = Math.max(1, settings.threads);

        // for now, hard code tag width based on enum value
        // From 2024 best guess is 6.5
        double tagWidth = Units.inchesToMeters(6.5);
        TargetModel tagModel = TargetModel.kAprilTag36h11;
        if (settings.tagFamily == AprilTagFamily.kTag16h5) {
            // 2023 tag, 6in
            tagWidth = Units.inchesToMeters(6);
            tagModel = TargetModel.kAprilTag16h5;
        }

        var config = new AprilTagDetector.Config();
        config.numThreads = settings.threads;
        config.refineEdges = settings.refineEdges;
        config.quadSigma = (float) settings.blur;
        config.quadDecimate = settings.decimate;

        var quadParams = new AprilTagDetector.QuadThresholdParameters();
        // 5 was the default minClusterPixels in WPILib prior to 2025
        // increasing it causes detection problems when decimate > 1
        quadParams.minClusterPixels = 5;
        // these are the same as the values in WPILib 2025
        // setting them here to prevent upstream changes from changing behavior of the detector
        quadParams.maxNumMaxima = 10;
        quadParams.criticalAngle = 45 * Math.PI / 180.0;
        quadParams.maxLineFitMSE = 10.0f;
        quadParams.minWhiteBlackDiff = 5;
        quadParams.deglitch = false;

        aprilTagDetectionPipe.setParams(
                new AprilTagDetectionPipeParams(settings.tagFamily, config, quadParams));

        lastQuadParams = quadParams;
        if (settings.farFieldEnabled) {
            if (farFieldSearch == null) farFieldSearch = new FarFieldSearch(getClass().getSimpleName());
            farFieldSearch.configure(
                    settings.tagFamily.getNativeName(), settings.farFieldThreads, quadParams);
        }
        if (settings.roiTrackEnabled) {
            if (roiTracker == null) roiTracker = new RoiTracker();
            roiTracker.configure(settings.tagFamily.getNativeName(), 1, quadParams);
        }

        // GPU detector: only when asked for and the native library is installed; otherwise the
        // CPU detector above stays in use and the setting is a no-op.
        useGpuDetector = settings.gpuDetector && GpuAprilTagJNI.isAvailable();
        if (useGpuDetector) {
            if (gpuDetectionPipe == null) gpuDetectionPipe = new AprilTagDetectionGpuPipe();
            gpuDetectionPipe.setParams(
                    new AprilTagDetectionGpuPipeParams(
                            settings.tagFamily,
                            settings.decimate,
                            settings.threads,
                            quadParams.minWhiteBlackDiff,
                            settings.refineEdges,
                            0.25f));
        }

        if (frameStaticProperties.cameraCalibration != null) {
            var cameraMatrix = frameStaticProperties.cameraCalibration.getCameraIntrinsicsMat();
            if (cameraMatrix != null && cameraMatrix.rows() > 0) {
                var cx = cameraMatrix.get(0, 2)[0];
                var cy = cameraMatrix.get(1, 2)[0];
                var fx = cameraMatrix.get(0, 0)[0];
                var fy = cameraMatrix.get(1, 1)[0];

                singleTagPoseEstimatorPipe.setParams(
                        new AprilTagPoseEstimatorPipeParams(
                                new Config(tagWidth, fx, fy, cx, cy),
                                frameStaticProperties.cameraCalibration,
                                settings.numIterations));

                // TODO global state ew
                var atfl = ConfigManager.getInstance().getConfig().getApriltagFieldLayout();
                multiTagPNPPipe.setParams(
                        new MultiTargetPNPPipeParams(frameStaticProperties.cameraCalibration, atfl, tagModel));
            }
        }
    }

    /**
     * Tier 2 (far-field band, background) and tier 3 (ROI re-detection, this frame), merged with the
     * main detector's results: an id re-detected at full resolution replaces the main detector's
     * version; ids only tier 3 found are added.
     */
    private List<AprilTagDetection> runTiers(
            Frame frame, AprilTagPipelineSettings settings, List<AprilTagDetection> tier1) {
        if (!settings.farFieldEnabled && !settings.roiTrackEnabled) return tier1;
        var grey = frame.processedImage.getMat();
        if (grey.empty()) return tier1;
        long seq = ++tierSeq;

        List<AprilTagDetection> seeds = List.of();
        if (settings.farFieldEnabled && farFieldSearch != null) {
            var band =
                    FarFieldBand.fromFractions(
                            grey.cols(), grey.rows(), settings.farFieldBandTop, settings.farFieldBandBottom);
            var cal = frameStaticProperties.cameraCalibration;
            if (settings.farFieldAutoBand && cal != null) {
                var k = cal.getCameraIntrinsicsMat();
                if (k != null && k.rows() > 0) {
                    band =
                            FarFieldBand.fromMount(
                                    grey.cols(),
                                    grey.rows(),
                                    k.get(1, 1)[0],
                                    k.get(1, 2)[0],
                                    settings.mountHeightMeters,
                                    settings.mountPitchDegrees,
                                    settings.farFieldMinDistanceMeters,
                                    settings.tagHeightMinMeters,
                                    settings.tagHeightMaxMeters,
                                    settings.tiltMarginDegrees,
                                    32);
                }
            }
            double interval = settings.farFieldRateHz > 0 ? 1.0 / settings.farFieldRateHz : 1.0;
            farFieldSearch.offer(grey, band, settings.farFieldUpsample, seq, interval);
            seeds = farFieldSearch.takeResults();
        }

        if (!settings.roiTrackEnabled || roiTracker == null) {
            // Without tier 3 the far-field results are used as they are (older frame)
            if (seeds.isEmpty()) return tier1;
            return merge(tier1, seeds);
        }

        var params =
                new RoiTracker.Params(
                        settings.roiMargin,
                        settings.roiPadPx,
                        settings.roiMaxCount,
                        settings.roiMaxMisses,
                        settings.roiUpsample);
        var tier3 = roiTracker.update(grey, seq, tier1, seeds, params);

        long now = System.nanoTime();
        if (now - lastTierLogNanos > 30_000_000_000L) {
            lastTierLogNanos = now;
            logger.info(
                    "tiers: main "
                            + tier1.size()
                            + ", far-field seeds "
                            + seeds.size()
                            + " (last search "
                            + (farFieldSearch != null
                                    ? String.format("%.1f", farFieldSearch.lastSearchMillis())
                                    : "-")
                            + " ms), roi "
                            + tier3.size()
                            + " of "
                            + roiTracker.trackCount()
                            + " tracked");
        }
        return merge(tier1, tier3);
    }

    /** Detections from {@code fullRes} replace same-id entries of {@code base}; others are added. */
    private static List<AprilTagDetection> merge(
            List<AprilTagDetection> base, List<AprilTagDetection> fullRes) {
        if (fullRes.isEmpty()) return base;
        var out = new ArrayList<AprilTagDetection>(base.size() + fullRes.size());
        var replaced = new java.util.HashSet<Integer>();
        for (var d : fullRes) replaced.add(d.getId());
        for (var d : base) if (!replaced.contains(d.getId())) out.add(d);
        out.addAll(fullRes);
        return out;
    }

    @Override
    protected CVPipelineResult process(Frame frame, AprilTagPipelineSettings settings) {
        long sumPipeNanosElapsed = 0L;

        if (frame.type != FrameThresholdType.GREYSCALE) {
            // We asked for a GREYSCALE frame, but didn't get one -- best we can do is give up
            return new CVPipelineResult(frame.sequenceID, 0, 0, List.of(), frame);
        }

        CVPipeResult<List<AprilTagDetection>> tagDetectionPipeResult =
                useGpuDetector
                        ? gpuDetectionPipe.run(frame.processedImage)
                        : aprilTagDetectionPipe.run(frame.processedImage);
        sumPipeNanosElapsed += tagDetectionPipeResult.nanosElapsed;

        List<AprilTagDetection> detections = tagDetectionPipeResult.output;
        detections = runTiers(frame, settings, detections);
        List<AprilTagDetection> usedDetections = new ArrayList<>();
        List<TrackedTarget> targetList = new ArrayList<>();

        // Filter out detections based on pipeline settings
        for (AprilTagDetection detection : detections) {
            // TODO this should be in a pipe, not in the top level here (Matt)
            if (detection.getDecisionMargin() < settings.decisionMargin) continue;
            if (detection.getHamming() > settings.hammingDist) continue;

            usedDetections.add(detection);

            // Populate target list for multitag
            // (TODO: Address circular dependencies. Multitag only requires corners and IDs, this should
            // not be necessary.)
            TrackedTarget target =
                    new TrackedTarget(
                            detection,
                            null,
                            new TargetCalculationParameters(
                                    false, null, null, null, null, frameStaticProperties));

            targetList.add(target);
        }

        // Do multi-tag pose estimation
        Optional<MultiTargetPNPResult> multiTagResult = Optional.empty();
        if (settings.solvePNPEnabled && settings.doMultiTarget) {
            var multiTagOutput = multiTagPNPPipe.run(targetList);
            sumPipeNanosElapsed += multiTagOutput.nanosElapsed;
            multiTagResult = multiTagOutput.output;
        }

        // Do single-tag pose estimation
        if (settings.solvePNPEnabled) {
            // Clear target list that was used for multitag so we can add target transforms
            targetList.clear();
            // TODO global state again ew
            var atfl = ConfigManager.getInstance().getConfig().getApriltagFieldLayout();

            for (AprilTagDetection detection : usedDetections) {
                AprilTagPoseEstimate tagPoseEstimate = null;
                // Do single-tag estimation when "always enabled" or if a tag was not used for multitag
                if (settings.doSingleTargetAlways
                        || !(multiTagResult.isPresent()
                                && multiTagResult.get().fiducialIDsUsed.contains((short) detection.getId()))) {
                    var poseResult = singleTagPoseEstimatorPipe.run(detection);
                    sumPipeNanosElapsed += poseResult.nanosElapsed;
                    tagPoseEstimate = poseResult.output;
                }

                // If single-tag estimation was not done, this is a multi-target tag from the layout
                if (tagPoseEstimate == null && multiTagResult.isPresent()) {
                    // compute this tag's camera-to-tag transform using the multitag result
                    var tagPose = atfl.getTagPose(detection.getId());
                    if (tagPose.isPresent()) {
                        var camToTag =
                                new Transform3d(
                                        new Pose3d().plus(multiTagResult.get().estimatedPose.best), tagPose.get());
                        // match expected AprilTag coordinate system
                        camToTag =
                                CoordinateSystem.convert(camToTag, CoordinateSystem.NWU(), CoordinateSystem.EDN());
                        // (AprilTag expects Z axis going into tag)
                        camToTag =
                                new Transform3d(
                                        camToTag.getTranslation(),
                                        new Rotation3d(0, Math.PI, 0).plus(camToTag.getRotation()));
                        tagPoseEstimate = new AprilTagPoseEstimate(camToTag, camToTag, 0, 0);
                    }
                }

                // populate the target list
                // Challenge here is that TrackedTarget functions with OpenCV Contour
                TrackedTarget target =
                        new TrackedTarget(
                                detection,
                                tagPoseEstimate,
                                new TargetCalculationParameters(
                                        false, null, null, null, null, frameStaticProperties));

                var correctedBestPose =
                        MathUtils.convertOpenCVtoPhotonTransform(target.getBestCameraToTarget3d());
                var correctedAltPose =
                        MathUtils.convertOpenCVtoPhotonTransform(target.getAltCameraToTarget3d());

                target.setBestCameraToTarget3d(
                        new Transform3d(correctedBestPose.getTranslation(), correctedBestPose.getRotation()));
                target.setAltCameraToTarget3d(
                        new Transform3d(correctedAltPose.getTranslation(), correctedAltPose.getRotation()));

                targetList.add(target);
            }
        }

        if (targetList.size() > Packet.MAX_ARRAY_LEN) {
            logger.error(
                    "We have " + targetList.size() + " targets! Arbitrarily dropping some on the floor");
            targetList = targetList.subList(0, Packet.MAX_ARRAY_LEN);
        }

        var fpsResult = calculateFPSPipe.run(null);
        var fps = fpsResult.output;

        return new CVPipelineResult(
                frame.sequenceID, sumPipeNanosElapsed, fps, targetList, multiTagResult, frame);
    }

    @Override
    public void release() {
        aprilTagDetectionPipe.release();
        if (gpuDetectionPipe != null) gpuDetectionPipe.release();
        if (farFieldSearch != null) farFieldSearch.close();
        if (roiTracker != null) roiTracker.close();
        singleTagPoseEstimatorPipe.release();
        super.release();
    }
}
