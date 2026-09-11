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

import com.fasterxml.jackson.annotation.JsonTypeName;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.target.TargetModel;

@JsonTypeName("AprilTagPipelineSettings")
public class AprilTagPipelineSettings extends AdvancedPipelineSettings {
    public AprilTagFamily tagFamily = AprilTagFamily.kTag36h11;
    public int decimate = 1;
    public double blur = 0;
    public int threads = 4; // Multiple threads seems to be better performance on most platforms
    public boolean debug = false;
    public boolean refineEdges = true;
    public int numIterations = 40;
    public int hammingDist = 0;
    public int decisionMargin = 35;
    public boolean doMultiTarget = false;
    public boolean doSingleTargetAlways = false;
    // Use the GPU detector (photon-gpu, NVIDIA Jetson) when it is installed
    public boolean gpuDetector = false;

    // Tier 2: full-resolution search of the far-field band on its own thread
    public boolean farFieldEnabled = false;
    public double farFieldRateHz = 10;
    public double farFieldUpsample =
            1.0; // 1.5 finds blurred tags one size step smaller at 2.25x cost
    public int farFieldThreads = 1;
    public boolean farFieldAutoBand = false; // from mount pose + calibration; else the fractions
    public double farFieldBandTop = 0.35; // fractions of image height
    public double farFieldBandBottom = 0.65;
    public double mountHeightMeters = 0.5;
    public double mountPitchDegrees = 0; // positive = tilted up
    public double farFieldMinDistanceMeters = 5.0;
    public double tagHeightMinMeters = 0.2;
    public double tagHeightMaxMeters = 1.5;
    public double tiltMarginDegrees = 3.0;

    // Tier 3: full-resolution re-detection of known tags in ROIs, every frame
    public boolean roiTrackEnabled = false;
    public double roiMargin = 2.0;
    public int roiPadPx = 16;
    public int roiMaxCount = 8;
    public int roiMaxMisses = 10;
    public double roiUpsample = 1.0;

    // 3d settings

    public AprilTagPipelineSettings() {
        super();
        pipelineType = PipelineType.AprilTag;
        targetModel = TargetModel.kAprilTag6p5in_36h11;
        cameraExposureRaw = 20;
        cameraAutoExposure = false;
        ledMode = false;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = super.hashCode();
        result = prime * result + ((tagFamily == null) ? 0 : tagFamily.hashCode());
        result = prime * result + decimate;
        long temp;
        temp = Double.doubleToLongBits(blur);
        result = prime * result + (int) (temp ^ (temp >>> 32));
        result = prime * result + threads;
        result = prime * result + (debug ? 1231 : 1237);
        result = prime * result + (gpuDetector ? 1231 : 1237);
        result = prime * result + (farFieldEnabled ? 1231 : 1237);
        result = prime * result + Double.hashCode(farFieldRateHz);
        result = prime * result + Double.hashCode(farFieldUpsample);
        result = prime * result + farFieldThreads;
        result = prime * result + (farFieldAutoBand ? 1231 : 1237);
        result = prime * result + Double.hashCode(farFieldBandTop);
        result = prime * result + Double.hashCode(farFieldBandBottom);
        result = prime * result + Double.hashCode(mountHeightMeters);
        result = prime * result + Double.hashCode(mountPitchDegrees);
        result = prime * result + Double.hashCode(farFieldMinDistanceMeters);
        result = prime * result + Double.hashCode(tagHeightMinMeters);
        result = prime * result + Double.hashCode(tagHeightMaxMeters);
        result = prime * result + Double.hashCode(tiltMarginDegrees);
        result = prime * result + (roiTrackEnabled ? 1231 : 1237);
        result = prime * result + Double.hashCode(roiMargin);
        result = prime * result + roiPadPx;
        result = prime * result + roiMaxCount;
        result = prime * result + roiMaxMisses;
        result = prime * result + Double.hashCode(roiUpsample);
        result = prime * result + (refineEdges ? 1231 : 1237);
        result = prime * result + numIterations;
        result = prime * result + hammingDist;
        result = prime * result + decisionMargin;
        result = prime * result + (doMultiTarget ? 1231 : 1237);
        result = prime * result + (doSingleTargetAlways ? 1231 : 1237);
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!super.equals(obj)) return false;
        if (getClass() != obj.getClass()) return false;
        AprilTagPipelineSettings other = (AprilTagPipelineSettings) obj;
        if (tagFamily != other.tagFamily) return false;
        if (decimate != other.decimate) return false;
        if (Double.doubleToLongBits(blur) != Double.doubleToLongBits(other.blur)) return false;
        if (threads != other.threads) return false;
        if (debug != other.debug) return false;
        if (refineEdges != other.refineEdges) return false;
        if (numIterations != other.numIterations) return false;
        if (hammingDist != other.hammingDist) return false;
        if (decisionMargin != other.decisionMargin) return false;
        if (doMultiTarget != other.doMultiTarget) return false;
        if (doSingleTargetAlways != other.doSingleTargetAlways) return false;
        if (gpuDetector != other.gpuDetector) return false;
        if (farFieldEnabled != other.farFieldEnabled) return false;
        if (farFieldRateHz != other.farFieldRateHz) return false;
        if (farFieldUpsample != other.farFieldUpsample) return false;
        if (farFieldThreads != other.farFieldThreads) return false;
        if (farFieldAutoBand != other.farFieldAutoBand) return false;
        if (farFieldBandTop != other.farFieldBandTop) return false;
        if (farFieldBandBottom != other.farFieldBandBottom) return false;
        if (mountHeightMeters != other.mountHeightMeters) return false;
        if (mountPitchDegrees != other.mountPitchDegrees) return false;
        if (farFieldMinDistanceMeters != other.farFieldMinDistanceMeters) return false;
        if (tagHeightMinMeters != other.tagHeightMinMeters) return false;
        if (tagHeightMaxMeters != other.tagHeightMaxMeters) return false;
        if (tiltMarginDegrees != other.tiltMarginDegrees) return false;
        if (roiTrackEnabled != other.roiTrackEnabled) return false;
        if (roiMargin != other.roiMargin) return false;
        if (roiPadPx != other.roiPadPx) return false;
        if (roiMaxCount != other.roiMaxCount) return false;
        if (roiMaxMisses != other.roiMaxMisses) return false;
        if (roiUpsample != other.roiUpsample) return false;
        return true;
    }
}
