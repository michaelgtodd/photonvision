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

import org.opencv.core.Rect;

/**
 * Where far tags can be in the image, as a full-width horizontal band of rows.
 *
 * <p>For a robot-mounted camera, every tag beyond some distance projects close to the horizon: the
 * angle above the optical axis of a tag at height {@code hTag} seen from a camera at height {@code
 * hCam} pitched up by {@code pitch} at distance {@code d} is {@code atan((hTag - hCam)/d) - pitch}.
 * The band spans that angle for the lowest and highest field tags at the near edge of the far
 * field, widened by a tilt margin for the robot rocking; farther tags fall inside it. Nearer tags
 * are larger and are the GPU tier's job.
 */
public final class FarFieldBand {
    private FarFieldBand() {}

    /** Manual band from fractions of the image height. */
    public static Rect fromFractions(int width, int height, double top, double bottom) {
        int y0 = (int) Math.round(Math.max(0, Math.min(1, top)) * height);
        int y1 = (int) Math.round(Math.max(0, Math.min(1, bottom)) * height);
        if (y1 <= y0) return new Rect(0, 0, width, 0);
        return new Rect(0, y0, width, y1 - y0);
    }

    /**
     * Band from the camera's mount pose and intrinsics.
     *
     * @param fy focal length in pixels (vertical)
     * @param cy principal point row
     * @param mountHeightM camera height above the floor
     * @param pitchDeg camera pitch, positive = tilted up
     * @param minDistanceM near edge of the far field
     * @param tagHeightMinM lowest field-tag centre above the floor
     * @param tagHeightMaxM highest field-tag centre above the floor
     * @param tiltMarginDeg extra angle either side for robot tilt
     * @param padPx extra rows either side so a tag at the edge is inside the band whole
     */
    public static Rect fromMount(
            int width,
            int height,
            double fy,
            double cy,
            double mountHeightM,
            double pitchDeg,
            double minDistanceM,
            double tagHeightMinM,
            double tagHeightMaxM,
            double tiltMarginDeg,
            int padPx) {
        double pitch = Math.toRadians(pitchDeg);
        double tilt = Math.toRadians(tiltMarginDeg);
        double d = Math.max(0.5, minDistanceM);
        // image rows grow downward; a target above the axis has a smaller row
        double aTop = Math.atan((tagHeightMaxM - mountHeightM) / d) - pitch + tilt;
        double aBot = Math.atan((tagHeightMinM - mountHeightM) / d) - pitch - tilt;
        double rowTop = cy - fy * Math.tan(aTop) - padPx;
        double rowBot = cy - fy * Math.tan(aBot) + padPx;
        int y0 = (int) Math.floor(Math.max(0, Math.min(height, rowTop)));
        int y1 = (int) Math.ceil(Math.max(0, Math.min(height, rowBot)));
        if (y1 <= y0) return new Rect(0, 0, width, 0);
        return new Rect(0, y0, width, y1 - y0);
    }
}
