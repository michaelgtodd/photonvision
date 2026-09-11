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

package org.photonvision.jni;

import edu.wpi.first.util.RawFrame;
import edu.wpi.first.util.TimestampSource;

public class CscoreExtras {
    /**
     * Fill {@param framePtr} with the latest image from the source this sink is connected to.
     *
     * <p>If lastFrameTime is provided and non-zero, the sink will fill image with the first frame
     * from the source that is not equal to lastFrameTime. If lastFrameTime is zero, the time of the
     * current frame owned by the CvSource is used, and this function will block until the connected
     * CvSource provides a new frame.
     *
     * @param sink Sink handle.
     * @param framePtr Pointer to a wpi::RawFrame.
     * @param timeout Timeout in seconds.
     * @param lastFrameTime Timestamp of the last frame - used to compare new frames against.
     * @return Frame time, in uS, of the incoming frame.
     */
    public static native long grabRawSinkFrameTimeoutLastTime(
            int sink, long framePtr, double timeout, long lastFrameTime);

    /**
     * Wrap the data owned by a RawFrame in a cv::Mat
     *
     * @param rawFramePtr
     * @return pointer to a cv::Mat
     */
    public static native long wrapRawFrame(long rawFramePtr);

    /**
     * Wrap the data owned by a RawFrame in a cv::Mat of a caller-chosen element type, after checking
     * that the frame really holds what the caller expects.
     *
     * <p>Unlike {@link #wrapRawFrame}, which picks the type from the frame's pixel format and views
     * Y16 as two 8-bit channels, this lets a 16-bit grey frame be viewed as CV_16UC1 so it can be
     * converted rather than misread. The Java-side fields of a RawFrame are not updated by {@link
     * #grabRawSinkFrameTimeoutLastTime}, so the check is done against the native frame.
     *
     * @param rawFramePtr Pointer to a wpi::RawFrame.
     * @param width Expected width, in pixels.
     * @param height Expected height, in pixels.
     * @param pixelFormat Expected WPI pixel format (PixelFormat.getValue()).
     * @param cvType OpenCV element type to view the data as (CvType.CV_16UC1, for instance).
     * @return pointer to a cv::Mat, or 0 if the frame does not match the expected geometry/format.
     */
    public static native long wrapRawFrameAs(
            long rawFramePtr, int width, int height, int pixelFormat, int cvType);

    private static native int getTimestampSourceNative(long rawFramePtr);

    public static TimestampSource getTimestampSource(RawFrame frame) {
        return TimestampSource.getFromInt(getTimestampSourceNative(frame.getNativeObj()));
    }
}
