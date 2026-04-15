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

package org.photonvision.common.util;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;

/**
 * Tracks camera disconnect/connect events across all cameras to detect simultaneous dropouts. This
 * is diagnostic instrumentation for debugging the all-cameras-drop-at-once issue observed with
 * B0578 (OV9281) cameras while driving.
 */
public class CameraHealthMonitor {
    private static final Logger logger =
            new Logger(CameraHealthMonitor.class, "CameraHealthMonitor", LogGroup.Camera);

    private static final Map<String, Long> lastDisconnectTimeNs = new ConcurrentHashMap<>();
    private static final Map<String, Long> lastConnectTimeNs = new ConcurrentHashMap<>();
    private static final Map<String, Integer> consecutiveGrabErrors = new ConcurrentHashMap<>();
    private static final Map<String, Long> firstGrabErrorTimeNs = new ConcurrentHashMap<>();
    private static final Map<String, Long> lastSuccessTimeNs = new ConcurrentHashMap<>();

    // Window within which disconnects are considered "simultaneous"
    private static final long SIMULTANEOUS_WINDOW_NS = 500_000_000L; // 500ms

    /**
     * Report that a camera has disconnected. Checks for simultaneous disconnects across cameras.
     *
     * @param cameraName Unique name/identifier for the camera
     */
    public static void reportDisconnect(String cameraName) {
        long now = System.nanoTime();
        lastDisconnectTimeNs.put(cameraName, now);

        // Check if other cameras disconnected within the simultaneous window
        StringBuilder concurrent = new StringBuilder();
        for (var entry : lastDisconnectTimeNs.entrySet()) {
            if (!entry.getKey().equals(cameraName)
                    && Math.abs(now - entry.getValue()) < SIMULTANEOUS_WINDOW_NS) {
                concurrent.append(entry.getKey()).append(" ");
            }
        }

        if (concurrent.length() > 0) {
            logger.error(
                    "SIMULTANEOUS DISCONNECT DETECTED: '"
                            + cameraName
                            + "' dropped within 500ms of: ["
                            + concurrent.toString().trim()
                            + "]");
        } else {
            logger.warn("Camera '" + cameraName + "' disconnected");
        }
    }

    /**
     * Report that a camera has reconnected.
     *
     * @param cameraName Unique name/identifier for the camera
     */
    public static void reportConnect(String cameraName) {
        long now = System.nanoTime();
        Long disconnectTime = lastDisconnectTimeNs.get(cameraName);
        lastConnectTimeNs.put(cameraName, now);

        if (disconnectTime != null) {
            long outageMs = (now - disconnectTime) / 1_000_000;
            logger.warn("Camera '" + cameraName + "' reconnected (outage duration: " + outageMs + "ms)");
        } else {
            logger.warn("Camera '" + cameraName + "' connected");
        }
    }

    /**
     * Report a frame grab error. Tracks consecutive errors and logs outage duration.
     *
     * @param cameraName Unique name/identifier for the camera
     * @param errorMsg The error message from cscore
     */
    public static void reportGrabError(String cameraName, String errorMsg) {
        int errorCount = consecutiveGrabErrors.compute(cameraName, (k, v) -> (v == null ? 0 : v) + 1);

        if (errorCount == 1) {
            firstGrabErrorTimeNs.put(cameraName, System.nanoTime());
        }

        Long lastSuccess = lastSuccessTimeNs.get(cameraName);
        long msSinceSuccess =
                (lastSuccess != null) ? (System.nanoTime() - lastSuccess) / 1_000_000 : -1;

        // Log first error, then every 10th to avoid flooding
        if (errorCount == 1 || errorCount % 10 == 0) {
            logger.warn(
                    "Frame grab error on '"
                            + cameraName
                            + "' (consecutive="
                            + errorCount
                            + ", "
                            + msSinceSuccess
                            + "ms since last success): "
                            + errorMsg);
        }
    }

    /**
     * Report a successful frame grab. Logs recovery if there were preceding errors.
     *
     * @param cameraName Unique name/identifier for the camera
     */
    public static void reportGrabSuccess(String cameraName) {
        Integer prevErrors = consecutiveGrabErrors.get(cameraName);

        if (prevErrors != null && prevErrors > 0) {
            Long firstError = firstGrabErrorTimeNs.get(cameraName);
            long outageMs = (firstError != null) ? (System.nanoTime() - firstError) / 1_000_000 : -1;
            logger.warn(
                    "Camera '"
                            + cameraName
                            + "' frame grab recovered after "
                            + prevErrors
                            + " consecutive errors (outage="
                            + outageMs
                            + "ms)");
        }

        consecutiveGrabErrors.put(cameraName, 0);
        lastSuccessTimeNs.put(cameraName, System.nanoTime());
    }
}
