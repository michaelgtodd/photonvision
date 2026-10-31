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

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Supplier;
import org.photonvision.vision.frame.FrameStaticProperties;

/**
 * Newest-frame captures (V4l2LatestFrame handles) by camera unique path, so a secondary feed can
 * read the frames its parent camera already captures instead of opening the device again.
 *
 * <p>The parent registers its capture once it exists and unregisters it before destroying it.
 * Readers hold the read lock across a grab; unregistering takes the write lock, so a handle is
 * never destroyed while another feed is reading from it.
 */
public final class SharedCaptures {
    public static final class Capture {
        public final long handle;
        private final Supplier<FrameStaticProperties> staticProperties;
        final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
        boolean closed = false; // guarded by lock

        private Capture(long handle, Supplier<FrameStaticProperties> staticProperties) {
            this.handle = handle;
            this.staticProperties = staticProperties;
        }

        /** The parent camera's current frame properties: resolution, FOV and calibration. */
        public FrameStaticProperties staticProperties() {
            return staticProperties.get();
        }
    }

    private static final ConcurrentHashMap<String, Capture> captures = new ConcurrentHashMap<>();

    private SharedCaptures() {}

    static Capture register(
            String uniquePath, long handle, Supplier<FrameStaticProperties> staticProperties) {
        var capture = new Capture(handle, staticProperties);
        captures.put(uniquePath, capture);
        return capture;
    }

    static void unregister(String uniquePath, Capture capture) {
        capture.lock.writeLock().lock();
        try {
            capture.closed = true;
            captures.remove(uniquePath, capture);
        } finally {
            capture.lock.writeLock().unlock();
        }
    }

    /** The capture registered for this camera unique path, or null. */
    public static Capture get(String uniquePath) {
        return captures.get(uniquePath);
    }
}
