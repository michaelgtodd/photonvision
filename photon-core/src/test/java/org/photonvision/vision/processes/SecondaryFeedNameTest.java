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

package org.photonvision.vision.processes;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import org.junit.jupiter.api.Test;

/** A secondary feed's name follows its camera's (VisionSourceManager.followingFeedName). */
public class SecondaryFeedNameTest {
    private static String follow(String feed, String oldName, String newName) {
        return VisionSourceManager.followingFeedName(feed, oldName, newName);
    }

    @Test
    public void feedNamedAfterItsCameraFollows() {
        assertEquals("front-left-full", follow("gmsl-linkA-full", "gmsl-linkA", "front-left"));
        assertEquals("gmsl-linkA-full", follow("front-left-full", "front-left", "gmsl-linkA"));
        assertEquals("Front Left (far)", follow("Cam A (far)", "Cam A", "Front Left"));
        assertEquals("left_full", follow("cam_full", "cam", "left"));
    }

    @Test
    public void feedWithANameOfItsOwnKeepsIt() {
        assertNull(follow("long-range", "gmsl-linkA", "front-left"));
        // a prefix of a longer word is not the camera's name
        assertNull(follow("gmsl-linkAB-full", "gmsl-linkA", "front-left"));
        assertNull(follow("Apple", "A", "B"));
        assertNull(follow("gmsl-linkA", "gmsl-linkA", "front-left"));
        assertNull(follow("gmsl-linkA-full", "", "front-left"));
        assertNull(follow(null, "gmsl-linkA", "front-left"));
    }
}
