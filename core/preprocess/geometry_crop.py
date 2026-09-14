import math

import cv2
import numpy as np

from core.geometry.haversine import haversine_distance


def find_nearest_pixels(corners, geometry_coords):
    """
    Find the pixel coordinates in a geometry table that are
    geographically closest to the supplied image corners.

    corners:
        Array-like of [latitude, longitude] points.

    geometry_coords:
        Array-like rows where:
        [longitude, latitude, scan, pixel]
    """
    corners = np.asarray(corners, dtype=float)
    geometry_coords = np.asarray(geometry_coords, dtype=float)

    if corners.shape != (4, 2):
        raise ValueError("corners must have shape (4, 2)")

    if geometry_coords.ndim != 2 or geometry_coords.shape[1] < 4:
        raise ValueError(
            "geometry_coords must contain longitude, latitude, scan and pixel"
        )

    pixel_coords = np.zeros((4, 2), dtype=np.int32)

    for i, (lat, lon) in enumerate(corners):
        distances = np.array(
            [
                haversine_distance(
                    lat,
                    lon,
                    row[1],
                    row[0],
                )
                for row in geometry_coords
            ]
        )

        nearest_index = int(np.argmin(distances))
        pixel_coords[i] = geometry_coords[nearest_index, 2:4]

    return pixel_coords


def crop_polygon(image, pixel_coords):
    """
    Crop the bounding box around a quadrilateral and mask
    everything outside the polygon.
    """
    image = np.asarray(image)
    pixel_coords = np.asarray(pixel_coords, dtype=np.int32)

    if pixel_coords.shape != (4, 2):
        raise ValueError("pixel_coords must have shape (4, 2)")

    min_x, min_y = pixel_coords.min(axis=0)
    max_x, max_y = pixel_coords.max(axis=0)

    min_x = max(0, int(min_x))
    min_y = max(0, int(min_y))
    max_x = min(image.shape[1], int(max_x))
    max_y = min(image.shape[0], int(max_y))

    if min_x >= max_x or min_y >= max_y:
        raise ValueError("Invalid crop coordinates")

    bounding_box = image[min_y:max_y, min_x:max_x].copy()

    local_coords = pixel_coords.copy()
    local_coords[:, 0] -= min_x
    local_coords[:, 1] -= min_y

    mask = np.zeros(bounding_box.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, local_coords, 1)

    cropped = np.zeros_like(bounding_box)
    cropped[mask.astype(bool)] = bounding_box[mask.astype(bool)]

    return cropped, local_coords


def rotate_image(image, angle_degrees):
    """Rotate an image around its center."""
    image = np.asarray(image)

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)

    matrix = cv2.getRotationMatrix2D(
        center,
        float(angle_degrees),
        1.0,
    )

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
    )


def estimate_bottom_edge_angle(pixel_coords):
    """
    Estimate the orientation of the bottom edge of a quadrilateral.
    """
    pixel_coords = np.asarray(pixel_coords, dtype=float)

    if pixel_coords.shape != (4, 2):
        raise ValueError("pixel_coords must have shape (4, 2)")

    # Expected order: UL, UR, LR, LL
    lower_right = pixel_coords[2]
    lower_left = pixel_coords[3]

    dx = lower_right[0] - lower_left[0]
    dy = lower_right[1] - lower_left[1]

    radians = math.atan2(dx, dy)

    return 90.0 - math.degrees(radians)