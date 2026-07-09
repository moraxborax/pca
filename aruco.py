from pathlib import Path

import cv2
import numpy as np

# A4 dimensions in millimeters (landscape)
A4_WIDTH_MM = 297.0
A4_HEIGHT_MM = 210.0

# Marker IDs: top-left, top-right, bottom-left, bottom-right, center
DEFAULT_MARKER_IDS = (0, 1, 2, 3, 4)


def mm_to_pixels(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


def generate_aruco_sheet(
    output_path: str | Path = "aruco_sheet.png",
    *,
    marker_ids: tuple[int, int, int, int, int] = DEFAULT_MARKER_IDS,
    marker_size_mm: float = 50.0,
    margin_mm: float = 15.0,
    dpi: int = 300,
    dictionary: int = cv2.aruco.DICT_4X4_50,
) -> np.ndarray:
    """
    Generate an A4 landscape sheet with five ArUco markers.

    Layout (marker IDs by default):
        0 (top-left)              1 (top-right)

                      4 (center)

        2 (bottom-left)           3 (bottom-right)
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    marker_px = mm_to_pixels(marker_size_mm, dpi)
    page_w = mm_to_pixels(A4_WIDTH_MM, dpi)
    page_h = mm_to_pixels(A4_HEIGHT_MM, dpi)
    margin_px = mm_to_pixels(margin_mm, dpi)

    sheet = np.full((page_h, page_w), 255, dtype=np.uint8)

    positions_mm = [
        (margin_mm, margin_mm),  # top-left
        (A4_WIDTH_MM - margin_mm - marker_size_mm, margin_mm),  # top-right
        (margin_mm, A4_HEIGHT_MM - margin_mm - marker_size_mm),  # bottom-left
        (
            A4_WIDTH_MM - margin_mm - marker_size_mm,
            A4_HEIGHT_MM - margin_mm - marker_size_mm,
        ),  # bottom-right
        (
            (A4_WIDTH_MM - marker_size_mm) / 2,
            (A4_HEIGHT_MM - marker_size_mm) / 2,
        ),  # center
    ]

    for marker_id, (x_mm, y_mm) in zip(marker_ids, positions_mm, strict=True):
        marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_px)
        x_px = mm_to_pixels(x_mm, dpi)
        y_px = mm_to_pixels(y_mm, dpi)
        sheet[y_px : y_px + marker_px, x_px : x_px + marker_px] = marker

    output_path = Path(output_path)
    cv2.imwrite(str(output_path), sheet)
    return sheet


if __name__ == "__main__":
    generate_aruco_sheet()

