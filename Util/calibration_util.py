import os
import cv2 
import numpy as np


class Pattern:
    """
    Chessboard pattern parameters
    """

    def __init__(self, height, width, square_size):
        self.pattern_height = height
        self.pattern_width = width
        self.square_size = float(square_size)  # in world coordinate system (m,mm,...)
        self.pattern_type = "chessboard"
        self.pattern_size = (
            self.pattern_width,
            self.pattern_height,
        )  # number of (inner) corners

        self.pattern_points = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        self.pattern_points[:, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)
        self.pattern_points *= square_size


def get_image_points(
    img, file_name, output_dir, pattern_params: Pattern, save_output=False
) -> np.ndarray:
    """
    Find corners of chessboard, approximate them (subpixel precision)

    Args:
        img: input image
        file_name: input filename
        output_dir: output directory for saving drawn corners
        pattern_params: parameters (height, width, ...) of the real world calibration pattern used
        save_output: save an image with the world points marked, default False

    Returns:
        image_points, object_points arrays for the input image
    """
    found = False
    corners = 0
    found, corners = cv2.findChessboardCorners(img, pattern_params.pattern_size)
    if found:
        print(f"corners found: {len(corners)}")
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 0.1)
        # refine corners
        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        # points in camera/image space
        frame_img_points = corners.reshape(-1, 2)

        # points in real world
        frame_obj_points = pattern_params.pattern_points
    else:
        print("corners not found")
        return None

    outfile = os.path.join(output_dir, file_name + "_board.png")
    if save_output:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis, pattern_params.pattern_size, corners, found)

        # draw real world origin on image
        vis = cv2.circle(
            vis,
            (int(frame_img_points[0][0]), int(frame_img_points[0][1])),
            10,
            (255, 0, 0),
            2,
        )
        cv2.imwrite(outfile, vis)

        print(f"{outfile}... OK")

    return [frame_img_points, frame_obj_points]
