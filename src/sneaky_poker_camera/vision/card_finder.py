import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class CardFinderConfig:
    """Configuration parameters for card detection"""
    min_card_area = 2000
    max_card_area = 2000000
    card_aspect_ratio = 1.4
    aspect_ratio_tolerance = 0.3
    target_size = (224, 224)
    max_working_dim = 1000


class CardFinder:
    """Handles detection and extraction of playing cards from images"""
    def __init__(self,
                 config: Optional[CardFinderConfig] = None,
                 debug: bool = False):

        self.config = config or CardFinderConfig()
        self.debug = debug


    def find_cards(self, image: NDArray) -> List[NDArray]:
        """
        Finds and extracts playing cards from an image.

        Args:
            image (NDArray): Input iamge

        Returns:
            List[NDArray]: List of extract card images
        """
        try:
            height, width = image.shape[:2]
            min_dim = min(height, width)
            if min_dim > self.config.max_working_dim:
                scale = self.config.max_working_dim / min_dim
                working_image = cv2.resize(image, None, fx=scale, fy=scale)
                self._show_debug("Resized for card identification", working_image)
            else:
                working_image = image.copy()
                scale = 1.0

            gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            thresh = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )

            edges = cv2.Canny(blurred, 30, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(edges, kernel, iterations=2)
            self._show_debug("Edge Detection", dilated)
            contours, _ = cv2.findContours(
                dilated,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cards = []
            for contour in contours:
                card = self._extract_card(contour, working_image, scale)
                if card is not None:
                    cards.append(card)
                    self._show_debug("Detected Card", card)

            if self.debug:
                debug_image = working_image.copy()
                cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 2)
                self._show_debug("All Contours", debug_image)

            return cards
        except Exception as e:
            print(f"ERROR - find_cards(): {e}")
            return []


    def _extract_card(self, contour: NDArray, image: NDArray, scale: float) -> Optional[NDArray]:
        """
        Extracts a single card from a contour.

        Args:
            contour (NDArray): Contour to analyze
            image (NDArray): Source image
            scale (float): Scale factor of the image

        Returns:
            Option[NDArray]: Extracted card image if valid, None otherwise
        """
        area = cv2.contourArea(contour)
        scaled_min_area = self.config.min_card_area * (scale ** 2)
        scaled_max_area = self.config.max_card_area * (scale ** 2)

        if not (scaled_min_area <= area <= scaled_max_area):
            print("WARNING - _extract_card(): contour area out of bounds")
            return None

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) != 4:
            print("WARNINIG - _extract_card(): contour is not rectangular")
            return None

        rect = cv2.minAreaRect(contour)
        width = min(rect[1])
        height = max(rect[1])
        if width == 0:
            print("WARNING - _extract_card(): contour width == 0")
            return None

        aspect_ratio = height / width
        expected_ratio = self.config.card_aspect_ratio
        """
        if not (expected_ratio - self.config.aspect_ratio_tolerance <=
                aspect ratio <=
                expected_ratio + self.config.aspect_ratio_tolerance):

            print("WARNING - _extract_card(): contour aspect ratio out of "
                  "bounds"
            return None
        """

        points = np.float32(approx.reshape(4, 2))
        center = np.mean(points, axis=0)
        sorted_points = sorted(points, key=lambda p:
            (np.arctan2(p[1] - center[1], p[0] - center[0]) + np.pi) % (2 * np.pi))
        points = np.array(sorted_points)

        target_width = int(self.config.target_size[0])
        target_height = int(target_width * self.config.card_aspect_ratio)
        target_points = np.float32([
            [0, 0],
            [target_width, 0],
            [target_width, target_height],
            [0, target_height]
        ])

        matrix = cv2.getPerspectiveTransform(points, target_points)
        card = cv2.warpPerspective(
            image,
            matrix,
            (target_width, target_height)
        )

        return cv2.resize(card, self.config.target_size)


    def _show_debug(self, name: str, image: NDArray, wait: bool = True) -> None:
        """Display debug images if debug mode is enabled"""
        if self.debug:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            height, width = image.shape[:2]
            max_dim = 800
            if max(height, width) > max_dim:
                scale = max_dim / max(height, width)
                cv2.resizeWindow(name, int(width * scale), int(height * scale))
            cv2.imshow(name, image)
            if wait:
                key = cv2.waitKey(0)
                if key == ord("q"):
                    self.debug = False
                cv2.destroyWindow(name)
