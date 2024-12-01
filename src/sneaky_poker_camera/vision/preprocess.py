import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional, List, Tuple
from card_finder import CardFinder, CardFinderConfig

class CardPreprocessor:
    """
    Preprocessors images to help neural network detect playing cards. Preprocessing
    steps are:
        1. Resize image
        2. Convert to grayscale
        3. Apply Gaussian Blur
        4. Apply Adaptive Threshold
        5. Apply Canny edges
        6. Apply dilation filter
    """
    def __init__(self, card_finder: Optional[CardFinder] = None, debug: bool = False):
        self.card_finder = card_finder or CardFinder(debug=debug)
        self.debug = debug

    
    def preprocess_cards(self, image: NDArray) -> List[NDArray]:
        """
        Find and preprocess all cards in the image.

        Args:
            image (NDArray): Input image

        Returns:
            List[NDArray]: List of preprocessed card images
        """
        cards = self.card_finder.find_cards(image)
        return [self.preprocess_single_card(card) for card in cards if card is not None]

    def preprocess_single_card(self, card: NDArray) -> Optional[NDArray]:
        """
        Apply preprocessing steps to a single card image.

        Args:
            card (NDArray): Card image to process

        Returns:
            Optional[NDArray]: Preprocessed card image
        """
        try:
            gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
            self._show_debug("Grayscale", gray)
            
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            self._show_debug("Blurred", blurred)

            thresh = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
            self._show_debug("Adaptive Threshold", thresh)

            edges = cv2.Canny(blurred, 30, 50)
            self._show_debug("Canny Edges", edges)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(edges, kernel, iterations=1)
            self._show_debug("Dilated Edges", dilated, wait=True)

            return dilated
        except Exception as e:
            print(f"ERROR - process_single_card(): {e}")
            return None


    def _show_debug(self, name: str, image: NDArray, wait: bool = True) -> None:
        """Display debug images if debug mode is enabled"""
        if self.debug:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            height, width = image.shape[:2]
            max_dim = 800
            if max(height, width) > max_dim:
                print(f"INFO - _show_debug(): scaling image")
                scale = max_dim / max(height, width)
                cv2.resizeWindow(name, int(width * scale), int(height * scale))
            cv2.imshow(name, image)
            if wait:
                key = cv2.waitKey(0)
                if key == ord("q"):
                    self.debug = False
                cv2.destroyWindow(name)
    

if __name__ == "__main__":
    """
    Example usage and testing
    """
    import sys
    from pathlib import Path

    try:
        image_path = Path("capture.jpg")
        if not image_path.exists():
            print(f"ERROR: Image not found at {image_path}")
            sys.exit(1)

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"ERROR: Failed to load test image")
            sys.exit(1)

        preprocessor = CardPreprocessor(debug=True)
        processed_cards = preprocessor.preprocess_cards(image)
        if processed_cards:
            print(f"Found and processed {len(processed_cards)} cards")
            for i, card in enumerate(processed_cards):
                cv2.imwrite(f"procseed_card_{i}.jpg", card)
                print(f"INFO - main: Saved to processed_card_{i}.jpg")
        else:
            print("WARN - main: No cards found or processed")
    except Exception as e:
        print(f"ERROR: {e}")
