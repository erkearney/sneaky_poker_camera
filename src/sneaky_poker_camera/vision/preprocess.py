import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple

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
    def __init__(self, target_size: Tuple[int, int] = (224, 224), debug: bool = False):
        self.target_size = target_size
        self.debug = debug

    
    def preprocess_image(self, image: NDArray) -> Optional[NDArray]:
        """
        Preprocessors images to help neural network detect playing cards.

        Args:
           image (NDArray): The image to preprocess

        Returns:
           NDArray: The preprocessed image
        """
        try:
            self.show_image("0. Original Image", image)

            height, width = image.shape[:2]
            min_dim = min(height, width)

            max_working_dim = 1000
            if min_dim > max_working_dim:
                scale = max_working_dim / min_dim
                working_image = cv2.resize(image, None, fx=scale, fy=scale)
                self.show_image("1. Resized image", working_image)
            else:
                working_image = image.copy()

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.show_image("2. Grayscale", gray)

            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            self.show_image("3. Blurred", blurred)

            thresh = cv2.adaptiveThreshold(
                    blurred,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11,
                    2
            )
            self.show_image("4. Adaptive Threshold", thresh)

            edges = cv2.Canny(blurred, 30, 50)
            self.show_image("5. Canny Edges", edges)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(edges, kernel, iterations=1)
            self.show_image("6. Dilated Edges", dilated)

            final = cv2.resize(dilated, self.target_size)
            self.show_image("7. Final result", final, wait=True)

            return final
        except Exception as e:
            print(f"Error during image preprocessing: {e}")
            return None


    def show_image(self, name: str, image: NDArray, wait: bool = True) -> None:
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
                if key == ord('q'):
                    self.debug = False
                cv2.destroyWindow(name)


if __name__ == "__main__":
    """
    Example usage and testing
    """
    import sys
    from pathlib import Path

    try:
        image_path = Path("captured_image.jpg")
        if not image_path.exists():
            print(f"ERROR: Image not found at {image_path}")
            sys.exit(1)

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"ERROR: Failed to load test image")
            sys.exit(1)

        card_preprocessor = CardPreprocessor(debug=True)
        processed_image = card_preprocessor.preprocess_image(image)

        if processed_image is not None:
            output_path = Path("preprocessed_card.jpg")
            cv2.imwrite(str(output_path), processed_image)
            print(f"Preprocseed image saved to {output_path}")
            print(f"Processed shape: {processed_image.shape}")
        else:
            print("ERROR: Failed to preprocess image")


    except Exception as e:
        print(f"ERROR: {e}")
