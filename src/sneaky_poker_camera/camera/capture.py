from picamera2 import Picamera2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import io
from typing import Optional

def capture_image() -> NDArray:
    """
    Captures an image from the camera, returns it as a numpy array

    Returns:
        NDArray: The captured image
    """
    picam2 = Picamera2()
    config = picam2.create_still_configuration()
    picam2.configure(config)

    picam2.start()
    image_array = picam2.capture_array()
    picam2.stop()

    return image_array


def save_image(image_array: NDArray, filename: str = 'captured_image.jpg') -> None:
    """
    Saves image_array to the filesystem at filename

    Args:
        image_array (NDArray): The image to save, should be shape (height, width, channels)
        filename (str): The filename to save the image to
    """
    image = Image.fromarray(image_array)
    image.save(filename)


if __name__ == "__main__":
    """
    Example usage and testing
    """
    captured_image = capture_image()
    save_image(captured_image)
    print(f"Image saved, shape: {captured_image.shape}")
