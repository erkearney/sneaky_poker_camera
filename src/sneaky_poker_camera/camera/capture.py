from typing import Optional, List, Dict, Any
import cv2
import time

from .config import CameraConfig

class CameraController:
    """Camera controller with focus optimization"""
    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.camera = None
        self.last_focus_score = 0.0


    def initialize(self) -> None:
        """Initialize camera"""
        try:
            self.camera = Picamera2()
            camera_config = self.camera.create_still_configuration(
                    main={"size": self.config.resolution,
                          "format": "RGB888"}
            )
            self.camera.configure(camera_config)
            self.camera.start()
            time.sleep(2)

        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            if self.camera and self.camera.started:
                self.camera.stop()
            raise


    def get_focus_score(self, image: NDArray) -> float:
        """Calculate the Laplcian variance to determine focus score"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        return cv2.Laplacian(gray, cv2.CV_64F).var()
        

    def capture_single(self) -> Optional[NDArray]:
        """Capture a single image with exposure testing"""
        try:
            if not self.camera:
                self.initialize()
            elif not self.camera.started:
                self.camera.start()
                time.sleep(2)

            image = self.camera.capture_array()
            focus_score = self.get_focus_score(image)
            self.last_focus_score = focus_score
            print(f"Image captured with focus score: {focus_score}")
            if focus_score < self.config.min_focus_score:
                print(f"WARNING, poor focus detected ({focus_score}). Attempting exposure adjustment...")
                current_exposure = self.camera.capture_metadata().get("ExposureTime", 10000)
                self.camera.set_controls({"ExposureTime": int(current_exposure * 1.5)})
                time.sleep(0.5)
                new_image = self.camera.capture_array()
                new_focus_score = self.get_focus_score(image)

                if new_focus_score > focus_score:
                    print(f"Improved focus score with adjusted exposure: {new_focus_score}")
                    self.last_focus_score = new_focus_score
                    image = new_image
                else:
                    self.camera.set_controls({"ExposureTime": current_exposure})

            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image_bgr
        except Exception as e:
            print(f"ERROR, failed to capture image: {e}")
            return None


    def capture_hdr(self) -> Optional[NDArray]:
        """Capture HDR image testing with multiple exposures"""
        try:
            if not self.camera:
                self.initialize()

            metadata = self.camera.capture_metadata()
            base_exposure = metadata.get("ExposureTime", 10000)

            exposures = []
            ev_steps = np.linspace(
                    -self.config.hdr_ev_steps,
                    self.config.hdr_ev_steps,
                    self.config.hdr_exposures
            )

            for ev in ev_steps:
                exposure_time = int(base_exposure * (2 ** ev))
                self.camera.set_controls({"ExposureTime": exposure_time})
                time.sleep(0.3)

                image = self.camera.capture_array()
                exposures.append(image)
                print(f"Captured HDR frame at EV {ev:.1f}")

            self.camera.set_controls({"ExposureTime": base_exposure})

            if not exposures:
                return None

            merge = cv2.createMergeMertens()
            hdr = merge.process(exposures)
            hdr_8bit = np.clip(hdr * 255, 0, 255).astype("uint8")
            hdr_bgr = cv2.cvtColor(hdr_8bit, cv2.COLOR_RGB2BGR)

            return hdr_bgr
        except Exception as e:
            print(f"ERROR, failed to capture HDR image: {e}")
            return None


    def capture_image(self) -> NDArray:
        """
        Captures an image from the camera, returns it as a numpy array

        Returns:
            NDArray: The captured image
        """
        return self.capture_hdr() if self.config.enable_hdr else self.capture_single()


    def __del__(self):
        """Clean up camera"""
        if self.camera and self.camera.started:
            self.camera.stop()
=======
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
>>>>>>> 34b19af5785edac2a74c39fc5a4ca0f6937460da


if __name__ == "__main__":
    """
    Example usage and testing
    """
<<<<<<< HEAD
    import argparse
    from pathlib import Path


    def test_camera():
        parser = argparse.ArgumentParser(description="Test camera capture with various settings")
        parser.add_argument("--output-dir", type=str, default=".", help="Output directory for captured images")
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")
        parser.add_argument("--hdr", action="store_true", help="Enable HDR capture")
        parser.add_argument("--min-focus", type=float, default=50.0, help="Minimum acceptable focus score")

        args = parser.parse_args()

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            config = CameraConfig(
                    min_focus_score=args.min_focus,
                    enable_hdr=args.hdr
            )

            camera = CameraController(config)

            print("Capturing image...")
            image = camera.capture_image()

            if image is not None:
                output_path = output_dir / "capture.jpg"
                success = cv2.imwrite(str(output_path), image)
                if success:
                    print(f"Image saved to {output_path}")
                    print(f"Focus score: {camera.last_focus_score}")
                    print(f"Image shape: {image.shape}")
                else:
                    print(f"ERROR, failed to save image to {output_path}")
            else:
                print(f"ERROR: Failed to capture image")
        except Exception as e:
            print(f"ERROR during camera test: {e}")
        finally:
            if "camera" in locals():
                del camera


    test_camera()
