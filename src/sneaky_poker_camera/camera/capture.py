from picamera2 import Picamera2
import numpy as np
from numpy.typing import NDArray
from typing import Optional, List, Dict, Any, Tuple
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
                      "format": "RGB888"},
                buffer_count=1,
                controls={
                    "NoiseReductionMode": 2,
                }
            )
            self.camera.configure(camera_config)
            self.camera.start()
            time.sleep(2)

        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            if self.camera and self.camera.started:
                self.camera.stop()
            raise


    def optimize_exposure(self, base_image: NDArray, base_exposure: int) -> Tuple[NDArray, float]:
        """Try different exposure settings to optimize image quality"""
        best_image = base_image
        best_score = self.get_focus_score(base_image)
        exposure_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

        for multiplier in exposure_multipliers:
            exposure_time = int(base_exposure * multiplier)
            self.camera.set_controls({
                "ExposureTime": exposure_time,
                # Reduce analog gain for longer exposures to minimize noise
                "AnalogueGain": max(1.0, 2.0 / multiplier)
            })
            time.sleep(0.3)

            image = self.camera.capture_array()
            focus_score = self.get_focus_score(image)
            contrast_score = self.get_contrast_score(image)
            exposure_score = self.get_exposure_score(image)

            quality_score = (focus_score * 0.5 + contrast_score * 0.3 + exposure_score * 0.2)

            print(f"Exposure multiplier: {multiplier:.2f}: "
                  f"focus={focus_score:.1f}, "
                  f"contrast={contrast_score:.1f}, "
                  f"exposure={exposure_score:.1f}, "
                  f"quality={quality_score:.1f}."
            )

            if focus_score > best_score:
                print(f"New best score: {best_score}")
                best_score = focus_score
                best_image = image
                self.last_focus_score = focus_score

        return best_image, best_score


    def get_focus_score(self, image: NDArray) -> float:
        """Calculate the Laplcian variance to determine focus score"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        return cv2.Laplacian(gray, cv2.CV_64F).var()


    def get_contrast_score(self, image: NDArray) -> float:
        """Calculate image contrast score"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return np.std(gray.astype(float))


    def get_exposure_score(self, image: NDArray) -> float:
        """Calculate exposure score based on histogram analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum() # Normalize
        
        # Calulate the fraction of pixels that are neither too dark to too bright
        dark_fraction = hist[:50].sum()    # Too dark
        bright_fraction = hist[200].sum()  # Too bright
        good_fraction = hist[50:200].sum() # Just right

        exposure_score = good_fraction * 100 - (dark_fraction + bright_fraction) * 50

        return max(0, exposure_score)
        

    def capture_single(self) -> Optional[NDArray]:
        """Capture a single image with exposure testing"""
        try:
            if not self.camera:
                self.initialize()
            elif not self.camera.started:
                self.camera.start()
                time.sleep(2)

            metadata = self.camera.capture_metadata()
            base_exposure = metadata.get("ExposureTime", 10000)

            image = self.camera.capture_array()
            initial_score = self.get_focus_score(image)
            print(f"Initial image captured with focus score: {initial_score:.1f}")
            if initial_score < self.config.min_focus_score:
                print(f"INFO: Poor image quality: {initial_score:.1f}, "
                        "Attempting to improve image quality with exposure optimization...")
                image, final_score = self.optimize_exposure(image, base_exposure)
                if final_score > self.config.min_focus_score:
                    print(f"INFO: Image quality improved to {final_score:.1f}")
                else:
                    print(f"WARN: Could not improve image quality enough with exposure adjustment."
                           "Adjusting noise reduction and sharpness as a last-resort...")

                    self.camera.set_controls({
                        "NoiseReductionMode": 2,
                        "Sharpness": 1.5
                        })

                    adjusted_image = self.camera.capture_array()
                    adjusted_score = self.get_focus_score(adjusted_image)
                    print(f"Final capture with noise reduction and sharpness increased: {adjusted_score:.1f}")
                    if adjusted_score > final_score:
                        print("INFO: Adjusting noise reduction and sharpness improved image quality")
                        image = adjusted_image
                    else:
                        print("INFO: Adjusting noise reduction and sharepness did NOT improve image quality")
            self.last_focus_score = self.get_focus_score(image)
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"ERROR - capture_single: Failed to capture image: {e}")
            return None


    def capture_hdr(self) -> Optional[NDArray]:
        """Capture HDR image testing with multiple exposures"""
        try:
            if not self.camera:
                self.initialize()

            metadata = self.camera.capture_metadata()
            base_exposure = metadata.get("ExposureTime", 10000)
            base_image = self.camera.capture_metadata()
            brightness = np.mean(cv2.cv2Color(base_image, cv2.COLOR_RGB2GRAY))
            if brightness < 64:
                # If too dark, bias towards longer exposure
                ev_range = (-1.0, 2.0)
            elif brightness > 192:
                # If too bright, bias towards shorter exposure
                ev_range = (-2.0, 1.0)
            else:
                ev_range = (-1.5, 1.5)


            exposures = []
            ev_steps = np.linspace(
                    ev_range[0],
                    ev_range[1],
                    self.config.hdr_exposures
            )

            for ev in ev_steps:
                exposure_time = int(base_exposure * (2 ** ev))
                self.camera.set_controls({
                    "ExposureTime": exposure_time,
                    "AnalogueGain": max(1.0, 1.0 / (2 ** ev))
                })
                time.sleep(0.3)

                image = self.camera.capture_array()
                exposures.append(image)
                print(f"HDR frame as EV {ev:.1f}: "
                      f"exposure={exposure_time}, "
                      f"focus={focus_score:.1f}"
                )

            self.camera.set_controls({
                "ExposureTime": base_exposure,
                "AnalogueGain": 1.0
            })

            if not exposures:
                return None

            merge = cv2.createMergeMertens(
                    contrast_weight=1.0,
                    saturation_weight=1.0,
                    exposure_weight=0.5
            )
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


if __name__ == "__main__":
    """
    Example usage and testing
    """
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
                    print(f"Focus score: {camera.last_focus_score:.1f}")
                    print(f"Image shape: {image.shape}")
                else:
                    print(f"ERROR, failed to save image to {output_path}")
            else:
                print(f"ERROR - test_camera: Failed to capture image")
        except Exception as e:
            print(f"ERROR - test_camera: {e}")
        finally:
            if "camera" in locals():
                del camera


    test_camera()
