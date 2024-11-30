from pathlib import Path
from typing import Optional
from .camera.capture import CameraController
from .vision.card_detection import CardDetector

class SneakyPokerCamera:
    """Main application class"""
    def __init__(self, model_path: Optional[str] = None):
        self.camera = CameraController()
        model_path = model_path or str(Path(__file__).parent / "models" / "card_model.tflite")
        self.detector = CardDetector(model_path)


    def process_single_frame(self) -> Optional[str]:
        """Capture and process a single frame"""
        try:
            image = self.camera.capture_image()
            if image is None:
                print("WARNING, camera capture_image() return None")
                return None

            result = self.detector.detec_card(image)
            return result

        except Exception as e:
            print(f"ERROR while processing frame: {e}")
            return None


    def main() -> int:
        """
        Application entry point

        Returns:
            1 if error
            0 otherwise"""
        try:
            camera_system = SneakyPokerCamera()
            result = camera_system.process_single_frame()

            if result:
                print(f"Predicted card: {result}")
            else:
                print("No card detected")

        except Exception as e:
            print(f"ERROR: {e}")
            return 1

        return 0


if __name__ == "__main__":
    exit(main())
