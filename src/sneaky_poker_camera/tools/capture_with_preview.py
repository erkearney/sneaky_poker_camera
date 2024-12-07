import os
import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parents[1])
sys.path.append(parent_dir)
from camera.capture import CameraController
from camera.config import CameraConfig
import cv2
import numpy as np
import time
from typing import Optional, List
import argparse


class CardImageCapture:
    """Capture card images with live preview"""
    def __init__(self, output_dir: str = "card_images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.camera = None
        self.window_name = "Card Capture Preview (Press SPACE to capture, Q to quit)"


    def initialize_camera(self) -> None:
        """Initialize camera"""
        config = CameraConfig(
            min_focus_score=100.0,
            enable_hdr=False
        )
        self.camera = CameraController(config)
        self.camera.initialize()


    def capture_cards(self, num_cards: int = 1) -> List[Path]:
        """
        Capture card image(s) with preview

        Args:
            num_cards (int): Number of cards images to capture

        Returns:
            List[Path]: Paths to captured images
        """
        try:
            if not self.camera:
                self.initialize_camera()
                captured_paths = []
                capture_count = 0
                print(f"\nINFO - capture_cards(): Ready to capture {num_cards} cards")
                print(f"Position each card and press SPACE to capture")
                print(f"Press Q to quit\n")

                while capture_count < num_cards:
                    frame = self.camera.camera.capture_array()
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    remaining = num_cards - capture_count
                    cv2.putText(display_frame,
                                f"Cards remaining: {remaining}",
                                (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2)
                    height, width = display_frame.shape[:2]
                    center_x, center_y = width // 2, height // 2
                    rect_width, rect_height = 300, 420
                    cv2.rectangle(display_frame,
                                 (center_x - rect_width//2, center_y - rect_height//2),
                                 (center_x + rect_width//2, center_y + rect_height//2),
                                 (0, 255, 0),
                                 2)

                    cv2.imshow(self.window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        print(f"\nINFO - capture_cards(): Capture cancelled by user input")
                        break
                    elif key == ord(" "):
                        print(f"\nINFO - capture_cards(): Capturing card {capture_count}/{num_cards}...")

                        optimized_image = self.camera.capture_image()
                        if optimized_image is not None:
                            output_path = self.output_dir / f"card_{capture_count}.jpg"
                            cv2.imwrite(str(output_path), optimized_image)
                            captured_paths.append(output_path)
                            capture_count += 1
                            print(f"INFO - capture_cards(): {capture_count} -> {output_path}")
                            print(f"INFO - capture_cards(): Focus score: {self.camera.last_focus_score:.1f}")

                            # Show captured image for one second
                            cv2.imshow("Captured Image", optimized_image)
                            cv2.waitKey(1000)
                            cv2.destroyWindow("Captured Image")
                        else:
                            print(f"ERROR - capture_cards(): Failed to capture image, try again")

                        time.sleep(0.5)     # Prevent accidental double-capture
                return captured_paths
        except Exception as e:
            print(f"ERROR - capture_cards(): {e}")
            return []
        finally:
            cv2.destroyAllWindows()
            if self.camera:
                del self.camera


def main():
    parser = argparse.ArgumentParser(description="Capture card images with preview")
    parser.add_argument("--output-dir", type=str, default="card_image",
                        help="Output directory for captured images")
    parser.add_argument("--num-cards", type=int, default=1,
                        help="Number of cards to capture")
    args = parser.parse_args()

    try:
        capturer = CardImageCapture(args.output_dir)
        captured_paths = capturer.capture_cards(args.num_cards)
        if len(captured_paths) == args.num_cards:
            print(f"\nINFO - main(): Success")
            print("INFO - main(): Captured images:")
            for path in captured_paths:
                print(f"    {path}")
        else:
            print(f"\nWARN - main(): Capture incomplete, captured {len(captured_paths)}/{args.num_cards} cards")
    except Exception as e:
        print(f"ERROR - main(): {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
