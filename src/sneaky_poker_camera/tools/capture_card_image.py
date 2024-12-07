import argparse
from pathlib import Path
import cv2
import time
import sys
from os.path import dirname, abspath

parent_dir = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir)

from camera.capture import CameraController
from vision.card_finder import CardFinder

def capture_card_image(card_name, output_dir = "../Pictures") -> None:
    """
    Capture and process a card image, saving both the original and processed image.

    Args"
        output_dir (str): Directory to save images
    """
    output_path = Path(output_dir)
    original_path = output_path / "original"
    processed_path = output_path / "processed"
    original_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    camera = CameraController()
    finder = CardFinder(debug=False)
    
    try:
        print("Capturing image...")
        image = camera.capture_single()
        if image is not None:
            original_file = original_path / f"{card_name}.jpg"
            cv2.imwrite(str(original_file), image)
            print(f"Original image saved to {original_file}")

            card = finder.find_cards(image)[0]
            if card is not None:
                processed_file = processed_path / f"{card_name}.jpg"
                cv2.imwrite(str(processed_file), card)
                print(f"Processed image saved to {processed_file}")
            else:
                print("ERROR - capture_card_image(): Failed to process image")
        else:
            print("ERROR - capture_card_image(): Failed to capture image")
    except Exception as e:
        print(f"ERROR - capture_card_image(): Error during capture, - exception thrown")
    finally:
        if "camera" in locals():
            del camera


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture and process a single playing card image")
    parser.add_argument("card_name", help="Name of the card (e.g. 'AS' for Ace of Spaces')")
    parser.add_argument("--output-dir", default="../Pictures", help="Output directory for saved image")
    args = parser.parse_args()

    capture_card_image(args.card_name, args.output_dir)
