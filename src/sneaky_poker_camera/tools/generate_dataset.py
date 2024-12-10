import argparse
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple
import random
import albumentations as A
import glob
import os

DECK_SIZE = 52

class CardDatasetGenerator:
    """Generates training data by augmenting extracted card images with
    textures and transformations"""
    def __init__(self,
                 cards_dir: str,
                 textures_dir: str,
                 output_dir: str,
                 samples_per_card: int = 100,
                 output_size: Tuple[int, int] = (800, 600),
                 demo_mode: bool = False):
        self.cards_dir = Path(cards_dir)
        self.textures_dir = Path(textures_dir)
        self.output_dir = Path(output_dir)
        self.samples_per_card = samples_per_card
        self.output_size = output_size
        self.demo_mode = demo_mode

        # Augmentation pipeline
        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.8),
            A.GaussNoise(p=0.6),
            A.RandomGamma(p=0.6),
            A.Blur(blur_limit=3, p=0.4),
            A.Rotate(limit=15, p=0.7),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                shear={"x": (-10, 10), "y": (-10, 10)},
                p=0.7
            ),
        ])


    def show_demo_image(self, title: str, image: np.ndarray, wait: bool = True) -> None:
        """Display images of each processing step"""
        if self.demo_mode:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            height, width = image.shape[:2]
            max_dim = 800
            if max(height, width) > max_dim:
                scale = max_dim / max(height, width)
                cv2.resizeWindow(title, int(width * scale), int(height * scale))
            cv2.imshow(title, image)
            if wait:
                print(f"\nINFO - show_demo_image(): Showing: {title}")
                print(f"    Press any key to continue, 'q' to quit demo mode")
                key = cv2.waitKey(0)
                cv2.destroyWindow(title)
                if key == ord("q"):
                    raise KeyboardInterrupt("INFO - show_demo_image(): Demo mode terminated by user")


    def load_textures(self) -> List[np.ndarray]:
        """Load the texture images from self.output_dir"""
        texture_images = []
        texture_paths = glob.glob(str(self.textures_dir / "*" / "*.jpg"))

        if self.demo_mode:
            # In demo mode, select a single texture at random
            if texture_paths:
                random_texture_path = random.choice(texture_paths)
                img = cv2.imread(random_texture_path)
                if img is not None:
                    texture_images.append(img)
                    texture_name = Path(random_texture_path).name
                    print(f"\nINFO - load_textures() -- demo_mode: Selected random texture: {texture_name}")
                    self.show_demo_image("DEMO MODE - sample texture", img)
        else:
            for path in texture_paths:
                img = cv2.imread(path)
                if img is not None:
                    texture_images.append(img)

        print(f"INFO - load_textures(): Loaded {len(texture_images)} textures")
        return texture_images


    def load_card_images(self) -> List[Tuple[str, np.ndarray]]:
        """Load all card images along with their labels"""
        card_images = []
        card_paths = list(self.cards_dir.glob("*.jpg"))

        if self.demo_mode:
            # In demo mode, randomly select one card
            if card_paths:
                random_card_path = random.choice(card_paths)
                img = cv2.imread(str(random_card_path))
                if img is not None:
                    label = random_card_path.stem
                    card_images.append((label, img))
                    self.show_demo_image("DEMO MODE - sample card", img)
        else:
            for card_path in card_paths:
                img = cv2.imread(str(card_path))
                if img is not None:
                    label = card_path.stem
                    card_images.append((label, img))
            if len(card_images) != DECK_SIZE and not self.demo_mode:
                print(f"WARN - load_card_images(): {len(card_images)} != {DECK_SIZE}")
            else:
                print("Loaded card images")
        return card_images


    def blend_with_texture(self,
                           card: np.ndarray,
                           texture: np.ndarray,
                           alpha: float = 0.85,
                           scale_range: Tuple[float, float] = (0.2, 0.4)) -> np.ndarray:
        """Blend card image with background texture"""
        background = cv2.resize(texture, self.output_size)
        self.show_demo_image("DEMO MODE - 1. Resized background", background)
        card_aspect_ratio = card.shape[1] / card.shape[0]

        scale = random.uniform(scale_range[0], scale_range[1])
        card_height = int(self.output_size[1] * scale)
        card_width = int(card_height * card_aspect_ratio)
        card_resized = cv2.resize(card, (card_width, card_height))
        self.show_demo_image("DEMO MODE - 2. Resized card", card_resized)

        gray = cv2.cvtColor(card_resized, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        self.show_demo_image("DEMO MODE - 3. Card mask", mask_3ch)

        max_x = self.output_size[0] - card_width
        max_y = self.output_size[1] - card_height
        if max_x < 0 or max_y < 0:
            raise ValueError("Card size too large for output image")
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        roi = background[y:y+card_height, x:x+card_width]
        blended = np.where(
            mask_3ch > 0,
            cv2.addWeighted(card_resized, alpha, roi, 1 - alpha, 0),
            roi
        )
        background[y:y+card_height, x:x+card_width] = blended
        self.show_demo_image("DEMO MODE - 4. Blended Result", background)

        position_info = {
            "x": x,
            "y": y,
            "width": card_width,
            "height": card_height
        }

        return background.astype(np.uint8), position_info


    def generate_dataset(self):
        """Generate augmented dataset"""
        textures = self.load_textures()
        cards = self.load_card_images()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        for card_idx, (label, card) in enumerate(cards):
            print(f"Processing card {card_idx+1}/{len(cards)}: {label}")
            card_output_dir = self.output_dir / label
            card_output_dir.mkdir(exist_ok=True)
            for i in range(self.samples_per_card):
                texture = random.choice(textures)
                blended, _ = self.blend_with_texture(card,
                                                  texture,
                                                  alpha=random.uniform(0.8, 0.95),
                                                  scale_range=(0.2, 0.4))
                augmented = self.transform(image=blended)["image"]
                if self.demo_mode:
                    self.show_demo_image("DEMO MODE - 5. Final result", augmented)
                    cv2.destroyAllWindows()
                    return None   # Only process one card in demo mode
                output_path = card_output_dir / f"{i:04d}.jpg"
                cv2.imwrite(str(output_path), augmented)
            print(f"Generated {self.samples_per_card} samples for {label}")


def main():
    parser = argparse.ArgumentParser(description="Generate augmented card dataset")
    parser.add_argument("--cards-dir", type=str, required=True,
                        help="Directory containing processed card images")
    parser.add_argument("--textures-dir", type=str, required=True,
                        help="Directory containing texture images")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--samples-per-card", type=int, required=True,
                        help="Number of samples to generate per card")
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode to visualize the generation process")
    args = parser.parse_args()

    generator = CardDatasetGenerator(
        cards_dir=args.cards_dir,
        textures_dir=args.textures_dir,
        output_dir=args.output_dir,
        samples_per_card=args.samples_per_card,
        demo_mode=args.demo
    )
    generator.generate_dataset()


if __name__ == "__main__":
    main()
