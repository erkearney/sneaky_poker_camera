import argparse
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple, Optional
import tflite_runtime.interpreter as tflite
import pandas as pd
from deuces import Card, Evaluator

class PokerHandTester:
    """Test poker hand recognition and evaluation"""
    def __init__(self, model_path: str, class_mapping_path: str):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.class_mapping = None
        self.evaluator = Evaluator()
        self._load_model(model_path)
        self._load_class_mapping(class_mapping_path)


    def _load_model(self, model_path: str) -> None:
        """Load TFLite model"""
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"INFO - _load_model(): Model successfully loaded from {model_path}")
        except Exception as e:
            print(f"ERROR - _load_model(): {e}")
            raise


    def _load_class_mapping(self, mapping_path: str) -> None:
        """Load class mapping from CSV"""
        try:
            df = pd.read_csv(mapping_path)
            self.class_mapping = dict(zip(df["class_idx"], df["card_name"]))
            print(f"INFO - _load_class_mapping(): Loaded {len(self.class_mapping)} classes")
        except Exception as e:
            print(f"ERROR - _load_class_mapping(): {e}")
            raise

    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input"""
        target_size = tuple(self.input_details[0]["shape"][1:3])
        if image.shape[:2] != target_size:
            image = cv2.resize(image, target_size)
        image = image.astype(np.float32) / 255.0
        return np.expand_dims(image, axis=0)


    def predict_card(self, image: np.ndarray) -> Tuple[str, float]:
        """Predict card from image"""
        try:
            processed_image = self.preprocess_image(image)
            self.interpreter.set_tensor(
                self.input_details[0]["index"],
                processed_image
            )
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(
                self.output_details[0]["index"]
            )
            pred_idx = np.argmax(output_data[0])
            confidence = float(output_data[0][pred_idx])
            pred_card = self.class_mapping.get(int(pred_idx), "Unknown")
            return pred_card, confidence
        except Exception as e:
            print(f"ERROR - predict_card(): {e}")
            return "Error", 0.0


    def convert_to_deuces_format(self, card_name: str) -> int:
        """Convert card name (e.g. 'AS' for Ace of Spades) to deuces format"""
        if len(card_name) != 2:
            raise ValueError(f"ERROR - convert_to_deuces_format(): Invalid card name - {card_name}")
        value, suit = card_name[0], card_name[1].lower()
        return Card.new(value + suit)


    def evaluate_hand(self, card_names: List[str]) -> Tuple[int, str]:
        """
        Evaluate poker hand strength using dueces

        Returns:
            Tuple[int, str]: Hand rank (lower is better) and hand description
        """
        try:
            deuces_cards = [self.convert_to_deuces_format(name) for name in card_names]
            rank = self.evaluator.evaluate(deuces_cards, [])
            hand_class = self.evaluator.get_rank_class(rank)
            hand_description = self.evaluator.class_to_string(hand_class)

            return rank, hand_description
        except Exception as e:
            print(f"ERROR - evaluate_hand(): {e}")
            return -1, "Error"


def test_poker_hand(model_path: str,
                    class_mapping_path: str,
                    image_paths: List[str]) -> None:
    """Test poker hand recognition and evaluation"""
    try:
        if len(image_paths) != 5:
            raise ValueError(f"ERROR - test_poker_hand(): Expected 5 images, got {len(image_paths)}")
        tester = PokerHandTester(model_path, class_mapping_path)
        recognized_cards = []

        print(f"\nRecognizing cards...")
        for i, image_path in enumerate(image_paths, 1):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"ERROR - test_poker_hand(): Could not read image from {image_path}")
            card_name, confidence = tester.predict_card(image)
            recognized_cards.append(card_name)
            print(f"INFO - test_poker_hand(): Card {i}: {card_name} (confidence: {confidence:.2%})")

        rank, description = tester.evaluate_hand(recognized_cards)
        print(f"\nINFO - test_poker_hand(): Recognized hand: {', '.join(recognized_cards)}")
        print(f"INFO - test_poker_hand(): Hand evaluation: {description}")
        print(f"INFO - test_poker_hand(): Hand rank: {rank} (lower is better)")
    except Exception as e:
        print(f"ERROR - test_poker_hand(): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test poker hand recognition")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the TFLite model file")
    parser.add_argument("--mapping", type=str, required=True,
                        help="Path to the class mapping CSV file")
    parser.add_argument("--images", type=str, required=True,
                        help="Path to the five card images")
    args = parser.parse_args()
    image_paths = [str(file) for file in list(Path(args.images).glob("*jpg"))]

    test_poker_hand(args.model, args.mapping, image_paths)
