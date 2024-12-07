import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Optional, Tuple, List
import argparse

class CardModelTester:
    """Test traned card recognition model on images"""
    def __init__(self,
                 model_path: str,
                 class_mapping_path: str,
                 target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.model = self._load_model(model_path)
        self.class_mapping = self._load_class_mapping(class_mapping_path)
        self.reverse_mapping = {v: k for k, v in self.class_mapping.items()}


    def _load_model(self, model_path: str) -> tf.keras.Model:
        """Load the trained model"""
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"ERROR - _load_model(): {e}")
            raise


    def _load_class_mapping(self, mapping_path: str) -> dict:
        """Load the class mapping"""
        try:
            df = pd.read_csv(mapping_path)
            return {row["card_name"]: row["class_idx"] for _, row in df.iterrows()}
        except Exception as e:
            print(f"ERROR - _load_class_mapping: {e}")
            raise


    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input"""
        resized = cv2.resize(image, self.target_size)
        normalized = resized.astype(np.float32) / 255.0

        return np.expand_dims(normalized, axis=0)


    def predict_card(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict card from image

        Returns:
            Tuple[str, float]: (predicted card name, confidence score)
        """
        processed_image = self.preprocess_image(image)

        predictions = self.model.predict(processed_image, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        card_name = self.reverse_mapping[predicted_idx]

        return card_name, confidence


    def test_image(self, image_path: str, display_result: bool = True) -> Optional[Tuple[str, float]]:
        """Test model on a single image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"ERROR - test_image(): Could not load image path {image_path}")
                return None
            card_name, confidence = self.predict_card(image)

            print(f"\nPredicted card: {card_name}")
            print(f"Confidence: {confidence:.2%}")

            if display_result:
                result_image = image.copy()
                text = f"{card_name} ({confidence:.2%})"
                cv2.putText(result_image, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Predicition Result", result_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return card_name, confidence
        except Exception as e:
            print(f"ERROR - test_image(): {e}")
            return None


    def test_directory(self, test_dir: str) -> dict:
        """Test model on all images in a directory"""
        results = {}
        test_dir = Path(test_dir)
        image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        total_images = len(image_files)
        print(f"\nINFO - test_directory(): Testing {total_images} images...")

        for i, image_path in enumerate(image_files, 1):
            print(f"\nINFO - test_directory(): Testing image {i}/{total_images}: {image_path.name}")
            result = self.test_image(str(image_path, display_result=True))
            if result:
                results[image_path.name] = {
                    "predicted_card": result[0],
                    "confidence": result[1]
                }

        return results


def main():
    parser = argparse.ArgumentParser(description="Test trained card recognition model")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model (.keras file)")  
    parser.add_argument("--mapping-path", type=str, required=True,
                        help="Path to class mapping CSV file")
    parser.add_argument("--test-path", type=str, required=True,
                        help="Path to test image or directory")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable visual display of results")
    args = parser.parse_args()

    try:
        tester = CardModelTester(args.model_path, args.mapping_path)
        test_path = Path(args.test_path)
        if test_path.is_file():
            tester.test_image(str(test_path), display_result=not args.no_display)
        elif test_path.is_dir():
            results = tester.test_directory(str(test_path))
            print("\nTesting Summary:")
            print(f"Total images tested: {len(results)}")

            results_df = pd.DataFrame.from_dict(results, orient="index")
            output_path = test_path / "test_results.csv"
            results_df.to_csv(output_path)
            print(f"Results saved to {output_path}")
        else:
            print(f"ERROR - main(): {test_path} is not a valid file or directory")
    except Exception as e:
        print(f"ERROR - main(): {e}")


if __name__ == "__main__":
    main()
