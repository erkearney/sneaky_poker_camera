import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Optional, Tuple, List, Dict
import argparse
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict

class ModelEvaluator:
    """Base class for model evaluation"""
    def __init__(self, class_mapping_path: str):
        self.class_mapping = self._load_class_mapping(class_mapping_path)
        self.reverse_mapping = {v: k for k, v in self.class_mapping.items()}


    def _load_class_mapping(self, mapping_path: str) -> dict:
        """Load the class mapping"""
        try:
            df = pd.read_csv(mapping_path)
            return {row["card_name"]: row["class_idx"] for _, row in df.iterrows()}
        except Exception as e:
            print(f"ERROR - _load_class_mapping: {e}")
            raise


    def evaluate_model(self, test_data_dir: str, batch_size: int = 32) -> Dict:
        """To be implemented by child class"""
        raise NotImplementedError


    def plot_suit_accuracy(self, results: Dict, output_dir: str):
        """Plot accuracy comparison between suits"""
        suit_metrics = defaultdict(list)
        for card, metrics in results.items():
            if isinstance(metrics, dict) and card in self.class_mapping:
                suit = card[-1]
                if suit in ["H", "S", "D", "C"]:
                    suit_metrics[suit].append(metrics["precision"])
        suit_names = {"H": "Hearts", "S": "Spades", "D": "Diamonds", "C": "Clubs"}
        avg_accuracy = {suit_names[suit]: np.mean(scores) for suit, scores in suit_metrics.items()}

        plt.figure(figsize=(10, 6))
        bars = plt.bar(avg_accuracy.keys(), avg_accuracy.values())
        plt.title("Accuracy by Suit")
        plt.ylabel("Accuracy")
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.,
                     height,
                     f"{height:.3f}",
                     ha="center",
                     va="bottom")
        plt.savefig(str(Path(output_dir) / "suit_accuracy.png"))
        plt.close()


    def plot_rank_accuracy(self, results: Dict, output_dir: str):
        """Plot accuracy comparison between ranks"""
        rank_metrics = defaultdict(list)
        for card, metrics in results.items():
            if isinstance(metrics, dict) and card in self.class_mapping:
                rank = card[:-1]
                rank_metrics[rank].append(metrics["precision"])
        avg_accuracy = {rank: np.mean(scores) for rank, scores in rank_metrics.items()}

        rank_order = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
        sorted_ranks = sorted(avg_accuracy.items(), key=lambda x: rank_order.index(x[0]))

        plt.figure(figsize=(12, 6))
        plt.plot(range(len(sorted_ranks)), [x[1] for x in sorted_ranks], "bo-")
        plt.xticks(range(len(sorted_ranks)), [x[0] for x in sorted_ranks])
        plt.title("Accuracy by Rank")
        plt.xlabel("Rank")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.savefig(str(Path(output_dir) / "rank_accuracy.png"))
        plt.close()


class KerasModelEvaluator(ModelEvaluator):
    """Evaluator for Keras model"""
    def __init__(self, model_path: str, class_mapping_path: str):
        super().__init__(class_mapping_path)
        self.model = tf.keras.models.load_model(model_path)


    def evaluate_model(self, test_data_dir: str, batch_size: int = 32) -> Dict:
        """Evaluate model on test dataset"""
        test_dir = Path(test_data_dir)
        all_predictions = []
        all_labels = []
        inference_times = []

        for class_dir in test_dir.iterdir():
            if not class_dir.is_dir():
                continue
            class_idx = self.class_mapping[class_dir.name]
            for img_path in class_dir.glob("*.jpg"):
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"WARN - KerasModelEvaluator.evaluate_model(): {img_path} failed to be read as an image, skipping...")
                    continue
                img = cv2.resize(img, (224, 224))
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=0)

                start_time = time.time()
                pred = self.model.predict(img, verbose=0)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                pred_class = np.argmax(pred[0])
                all_predictions.append(pred_class)
                all_labels.append(class_idx)
        avg_inference_time = np.mean(inference_times) * 1000 # Convert to ms
        print(f"\nINFO - KerasModelEvaluator.evaluate_model(): Average inference time: {avg_inference_time:.2f} ms")

        label_names = [self.reverse_mapping[i] for i in range(len(self.class_mapping))]
        results = classification_report(all_labels,
                                        all_predictions,
                                        target_names=label_names,
                                        output_dict=True)
        results["inference_time"] = avg_inference_time

        return results


class TFLiteModelEvaluator(ModelEvaluator):
    """Evaluator for TFLite model"""
    def __init__(self, model_path: str, class_mapping_path: str):
        super().__init__(class_mapping_path)
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def evaluate_model(self, test_data_dir: str, batch_size: int = 32) -> Dict:
        """Evaluate TFLite model on test dataset"""
        test_dir = Path(test_data_dir)
        all_predictions = []
        all_labels = []
        inference_times = []

        for class_dir in test_dir.iterdir():
            if not class_dir.is_dir():
                continue
            class_idx = self.class_mapping[class_dir.name]
            for img_path in class_dir.glob("*.jpg"):
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"WARN - TFLiteModelEvaluator.evaluate_model(): {img_path} could not be read as an image, skipping...")
                    continue
                img = cv2.resize(img, (224, 224))
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=0)
                self.interpreter.set_tensor(self.input_details[0]["index"], img)

                start_time = time.time()
                self.interpreter.invoke()
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
                pred_class = np.argmax(output_data[0])
                all_predictions.append(pred_class)
                all_labels.append(class_idx)
        avg_inference_time = np.mean(inference_times) * 1000    # Convert to ms
        print(f"\nINFO - TFLiteModelEvaluator.evaluate_model(): Average inference time: {avg_inference_time:.2f} ms")

        label_names = [self.reverse_mapping[i] for i in range(len(self.class_mapping))]
        results = classification_report(all_labels,
                                        all_predictions,
                                        target_names=label_names,
                                        output_dict=True)
        results["inference_time"] = avg_inference_time

        return results


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
            result = self.test_image(str(image_path), display_result=True)
            if result:
                results[image_path.name] = {
                    "predicted_card": result[0],
                    "confidence": result[1]
                }

        return results


def main():
    parser = argparse.ArgumentParser(description="Test trained card recognition model")
    parser.add_argument("--keras-model", type=str, required=True,
                        help="Path to Keras model file")
    parser.add_argument("--tflite-model", type=str, required=True,
                        help="Path to TFLite model file")
    parser.add_argument("--mapping-path", type=str, required=True,
                        help="Path to class mapping CSV file")
    parser.add_argument("--test-path", type=str, required=True,
                        help="Path to test image or directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save evaluation results")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    if args.keras_model:
        print(f"INFO - main(): Evaluating Keras model...")
        keras_evaluator = KerasModelEvaluator(args.keras_model, args.mapping_path)
        keras_results = keras_evaluator.evaluate_model(args.test_path)
        keras_df = pd.DataFrame(keras_results).transpose()
        keras_df.to_csv(output_dir / "keras_results.csv")
        keras_evaluator.plot_suit_accuracy(keras_results, str(output_dir / "keras"))
        keras_evaluator.plot_rank_accuracy(keras_results, str(output_dir / "keras"))
        results["keras"] = keras_results

    if args.tflite_model:
        print(f"INFO - main(): Evaluating TFLite model...")
        tflite_evaluator = TFLiteModelEvaluator(args.tflite_model, args.mapping_path)
        tflite_results = tflite_evaluator.evaluate_model(args.test_path)
        tflite_df = pd.DataFrame(tflite_results).transpose()
        tflite_df.to_csv(output_dir / "tflite_results.csv")
        tflite_evaluator.plot_suit_accuracy(tflite_results, str(output_dir / "tflite"))
        tflite_evaluator.plot_rank_accuracy(tflite_results, str(output_dir / "tflite"))
        results["tflite"] = tflite_results

    if len(results) == 2:
        keras_time = results["keras"]["inference_time"]
        tflite_time = results["tflite"]["inference_time"]

        plt.figure(figsize=(8, 6))
        plt.bar(["Keras", "TFlite"], [keras_time, tflite_time])
        plt.title("Average Inference Time Comparison")
        plt.ylabel("Time (ms)")
        plt.savefig(str(output_dir / "inference_time_comparison.png"))
        plt.close()


if __name__ == "__main__":
    main()
