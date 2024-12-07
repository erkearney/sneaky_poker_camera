import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from pathlib import Path
import cv2
from typing import Tuple, List, Iterator, Dict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math

physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    print(f"Found {len(physical_devices)} GPU(s)")
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
            tf.config.set_logical_device_configuration(
                device,
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024*4)]
            )
            print(f"INFO - train_model.py: Memory growth enabled for {device}")
        except RuntimeError as e:
            print(f"Error enabling memory growth: {e}")
else:
    print("No GPUs found. Running on CPU")


class CardDataGenerator(tf.keras.utils.Sequence):
    """Memory-efficient data generator for card images"""
    def __init__(self,
                 image_paths: List[Path],
                 labels: List[int],
                 target_size: Tuple[int, int] = (224, 224),
                 batch_size: int = 32,
                 shuffle: bool = True):
        self.image_paths = image_paths
        self.labels = labels
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indicies = np.arange(len(self.image_paths))

        print(f"Initialized generator with {len(self.image_paths)} images")
        self.on_epoch_end()


    def __len__(self):
        return math.ceil(len(self.image_paths) / self.batch_size)


    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.image_paths))
        batch_indicies = self.indicies[start_idx: end_idx]

        batch_x = np.zeros((len(batch_indicies), *self.target_size, 3), dtype=np.float32)
        batch_y = np.zeros(len(batch_indicies), dtype=np.float32)

        for i, idx in enumerate(batch_indicies):
            img_path = self.image_paths[idx]
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.resize(img, self.target_size)
                img = img.astype(np.float32) / 255.0
                batch_x[i] = img
                batch_y[i] = self.labels[idx]

        return batch_x, batch_y


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indicies)


class CardRecognitionModel:
    """CNN model for recognizing cards"""
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 52):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()


    def _build_model(self) -> tf.keras.Model:
        """Build the CNN model"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(256, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation="softmax")
        ])

        return model


    def compile_model(self, learning_rate: float = 0.001):
        """Compiule the model with optmizer and loss function"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )


    def train(self,
        train_data: Tuple[np.ndarray, np.ndarray],
        validation_data: Tuple[np.ndarray, np.ndarray],
        epochs: int = 50,
        batch_size: int = 32,
        callbacks: List[tf.keras.callbacks.Callback] = None):
        """Train the model"""
        return self.model.fit(
            train_data[0], train_data[1],
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )


    def save_model(self, model_path: str):
        """Save the trained model"""
        self.model.save(model_path)


    def convert_to_tflite(self, tflite_path: str):
        """Convert the model to TFLite"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)


def create_data_splits(data_dir: str,
                       target_size: Tuple[int, int] = (224, 224),
                       batch_size: int = 32,
                       validation_split: float = 0.15,
                       test_split: float = 0.15) -> Tuple[CardDataGenerator,
                                                          CardDataGenerator,
                                                          CardDataGenerator,
                                                          dict]:
    data_dir = Path(data_dir)
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_mapping = {class_name: idx for idx, class_name in enumerate(classes)}
    print(f"INFO - create_data_splits(): Found {len(classes)} classes: {classes}")

    image_paths = []
    labels = []
    for class_dir in data_dir.iterdir():
        if not class_dir.is_dir():
            continue
        class_idx = class_mapping[class_dir.name]
        class_images = list(class_dir.glob("*.jpg"))
        print(f"INFO - create_data_splits(): Found {len(class_images)} images for class {class_dir.name}")
        image_paths.extend(class_images)
        labels.extend([class_idx] * len(class_images))

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_split, random_state=451, stratify=labels
    )

    val_size = validation_split / (1 - test_split)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_size, random_state=451,
        stratify=train_val_labels
    )

    print(f"\nINFO - create_data_splits(): Data split summary:")
    print(f"    Training set: {len(train_paths)} images")
    print(f"    Validation set: {len(val_paths)} images")
    print(f"    Test set: {len(test_paths)} images")

    train_gen = CardDataGenerator(train_paths, train_labels, target_size, batch_size, shuffle=True)
    val_gen = CardDataGenerator(val_paths, val_labels, target_size, batch_size, shuffle=False)
    test_gen = CardDataGenerator(test_paths, test_labels, target_size, batch_size, shuffle=False)

    return train_gen, val_gen, test_gen, class_mapping


def create_data_generators(data_dir: str,
                           target_size: Tuple[int, int] = (224, 224),
                           batch_size: int = 32,
                           validation_split: float = 0.2) -> Tuple[CardDataGenerator, CardDataGenerator, dict]:
    """Create train and validation data generaotrs"""
    data_dir = Path(data_dir)
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_mapping = {class_name: idx for idx, class_name in enumerate(classes)}
    print(f"Found {len(classes)} classes: {classes}")

    image_paths = []
    labels = []
    for class_dir in data_dir.iterdir():
        if not class_dir.is_dir():
            continue
        class_idx = class_mapping[class_dir.name]
        class_images = list(class_dir.glob("*.jpg"))
        print(f"Found {len(class_images)} images for class {class_dir.name}")
        image_paths.extend(class_images)
        labels.extend([class_idx] * len(class_images))

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=validation_split, random_state=451
    )

    print(f"Training set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")

    train_gen = CardDataGenerator(
        train_paths,
        train_labels,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True
    )

    val_gen = CardDataGenerator(
        val_paths,
        val_labels,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False
    )

    return train_gen, val_gen, class_mapping


class DataLoader:
    """Load and preprocess card images for training"""
    def __init__(self,
                 data_dir: str,
                 target_size: Tuple[int, int] = (224, 224),
                 batch_size: int = 32):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.batch_size = batch_size
        self.class_mapping = self._create_class_mapping()


    def _create_class_mapping(self) -> dict:
        """Create a mapping of card names to numeric indicies"""
        classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        return {class_name: idx for idx, class_name in enumerate(classes)}


    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load all images and labels"""
        images = []
        labels = []

        for class_dir in self.data_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_idx = self.class_mapping[class_dir.name]
            for img_path in class_dir.glob("*.jpg"):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.resize(img, self.target_size)
                img = img / 255.0   # Normalize
                images.append(img)
                labels.append(class_idx)
        return np.array(images), np.array(labels)


def train_card_model(data_dir: str,
                     model_output_dir: str,
                     input_shape: Tuple[int, int, int] = (224, 224, 3),
                     batch_size: int = 32,
                     epochs: int = 50,
                     learning_rate: float = 0.001):
    """Train and save the card recognition model"""
    output_dir = Path(model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_gen, val_gen, test_gen, class_mapping = create_data_splits(
        data_dir,
        target_size=input_shape[:2],
        batch_size=batch_size
    )

    data_loader = DataLoader(data_dir, target_size=input_shape[:2], batch_size=batch_size)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(output_dir / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(output_dir / "logs"),
            histogram_freq=1
        ) 
    ]

    model = CardRecognitionModel(input_shape=input_shape)
    model.compile_model(learning_rate=learning_rate)
    model.model.summary()

    history = model.model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
    )

    plot_training_history(history.history, output_dir)
    evaluation_results = evaluate_model(model.model, test_gen, class_mapping, output_dir)

    model.save_model(str(output_dir / "card_model.keras"))
    model.convert_to_tflite(str(output_dir / "card_model.tflite"))

    class_mapping_df = pd.DataFrame(
        list(data_loader.class_mapping.items()),
        columns=["card_name", "class_idx"]
    )
    class_mapping_df.to_csv(output_dir / "class_mapping.csv", index=False)

    return model, history, evaluation_results


def plot_training_history(history: Dict, output_dir: Path):
    """Plot training history"""
    matplotlib.use("Agg")
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Training Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png")
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_mapping: dict,
                          output_dir: Path):
    """Plot confusion matrix"""
    matplotlib.use("Agg")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()


def evaluate_model(model,
                    test_gen: CardDataGenerator,
                    class_mapping: dict,
                    output_dir: Path) -> dict:
    """Evaluate model on test test"""
    all_predictions = []
    all_labels = []
    print(f"\nINFO - evaluate_model(): Evaluating model on test set")
    for i in range(len(test_gen)):
        x, y = test_gen[i]
        predictions = model.predict(x, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        all_predictions.extend(pred_classes)
        all_labels.extend(y)

    reverse_mapping = {v: k for k, v in class_mapping.items()}
    label_names = [reverse_mapping[i] for i in range(len(class_mapping))]
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=label_names,
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_dir / "classification_report.csv")
    plot_confusion_matrix(all_labels, all_predictions, class_mapping, output_dir)
    print(f"\nINFO - evaluate_model(): Test set Evaluation:")
    print(f"    Overall accuracy: {report['accuracy']:.4f}")
    print("     \nPer-class metrics:")
    for card, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"    {card}: Precision={metrics['precision']:.4f}, "
                  f"    Recall={metrics['recall']:.4f}, "
                  f"    F1-score={metrics['f1-score']:.4f}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train card recognition model")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing the card dataset")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Initial learning rate")
    args = parser.parse_args()

    model, history, evaluation = train_card_model(
        args.data_dir,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
