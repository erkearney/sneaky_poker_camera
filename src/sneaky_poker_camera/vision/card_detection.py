import tensorflow as tf
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Dict, List
from .preprocessing import CardPreprocessor

class CardDectector:
    """Card detection and recognition TFLite model"""
    def __init__(self, model_path: str):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.preprocessor = None

        try:
            self._initialize(model_path)
        except Exception as e:
            print(f"ERROR: Failed to initialze card detector {e}")
            raise


    def _initialize_model(self, model_path: str) -> None:
        """
        Initialze TFLite model with input output details

        Args:
            model_path (str)
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        input_shape = tuple(self.input_details[0]['shape'][1:3])
        self.preprocessor = CardPreprocessor(input_shape)
