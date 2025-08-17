
###Loss Function: I will use sigmoid activation function, we convert the multi-label problem into multiple binary classification problems, where each class is predicted independently, and we apply binary crossentropy loss to optimize the model's performance.
import os
import sys
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dropout, Flatten, Dense
)
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.exception import CustomException
from src.logger import logging


class ModelTrainer:
    """
    Handles model creation, training on batches, and saving weights.
    """

    def __init__(self, img_size: Tuple[int, int] = (128, 128), epochs: int = 5,
                 n_class: int = 14, learning_rate: float = 1e-3,
                 weights_path: str = "checkpoints/best_model.weights.h5"):
        self.img_size = img_size
        self.epochs = epochs
        self.n_class = n_class
        self.learning_rate = learning_rate
        self.weights_path = weights_path
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)

        # Build or load model
        self.model = self._build_model()
        if os.path.exists(self.weights_path):
            logging.info(f"Loading existing weights from {self.weights_path}")
            self.model.load_weights(self.weights_path)

    def _build_model(self) -> Sequential:
        """Build transfer learning model with MobileNet backbone."""
        try:
            base_model = MobileNet(
                input_shape=(self.img_size[0], self.img_size[1], 1),
                include_top=False,
                weights=None  # no pretrained ImageNet on grayscale
            )
            model = Sequential([
                base_model,
                GlobalAveragePooling2D(),
                Dropout(0.3),
                Flatten(),
                Dense(140, activation='relu'),
                Dropout(0.3),
                Dense(self.n_class, activation='sigmoid')
            ])
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )
            logging.info("Model built successfully")
            return model
        except Exception as e:
            logging.error("Failed to build model")
            raise CustomException(e, sys)

    def train_batch(self, train_gen, valid_gen, batch_num: int):
        """Train the model on a single batch of images."""
        try:
            checkpoint = ModelCheckpoint(
                filepath=self.weights_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
            earlystopping = EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )

            history = self.model.fit(
                train_gen,
                validation_data=valid_gen,
                epochs=self.epochs,
                callbacks=[earlystopping, checkpoint],
                verbose=1
            )
            logging.info(f"Batch {batch_num} training completed")
            return history

        except Exception as e:
            logging.error(f"Failed training on batch {batch_num}")
            raise CustomException(e, sys)

    def save_model(self, path: str = "artifacts/final_model.h5"):
        """Save the final model after all batches."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
            logging.info(f"Final model saved at {path}")
        except Exception as e:
            logging.error("Failed to save final model")
            raise CustomException(e, sys)
