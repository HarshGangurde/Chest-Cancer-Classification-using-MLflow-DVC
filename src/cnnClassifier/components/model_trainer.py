import os
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import (DataIngestionConfig,PrepareBaseModelConfig)
from cnnClassifier.entity.config_entity import TrainingConfig
import tensorflow as tf
import os

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        # Load the pre-trained model
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
        
        # Modify the output layer to 4 classes
        x = self.model.output
        output_layer = tf.keras.layers.Dense(3, activation='softmax',name='final_output')(x)
        self.model = tf.keras.models.Model(inputs=self.model.input, outputs=output_layer)

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            class_mode='categorical',
            interpolation="bilinear"
        )

        # Training generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
