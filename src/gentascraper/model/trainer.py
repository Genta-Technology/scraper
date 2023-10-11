"""
## Module for training the AI model used to scrape article from the extracted visible text.

Copyright (C) 2023 Genta Technology

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

import os

import transformers
import tensorflow as tf

from typing import List, Tuple
from transformers import DefaultDataCollator, create_optimizer
from datasets import dataset_dict

from .model import Model


class Trainer:
    def __init__(self, checkpoint: str, **kwargs) -> None:
        self.checkpoint = checkpoint
        self.batch_size = kwargs.get("batch_size", 32)
        self.num_epochs = kwargs.get("num_epochs", 5)
        self.init_lr = kwargs.get("init_lr", 2e-5)
        self.warmup_steps = kwargs.get("warmup_steps", 0)

    def load_optimizer(
        self,
        train_dataset: dataset_dict.DatasetDict
    ) -> Tuple[tf.keras.optimizers.Optimizer, transformers.data.data_collator.DataCollator]:
        """
        Returns an optimizer and data collator for training a machine learning model using the given `train_dataset`.

        :param train_dataset: The dataset to be used for training.
        :param batch_size: The batch size.
        :param num_epochs: The number of epochs.

        :return: A tuple of optimizer and data collator.
        """

        data_collator = DefaultDataCollator(return_tensors="tf")
        total_train_steps = (len(train_dataset) // self.batch_size) * self.num_epochs
        optimizer, schedule = create_optimizer(
            init_lr=self.init_lr,
            num_warmup_steps=self.warmup_steps,
            num_train_steps=total_train_steps,
        )

        return optimizer, data_collator

    def finetune(
            self,
            data: dataset_dict.DatasetDict,
            max_length: int = 384,
            stride: int = 128,
            callbacks: List[tf.keras.callbacks.Callback] = None,
            save_dir: str = None,
            save_filename: str = "model"
        ) -> tf.keras.Model:
        """
        Fine-tunes a pre-trained transformer-based language model on a QA task using the provided dataset.
        
        :param data: The dataset to be used for fine-tuning. Should contain train and test column.
        :param model_checkpoint: The name of the pre-trained model to be fine-tuned.
        :param max_length: The maximum length of the input sequence.
        :param stride: The stride of the sliding window.
        :param batch_size: The batch size.
        :param num_epochs: The number of epochs.
        :param callbacks: A list of callbacks to be used during training.
        :param save_dir: The directory to save the trained model.
        :param save_filename: The filename to save the trained model.
        :return: The trained model.
        """
        
        if not ("train" in data and "test" in data):
            raise ValueError("Data should contain train and test column.")
        
        def dataValid(data):
            return ("id"        in data and
                    "context"   in data and
                    "question"  in data and
                    "answer"    in data and
                    ("answer_start" in data["answer"] and
                    "text" in data["answer"]))
        
        if not dataValid(data["train"]):
            raise ValueError("Each data should contain id, context, question, answer. \
                             Each answer in data should contain answer_start and text.")
        
        _model = Model(self.checkpoint, max_length=max_length, stride=stride)
        model, tokenizer = _model.model, _model.tokenizer

        tokenized_data = data.map(
            lambda example: _model.preprocess_training(example, tokenizer, max_length, stride),
            batched=True,
            remove_columns=data["train"].column_names
        )

        optimizer, data_collator = self.load_optimizer(data["train"], self.batch_size, self.num_epochs)

        tf_train_set = model.prepare_tf_dataset(
            tokenized_data["train"],
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=data_collator,
        )

        tf_validation_set = model.prepare_tf_dataset(
            tokenized_data["test"],
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=data_collator,
        )

        model.compile(optimizer=optimizer)
        
        model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=self.num_epochs, callbacks=callbacks)

        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, save_filename)
            model.save(save_path)
            print(f"Trained model saved to {save_path}")

        return model
