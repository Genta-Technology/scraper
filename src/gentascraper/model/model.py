"""
## Module containing the model used to scrape article from the extracted visible text.

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

import collections
import datasets
import numpy as np

from datasets import Dataset
from typing import List, Dict, Any, Tuple
from transformers import TFBertForQuestionAnswering, BertTokenizerFast, BertConfig


class Model:
    """
    A class used to represent a BERT-based question-answering model.

    :param checkpoint: The name of the pre-trained model to load.
    :type checkpoint: str
    :param max_length: The maximum length of the input sequences.
    :type max_length: int

    :ivar checkpoint: The name of the pre-trained model to load.
    :vartype checkpoint: str
    :ivar max_length: The maximum length of the input sequences.
    :vartype max_length: int
    :ivar model: The loaded model.
    :ivar tokenizer: The loaded tokenizer.
    :ivar stride: The stride to use when splitting the input sequences.
    :ivar n_best: The number of best answers to extract.
    :ivar max_answer_length: The maximum length of the answer.
    :ivar model_checkpoint: The name of the pre-trained model to load.
    :ivar from_pt: Whether to load the model from PyTorch weights.

    :return: A BERT-based question-answering model.
    :rtype: Model
    """

    def __init__(self, checkpoint: str, **kwargs) -> None:
        self.checkpoint = checkpoint
        self.max_length = kwargs.get("max_length", 512)
        self.stride = kwargs.get("stride", 128)
        self.n_best = kwargs.get("n_best", 20)
        self.max_answer_length = kwargs.get("max_answer_length", 30)

        self.model, self.tokenizer = self.load_model(self.checkpoint)

    def load_model(
        model_checkpoint: str = "Rifky/IndoBERT-Large-P2-QA",
        from_pt: bool = False
    ) -> Tuple[TFBertForQuestionAnswering, BertTokenizerFast]:
        """
        Loads a pre-trained BERT model and its corresponding tokenizer.

        :param model_checkpoint: The name of the pre-trained model to load.
        :type model_checkpoint: str
        :param from_pt: Whether to load the model from PyTorch weights.
        :type from_pt: bool
        :return: The loaded model and tokenizer.
        :rtype: Tuple[TFBertForQuestionAnswering, BertTokenizerFast]
        """

        tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
        config = BertConfig.from_pretrained(model_checkpoint)

        # if using IndoBERT model by IndoNLU (indobenchmark)
        if (model_checkpoint.split('/')[0] == "indobenchmark"):
            config.num_labels = 2
            
        model = TFBertForQuestionAnswering.from_pretrained(model_checkpoint, config=config, from_pt=from_pt)

        return model, tokenizer

    def predict(self, inputs: dict) -> List[Dict[str, str]]:
        """
        Uses a BERT-based question-answering model to predict answers to a set of input questions and contexts. The input
        is first preprocessed to tokenize and split the sequences into windows, then passed through the model to obtain
        answer probabilities for each window. The resulting probabilities are converted into answer strings, and the top
        N answers with the highest probabilities are returned.

        :param inputs: A dictionary containing the input questions and contexts.
        :type inputs: dict
        :return: A list of dictionaries containing the predicted answers.
        :rtype: List[Dict[str, str]]
        """

        inputs = Dataset.from_dict(inputs)
        inputs_tokenized = Dataset.from_dict(self.preprocess(inputs))

        # remove unnecessary columns and set the format convert to tf.Tensor
        inputs_model = inputs_tokenized.remove_columns(["example_id", "offset_mapping"])
        inputs_model.set_format("tf")

        # get model predictions
        inputs_model = {k: inputs_model[k] for k in inputs_model.column_names}
        outputs = self.model(**inputs_model)

        return self.postprocess(outputs, inputs_tokenized, inputs, self.n_best, self.max_answer_length)

    def preprocess_training(self, examples: Dataset):
        """
        Preprocesses training examples for the model.

        :param examples: Training examples.
        :type examples: List[Dict[str, Any]]
        :return: Preprocessed training examples.
        """

        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs
    
    def preprocess(self, examples: datasets.arrow_dataset.Dataset) -> dict:
        """
        Preprocesses the input questions and contexts for the model.

        :param examples: The input questions and contexts.
        :type examples: datasets.arrow_dataset.Dataset
        :return: The preprocessed input questions and contexts.
        :rtype: dict
        """

        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs
    
    def postprocess(
        self,
        logits,
        features,
        examples,
        n_best: int = 20,
        max_answer_length: int = 30
    ):
        """
        Postprocesses the model's output to extract the best answers to the given questions.

        :param logits: The model's output.
        :type logits: transformers.modeling_outputs.QuestionAnsweringModelOutput
        :param features: The features used to generate the model's output.
        :type features: List[Dict[str, Any]]
        :param examples: The examples used to generate the model's output.
        :type examples: List[Dict[str, Any]]
        :param n_best: The number of best answers to extract.
        :type n_best: int
        :param max_answer_length: The maximum length of the answer.
        :type max_answer_length: int
        :return: The best answers to the given questions.
        :rtype: List[Dict[str, Any]]
        """

        # Map the example ID to its associated features
        start_logits = logits.start_logits
        end_logits = logits.end_logits

        # Map the example ID to its associated features
        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in examples:
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                            "start_position": offsets[start_index][0],
                            "end_position": offsets[end_index][1]
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {
                        "id": example_id,
                        "prediction_text": best_answer["text"],
                        "start_position": best_answer["start_position"],
                        "end_position": best_answer["end_position"],
                        "score": best_answer["logit_score"]
                    }
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": "", "start_position": -1, "end_position": -1, "score": -1})

        return predicted_answers
