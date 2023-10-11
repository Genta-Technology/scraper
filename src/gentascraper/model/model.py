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

from typing import List, Dict, Any
from transformers import BertForQuestionAnswering, BertTokenizer


class Model:
    def __init__(self, checkpoint: str, **kwargs) -> None:
        self.checkpoint = checkpoint

        # Default values
        self.max_length = 512
        self.stride = 128
        self.n_best = 20
        self.max_answer_length = 512

        if kwargs["max_length"] is not None:
            self.max_length = kwargs["max_length"]
        if kwargs["stride"] is not None:
            self.stride = kwargs["stride"]
        if kwargs["n_best"] is not None:
            self.n_best = kwargs["n_best"]
        if kwargs["max_answer_length"] is not None:
            self.max_answer_length = kwargs["max_answer_length"]

        self.model = BertForQuestionAnswering.from_pretrained(checkpoint)
        self.tokenizer = BertTokenizer.from_pretrained(checkpoint)

    def preprocess_training_examples(self, examples):
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
    
    def preprocess(self, examples: datasets.arrow_dataset.Dataset):
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
