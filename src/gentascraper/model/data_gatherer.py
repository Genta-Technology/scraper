"""
## Module to gather data from news sites based on newspaper3k.

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

import multiprocessing.dummy as mp

from typing import List, Tuple

from newspaper import Article
from transformers import hf_argparser, BertTokenizerFast

from gentascraper import Scraper

class DataGatherer():
    def __init__(self) -> None:
        self.__dict__ = {
            'id': [],
            'url': [],
            'input': [],
            'label': [],
            'start_pos': [],
            'end_pos': [],
        }

        self.tokenizer = BertTokenizerFast.from_pretrained('indobenchmark/indobert-base-p2')

    def __get_article(self, url: str) -> str:
        """
        Get article from url using newspaper3k.
        
        :param url: url of the article
        :type url: str
        :return: article text and title
        :rtype: Tuple[str, str]
        """

        article = Article(url)
        article.download()
        article.parse()
        return article.text

    def __get_articles(self, urls: List[str]) -> None:
        """
        Get articles from urls using newspaper3k.
        
        :param urls: urls of the articles
        :type urls: List[str]
        """

        with mp.Pool() as pool:
            self.__dict__['label'].append(pool.map(self.get_article, urls))
    
    def __get_input(self, url: str) -> str:
        """
        Get input from url using gentascraper.
        
        :param url: url of the article
        :type url: str
        :return: input text
        :rtype: str
        """

        scraper = Scraper(url)
        return scraper.article_raw
    
    def __get_inputs(self, urls: List[str]) -> None:
        """
        Get inputs from urls using gentascraper.
        
        :param urls: urls of the articles
        :type urls: List[str]
        """

        with mp.Pool() as pool:
            self.__dict__['input'].append(pool.map(self.get_input, urls))

    def __get_index(self, input: str, label: str) -> Tuple[int, int]:
        """
        Get start and end index of label in input.

        :param input: input text
        :type input: str
        :param label: label text
        :type label: str
        :return: start and end index of label in input
        :rtype: Tuple[int, int]
        """

        input_tokens = self.tokenizer.encode(input, add_special_tokens=False, return_offsets_mapping=True)
        label_tokens = self.tokenizer.encode(label, add_special_tokens=False, return_offsets_mapping=True)

        return self.__get_start_index(input_tokens['input_ids'], label_tokens['input_ids']), \
            self.__get_end_index(input_tokens['input_ids'], label_tokens['input_ids'])
    
    def __get_indexes(self) -> None:
        """
        Get start and end indexes of labels in inputs.
        """

        with mp.Pool() as pool:
            self.__dict__['start_pos'], self.__dict__['end_pos'] = zip(*pool.starmap(self.get_index, zip(self.__dict__['input'], self.__dict__['label'])))

    def __get_start_index(self, input_token_ids: List[int], label_token_ids: List[int]) -> int:
        """
        Get start index of label in input.
        
        :param input_token_ids: token ids of input
        :type input_token_ids: List[int]
        :param label_token_ids: token ids of label
        :type label_token_ids: List[int]
        :return: start index of label in input
        :rtype: int
        """
        
        # # tokenize input using BPE tokenizer
        # input_tokens = self.tokenizer.tokenize(input, add_special_tokens=False, return_offsets_mapping=True)
        # # tokenize label using BPE tokenizer
        # label_tokens = self.tokenizer.tokenize(label, add_special_tokens=False, return_offsets_mapping=True)

        # take the first 5 tokens of label
        head_label_tokens = label_token_ids[:5]

        # get start index of label in input
        match_tokens = 0
        for i, token in enumerate(input_token_ids):
            if token == head_label_tokens[match_tokens]:
                match_tokens += 1
                if match_tokens == 5:
                    return i - len(head_label_tokens) + 1
            else:
                match_tokens = 0

        return -1
    
    def __get_end_index(self, input_token_ids: List[int], label_token_ids: List[int]) -> int:
        """
        Get end index of label in input.
        
        :param input_token_ids: token ids of input
        :type input_token_ids: List[int]
        :param label_token_ids: token ids of label
        :type label_token_ids: List[int]
        :return: end index of label in input
        :rtype: int
        """

        # # tokenize input using BPE tokenizer
        # input_tokens = self.tokenizer.tokenize(input, add_special_tokens=False, return_offsets_mapping=True)
        # # tokenize label using BPE tokenizer
        # label_tokens = self.tokenizer.tokenize(label, add_special_tokens=False, return_offsets_mapping=True)

        # take the last 5 tokens of label
        tail_label_tokens = label_token_ids[-5:]

        # get end index of label in input
        match_tokens = 0
        for i, token in enumerate(reversed(input_token_ids)):
            if token == tail_label_tokens[match_tokens]:
                match_tokens += 1
                if match_tokens == 5:
                    return len(input_token_ids) - i + len(tail_label_tokens) - 1
            else:
                match_tokens = 0

        return -1
