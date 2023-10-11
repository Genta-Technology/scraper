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
import pandas as pd

from typing import List, Tuple

from newspaper import Article
from transformers import hf_argparser, BertTokenizerFast
from datasets import Dataset

from gentascraper import Scraper


class DataGatherer():
    """
    Class to gather data from news sites based on newspaper3k.

    :param urls: urls of the articles
    :type urls: List[str]
    
    :ivar id: id of the article
    :vartype id: List[int]
    :ivar url: url of the article
    :vartype url: List[str]
    :ivar input: input text
    :vartype input: List[str]
    :ivar label: label text
    :vartype label: List[str]
    :ivar start_pos: start position of label in input
    :vartype start_pos: List[int]
    :ivar end_pos: end position of label in input
    :vartype end_pos: List[int]

    :ivar tokenizer: tokenizer to tokenize input and label
    :vartype tokenizer: BertTokenizerFast

    :return: data gatherer object
    :rtype: DataGatherer

    :example:
    >>> from gentascraper import DataGatherer
    >>> urls = [
    ...     "https://nasional.kompas.com/read/2023/10/10/14253131/mk-putuskan-usia-capres-cawapres-16-oktober",
    ...     "https://nasional.kompas.com/read/2023/10/10/17002281/jokowi-disebut-akan-kunjungi-china-dan-arab-saudi"
    ... ]
    >>> data_gatherer = DataGatherer(urls)
    >>> data_gatherer.to_pandas()
    ... id	url	                                                title                                               ...
    ... 0	https://nasional.kompas.com/read/2023/10/10/14...	JAKARTA, KOMPAS.com - Mahkamah Konstitusi (MK)...	...
    ... 1	https://nasional.kompas.com/read/2023/10/10/17...	JAKARTA, KOMPAS.com - Presiden Joko Widodo aka...	...
    """

    def __init__(self, urls: List[str]) -> None:
        self.__data = {
            'id': [],
            'url': [],
            'question': [],
            'context': [],
            'answers': []
        }

        self.tokenizer = BertTokenizerFast.from_pretrained('indobenchmark/indobert-base-p2')

        articles, self.__data['question'] = self.__get_articles(urls)
        start_indexes = self.__get_answers(articles)
        self.__data['answers'] = [{'text': article, 'answer_start': start} \
                                  for article, start in zip(articles, start_indexes)]

        self.__get_inputs(urls)

        for i, url in enumerate(urls):
            self.__data['id'].append(i)
            self.__data['url'].append(url)

    
    def to_pandas(self) -> pd.DataFrame:
        """
        Save data to pandas dataframe.
        
        :return: data in pandas dataframe
        :rtype: pd.DataFrame
        """

        return pd.DataFrame(self.__data)

    def to_csv(self, path: str) -> None:
        """
        Save data to csv file.
        
        :param path: path to save csv file
        :type path: str
        """

        df = pd.DataFrame(self.__data)
        df.to_csv(path, index=False)

    def to_json(self, path: str) -> None:
        """
        Save data to json file.
        
        :param path: path to save json file
        :type path: str
        """

        df = pd.DataFrame(self.__data)
        df.to_json(path, orient='records')

    def to_pickle(self, path: str) -> None:
        """
        Save data to pickle file.
        
        :param path: path to save pickle file
        :type path: str
        """

        df = pd.DataFrame(self.__data)
        df.to_pickle(path)

    def to_dict(self) -> dict:
        """
        Save data to dictionary.
        
        :return: data in dictionary
        :rtype: dict
        """

        return self.__data
    
    def to_list(self) -> List:
        """
        Save data to list.
        
        :return: data in list
        :rtype: List
        """

        return list(self.__data.values())
    
    def to_dataset(self) -> Dataset:
        """
        Save data to dataset.
        
        :return: data in dataset
        :rtype: Dataset
        """

        return Dataset.from_dict(self.__data)
    
    def push_to_hub(self, repo_name: str) -> None:
        """
        Push data to huggingface hub.
        
        :param repo_name: name of the repository
        :type repo_name: str
        """

        dataset = self.to_dataset()
        dataset.push_to_hub(repo_name)

    def __str__(self) -> str:
        return str(self.__data)

    def __repr__(self) -> str:
        return str(self.__data)

    def __len__(self) -> int:
        return len(self.__data['id'])

    def __getitem__(self, key: str) -> List:
        return self.__data[key]

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
        return article.text, article.title

    def __get_articles(self, urls: List[str]) -> Tuple[List[str], List[str]]:
        """
        Get articles from urls using newspaper3k.
        
        :param urls: urls of the articles
        :type urls: List[str]
        :return: articles text and titles
        :rtype: Tuple[List[str], List[str]]
        """

        with mp.Pool() as pool:
            return zip(*pool.starmap(self.__get_article, zip(urls, self.__data['question'])))
    
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
            self.__data['context'] = pool.map(self.__get_input, urls)

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

        input_tokens = self.tokenizer(input, add_special_tokens=False, return_offsets_mapping=True)
        label_tokens = self.tokenizer(label, add_special_tokens=False, return_offsets_mapping=True)

        return input_tokens['offset_mapping'][self.__get_start_index(input_tokens['input_ids'], label_tokens['input_ids'])][0]
    
    def __get_answers(self, articles: List[str]) -> None:
        """
        Get start and end indexes of labels in inputs.
        """

        with mp.Pool() as pool:
            return pool.map(self.__get_index, self.__data['context'], articles)

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
        
        # take the first 5 tokens of label
        head_label_tokens = label_token_ids[:5]

        # get start index of label in input
        match_tokens = 0
        for i, token in enumerate(input_token_ids):
            if token == head_label_tokens[match_tokens]:
                match_tokens += 1
                if match_tokens == len(head_label_tokens):
                    return i - len(head_label_tokens) + 1
            else:
                match_tokens = 0

        return -1
