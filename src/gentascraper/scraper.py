"""
## Module for scraping HTML string from a URL and preprocess it.

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

import re
import requests

import bs4.element as elem

from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from bs4.element import Comment

from .utilities import is_numeric


class Scraper:
    """
    The Scraper class scrapes the HTML string from a URL and preprocesses it.

    :param url: The URL to scrape.
    :type url: str
    """

    def __init__(self, data: str) -> None:
        """
        Initializes the Scraper class.

        :param data: The URL or HTML to scrape.
        :type url: str
        """

        # check if data is a URL or HTML string
        if data.startswith('http'):
            self.__get_html(data)
        else:
            self.__html = data
        
        self.__html = re.sub(r'<!(.*?)->','', self.__html)
        self.__html = re.sub(r'href=".*?"', '', self.__html)
        self.__soup = BeautifulSoup(
            self.__html, 
            'lxml'
        )

        self.__get_meta()
        self.__preprocess()
        self.__get_article_raw()

    def __get_html(self, url: str) -> None:
        """
        Gets the HTML string from the URL.

        :param url: The URL to get the HTML string from.
        :type url: str
        """

        self.__html = requests.get(url).text

    def __get_date(self) -> str:
        """
        Gets the date from the meta tags.
        """

        return next((self.__meta[key] for key in self.__meta \
            if 'date' in key.lower() or 'time' in key.lower()), None)
    
    def __get_site_name(self) -> str:
        """
        Gets the site name from the meta tags.
        """

        return next((self.__meta[key] for key in self.__meta \
            if 'site' in key.lower()), None)

    def __get_author(self) -> List[str]:
        """
        Gets the author from the meta tags.
        """

        return [self.__meta[key] for key in self.__meta \
            if ('author' in key.lower() or 'writer' in key.lower()) and \
               (not is_numeric(self.__meta[key]))]

    def __get_title(self) -> str:
        """
        Gets the title from the meta tags.
        """

        title = next((self.__meta[key] for key in self.__meta \
            if 'title' in key.lower()), None)
    
        if title is None:
            title = self.__soup.title.string
    
        return title

    def __get_meta(self) -> None:
        """
        Gets the meta tags from the HTML string.
        """
        
        self.__meta = {tag['name']: tag['content'] for tag in \
            self.__soup.find_all('meta') if tag.has_attr('name')}

    def __preprocess(self) -> None:
        """
        Preprocesses the HTML string by removing all script tags, links, styles, meta tags, item list, and comments.
        """

        body = self.__soup.body

        unwanted_tags = ['script', 'style', 'meta', 'li', 'ul', 
                         'iframe', 'br', 'noscript', 'aside', 'nav',
                         'form', 'input', 'button', 'select', 'option',
                         'textarea', 'label', 'fieldset', 'legend', 'datalist',
                         'output', 'progress', 'meter', 'details', 'summary',
                         'menu', 'menuitem', 'dialog']
        for tag in body(unwanted_tags):
            tag.extract()

        id_class_pattern = re.compile(r'share|related|socmed|social|media|' + \
                                      r'social|facebook|twitter|instagram|' + \
                                      r'youtube|linkedin|whatsapp|telegram|' + \
                                      r'line|tiktok|pinterest|tumblr|reddit|vimeo')

        tags_with_attr = [('div', 'id'), ('div', 'class'), ('span', 'id'), ('span', 'class')]
        for tag, attr in tags_with_attr:
            [s.decompose() for s in body.find_all(tag, {attr: id_class_pattern})]

        self.__body = body.prettify()

    def __get_article_raw(self) -> None:
        """
        Gets the article from the HTML string.
        """

        # extract text from the body
        self.__article_raw = self.text_from_html(self.__soup)
        
    @staticmethod
    def tag_visible(element: elem.Tag) -> bool:
        """
        Checks if the tag is visible.

        :param element: The element to check.
        :type element: bs4.element.Tag
        :return: True if the tag is visible, False otherwise.
        """

        return not (element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]'] \
                    or isinstance(element, Comment))

    @staticmethod
    def text_from_html(soup: BeautifulSoup) -> str:
        """
        Gets the text from the HTML string.

        :param soup: The HTML string.
        :type soup: bs4.BeautifulSoup
        :return: The text from the HTML string.
        :rtype: str
        """

        texts = soup.findAll(text=True)
        visible_texts = filter(Scraper.tag_visible, texts)  
        return u" ".join(t.strip() for t in visible_texts)
    
    @property
    def html(self) -> str:
        """
        Gets the HTML string.

        :return: The HTML string.
        :rtype: str
        """

        return self.__html

    @property
    def meta(self) -> Dict[str, str]:
        """
        Gets the meta tags.

        :return: The meta tags.
        :rtype: dict
        """

        return self.__meta

    @property
    def site_name(self) -> str:
        """
        Gets the site name.

        :return: The site name.
        :rtype: str
        """

        return self.__get_site_name()
    
    @property
    def author(self) -> List[str]:
        """
        Gets the author.

        :return: The author.
        :rtype: str
        """

        return self.__get_author()

    @property
    def date(self) -> str:
        """
        Gets the date.

        :return: The date.
        :rtype: str
        """

        return self.__get_date()

    @property
    def title(self) -> str:
        """
        Gets the title.

        :return: The title.
        :rtype: str
        """

        return self.__get_title()

    @property
    def body(self) -> str:
        """
        Gets the body.

        :return: The body.
        :rtype: str
        """

        return self.__body
    
    @property
    def article_raw(self) -> str:
        """
        Gets the article before fed to AI model.

        :return: The article.
        :rtype: str
        """

        return self.__article_raw
