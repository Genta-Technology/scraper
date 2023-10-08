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
import numbers

import cchardet as chardet

from typing import Dict, Optional
from bs4 import BeautifulSoup, SoupStrainer

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
        
        self.__html = re.sub(r'<!.*?->','', self.__html)
        self.__soup = BeautifulSoup(
            self.__html, 
            'lxml'
        )

        self.__get_meta()
        self.__get_site_name()
        self.__get_author()
        self.__get_date()
        self.__get_title()
        self.__preprocess()

    def __get_html(self, url: str) -> None:
        """
        Gets the HTML string from the URL.

        :param url: The URL to get the HTML string from.
        :type url: str
        """

        self.__html = requests.get(url).text

    def __get_date(self) -> None:
        """
        Gets the date from the meta tags.
        """

        self.__date = next((self.__meta[key] for key in self.__meta \
            if 'date' in key.lower() or 'time' in key.lower()), None)
    
    def __get_site_name(self) -> None:
        """
        Gets the site name from the meta tags.
        """

        self.__site_name = next((self.__meta[key] for key in self.__meta \
            if 'site' in key.lower()), None)

    def __get_author(self) -> None:
        """
        Gets the author from the meta tags.

        :param meta: The meta tags.
        :type meta: dict
        """

        self.__author = next((self.__meta[key] for key in self.__meta \
            if ('author' in key.lower() or 'writer' in key.lower()) and \
               ('content' in key.lower() or 'article' in key.lower()) and \
               (not is_numeric(self.__meta[key]))), None)

    def __get_title(self) -> None:
        """
        Gets the title from the meta tags.

        :param meta: The meta tags.
        :type meta: dict
        """

        self.__title = next((self.__meta[key] for key in self.__meta \
            if 'title' in key.lower()), None)
    
        if self.__title is None:
            self.__title = self.__soup.title.string

    def __get_meta(self) -> None:
        """
        Gets the meta tags from the HTML string.

        :param html: The HTML string to get the meta tags from.
        :type html: str
        """

        self.__meta = {tag['name']: tag['content'] for tag in \
            self.__soup.find_all('meta') if tag.has_attr('name')}

    def __preprocess(self) -> None:
        """
        Preprocesses the HTML string by removing all script tags, links, styles, meta tags, item list, and comments.

        :param html: The HTML string to preprocess.
        :type html: str
        """

        body = self.__soup.body

        unwanted_elements = ['script', 'link', 'style', 'meta', 'ul', 'iframe', 'i', 'br', 'noscript']
        for element in unwanted_elements:
            [s.extract() for s in body(element)]

        id_class_pattern = re.compile(r'ad|ads|comment|disqus|share|related|' + \
                                    r'google|sso|recommendation|pagination|' + \
                                    r'footer|header|menu|nav|sidebar|widget|' + \
                                    r'social|facebook|twitter|instagram|' + \
                                    r'youtube|linkedin|whatsapp|telegram|' + \
                                    r'line|tiktok|pinterest|tumblr|reddit|vimeo|' + \
                                    r'snap|modal|overlay|popup|banner|' + \
                                    r'cookie|consent|gdpr|privacy|terms|' + \
                                    r'login|register|subscribe|newsletter|' + \
                                    r'contact|search|form|input|button|' + \
                                    r'video|audio|image|picture|gallery|')

        tags_with_attr = [('div', 'id'), ('div', 'class'), ('span', 'id'), ('span', 'class')]
        for tag, attr in tags_with_attr:
            [s.extract() for s in body.find_all(tag, {attr: id_class_pattern})]

        self.__body = body.prettify()
    
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

        return self.__site_name
    
    @property
    def author(self) -> str:
        """
        Gets the author.

        :return: The author.
        :rtype: str
        """

        return self.__author

    @property
    def date(self) -> str:
        """
        Gets the date.

        :return: The date.
        :rtype: str
        """

        return self.__date

    @property
    def title(self) -> str:
        """
        Gets the title.

        :return: The title.
        :rtype: str
        """

        return self.__title

    @property
    def body(self) -> str:
        """
        Gets the body.

        :return: The body.
        :rtype: str
        """

        return self.__body
