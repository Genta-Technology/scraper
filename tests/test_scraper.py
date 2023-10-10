"""
Test scraper.py
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

import sys
import os
import re
import requests

from time import time

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from gentascraper import Scraper

test_case = Scraper("https://nasional.kompas.com/read/2023/10/08/13173901/" + \
                    "jokowi-benarkan-akan-bertemu-syahrul-yasin-limpo-di-istana-malam-ini")

def test_get_title():
    assert test_case.title.lower() == "Jokowi Benarkan Akan Bertemu Syahrul Yasin Limpo di Istana Malam Ini".lower()

def test_get_author():
    assert "Nicholas Ryan Aditya" in test_case.author

def test_get_date():
    assert test_case.date.lower() == "2023-10-08 13:17:39".lower() or \
           test_case.date.lower() == "2023-10-08T06:17:39+00:00".lower()
    
def test_length_preprocessed():
    assert len(test_case.article_raw.split()) < 1000

def test_time_to_scrape():
    html = requests.get("https://nasional.kompas.com/read/2023/10/08/13173901/" + \
                        "jokowi-benarkan-akan-bertemu-syahrul-yasin-limpo-di-istana-malam-ini").text
    start = time()
    Scraper(html)
    end = time()
    time_elapsed = end - start
    assert time_elapsed < 0.6

test_case_2 = Scraper("https://www.merdeka.com/peristiwa/reaksi-muhammadiyah-" + \
                      "hingga-mui-soal-konflik-hamas-vs-israel-memanas-33021-mvk.html")

def test_get_title_2():
    assert test_case_2.title.lower() == "Reaksi Muhammadiyah hingga MUI Soal Konflik Hamas Vs Israel Memanas".lower()

def test_length_preprocessed_2():
    assert len(test_case_2.article_raw.split()) < 1000

def test_time_to_scrape_2():
    html = requests.get("https://www.merdeka.com/peristiwa/reaksi-muhammadiyah" + \
                        "-hingga-mui-soal-konflik-hamas-vs-israel-memanas-33021-mvk.html").text
    start = time()
    Scraper(html)
    end = time()
    time_elapsed = end - start
    assert time_elapsed < 0.06

test_case_3 = Scraper("https://www.cnnindonesia.com/nasional/20231008174210-617-" + \
                      "1008627/anies-usul-pemilik-lahan-terdampak-pembangunan-tol-dapat-bagian-saham")

def test_get_title_3():
    assert test_case_3.title.lower() == "Anies Usul Pemilik Lahan Terdampak Pembangunan Tol Dapat Bagian Saham".lower()

def test_get_author_3():
    assert "CNN Indonesia" in test_case_3.author

def test_get_date_3():
    assert test_case_3.date.lower() == "2023-10-08 17:42:10".lower() or \
           test_case_3.date.lower() == "2023-10-08 18:20:04".lower()

def test_length_preprocessed_3():
    assert len(test_case_3.article_raw.split()) < 1000

def test_time_to_scrape_3():
    html = requests.get("https://www.cnnindonesia.com/nasional/20231008174210-617-1008627/" + \
                        "anies-usul-pemilik-lahan-terdampak-pembangunan-tol-dapat-bagian-saham").text
    
    start = time()
    Scraper(html)
    end = time()

    time_elapsed = end - start
    assert time_elapsed < 0.06
