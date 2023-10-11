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

from transformers import BertForQuestionAnswering, BertTokenizer


class Model:
    def __init__(self, checkpoint: str) -> None:
        self.model = BertForQuestionAnswering.from_pretrained(checkpoint)
        self.tokenizer = BertTokenizer.from_pretrained(checkpoint)