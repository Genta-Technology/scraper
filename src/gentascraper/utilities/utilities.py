"""
## This module contains utility functions that are used by other modules in the package.

Functions:
    is_numeric(string: str) -> bool: Checks if a string is numeric.

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

def is_numeric(string: str) -> bool:
    """
    Checks if a string is numeric.

    :param string: The string to check.
    :type string: str
    :return: True if the string is numeric, False otherwise.
    """
    # Use a regular expression to match a string that contains only digits, decimal point, or sign
    pattern = r'^[+-]?\d*\.?\d+$'
    # Return True if the string matches the pattern, False otherwise
    return bool(re.match(pattern, string))