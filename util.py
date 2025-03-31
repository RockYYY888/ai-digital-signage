# Copyright (c) 2025 Team2024.06
# All rights reserved.
#
# This file is part of Targeted Digital Signage.
# Licensed under the MIT license.
# See the LICENSE file in the project root for full license information.

import os
import sys


def get_resource_path(relative_path):
    """Retrieve the absolute path to a resource, handling both development and PyInstaller environments.

    This function resolves the correct path to a resource file, whether the script is running in a standard
    Python environment or bundled as a PyInstaller executable. When bundled, it uses the temporary
    extraction directory (`_MEIPASS`). Otherwise, it resolves the path relative to the script's location.

    Args:
        relative_path (str): The relative path to the resource file.

    Returns:
        str: The absolute path to the resource file.
    """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        return os.path.join(os.path.abspath("."), relative_path)