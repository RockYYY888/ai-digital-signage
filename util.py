import os
import sys


def get_resource_path(relative_path):
    """获取资源的绝对路径，兼容开发和打包环境"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        return os.path.join(os.path.dirname(__file__), relative_path)